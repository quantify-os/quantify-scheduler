# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing quantify YAML utilities."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from typing_extensions import Self

import numpy as np
import ruamel.yaml as ry
from qcodes.instrument import Instrument
from qcodes.instrument.instrument_meta import InstrumentMeta

from quantify_core.data.handling import get_datadir

if TYPE_CHECKING:
    from pathlib import Path


def represent_enum(representer: ry.Representer, data: Enum) -> ry.ScalarNode:
    """Provide a value-based representation for Enum instances in YAML."""
    # An enum value can be any scalar type; if there's no matching representer, default to str.
    repr_method = representer.yaml_representers.get(type(data.value), str)
    return repr_method(representer, data.value)


def represent_ndarray(representer: ry.Representer, data: np.ndarray) -> ry.MappingNode:
    """Represent a NumPy array as a mapping, including its type and shape."""
    node = representer.represent_mapping(
        "!numpy.ndarray",
        {
            "dtype": str(data.dtype),
            "shape": data.shape,
            "data": data.tolist(),
        },
    )
    # Set flow_style for "shape" and "data", so they aren't serialized one element per line.
    for key in node.value:
        if key[0].value in ("shape", "data"):
            key[1].flow_style = True
    return node


def construct_ndarray(constructor: ry.Constructor, node: ry.MappingNode) -> np.ndarray:
    """Restore a NumPy array from a mapping with its proper type and shape."""
    if isinstance(constructor, ry.RoundTripConstructor):
        data = ry.CommentedMap()
        constructor.construct_mapping(node, maptyp=data, deep=True)
    else:
        data = constructor.construct_mapping(node, deep=True)
    return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])


# The "rt" (round-trip) loader can be advantageous compared to the "safe" loader,
# particularly when working with complex, nested Python classes:
# - Comments: it retains comments present in the YAML file
# - Key Order: it preserves the order of keys in YAML mappings (dictionaries)
# - Formatting: it attempts to maintain the original indentation and whitespace
# - Anchors and Aliases: it preserves YAML anchors and aliases
# - Tags: it handles tags, including custom tags that are used to represent your custom classes
yaml = ry.YAML(typ="rt")

# Support Enum and its subclasses
yaml.representer.add_multi_representer(Enum, represent_enum)
# Support NumPy arrays
yaml.representer.add_representer(np.ndarray, represent_ndarray)
yaml.constructor.add_constructor("!numpy.ndarray", construct_ndarray)


class YAMLSerializableMeta(InstrumentMeta):
    """
    Metaclass to register mixed in classes with the YAML parser.

    Needs to inherit from ``InstrumentMeta`` due to metaclass conflict.
    """

    def __init__(cls, name, bases, dct) -> None:  # noqa: ANN001, N805
        """
        Register the class with the YAML parser so that (de)serialization can happen
        automatically.
        """
        super().__init__(name, bases, dct)
        yaml.register_class(cls)


class YAMLSerializable(metaclass=YAMLSerializableMeta):
    """
    Mixin to allow (de)serialization of instruments from/to YAML.

    NOTE: Only works with ``Instrument`` subclasses, for others use `@yaml.register_class`.

    NOTE: `to_yaml` and `from_yaml` methods cannot be created because they would be found
      and used by the ``ruamel.yaml`` representers and constructors.
    """

    def to_yaml_file(
        self,
        path: str | Path | None = None,
        add_timestamp: bool = True,
    ) -> str:
        """
        Convert the object's data structure to a YAML string and store it in a file.

        Parameters
        ----------
        path
            The path to the directory where the file is created. Default
            is `None`, in which case the file will be saved in the directory
            determined by :func:`~quantify_core.data.handling.get_datadir()`.

        add_timestamp
            Specify whether to append timestamp to the filename.
            Default is True.

        Returns
        -------
            The name of the file containing the serialized object.

        """
        if path is None:
            path = get_datadir()

        name = getattr(self, "name")  # noqa: B009 This is to shut up the linter about self.name

        if add_timestamp:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%Z")
            filename = os.path.join(path, f"{name}_{timestamp}.yaml")
        else:
            filename = os.path.join(path, f"{name}.yaml")

        with open(filename, "w") as file:
            yaml.dump(self, file)

        return filename

    @classmethod
    def from_yaml_file(cls, filename: str | Path) -> Self:
        """
        Read YAML data from a file and convert it to an instance of the attached class.

        Parameters
        ----------
        filename
            The name of the file containing the serialized object.

        Returns
        -------
            The deserialized object.

        """
        with open(filename) as file:
            deserialized_obj = yaml.load(file)
        return deserialized_obj

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        When deserializing an ``Instrument``, add it to the qcodes global registry.

        Must be invoked by subclasses if :meth:`__setstate__` is overridden.
        """
        if isinstance(self, Instrument):
            type(self).record_instance(self)


__all__ = ["YAMLSerializable", "yaml"]
