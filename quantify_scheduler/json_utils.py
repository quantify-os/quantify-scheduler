# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing quantify JSON utilities."""
from __future__ import annotations

import functools
import json
import pathlib
import sys
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, List, Type, Union

import fastjsonschema
import numpy as np
from qcodes import Instrument

from quantify_scheduler import enums
from quantify_scheduler.helpers import inspect as inspect_helpers
from quantify_scheduler.helpers.importers import import_python_object_from_string

current_python_version = sys.version_info

lru_cache = functools.lru_cache(maxsize=200)

DEFAULT_TYPES = [
    complex,
    float,
    int,
    bool,
    str,
    np.ndarray,
    np.complex128,
    np.int32,
    np.uint32,
    np.int64,
]


def validate_json(data, schema):
    """Validate schema using jsonschema-rs."""
    return fastjsonschema.validate(schema, data)


def load_json_schema(relative_to: Union[str, pathlib.Path], filename: str):
    """
    Load a JSON schema from file. Expects a 'schemas' directory in the same directory
    as ``relative_to``.

    .. tip::

        Typical usage of the form
        ``schema = load_json_schema(__file__, 'definition.json')``

    Parameters
    ----------
    relative_to
        the file to begin searching from
    filename
        the JSON file to load

    Returns
    -------
    dict
        the schema
    """
    path = pathlib.Path(relative_to).resolve().parent.joinpath("schemas", filename)
    with path.open(mode="r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache
def load_json_validator(
    relative_to: Union[str, pathlib.Path], filename: str
) -> Callable:
    """
    Load a JSON validator from file. Expects a 'schemas' directory in the same directory
    as ``relative_to``.


    Parameters
    ----------
    relative_to
        the file to begin searching from
    filename
        the JSON file to load

    Returns
    -------
    Callable
        The validator
    """
    definition = load_json_schema(relative_to, filename)
    validator = fastjsonschema.compile(definition, handlers={}, formats={})
    return validator


class UnknownDeserializationTypeError(Exception):
    """Raised when an unknown deserialization type is encountered."""


class JSONSchemaValMixin:
    """
    A mixin that adds validation utilities to classes that have
    a data attribute like a :class:`UserDict` based on JSONSchema.

    This requires the class to have a class variable "schema_filename"
    """

    @classmethod
    def is_valid(cls, object_to_be_validated) -> bool:
        """
        Checks if the object is valid according to its schema.

        Raises
        ------
        fastjsonschema.JsonSchemaException
            if the data is invalid

        Returns
        -------
        :

        """
        validator_method = load_json_validator(__file__, cls.schema_filename)
        validator_method(object_to_be_validated.data)
        return True  # if no exception was raised during validation


class SchedulerJSONDecoder(json.JSONDecoder):
    """
    The Quantify Scheduler JSONDecoder.

    The SchedulerJSONDecoder is used to convert a string with JSON content into
    instances of classes in quantify-scheduler.

    For a few types, :data:`~.DEFAULT_TYPES` contains the mapping from type name to the
    python object. This dictionary can be expanded with classes from modules specified
    in the keyword argument ``modules``.

    Classes not contained in :data:`~.DEFAULT_TYPES` by default must implement
    ``__getstate__``, such that it returns a dictionary containing at least the keys
    ``"deserialization_type"`` and ``"data"``, and ``__setstate__``, which should be
    able to parse the data from ``__getstate__``.

    The value of ``"deserialization_type"`` must be either the name of the class
    specified in :data:`~.DEFAULT_TYPES` or the fully qualified name of the class, which
    can be obtained from
    :func:`~quantify_scheduler.helpers.importers.export_python_object_to_path_string`.

    Keyword Arguments
    -----------------
    modules : List[ModuleType], *optional*
        A list of custom modules containing serializable classes, by default []
    """  # noqa: D416

    _classes: Dict[str, Type[Any]]

    def __init__(self, *args, **kwargs) -> None:
        extended_modules: List[ModuleType] = kwargs.pop("modules", [])
        invalid_modules = list(
            filter(lambda o: not isinstance(o, ModuleType), extended_modules)
        )
        if invalid_modules:
            raise ValueError(
                f"Attempting to create a Schedule decoder class SchedulerJSONDecoder. "
                f"The following modules provided are not an instance of the ModuleType:"
                f" {invalid_modules} ."
            )

        super().__init__(
            object_hook=self.custom_object_hook,
            *args,
            **kwargs,
        )

        self._classes = inspect_helpers.get_classes(*[enums, *extended_modules])
        self._classes.update({t.__name__: t for t in DEFAULT_TYPES})

    def decode_dict(
        self, obj: Dict[str, Any]
    ) -> Union[Dict[str, Any], np.ndarray, type]:
        """
        Returns the deserialized JSON dictionary.

        Parameters
        ----------
        obj
            The dictionary to deserialize.

        Returns
        -------
        :
            The deserialized result.
        """
        # If "deserialization_type" is present in `obj` it means the object was
        # serialized using `__getstate__` and should be deserialized using
        # `__setstate__`.
        if "deserialization_type" in obj:
            try:
                class_type = self._get_type_from_string(obj["deserialization_type"])
            except UnknownDeserializationTypeError as exc:
                raise UnknownDeserializationTypeError(
                    f"Object '{obj}' cannot be deserialized to type '{obj['deserialization_type']}'"
                ) from exc

            if "mode" in obj and obj["mode"] == "__init__":
                if class_type == np.ndarray:
                    return np.array(obj["data"])
                if issubclass(class_type, Instrument):
                    return class_type(**obj["data"])
                return class_type(obj["data"])

            if "mode" in obj and obj["mode"] == "type":
                return class_type

            new_obj = class_type.__new__(class_type)
            new_obj.__setstate__(obj)
            return new_obj

        return obj

    def custom_object_hook(self, obj: object) -> object:
        """
        The ``object_hook`` hook will be called with the result of every JSON object
        decoded and its return value will be used in place of the given ``dict``.

        Parameters
        ----------
        obj
            A pair of JSON objects.

        Returns
        -------
        :
            The deserialized result.
        """
        if isinstance(obj, dict):
            return self.decode_dict(obj)
        return obj

    def _get_type_from_string(self, deserialization_type: str) -> Type:
        """
        Get the python type based on the description string.

        The following methods are tried, in order:

            1. Try to find the string in :data:`~.DEFAULT_TYPES` or the extended modules
                passed to this class' initializer.
            2. Try to import the type. This works only if ``deserialization_type`` is
                formatted as a dot-separated path to the type. E.g.
                ``quantify_scheduler.json_utils.SchedulerJSONDecoder``.
            3. (deprecated) Try to find the class by its ``__name__`` in a predefined
                selection of types present in ``quantify_scheduler``.

        Parameters
        ----------
        deserialization_type
            Description of a type.

        Raises
        ------
        UnknownDeserializationTypeError
            If the type cannot be found by any of the methods described.

        Returns
        -------
        Type
            The ``Type`` found.
        """
        try:
            return self._classes[deserialization_type]
        except KeyError:
            pass

        try:
            return import_python_object_from_string(deserialization_type)
        except (AttributeError, ModuleNotFoundError, ValueError):
            pass

        try:
            return _get_type_from_string_deprecated(deserialization_type)
        except KeyError:
            raise UnknownDeserializationTypeError(
                f"Type '{deserialization_type}' is not a known deserialization type."
            )


def _get_type_from_string_deprecated(deserialization_type: str) -> Type:
    # Use local import to void Error('Operation' from partially initialized module
    # 'quantify_scheduler')

    from quantify_scheduler import resources
    from quantify_scheduler.backends.qblox.operations.stitched_pulse import (
        StitchedPulse as QbloxStitchedPulse,
    )
    from quantify_scheduler.device_under_test import transmon_element
    from quantify_scheduler.device_under_test.composite_square_edge import (
        CompositeSquareEdge,
    )
    from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
    from quantify_scheduler.operations import (
        acquisition_library,
        gate_library,
        nv_native_library,
        operation,
        pulse_library,
        shared_native_library,
    )
    from quantify_scheduler.schedules.schedule import AcquisitionMetadata, Schedulable

    classes = inspect_helpers.get_classes(
        operation,
        transmon_element,
        acquisition_library,
        gate_library,
        pulse_library,
        nv_native_library,
        shared_native_library,
        resources,
    )
    classes.update(
        {
            c.__name__: c
            for c in [
                AcquisitionMetadata,
                Schedulable,
                QuantumDevice,
                CompositeSquareEdge,
            ]
        }
    )
    classes.update({"StitchedPulse": QbloxStitchedPulse})

    class_type = classes[deserialization_type]
    # Only warn if we succeed
    warnings.warn(
        "Having only the class name as the deserialization type is deprecated "
        "and this feature will be removed in quantify-scheduler >= 0.20.0. "
        "Please re-serialize the object to use the fully qualified class name.",
        FutureWarning,
    )
    return class_type


class SchedulerJSONEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder which encodes a Quantify Scheduler object into a JSON file format
    string.
    """

    def default(self, o):
        """
        Overloads the json.JSONEncoder default method that returns a serializable
        object. It will try 3 different serialization methods which are, in order,
        check if the object is to be serialized to a string using repr. If not, try
        to use ``__getstate__``. Finally, try to serialize the ``__dict__`` property.
        """
        if isinstance(o, (complex, np.int32, np.complex128, np.int64, enums.BinMode)):
            return {
                "deserialization_type": type(o).__name__,
                "mode": "__init__",
                "data": str(o),
            }
        if isinstance(o, (np.ndarray,)):
            return {
                "deserialization_type": type(o).__name__,
                "mode": "__init__",
                "data": list(o),
            }
        if o in DEFAULT_TYPES:
            return {"deserialization_type": o.__name__, "mode": "type"}
        if hasattr(o, "__getstate__"):
            return o.__getstate__()
        if hasattr(o, "__dict__"):
            return o.__dict__

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)
