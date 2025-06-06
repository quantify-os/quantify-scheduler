# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""The module contains definitions for edges."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ruamel.yaml as ry
from qcodes.instrument.base import Instrument

from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.helpers.importers import export_python_object_to_path_string
from quantify_scheduler.json_utils import JSONSerializable
from quantify_scheduler.yaml_utils import YAMLSerializable

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import OperationCompilationConfig


class Edge(YAMLSerializable, JSONSerializable, Instrument):
    """
    Create an Edge.

    This class encapsulates the connection information between DeviceElements in the
    QuantumDevice. It provides an interface for the QuantumDevice to generate the
    edge information for use in the device compilation step. See
    :class:`quantify_scheduler.device_under_test.composite_square_edge` for an example
    edge implementation.
    """

    def __init__(
        self,
        parent_element_name: str,
        child_element_name: str,
        **kwargs,
    ) -> None:
        self._parent_element_name = parent_element_name
        self._child_element_name = child_element_name

        super().__init__(name=f"{parent_element_name}_{child_element_name}", **kwargs)

    def __json_getstate__(self) -> dict[str, Any]:  # type: ignore
        """
        Serialize :class:`~Edge` into a dictionary.

        Serialization is performed by converting submodules of the object into
        the dictionary containing the parent and child element names of
        this edge and a dict for each submodule containing its parameter names
        and corresponding values.
        """
        snapshot = self.snapshot()

        edge_data: dict[str, Any] = {
            "parent_element_name": self._parent_element_name,
            "child_element_name": self._child_element_name,
        }

        for submodule_name, submodule_data in snapshot["submodules"].items():
            edge_data[submodule_name] = {
                name: data["value"] for name, data in submodule_data["parameters"].items()
            }

        state = {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "mode": "__init__",
            "data": edge_data,
        }
        return state

    def __getstate__(self) -> dict[str, Any]:  # type: ignore[override]
        """Get the state of :class:`~Edge` (used for YAML serialization)."""
        snapshot = self.snapshot()

        edge_data: dict[str, Any] = {
            "parent_element_name": self._parent_element_name,
            "child_element_name": self._child_element_name,
        }

        for submodule_name, submodule_data in snapshot["submodules"].items():
            edge_data[submodule_name] = {
                name: data["value"] for name, data in submodule_data["parameters"].items()
            }

        return edge_data

    @classmethod
    def from_yaml(cls, constructor: ry.Constructor, node: ry.MappingNode) -> Instrument:
        """YAML loading logic."""
        if isinstance(constructor, ry.RoundTripConstructor):
            data = ry.CommentedMap()
            constructor.construct_mapping(node, maptyp=data, deep=True)
        else:
            data = constructor.construct_mapping(node, deep=True)

        if "parent_element_name" in data:
            return cls(**data)
        else:
            return cls(name=data.pop("name"), **data)

    @property
    def parent_device_element(self) -> Instrument:
        """The parent DeviceElement connected by the edge."""
        return self.find_instrument(name=self._parent_element_name, instrument_class=DeviceElement)

    @property
    def child_device_element(self) -> Instrument:
        """The child DeviceElement connected by the edge."""
        return self.find_instrument(name=self._child_element_name, instrument_class=DeviceElement)

    def generate_edge_config(self) -> dict[str, dict[str, OperationCompilationConfig]]:
        """
        Generate the device configuration for an edge.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        raise NotImplementedError
