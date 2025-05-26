# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""The module contains definitions for device elements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ruamel.yaml as ry
from qcodes.instrument.base import Instrument

from quantify_scheduler.helpers.importers import export_python_object_to_path_string
from quantify_scheduler.json_utils import JSONSerializable
from quantify_scheduler.yaml_utils import YAMLSerializable

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import DeviceCompilationConfig


class DeviceElement(YAMLSerializable, JSONSerializable, Instrument):
    """
    Create a device element for managing parameters.

    The :class:`~DeviceElement` is responsible for compiling operations applied to that
    specific device element from the quantum-circuit to the quantum-device layer.
    """

    def __init__(self, name: str, **kwargs) -> None:
        if "-" in name or "_" in name:
            raise ValueError(
                f"Invalid DeviceElement name '{name}'. Hyphens and "
                f"underscores are not allowed due to naming conventions"
            )
        super().__init__(name, **kwargs)

    def __json_getstate__(self) -> dict[str, Any]:  # type: ignore
        """
        Serialize :class:`~DeviceElement` and derived classes.

        Serialization is performed by converting submodules into a dict containing
        the name of the device element and a dict for each submodule containing its
        parameter names and corresponding values.
        """
        snapshot = self.snapshot()

        element_data: dict[str, Any] = {"name": self.name}
        for submodule_name, submodule_data in snapshot["submodules"].items():
            element_data[submodule_name] = {
                name: data["value"] for name, data in submodule_data["parameters"].items()
            }

        state = {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "mode": "__init__",
            "data": element_data,
        }
        return state

    def __getstate__(self) -> dict[str, Any]:  # type: ignore[override]
        """Get the state of :class:`~DeviceElement` (used for YAML serialization)."""
        snapshot = self.snapshot()

        element_data: dict[str, Any] = {"name": self.name}
        for submodule_name, submodule_data in snapshot["submodules"].items():
            element_data[submodule_name] = {
                name: data["value"] for name, data in submodule_data["parameters"].items()
            }

        return element_data

    @classmethod
    def from_yaml(cls, constructor: ry.Constructor, node: ry.MappingNode) -> Instrument:
        """YAML loading logic."""
        if isinstance(constructor, ry.RoundTripConstructor):
            data = ry.CommentedMap()
            constructor.construct_mapping(node, maptyp=data, deep=True)
        else:
            data = constructor.construct_mapping(node, deep=True)
        # The name attribute needs to be passed explicitly and separately to this `Instrument`.
        return cls(name=data.pop("name"), **data)

    def generate_device_config(self) -> DeviceCompilationConfig:
        """Generate the device configuration."""
        raise NotImplementedError
