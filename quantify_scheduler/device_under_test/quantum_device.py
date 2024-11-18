# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing the QuantumDevice object."""
from __future__ import annotations

import json
from typing import Any

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import InstrumentRefParameter, ManualParameter
from qcodes.utils import validators

from quantify_scheduler.backends.graph_compilation import (
    DeviceCompilationConfig,
    SerialCompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig
from quantify_scheduler.backends.types.common import (
    HardwareCompilationConfig,
    HardwareOptions,
)
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.device_under_test.edge import Edge
from quantify_scheduler.device_under_test.hardware_config import HardwareConfig
from quantify_scheduler.helpers.importers import (
    export_python_object_to_path_string,
    import_python_object_from_string,
)
from quantify_scheduler.json_utils import (
    JSONSerializableMixin,
    SchedulerJSONDecoder,
    SchedulerJSONEncoder,
)


class QuantumDevice(JSONSerializableMixin, Instrument):
    """
    The QuantumDevice directly represents the device under test (DUT).

    This contains a description of the connectivity to the control hardware as
    well as parameters specifying quantities like cross talk, attenuation and
    calibrated cable-delays. The QuantumDevice also contains references to
    individual DeviceElements, representations of elements on a device (e.g, a
    transmon qubit) containing the (calibrated) control-pulse parameters.

    This object can be used to generate configuration files for the compilation step
    from the gate-level to the pulse level description.
    These configuration files should be compatible with the
    :meth:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler.compile`
    function.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

        self.elements = ManualParameter(
            "elements",
            initial_value=list(),
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all elements that"
            " are located on this QuantumDevice.",
            instrument=self,
        )

        self.edges = ManualParameter(
            "edges",
            initial_value=list(),
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all the edges which connect the"
            " DeviceElements within this QuantumDevice",
            instrument=self,
        )

        self.instr_measurement_control = InstrumentRefParameter(
            "instr_measurement_control",
            docstring="A reference to the measurement control instrument.",
            vals=validators.MultiType(validators.Strings(), validators.Enum(None)),
            instrument=self,
        )

        self.instr_instrument_coordinator = InstrumentRefParameter(
            "instr_instrument_coordinator",
            docstring="A reference to the instrument_coordinator instrument.",
            vals=validators.MultiType(validators.Strings(), validators.Enum(None)),
            instrument=self,
        )

        self.cfg_sched_repetitions = ManualParameter(
            "cfg_sched_repetitions",
            initial_value=1024,
            docstring=(
                "The number of times execution of the schedule gets repeated when "
                "performing experiments, i.e. used to set the repetitions attribute of "
                "the Schedule objects generated."
            ),
            vals=validators.Ints(min_value=1),
            instrument=self,
        )

        self.keep_original_schedule = ManualParameter(
            "keep_original_schedule",
            initial_value=True,
            docstring=(
                "If `True`, the compiler will not modify the schedule argument. "
                "If `False`, the compilation modifies the schedule, thereby "
                "making the original schedule unusable for further usage; this "
                "improves compilation time. Warning: if `False`, the returned schedule "
                "references objects from the original schedule, please refrain from modifying "
                "the original schedule after compilation in this case!"
            ),
            vals=validators.Bool(),
            instrument=self,
        )

        self.hardware_config: HardwareConfig = HardwareConfig(instrument=self)
        """
        The input dictionary used to generate a valid HardwareCompilationConfig using
        :meth:`~.generate_hardware_compilation_config`.
        This configures the compilation from the quantum-device layer to the control-hardware layer.


        Useful methods to write and reload the configuration from a json file are
        :meth:`~.HardwareConfig.load_from_json_file` and
        :meth:`~.HardwareConfig.write_to_json_file`.

        """

        self.scheduling_strategy = ManualParameter(
            "scheduling_strategy",
            docstring="Scheduling strategy used to calculate absolute timing.",
            vals=validators.Enum("asap", "alap"),
            initial_value="asap",
        )

        # Store refs to prevent them from being garbage collected.
        self._instrument_references = {}

    def __getstate__(self) -> dict[str, Any]:  # type: ignore
        """
        Serializes :class:`~QuantumDevice` into a dict containing serialized :class:`~DeviceElement`
        and :class:`~Edge` objects plus ``cfg_sched_repetitions``.
        """
        data: dict[str, Any] = {"name": self.name}

        data["elements"] = {
            element_name: json.dumps(self.get_element(element_name), cls=SchedulerJSONEncoder)
            for element_name in self.elements()
        }

        data["edges"] = {
            edge_name: json.dumps(self.get_edge(edge_name), cls=SchedulerJSONEncoder)
            for edge_name in self.edges()
        }

        data["cfg_sched_repetitions"] = str(self.cfg_sched_repetitions())

        state = {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": data,
        }

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Deserializes a dict of serialized :class:`~DeviceElement` and :class:`~Edge` objects
        into a `QuantumDevice`.
        """
        self.__init__(state["data"]["name"])

        for element_name, serialized_element in state["data"]["elements"].items():
            self._instrument_references[element_name] = json.loads(
                serialized_element, cls=SchedulerJSONDecoder
            )
            self.add_element(self._instrument_references[element_name])

        for edge_name, serialized_edge in state["data"]["edges"].items():
            self._instrument_references[edge_name] = json.loads(
                serialized_edge, cls=SchedulerJSONDecoder
            )
            self.add_edge(self._instrument_references[edge_name])

        self.cfg_sched_repetitions(int(state["data"]["cfg_sched_repetitions"]))

    def to_json(self) -> str:
        """
        Convert the :class:`~QuantumDevice` data structure to a JSON string.
        Overrides the base mixin method to perform additional checks.

        Returns
        -------
        :
            The json string containing the serialized `QuantumDevice`.

        """
        # Check whether there are closed instruments that prevent serialization.
        device_instruments = []
        if hasattr(self, "elements"):
            device_instruments += self.elements()
        if hasattr(self, "edges"):
            device_instruments += self.edges()
        if not device_instruments:
            raise RuntimeError(
                f"Cannot serialize '{self.name}'. All attached instruments have been "
                f"closed and their information cannot be retrieved any longer."
            )

        closed_instruments = []
        for device_name in device_instruments:
            try:
                Instrument.find_instrument(device_name)
            except KeyError:
                closed_instruments.append(device_name)
        if closed_instruments:
            raise RuntimeError(
                f"Cannot serialize '{self.name}'. Instruments '{closed_instruments}' have "
                f"been closed and their information cannot be retrieved any longer. "
                f"If you do not wish to include these in the "
                f"serialization, please remove using `QuantumDevice.remove_element` or "
                f"`QuantumDevice.remove_edge`."
            )

        # Let the JSON mixin handle serialization.
        return super().to_json()

    def generate_compilation_config(self) -> SerialCompilationConfig:
        """Generate a config for use with a :class:`~.graph_compilation.QuantifyCompiler`."""
        return SerialCompilationConfig(
            name="QuantumDevice-generated SerialCompilationConfig",
            keep_original_schedule=self.keep_original_schedule(),
            device_compilation_config=self.generate_device_config(),
            hardware_compilation_config=self.generate_hardware_compilation_config(),
        )

    def generate_hardware_config(self) -> dict[str, Any]:
        """
        Generate a valid hardware configuration describing the quantum device.

        Returns
        -------
            The hardware configuration file used for compiling from the quantum-device
            layer to a hardware backend.

        .. warning:

            The config currently has to be specified by the user using the
            :code:`hardware_config` parameter.

        """
        return self.hardware_config()

    def generate_device_config(self) -> DeviceCompilationConfig:
        """
        Generate a device config.

        This config is used to compile from the quantum-circuit to the
        quantum-device layer.
        """
        clocks = {}
        elements_cfg = {}
        edges_cfg = {}

        # iterate over the elements on the device
        for element_name in self.elements():
            element = self.get_element(element_name)
            element_cfg = element.generate_device_config()
            clocks.update(element_cfg.clocks)
            elements_cfg.update(element_cfg.elements)

        # iterate over the edges on the device
        for edge_name in self.edges():
            edge = self.get_edge(edge_name)
            edge_cfg = edge.generate_edge_config()
            edges_cfg.update(edge_cfg)

        device_config = DeviceCompilationConfig(
            elements=elements_cfg,
            clocks=clocks,
            edges=edges_cfg,
            scheduling_strategy=self.scheduling_strategy(),
        )

        return device_config

    def generate_hardware_compilation_config(self) -> HardwareCompilationConfig | None:
        """
        Generate a hardware compilation config.

        The compilation config is used to compile from the quantum-device to the
        control-hardware layer.
        """
        hardware_config = self.hardware_config()
        if hardware_config is None:
            return None
        elif isinstance(hardware_config, HardwareCompilationConfig):
            # Hardware config is already a valid HardwareCompilationConfig DataStructure
            return hardware_config
        elif not any(
            [
                key in hardware_config
                for key in [
                    "config_type",
                    "hardware_description",
                    "hardware_options",
                    "connectivity",
                ]
            ]
        ):
            # Legacy support for the old hardware config dict:
            if (
                hardware_config["backend"]
                == "quantify_scheduler.backends.qblox_backend.hardware_compile"
            ):
                hardware_compilation_config = QbloxHardwareCompilationConfig.model_validate(
                    hardware_config
                )

            elif (
                hardware_config["backend"]
                == "quantify_scheduler.backends.zhinst_backend.compile_backend"
            ):
                compilation_passes = [
                    SimpleNodeConfig(
                        name="zhinst_compile_backend",
                        compilation_func=hardware_config["backend"],
                    ),
                ]
                hardware_compilation_config = HardwareCompilationConfig(
                    hardware_description={},
                    hardware_options=HardwareOptions(),
                    connectivity=hardware_config,
                    compilation_passes=compilation_passes,
                )

            else:
                compilation_passes = [
                    SimpleNodeConfig(
                        name="custom_hardware_backend",
                        compilation_func=hardware_config["backend"],
                    ),
                ]
                hardware_compilation_config = HardwareCompilationConfig(
                    hardware_description={},
                    hardware_options=HardwareOptions(),
                    connectivity=hardware_config,
                    compilation_passes=compilation_passes,
                )
        else:
            # Parse a (backend-specific) HardwareCompilationConfig
            if "backend" in hardware_config:
                raise ValueError(
                    f"`{HardwareCompilationConfig.__name__}` no longer takes a"
                    f" 'backend' field; instead, specify the 'config_type', which should"
                    " contain a string reference to the backend-specific datastructure"
                    " that should be parsed."
                )
            hardware_compilation_config_model = import_python_object_from_string(
                hardware_config["config_type"]
            )
            hardware_compilation_config = hardware_compilation_config_model.model_validate(
                hardware_config
            )

        return hardware_compilation_config

    def get_element(self, name: str) -> DeviceElement:
        """
        Return a :class:`~quantify_scheduler.device_under_test.device_element.DeviceElement`
        by name.

        Parameters
        ----------
        name
            The element name.

        Returns
        -------
        :
            The element.

        Raises
        ------
        KeyError
            If key ``name`` is not present in `self.elements`.

        """
        if name in self.elements():
            return self.find_instrument(name)  # type: ignore
        raise KeyError(f"'{name}' is not an element of {self.name}.")

    def add_element(
        self,
        element: DeviceElement,
    ) -> None:
        """
        Add an element to the elements collection.

        Parameters
        ----------
        element
            The element to add.

        Raises
        ------
        ValueError
            If a element with a duplicated name is added to the collection.
        TypeError
            If :code:`element` is not an instance of the base element.

        """
        if element.name in self.elements():
            raise ValueError(f"'{element.name}' has already been added.")

        if not isinstance(element, DeviceElement):
            raise TypeError(f"{repr(element)} is not a DeviceElement.")

        self.elements().append(element.name)  # list gets updated in place
        self._instrument_references[element.name] = element

    def remove_element(self, name: str) -> None:
        """
        Removes an element by name.

        Parameters
        ----------
        name
            The element name.

        """
        self.elements().remove(name)  # list gets updated in place

    def get_edge(self, name: str) -> Instrument:
        """
        Returns an edge by name.

        Parameters
        ----------
        name
            The edge name.

        Returns
        -------
        :
            The edge.

        Raises
        ------
        KeyError
            If key ``name`` is not present in ``self.edges``.

        """
        if name in self.edges():
            return self.find_instrument(name)
        raise KeyError(f"'{name}' is not an edge of {self.name}.")

    def add_edge(self, edge: Edge) -> None:
        """
        Add the edges.

        Parameters
        ----------
        edge
            The edge name connecting the elements. Has to follow the convention
            'element_0'-'element_1'

        """
        if edge.name in self.edges():
            raise ValueError(f"'{edge.name}' has already been added.")

        if not isinstance(edge, Edge):
            raise TypeError(f"{repr(edge)} is not an Edge.")

        self.edges().append(edge.name)
        self._instrument_references[edge.name] = edge

    def remove_edge(self, edge_name: str) -> None:
        """
        Remove an edge by name.

        Parameters
        ----------
        edge_name
            The edge name.

        """
        self.edges().remove(edge_name)  # list gets updated in place
