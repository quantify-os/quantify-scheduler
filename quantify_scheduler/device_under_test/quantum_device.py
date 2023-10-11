# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing the QuantumDevice object.
"""
from __future__ import annotations

from typing import Any, Dict
from functools import partial

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import InstrumentRefParameter, ManualParameter
from qcodes.utils import validators
from quantify_scheduler.backends.circuit_to_device import compile_circuit_to_device
from quantify_scheduler.backends.qblox_backend import hardware_compile as qblox_backend
from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig
from quantify_scheduler.backends.zhinst_backend import compile_backend as zhinst_backend
from quantify_scheduler.backends.zhinst_backend import ZIHardwareCompilationConfig
from quantify_scheduler.backends.graph_compilation import (
    SerialCompilationConfig,
    SimpleNodeConfig,
    DeviceCompilationConfig,
)
from quantify_scheduler.backends.types.common import HardwareCompilationConfig
from quantify_scheduler.backends.qblox_backend import (
    compile_long_square_pulses_to_awg_offsets,
)
from quantify_scheduler.compilation import (
    determine_absolute_timing,
    flatten_schedule,
    resolve_control_flow,
)
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.device_under_test.edge import Edge


class QuantumDevice(Instrument):
    """
    The QuantumDevice directly represents the device under test (DUT) and contains a
    description of the connectivity to the control hardware as well as parameters
    specifying quantities like cross talk, attenuation and calibrated cable-delays.
    The QuantumDevice also contains references to individual DeviceElements,
    representations of elements on a device (e.g, a transmon qubit) containing
    the (calibrated) control-pulse parameters.

    This object can be used to generate configuration files for the compilation step
    from the gate-level to the pulse level description.
    These configuration files should be compatible with the
    :meth:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler.compile` function.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

        self.elements = ManualParameter(
            "elements",
            initial_value=dict(),
            vals=validators.Dict(),
            docstring="A dictionary with the names of all elements located on this"
            "QuantumDevice as keys and the corresponding `DeviceElement` objects as values.",
            instrument=self,
        )

        self.edges = ManualParameter(
            "edges",
            initial_value=dict(),
            vals=validators.Dict(),
            docstring="A dictionary with the names of all the edges which connect the"
            " `DeviceElement`s within this QuantumDevice as keys and the corresponding"
            " `Edge` objects as values.",
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

        self.hardware_config = ManualParameter(
            "hardware_config",
            docstring=(
                "The input dictionary used to generate a valid HardwareCompilationConfig "
                "using quantum_device.generate_hardware_compilation_config(). This configures "
                "the compilation from the quantum-device layer to the control-hardware layer."
            ),
            vals=validators.Dict(),
            initial_value=None,
            instrument=self,
        )

        self.scheduling_strategy = ManualParameter(
            "scheduling_strategy",
            docstring=("Scheduling strategy used to calculate absolute timing."),
            vals=validators.Enum("asap", "alap"),
            initial_value="asap",
        )

    def generate_compilation_config(self) -> SerialCompilationConfig:
        """
        Generates a compilation config for use with a
        :class:`~.graph_compilation.QuantifyCompiler`.
        """

        # Part that is always the same
        dev_cfg = self.generate_device_config()
        compilation_passes = [
            SimpleNodeConfig(
                name="circuit_to_device",
                compilation_func=dev_cfg.backend,
            ),
            SimpleNodeConfig(
                name="set_pulse_and_acquisition_clock",
                compilation_func="quantify_scheduler.backends.circuit_to_device."
                + "set_pulse_and_acquisition_clock",
            ),
            SimpleNodeConfig(
                name="resolve_control_flow",
                compilation_func=resolve_control_flow,
            ),
            SimpleNodeConfig(
                name="determine_absolute_timing",
                compilation_func=partial(
                    determine_absolute_timing, keep_original_schedule=False
                ),
            ),
            SimpleNodeConfig(
                name="flatten",
                compilation_func=flatten_schedule,
            ),
        ]

        # If statements to support the different hardware compilation configs.
        hw_comp_cfg = self.generate_hardware_compilation_config()
        if hw_comp_cfg is None:
            backend_name = "Device compiler"
        elif hw_comp_cfg.backend == qblox_backend:
            backend_name = "Qblox compiler"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="compile_long_square_pulses_to_awg_offsets",
                    compilation_func=compile_long_square_pulses_to_awg_offsets,
                )
            )
            compilation_passes.append(
                SimpleNodeConfig(
                    name="qblox_hardware_compile",
                    compilation_func=hw_comp_cfg.backend,
                )
            )
        elif hw_comp_cfg.backend == zhinst_backend:
            backend_name = "Zhinst compiler"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="zhinst_hardware_compile",
                    compilation_func=hw_comp_cfg.backend,
                )
            )

        else:
            backend_name = "Custom compiler"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="custom_hardware_compile",
                    compilation_func=hw_comp_cfg.backend,
                )
            )

        compilation_config = SerialCompilationConfig(
            name=backend_name,
            device_compilation_config=dev_cfg,
            hardware_compilation_config=hw_comp_cfg,
            compilation_passes=compilation_passes,
        )

        return compilation_config

    def generate_hardware_config(self) -> Dict[str, Any]:
        """
        Generates a valid hardware configuration describing the quantum device.

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
        Generates a device config to compile from the quantum-circuit to the
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
            backend=partial(compile_circuit_to_device, keep_original_schedule=False),
            elements=elements_cfg,
            clocks=clocks,
            edges=edges_cfg,
            scheduling_strategy=self.scheduling_strategy(),
        )

        return device_config

    def generate_hardware_compilation_config(self) -> HardwareCompilationConfig | None:
        """
        Generates a hardware compilation config to compile from the quantum-device to the
        control-hardware layer.
        """

        hardware_config = self.hardware_config()
        if hardware_config is None:
            return None

        if (
            hardware_config["backend"]
            == "quantify_scheduler.backends.qblox_backend.hardware_compile"
        ):
            if not any(
                [
                    key in hardware_config
                    for key in [
                        "hardware_description",
                        "hardware_options",
                        "connectivity",
                    ]
                ]
            ):
                # Legacy support for the old hardware config dict:
                hardware_compilation_config = QbloxHardwareCompilationConfig(
                    backend=hardware_config["backend"],
                    hardware_description={},
                    hardware_options={},
                    connectivity=hardware_config,
                )
            else:
                hardware_compilation_config = QbloxHardwareCompilationConfig.parse_obj(
                    hardware_config
                )
        elif (
            hardware_config["backend"]
            == "quantify_scheduler.backends.zhinst_backend.compile_backend"
        ):
            if not any(
                [
                    key in hardware_config
                    for key in [
                        "hardware_description",
                        "hardware_options",
                        "connectivity",
                    ]
                ]
            ):
                # Legacy support for the old hardware config dict:
                hardware_compilation_config = ZIHardwareCompilationConfig(
                    backend=hardware_config["backend"],
                    hardware_description={},
                    hardware_options={},
                    connectivity=hardware_config,
                )
            else:
                hardware_compilation_config = ZIHardwareCompilationConfig.parse_obj(
                    hardware_config
                )
        else:
            if not any(
                [
                    key in hardware_config
                    for key in [
                        "hardware_description",
                        "hardware_options",
                        "connectivity",
                    ]
                ]
            ):
                # Legacy support for the old hardware config dict:
                hardware_compilation_config = HardwareCompilationConfig(
                    backend=hardware_config["backend"],
                    hardware_description={},
                    hardware_options={},
                    connectivity=hardware_config,
                )
            else:
                hardware_compilation_config = HardwareCompilationConfig.parse_obj(
                    hardware_config
                )

        return hardware_compilation_config

    def get_element(self, name: str) -> DeviceElement:
        """
        Returns a
        :class:`~quantify_scheduler.device_under_test.device_element.DeviceElement`
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
            If key `name` is not present in `self.elements`.
        """
        if name in self.elements():
            element = self.elements()[name]
            if not Instrument.is_valid(element):
                raise ValueError(
                    f"{element} of {self.name} is not a valid device element."
                    f" Please re-create and re-add it."
                )
            return element
        raise KeyError(f"'{name}' is not an element of {self.name}.")

    def add_element(
        self,
        element: DeviceElement,
    ) -> None:
        """
        Adds an element to the elements collection.

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
            if Instrument.is_valid(self.elements()[element.name]):
                # Only raise an error if the element is still valid
                raise ValueError(f"'{element.name}' has already been added.")

        if not isinstance(element, DeviceElement):
            raise TypeError(f"{repr(element)} is not a DeviceElement.")

        self.elements()[element.name] = element  # dict gets updated in place

    def remove_element(self, name: str) -> None:
        """
        Removes a element by name.

        Parameters
        ----------
        name
            The element name.
        """

        self.elements().pop(name)  # dict gets updated in place

    def get_edge(self, name: str) -> Instrument:
        """
        Returns a edge by name.

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
            If key `name` is not present in `self.edges`.
        """
        if name in self.edges():
            edge = self.edges()[name]
            if not Instrument.is_valid(edge):
                raise ValueError(
                    f"{edge} of {self.name} is not a valid edge."
                    f" Please re-create and re-add it."
                )
            return edge
        raise KeyError(f"'{name}' is not an edge of {self.name}.")

    def add_edge(self, edge: Edge) -> None:
        """
        Adds the edges.

        Parameters
        ----------
        edge
            The edge name connecting the elements. Has to follow the convention
            'element_0'-'element_1'
        """
        if edge.name in self.edges():
            if Instrument.is_valid(self.edges()[edge.name]):
                # Only raise an error if the edge is still valid
                raise ValueError(f"'{edge.name}' has already been added.")

        if not isinstance(edge, Edge):
            raise TypeError(f"{repr(edge)} is not a Edge.")

        self.edges()[edge.name] = edge  # dict gets updated in place

    def remove_edge(self, edge_name: str) -> None:
        """
        Removes an edge by name.

        Parameters
        ----------
        edge_name
            The edge name.
        """

        self.edges().pop(edge_name)  # dict gets updated in place
