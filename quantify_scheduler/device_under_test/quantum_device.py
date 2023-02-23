# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing the QuantumDevice object.
"""

from typing import Any, Dict

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import InstrumentRefParameter, ManualParameter
from qcodes.utils import validators
from quantify_scheduler.backends.circuit_to_device import (
    DeviceCompilationConfig,
    compile_circuit_to_device,
)
from quantify_scheduler.backends.graph_compilation import (
    SerialCompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.compilation import determine_absolute_timing
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
    :func:`~quantify_scheduler.compilation.qcompile` function.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

        self.add_parameter(
            "elements",
            initial_value=list(),
            parameter_class=ManualParameter,
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all elements that"
            " are located on this QuantumDevice.",
        )

        self.add_parameter(
            "edges",
            initial_value=list(),
            parameter_class=ManualParameter,
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all the edges which connect the"
            " DeviceElements within this QuantumDevice",
        )

        self.add_parameter(
            "instr_measurement_control",
            docstring="A reference to the measurement control instrument.",
            parameter_class=InstrumentRefParameter,
            vals=validators.MultiType(validators.Strings(), validators.Enum(None)),
        )

        self.add_parameter(
            "instr_instrument_coordinator",
            docstring="A reference to the instrument_coordinator instrument.",
            parameter_class=InstrumentRefParameter,
            vals=validators.MultiType(validators.Strings(), validators.Enum(None)),
        )

        self.add_parameter(
            "cfg_sched_repetitions",
            initial_value=1024,
            parameter_class=ManualParameter,
            docstring=(
                "The number of times execution of the schedule gets repeated when "
                "performing experiments, i.e. used to set the repetitions attribute of "
                "the Schedule objects generated."
            ),
            vals=validators.Ints(min_value=1),
        )

        self.add_parameter(
            "hardware_config",
            docstring="The hardware configuration file used for compiling from the "
            "quantum-device layer to a hardware backend.",
            parameter_class=ManualParameter,
            vals=validators.Dict(),
            initial_value=None,
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
                compilation_options=dev_cfg,
            ),
            SimpleNodeConfig(
                name="set_pulse_and_acquisition_clock",
                compilation_func="quantify_scheduler.backends.circuit_to_device."
                + "set_pulse_and_acquisition_clock",
                compilation_options=dev_cfg,
            ),
            SimpleNodeConfig(
                name="determine_absolute_timing",
                compilation_func=determine_absolute_timing,
            ),
        ]

        # If statements to support the different (currently unstructured) hardware
        # configs.
        hardware_config = self.generate_hardware_config()
        if hardware_config is None:
            backend_name = "Device compiler"
        elif (
            hardware_config["backend"]
            == "quantify_scheduler.backends.qblox_backend.hardware_compile"
        ):
            backend_name = "Qblox compiler"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="qblox_hardware_compile",
                    compilation_func=hardware_config["backend"],
                    compilation_options=hardware_config,
                )
            )
        elif (
            hardware_config["backend"]
            == "quantify_scheduler.backends.zhinst_backend.compile_backend"
        ):
            backend_name = "Zhinst compiler"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="zhinst_hardware_compile",
                    compilation_func=hardware_config["backend"],
                    compilation_options=hardware_config,
                )
            )

        else:
            backend_name = "Custom compiler"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="custom_hardware_compile",
                    compilation_func=hardware_config["backend"],
                    compilation_options=hardware_config,
                )
            )

        compilation_config = SerialCompilationConfig(
            name=backend_name,
            device_compilation_config=dev_cfg,
            hardware_options=[],
            connectivity=hardware_config,
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


        The hardware config should be valid input for the
        :func:`quantify_scheduler.compilation.qcompile` function.

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
            backend=compile_circuit_to_device,
            elements=elements_cfg,
            clocks=clocks,
            edges=edges_cfg,
        )

        return device_config

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
            return self.find_instrument(name)
        raise KeyError(f"'{name}' is not a element of {self.name}.")

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
            raise ValueError(f"'{element.name}' has already been added.")

        if not isinstance(element, DeviceElement):
            raise TypeError(f"{repr(element)} is not a DeviceElement.")

        self.elements().append(element.name)  # list gets updated in place

    def remove_element(self, name: str) -> None:
        """
        Removes a element by name.

        Parameters
        ----------
        name
            The element name.
        """

        self.elements().remove(name)  # list gets updated in place

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
            return self.find_instrument(name)
        raise KeyError(f"'{name}' is not a edge of {self.name}.")

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
            raise ValueError(f"'{edge.name}' has already been added.")

        if not isinstance(edge, Edge):
            raise TypeError(f"{repr(edge)} is not a Edge.")

        self.edges().append(edge.name)

    def remove_edge(self, edge_name: str) -> None:
        """
        Removes an edge by name.

        Parameters
        ----------
        edge_name
            The edge name.
        """

        self.edges().remove(edge_name)  # list gets updated in place
