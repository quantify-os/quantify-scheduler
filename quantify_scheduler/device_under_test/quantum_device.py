# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing the QuantumDevice object.
"""

from typing import Any, Dict

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import InstrumentRefParameter, ManualParameter
from qcodes.utils import validators

from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig


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
            "components",
            initial_value=list(),
            parameter_class=ManualParameter,
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all elements that"
            " are located on this QuantumDevice.",
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
        )

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

        .. note:

            The config currently does not support two-qubit gates.
        """

        clocks = {}
        elements_cfg = {}

        # iterate over the elements on the device
        for element_name in self.components():
            element = self.get_component(element_name)
            element_cfg = element.generate_device_config()
            clocks.update(element_cfg.clocks)
            elements_cfg.update(element_cfg.elements)

        # iterate over the edges on the device
        edges_cfg: dict = {}
        # FIXME: add support for operations acting on edges.

        device_config = DeviceCompilationConfig(
            backend="quantify_scheduler.backends"
            ".circuit_to_device.compile_circuit_to_device",
            elements=elements_cfg,
            clocks=clocks,
            edges=edges_cfg,
        )

        return device_config

    def get_component(self, name: str) -> Instrument:
        """
        Returns a component by name.

        Parameters
        ----------
        name
            The component name.

        Returns
        -------
        :
            The component.

        Raises
        ------
        KeyError
            If key `name` is not present in `self.components`.
        """
        if name in self.components():
            return self.find_instrument(name)
        raise KeyError(f"'{name}' is not a component of {self.name}.")

    def add_component(
        self,
        component: Instrument,
    ) -> None:
        """
        Adds a component to the components collection.

        Parameters
        ----------
        component
            The component to add.

        Raises
        ------
        ValueError
            If a component with a duplicated name is added to the collection.
        TypeError
            If :code:`component` is not an instance of the base component.
        """
        if component.name in self.components():
            raise ValueError(f"'{component.name}' has already been added.")

        if not isinstance(component, Instrument):
            # FIXME: check if it is also a valid device element. # pylint: disable=fixme
            # This requires a base class for device elements that does not exist yet.
            # See also `InstrumentCoordinatorComponentBase`.
            raise TypeError(f"{repr(component)} is not a QCoDeS instrument.")

        self.components().append(component.name)  # list gets updated in place

    def remove_component(self, name: str) -> None:
        """
        Removes a component by name.

        Parameters
        ----------
        name
            The component name.
        """

        self.components().remove(name)  # list gets updated in place
