# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Module containing the QuantumDevice object.
"""

from typing import Dict, Any, List
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import (
    ManualParameter,
    InstrumentRefParameter,
)
from qcodes.utils import validators


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
        super().__init__(name)

        self.add_parameter(
            "components",
            initial_value=list(),
            parameter_class=ManualParameter,
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all elements that"
            " are located on this QuantumDevice.",
        )

        device_cfg_backend_validator = validators.Enum(
            "quantify_scheduler.compilation.add_pulse_information_transmon"
        )
        self.add_parameter(
            "device_cfg_backend",
            initial_value=(
                "quantify_scheduler.compilation.add_pulse_information_transmon"
            ),
            parameter_class=ManualParameter,
            vals=device_cfg_backend_validator,
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
            "cfg_nr_averages",
            initial_value=1024,
            parameter_class=ManualParameter,
            docstring=(
                "The number of averages when performing experiments. Used to"
                " set the repetitions attribute of a Schedule."
            ),
            vals=validators.Ints(min_value=1),
        )

        # fixme, this should be generated and not be user-provided
        self.add_parameter("hardware_config", parameter_class=ManualParameter)

    def generate_hardware_config(self) -> Dict[str, Any]:
        """
        Generates a valid hardware configuration describing the quantum device.

        The hardware config should be valid input for the
        :func:`quantify_scheduler.compilation.qcompile` function.

        .. note:

            The config currently has to be specified by the user using the
            :code:`hardware_config` parameter.
        """

        # currently this has to be set by the user, in the future this should be
        # code generated.
        return self.hardware_config()

    def generate_device_config(self) -> Dict[str, Any]:
        """
        Generates a valid device config for the quantify-scheduler making use of the
        :func:`quantify_scheduler.compilation.add_pulse_information_transmon` function.

        .. note:

            The config currently does not support two-qubit gates.

        """

        # initialize an dictionary with the right structure
        device_configuration = {
            "backend": self.device_cfg_backend(),
            "qubits": {},
            "edges": {},
        }

        # iterate over all components. For now, all are assumed to be qubits.
        for comp_name in self.components():
            comp = self.get_component(comp_name)
            device_configuration["qubits"].update(comp.generate_config())

        return device_configuration

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
        raise KeyError(f"'{name}' is not a component of {self.name}!")

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
            raise ValueError(f"'{component.name}' has already been added!")

        if not isinstance(component, Instrument):
            # check can be improved to see if it is also a valid device element.
            # this requires a base class for device elements that does not exist yet.
            raise TypeError(f"{repr(component)} is not a QCoDeS instrument.")

        components: List[str] = self.components()
        # add the component by name
        components.append(component.name)
        self.components.set(components)

    def remove_component(self, name: str) -> None:
        """
        Removes a component by name.

        Parameters
        ----------
        name
            The component name.
        """

        # list gets updated in place
        self.components().remove(name)
