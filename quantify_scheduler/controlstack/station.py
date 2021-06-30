# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the main ControlStack Component."""
from __future__ import annotations

from typing import Any, Dict, List
from collections import OrderedDict

from qcodes.utils import validators
from qcodes.instrument import parameter
from qcodes.instrument import base as qcodes_base
from quantify_scheduler.controlstack.components import base


class ControlStack(qcodes_base.Instrument):
    """
    The ControlStack class is a collection of ControlStack components.

    This class provides a high level interface to:

    1. Arm instruments with sequence programs,
       waveforms and other settings.
    2. Start and stop the components
    3. Get the results.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.add_parameter(
            "components",
            initial_value=list(),
            parameter_class=parameter.ManualParameter,
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all components that"
            " are part of this ControlStack.",
        )

    @property
    def is_running(self) -> bool:
        """
        Returns if any of the ControlStack components is running.

        Returns
        -------
        :
            The ControlStack's running state.
        """
        return any(
            self.find_instrument(c_name).is_running is True
            for c_name in self.components()
        )

    def get_component(self, name: str) -> base.ControlStackComponentBase:
        """
        Returns the ControlStack component by name.

        Parameters
        ----------
        name
            The ControlStack component name.

        Returns
        -------
        :
            The ControlStack component.

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
        component: base.ControlStackComponentBase,
    ) -> None:
        """
        Adds a ControlStack component to the ControlStack components collection.

        Parameters
        ----------
        component
            The ControlStack component to add.

        Raises
        ------
        ValueError
            If a component with a duplicated name is added to the collection.
        TypeError
            If 'component' is not an instance of 'ControlStackComponentBase'.
        """
        if component.name in self.components():
            raise ValueError(f"'{component.name}' has already been added!")

        if not isinstance(component, base.ControlStackComponentBase):
            raise TypeError(
                f"{repr(component)} is not "
                f"{base.__name__}.{base.ControlStackComponentBase.__name__}."
            )

        components: List[str] = self.components()
        # add the component by name
        components.append(component.name)
        self.components.set(components)

    def remove_component(self, name: str) -> None:
        """
        Removes a ControlStack component by name.

        Parameters
        ----------
        name
            The ControlStack component name.
        """

        # list gets updated in place
        self.components().remove(name)

    def prepare(
        self,
        options: Dict[str, Any],
    ) -> None:
        """
        Prepares each ControlStack component in the list by name with
        parameters.

        The parameters such as sequence programs, waveforms and other
        settings are used to arm the instrument so it is ready to execute
        the experiment.

        Parameters
        ----------
        options
            The arguments per ControlStack component required to arm the
            instrument.

        Raises
        ------
        KeyError
            Undefined component name.
        """
        for instrument_name, args in options.items():
            self.get_component(instrument_name).prepare(args)

    def start(self) -> None:
        """
        Start all of the ControlStack components.

        The components are started in the order in which they were added
        to the ControlStack.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            instrument.start()

    def stop(self) -> None:
        """
        Stops all ControlStack Components.

        The components are stopped in the order in which they were added
        to the ControlStack.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            instrument.stop()

    def retrieve_acquisition(self) -> Dict[str, Any]:
        """
        Retrieves the latest acquisition results of ControlStack components
        with acquisition capabilities.

        Returns
        -------
        :
            The acquisition data per ControlStack component.
        """
        acq_dict = OrderedDict()
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            acq = instrument.retrieve_acquisition()
            if acq is not None:
                acq_dict[instrument.name] = acq
        return acq_dict

    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Awaits each ControlStack component until it has stopped running or until it has
        exceeded the amount of time to run.

        The timeout in seconds specifies the allowed amount of time to run before
        it times out.

        Parameters
        ----------
        timeout_sec :
            The maximum amount of time in seconds before a timeout.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            self.get_component(instrument.name).wait_done(timeout_sec)
