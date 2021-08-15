# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the main InstrumentCoordinator Component."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from qcodes.utils import validators
from qcodes.instrument import parameter
from qcodes.instrument import base as qcodes_base
from quantify_scheduler.instrument_coordinator.components import base


from quantify_scheduler.types import CompiledSchedule


class InstrumentCoordinator(qcodes_base.Instrument):
    """
    The InstrumentCoordinator serves as the central interface of the hardware
    abstraction layer. It provides a standardized interface to execute Schedules on
    control hardware.

    The InstrumentCoordinator has two main functionalities exposed to the user,
    the ability to configure the instrument coordinator
    :mod:`~quantify_scheduler.instrument_coordinator.components`
    representing physical instruments,  and the ability to execute experiments.

    .. todo::

        add code example on adding and removing instruments

        add code example on executing an experiment.


    class is a collection of InstrumentCoordinator components.

    This class provides a high level interface to:

    1. Arm instruments with sequence programs,
       waveforms and other settings.
    2. Start and stop the components.
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
            " are part of this InstrumentCoordinator.",
        )

    @property
    def is_running(self) -> bool:
        """
        Returns if any of the InstrumentCoordinator components is running.

        Returns
        -------
        :
            The InstrumentCoordinator's running state.
        """
        return any(
            self.find_instrument(c_name).is_running is True
            for c_name in self.components()
        )

    def get_component(self, name: str) -> base.InstrumentCoordinatorComponentBase:
        """
        Returns the InstrumentCoordinator component by name.

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
        component: base.InstrumentCoordinatorComponentBase,
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

        if not isinstance(component, base.InstrumentCoordinatorComponentBase):
            raise TypeError(
                f"{repr(component)} is not "
                f"{base.__name__}.{base.InstrumentCoordinatorComponentBase.__name__}."
            )

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

    def prepare(
        self,
        compiled_schedule: CompiledSchedule,
    ) -> None:
        """
        Prepares each component for execution of a schedule.

        It attempts to configure all instrument coordinator components for which
        compiled instructions, typically consisting of a combination of sequence
        programs, waveforms and other instrument settings, are available in the
        compiled schedule.


        Parameters
        ----------
        compiled_schedule
            A schedule containing the information required to execute the program.

        Raises
        ------
        KeyError
            Undefined component name if the compiled schedule contains instructions
        """
        if not CompiledSchedule.is_valid(compiled_schedule):
            raise TypeError(f"{compiled_schedule} is not a valid CompiledSchedule")

        # N.B. this would a good place to store a reference to the last executed
        # schedule that the InstrumentCoordinator has touched.

        compiled_instructions = compiled_schedule["compiled_instructions"]
        for instrument_name, args in compiled_instructions.items():
            self.get_component(instrument_name).prepare(args)

    def start(self) -> None:
        """
        Start all of the components.

        The components are started in the order in which they were added.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            instrument.start()

    def stop(self) -> None:
        """
        Stops all components.

        The components are stopped in the order in which they were added.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            instrument.stop()

    def retrieve_acquisition(self) -> Dict[Tuple[int, int], Any]:
        """
        Retrieves the latest acquisition results of the components
        with acquisition capabilities.

        Returns
        -------
        :
            The acquisition data per component.
        """
        # Temporary. Will probably be replaced by an xarray object
        # See quantify-core#187, quantify-core#233, quantify-scheduler#36
        acquisitions: Dict[Tuple[int, int], Any] = dict()
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            acqs = instrument.retrieve_acquisition()
            if acqs is not None:
                acquisitions.update(acqs)
        return acquisitions

    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Awaits each component until it has stopped running or
        until it has exceeded the amount of time to run.

        The timeout in seconds specifies the allowed amount of time to run before
        it times out.

        Parameters
        ----------
        timeout_sec
            The maximum amount of time in seconds before a timeout.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            self.get_component(instrument.name).wait_done(timeout_sec)
