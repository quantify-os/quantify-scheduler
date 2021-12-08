# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the InstrumentCoordinator interface."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

from qcodes.instrument import base, parameter
from qcodes.utils import validators


class InstrumentCoordinatorComponentBase(base.Instrument):
    """The InstrumentCoordinator component abstract interface."""

    # NB `_instances` also used by `Instrument` class
    _no_gc_instances: Dict[str, InstrumentCoordinatorComponentBase] = dict()

    def __new__(
        cls, instrument: base.InstrumentBase
    ) -> InstrumentCoordinatorComponentBase:
        """Keeps track of the instances of this class.

        NB This is done intentionally to prevent the instances from being garbage
        collected.
        """
        instance = super().__new__(cls)
        cls._no_gc_instances[instrument.name] = instance
        return instance

    def close(self) -> None:
        """Makes sure the instances reference is released so that garbage collector can
        claim the object"""
        _ = self._no_gc_instances.pop(self.instrument_ref())
        super().close()

    def __init__(self, instrument: base.InstrumentBase, **kwargs: Any) -> None:
        """Instantiates the InstrumentCoordinatorComponentBase base class."""
        super().__init__(f"ic_{instrument.name}", **kwargs)

        self.add_parameter(
            "instrument_ref",
            initial_value=instrument.name,
            parameter_class=parameter.InstrumentRefParameter,
            docstring="A reference of an instrument associated to this component.",
            vals=validators.MultiType(validators.Strings(), validators.Enum(None)),
        )

        self.add_parameter(
            "force_set_parameters",
            initial_value=False,
            parameter_class=parameter.ManualParameter,
            docstring=(
                "A switch to force the setting of a parameter, "
                + "bypassing the lazy_set utility."
            ),
            vals=validators.Bool(),
        )

    @property
    def instrument(self) -> base.InstrumentBase:
        """Returns the instrument referenced by `instrument_ref`."""
        return self.instrument_ref.get_instr()

    @instrument.setter
    def instrument(self, instrument: base.InstrumentBase) -> None:
        """Sets a new Instrument as reference."""
        self.instrument_ref(instrument.name)

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """
        Returns if the InstrumentCoordinator component is running.

        The property `is_running` is evaluated each time it is accessed. Example:

        .. code-block::

            while my_instrument_coordinator_component.is_running:
                print('running')

        Returns
        -------
        :
            The components' running state.
        """

    @abstractmethod
    def start(self) -> None:
        """Starts the InstrumentCoordinator Component."""

    @abstractmethod
    def stop(self) -> None:
        """Stops the InstrumentCoordinator Component."""

    @abstractmethod
    def prepare(self, options: Any) -> None:
        """Initializes the InstrumentCoordinator Component with parameters."""

    @abstractmethod
    def retrieve_acquisition(self) -> Any:
        """Gets and returns acquisition data."""

    @abstractmethod
    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Waits until the InstrumentCoordinator component has stopped running or until it
        has exceeded the amount of time to run.

        The maximum amount of time, in seconds, before it times out is set via the
        timeout_sec parameter.

        Parameters
        ----------
        timeout_sec :
            The maximum amount of time in seconds before a timeout.
        """
