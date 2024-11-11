# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing the InstrumentCoordinator interface."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from qcodes.instrument import Instrument, InstrumentBase, parameter
from qcodes.utils import validators

if TYPE_CHECKING:
    from xarray import Dataset

    from quantify_scheduler.schedules.schedule import CompiledSchedule


def instrument_to_component_name(instrument_name: str) -> str:
    """
    Give the name of the instrument coordinator component.

    Parameters
    ----------
    instrument_name
        The name of the instrument.

    Returns
    -------
    :
        The name of the instrument coordinator component.

    """
    return f"ic_{instrument_name}"


class InstrumentCoordinatorComponentBase(Instrument):
    """The InstrumentCoordinator component abstract interface."""

    # NB `_instances` also used by `Instrument` class
    _no_gc_instances: dict[str, InstrumentCoordinatorComponentBase] = dict()

    def __new__(cls, instrument: InstrumentBase) -> InstrumentCoordinatorComponentBase:
        """
        Keeps track of the instances of this class.

        NB This is done intentionally to prevent the instances from being garbage
        collected.
        """
        instance = super().__new__(cls)
        cls._no_gc_instances[instrument.name] = instance
        return instance

    def close(self) -> None:
        """Release instance so that garbage collector can claim the object."""
        _ = self._no_gc_instances.pop(self.instrument_ref())
        super().close()

    def __init__(
        self,
        instrument: InstrumentBase,
        **kwargs: Any,  # noqa ANN401 (complicated subclass overrides)
    ) -> None:
        super().__init__(instrument_to_component_name(instrument.name), **kwargs)

        self.instrument_ref = parameter.InstrumentRefParameter(
            "instrument_ref",
            initial_value=instrument.name,
            docstring="A reference of an instrument associated to this component.",
            vals=validators.MultiType(validators.Strings(), validators.Enum(None)),
            instrument=self,
        )

        self.force_set_parameters = parameter.ManualParameter(
            "force_set_parameters",
            initial_value=False,
            docstring=(
                "A switch to force the setting of a parameter, " + "bypassing the lazy_set utility."
            ),
            vals=validators.Bool(),
            instrument=self,
        )

    @property
    def instrument(self) -> InstrumentBase:
        """Returns the instrument referenced by `instrument_ref`."""
        return self.instrument_ref.get_instr()

    @instrument.setter
    def instrument(self, instrument: InstrumentBase) -> None:
        """Sets a new Instrument as reference."""
        self.instrument_ref(instrument.name)

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """
        Returns if the InstrumentCoordinator component is running.

        The property ``is_running`` is evaluated each time it is accessed. Example:

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
    def prepare(self, options: Any) -> None:  # noqa: ANN401 (Complicated subclass overrides)
        """Initializes the InstrumentCoordinator Component with parameters."""

    @abstractmethod
    def retrieve_acquisition(self) -> Dataset | None:
        """Gets and returns acquisition data."""

    @abstractmethod
    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Wait until the InstrumentCoordinator is done.

        The coordinator is ready when it has stopped running or until it
        has exceeded the amount of time to run.

        The maximum amount of time, in seconds, before it times out is set via the
        timeout_sec parameter.

        Parameters
        ----------
        timeout_sec :
            The maximum amount of time in seconds before a timeout.

        """

    @abstractmethod
    def get_hardware_log(
        self,
        compiled_schedule: CompiledSchedule,
    ) -> dict | None:
        """Retrieve the hardware logs of the instrument associated to this component."""
