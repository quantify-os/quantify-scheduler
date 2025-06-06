# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
r"""
Module containing :class:`quantify_core.measurement.types.Gettable`\s for use with
quantify-scheduler.

.. warning::

    The gettable module is expected to change significantly as the
    acquisition protocols (#36 and #80) get fully supported by the scheduler.
    Currently different Gettables are required for different acquisition modes.
    The intent is to have one generic ``ScheduleGettable``.
    Expect breaking changes.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from qcodes.parameters import Parameter

from quantify_core.measurement.control import MeasurementControl
from quantify_scheduler.helpers.diagnostics_report import _generate_diagnostics_report

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
    from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule

logger = logging.getLogger(__name__)


class ScheduleGettable:
    """
    Generic gettable for a quantify schedule using vector (I,Q) acquisition.

    The gettable evaluates the parameters passed as ``schedule_kwargs``, then generates
    the :class:`quantify_scheduler.schedules.schedule.Schedule` using the
    ``schedule_function``, this is then compiled and finally executed by the
    :class:`~.InstrumentCoordinator`.

    ``ScheduleGettable`` can be set to return either static (demodulated) I and Q
    values or magnitude and phase.

    Parameters
    ----------
    quantum_device
        The qcodes instrument representing the quantum device under test (DUT)
        containing quantum device properties and setup configuration information.
    schedule_function
        A function which returns a
        :class:`quantify_scheduler.schedules.schedule.Schedule`. The
        function is required to have the ``repetitions`` keyword argument.
    schedule_kwargs
        The schedule function keyword arguments, when a value in this dictionary is
        a :class:`~qcodes.instrument.parameter.Parameter`, this parameter will be
        evaluated every time :code:`.get()` is called before being passed to the
        :code:`schedule_function`.
    num_channels
        The number of channels to expect in the acquisition data.
    data_labels
        Allows to specify custom labels. Needs to be precisely 2*num_channels if
        specified. The order is [Voltage I 0, Voltage Q 0, Voltage I 1, Voltage Q 1,
        ...], in case real_imag==True, otherwise [Magnitude 0, Phase 0, Magnitude 1,
        Phase 1, ...].
    real_imag
        If true, the gettable returns I, Q values. Otherwise, magnitude and phase
        (degrees) are returned.
    batched
        Used to indicate if the experiment is performed in batches or in an
        iterative fashion.
    max_batch_size:
        Determines the maximum number of points to acquire when acquiring in batched
        mode. Can be used to split up a program in parts if required due to hardware
        constraints.
    always_initialize:
        If True, then reinitialize the schedule on each invocation of ``get``. If
        False, then only initialize the first invocation of ``get``.

    """

    def __init__(
        self,
        quantum_device: QuantumDevice,
        schedule_function: Callable[..., Schedule],
        schedule_kwargs: dict[str, Any],
        num_channels: int = 1,
        data_labels: list[str] | None = None,
        real_imag: bool = True,
        batched: bool = False,
        max_batch_size: int = 1024,
        always_initialize: bool = True,
        return_xarray: bool = False,
    ) -> None:
        self._data_labels_specified = data_labels is not None

        self.always_initialize = always_initialize
        self.is_initialized = False
        self._compiled_schedule: CompiledSchedule | None = None

        self.real_imag = real_imag
        if self.real_imag:
            self.name = ["I", "Q"] * num_channels
            if data_labels:
                self.label = data_labels
            else:
                self.label = [
                    f"Voltage {iq}{ch}" for ch in range(num_channels) for iq in ["I", "Q"]
                ]
                logger.info(f"Auto-generating labels. {self.label}")
            self.unit = ["V", "V"] * num_channels
        else:
            self.name = ["magn", "phase"] * num_channels
            if data_labels:
                self.label = data_labels
            else:
                self.label = [
                    f"{val_label}{ch}"
                    for ch in range(num_channels)
                    for val_label in ["Magnitude", "Phase"]
                ]
                logger.info(f"Auto-generating labels. {self.label}")
            self.unit = ["V", "deg"] * num_channels

        self.batched = batched
        self.batch_size = max_batch_size
        self._return_xarray = return_xarray  # to support legacy mode

        # schedule arguments
        self.schedule_function = schedule_function
        self.schedule_kwargs = schedule_kwargs
        self._evaluated_sched_kwargs = {}

        # the quantum device object containing setup configuration information
        self.quantum_device = quantum_device

        # The backend used for compilation. Available as a private variable
        # to facilitate debugging. Will be assigned upon compilation in self.initialize
        self._backend = None

        # Indicates whether compilation is done using the debug mode.
        # When using with diagnostics report, its value is set to `True`.
        self._debug_mode: bool = False

    def __call__(self) -> tuple[float, ...] | tuple[np.ndarray, ...]:
        """Acquire and return data."""
        return self.get()

    def compile(self) -> CompiledSchedule:
        """
        Compile the schedule without preparing and running it.
        The returned compiled schedule can be used to
        plot the circuit or pulse diagrams for example.

        Returns
        -------
        :
            The compiled schedule.

        """
        schedule = self.schedule_function(
            **self._evaluated_sched_kwargs,
            repetitions=self.quantum_device.cfg_sched_repetitions(),
        )
        compilation_config = self.quantum_device.generate_compilation_config()
        compilation_config.debug_mode = self._debug_mode

        # made into a private variable for debugging and future caching functionality
        self._backend = compilation_config.backend(name=compilation_config.name)
        self._compiled_schedule = self._backend.compile(
            schedule=schedule, config=compilation_config
        )
        return self._compiled_schedule

    def initialize(self) -> None:
        """
        Generates the schedule and uploads the compiled instructions to the
        hardware using the instrument coordinator.
        """
        logger.debug("Initializing schedule gettable.")
        self._evaluated_sched_kwargs = _evaluate_parameter_dict(self.schedule_kwargs)

        # generate a schedule using the evaluated keyword arguments dict
        self.compile()

        instr_coordinator = self.quantum_device.instr_instrument_coordinator.get_instr()
        instr_coordinator.prepare(self._compiled_schedule)

        self.is_initialized = True

    @property
    def compiled_schedule(self) -> CompiledSchedule | None:
        """Get the latest compiled schedule, or None if nothing has been compiled yet."""
        return self._compiled_schedule

    def get(self) -> tuple[np.ndarray, ...]:
        """
        Start the experimental sequence and retrieve acquisition data.

        The data format returned is dependent on the type of acquisitions used
        in the schedule. These data formats can be found in the :ref:`user guide
        <sec-user-guide-acquisition-data-schedulegettable>`.

        Returns
        -------
        :
            A tuple of acquisition data per acquisition channel as specified above.

        """
        instr_coordinator = self.quantum_device.instr_instrument_coordinator.get_instr()

        # ensure the instruments are not running and we are starting from a clean state
        # We switch on allow_failure so that some instruments which require the prepare
        # before the stop can fail.
        instr_coordinator.stop(allow_failure=True)

        if not self.is_initialized or self.always_initialize:
            self.initialize()

        if self.compiled_schedule is None:
            raise RuntimeError(
                "No compiled schedule was found. Either the schedule was not "
                "compiled, or the compiled schedule was not assigned to the "
                "correct attribute."
            )

        instr_coordinator.start()
        acquired_data = instr_coordinator.retrieve_acquisition()
        instr_coordinator.stop()

        if len(acquired_data) == 0:
            raise RuntimeError(
                f"InstrumentCoordinator.retrieve_acquisition() "
                f"('{instr_coordinator.name}') "
                f"did not return any data, but was expected to return data."
            )
        # Return xarray similar to Instrument coordinator
        if self._return_xarray:
            return acquired_data

        # Use Adapter for the old interface
        return MeasurementControl._process_acquired_data(
            acquired_data, self.batched, self.real_imag
        )

    def initialize_and_get_with_report(self) -> str:
        """
        Create a report that saves all information from this experiment in a zipfile.

        Run :meth:`~.ScheduleGettable.initialize` and :meth:`~.ScheduleGettable.get`
        and capture all information from the experiment in a zipfile in the quantify
        datadir.
        The basic information in the report includes the schedule, device config and
        hardware config. The method attempts to compile the schedule, and if it
        succeeds, it runs the experiment and adds the compiled schedule, a snapshot of
        the instruments, and logs from the actual hardware (only Qblox instruments
        supported currently) to the zipfile.
        A full error trace is also included if any of these steps fail.

        Returns
        -------
        :
            A path to the generated report. Directory name includes a flag indicating at
            which state the experiment and report retrieval stopped.

            Flags (defined in :func: `~._generate_diagnostics_report`):

            - ``failed_initialization``: The experiment failed during \
            :meth:`~.ScheduleGettable.initialize`.
            - ``failed_exp``: The experiment initialized failed during \
            :meth:`~.ScheduleGettable.get`.
            - ``failed_connection_to_hw``: The experiment initialized but both \
            :meth:`~.ScheduleGettable.get` and \
            :meth:`~.InstrumentCoordinator.retrieve_hardware_logs` failed. Connection \
            to hardware was likely interrupted during runtime.
            - ``failed_hw_log_retrieval``: The experiment succeeded but \
            :meth:`~.InstrumentCoordinator.retrieve_hardware_logs` failed.
            - ``completed_exp``: The experiment succeeded.

        """
        if not self.quantum_device.instr_instrument_coordinator:
            raise AttributeError(
                "`InstrumentCoordinator` not found in the `QuantumDevice`, please\
                add using\
                `QuantumDevice.instr_instrument_coordinator(instrument_coordinator)`."
            )
        if not self.quantum_device.hardware_config():
            raise AttributeError(
                "Hardware configuration not found in the `QuantumDevice`, please add\
                using `QuantumDevice.hardware_config(hardware_cfg)`."
            )

        exception = None
        initialized = False
        acquisition_data = None
        self._debug_mode = True

        # Make sure only a compiled schedule generated in the block below is included in
        # the report
        self._compiled_schedule = None
        try:
            self.initialize()
            initialized = True
            acquisition_data = self.get()
        except Exception:
            exception = sys.exc_info()

        gettable_config = {
            "repetitions": self.quantum_device.cfg_sched_repetitions(),
            "evaluated_schedule_kwargs": self._evaluated_sched_kwargs,
        }

        try:
            schedule = self.schedule_function(
                **self._evaluated_sched_kwargs,
                repetitions=self.quantum_device.cfg_sched_repetitions(),
            )
        except TypeError:  # Not tested, included for guiding users with common MeasurementControl mistake  # noqa: E501
            raise TypeError(
                f"One or more keyword arguments of the schedule function in "
                f"schedule_kwargs are of unexpected type, please check the arguments "
                f"and try again.\n{self._evaluated_sched_kwargs=}"
            ) from None

        instrument_coordinator = self.quantum_device.instr_instrument_coordinator.get_instr()
        return _generate_diagnostics_report(
            quantum_device=self.quantum_device,
            gettable_config=gettable_config,
            schedule=schedule,
            instrument_coordinator=instrument_coordinator,  # type: ignore
            compiled_schedule=self.compiled_schedule,
            acquisition_data=acquisition_data,
            initialized=initialized,
            experiment_exception=exception,
        )


def _evaluate_parameter_dict(parameters: dict[str, Any]) -> dict[str, Any]:
    r"""
    Loop over the keys and values in a dict and replaces parameters with their current
    value.

    Parameters
    ----------
    parameters
        A dictionary containing a mix of
        :class:`~qcodes.instrument.parameter.Parameter`\s and normal values.

    Returns
    -------
    :
        The ``parameters`` dictionary, but with the parameters replaced by their current
        value.

    Raises
    ------
    TypeError
        If a parameter returns None

    """
    evaluated_parameters = dict()

    for key, val in parameters.items():
        if isinstance(val, Parameter):
            # evaluate the parameter
            evaluated_parameters[key] = val.get()
            # verify that the parameter has a value, a missing value typically indicates
            # that it was not initialized.
            if evaluated_parameters[key] is None:
                raise TypeError(
                    f"{key}: parameter {val} returns None. "
                    "It is possible this parameter was not configured correctly."
                )

        else:
            evaluated_parameters[key] = val

    return evaluated_parameters
