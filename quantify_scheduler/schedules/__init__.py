# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing a standard library of schedules for common experiments as well as the
:class:`.ScheduleBase`, :class:`.Schedule`, and :class:`.CompiledSchedule` classes.


.. tip::

    The source code of the schedule generating functions in this module can
    serve as examples when creating schedules for custom experiments.

"""
from .schedule import CompiledSchedule, Schedulable, Schedule
from .spectroscopy_schedules import (
    heterodyne_spec_sched,
    heterodyne_spec_sched_nco,
    nv_dark_esr_sched,
    two_tone_spec_sched,
    two_tone_spec_sched_nco,
)
from .timedomain_schedules import (
    allxy_sched,
    echo_sched,
    rabi_pulse_sched,
    rabi_sched,
    ramsey_sched,
    readout_calibration_sched,
    t1_sched,
)
from .trace_schedules import (
    trace_schedule,
    trace_schedule_circuit_layer,
    two_tone_trace_schedule,
)

__all__ = [
    "CompiledSchedule",
    "Schedulable",
    "Schedule",
    "allxy_sched",
    "echo_sched",
    "heterodyne_spec_sched",
    "heterodyne_spec_sched_nco",
    "nv_dark_esr_sched",
    "rabi_pulse_sched",
    "rabi_sched",
    "ramsey_sched",
    "readout_calibration_sched",
    "t1_sched",
    "trace_schedule",
    "trace_schedule_circuit_layer",
    "two_tone_spec_sched",
    "two_tone_spec_sched_nco",
    "two_tone_trace_schedule",
]
