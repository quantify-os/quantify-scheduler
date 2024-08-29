# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler for the quantify_scheduler."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.helpers.waveforms import area_pulse
from quantify_scheduler.operations.control_flow_library import (
    ControlFlowOperation,
    LoopOperation,
)
from quantify_scheduler.resources import BasebandClockResource
from quantify_scheduler.schedules.schedule import Schedule, ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler.operations.operation import Operation


PortClock = Tuple[str, str]


@dataclass
class SumEnd:
    """Class to store the sum and end as floats."""

    sum: float = 0.0
    end: float = 0.0

    def merge(self, other: SumEnd) -> SumEnd:
        """Merge two `SumEnd` objects together: `sum` are added, `end` are maxed."""
        return SumEnd(
            self.sum + other.sum,
            max(self.end, other.end),
        )


def _merge_sum_and_end(
    pulses_sum_end_1: dict[PortClock, SumEnd], pulses_sum_end_2: dict[PortClock, SumEnd]
) -> dict[PortClock, SumEnd]:
    merged_pulses_sum_end: dict[PortClock, SumEnd] = {}

    all_port_clocks: set[PortClock] = pulses_sum_end_1.keys() | pulses_sum_end_2.keys()
    for port_clock in all_port_clocks:
        if (port_clock in pulses_sum_end_1) and (port_clock in pulses_sum_end_2):
            merged_pulses_sum_end[port_clock] = pulses_sum_end_1[port_clock].merge(
                pulses_sum_end_2[port_clock]
            )
        elif port_clock in pulses_sum_end_1:
            merged_pulses_sum_end[port_clock] = pulses_sum_end_1[port_clock]
        elif port_clock in pulses_sum_end_2:
            merged_pulses_sum_end[port_clock] = pulses_sum_end_2[port_clock]
    return merged_pulses_sum_end


def _determine_sum_and_end_of_all_pulses(
    operation: Schedule | Operation,
    sampling_rate: float,
    time_offset: float,
) -> dict[PortClock, SumEnd]:
    """
    Calculates the sum (or integral) of the amplitudes of all pulses in the operation,
    and the end time of the last pulse in the operation.
    The function assumes there is no operation which need to be pulse compensated inside.
    The function also assumes that the absolute timings are already calculated in the schedule.

    Parameters
    ----------
    operation
        The schedule or operation to calculate sum and end of pulses.
    sampling_rate
        Sampling rate of the pulses.
    time_offset
        Time offset for the operation with regards to the start of the whole schedule.

    Returns
    -------
    :
        The sum and end time of all the pulses as a `SumEnd`.
    """
    # TODO: uncomment in SE-540.
    # assert not isinstance(operation, PulseCompensation)

    if isinstance(operation, ScheduleBase):
        pulses_sum_end: dict[PortClock, SumEnd] = {}
        for schedulable in operation.schedulables.values():
            abs_time = schedulable["abs_time"]
            inner_operation = operation.operations[schedulable["operation_id"]]
            new_pulses_sum_end: dict[PortClock, SumEnd] = (
                _determine_sum_and_end_of_all_pulses(
                    inner_operation, sampling_rate, time_offset + abs_time
                )
            )
            pulses_sum_end = _merge_sum_and_end(pulses_sum_end, new_pulses_sum_end)
        return pulses_sum_end
    elif isinstance(operation, ControlFlowOperation):
        if isinstance(operation, LoopOperation):
            body_pulses_sum_end: dict[PortClock, SumEnd] = (
                _determine_sum_and_end_of_all_pulses(
                    operation.body, sampling_rate, time_offset
                )
            )
            repetitions = operation.data["control_flow_info"]["repetitions"]
            assert repetitions != 0
            looped_pulses_sum_end: dict[PortClock, SumEnd] = {}
            for port_clock, body_sum_end in body_pulses_sum_end.items():
                looped_pulses_sum_end[port_clock] = SumEnd(
                    sum=(repetitions * body_sum_end.sum),
                    end=(repetitions - 1) * operation.body.duration + body_sum_end.end,
                )
            return looped_pulses_sum_end
        else:
            raise ValueError(
                f"Error calculating compensating pulse amplitude for '{operation}'. "
                f"This control flow operation type is not allowed "
                f"in a pulse compensating structure. "
            )
    elif operation.has_voltage_offset:
        raise ValueError(
            f"Error calculating compensating pulse amplitude for '{operation}'. "
            f"Voltage offset operation type is not allowed "
            f"in a pulse compensating structure. "
        )
    elif operation.valid_pulse:
        pulses_sum_end: dict[PortClock, SumEnd] = {}
        for pulse_info in operation["pulse_info"]:
            if pulse_info["clock"] != BasebandClockResource.IDENTITY:
                raise ValueError(
                    f"Error calculating compensating pulse amplitude for '{operation}'. "
                    f"Clock must be the baseband clock. "
                )
            port_clock: PortClock = (pulse_info["port"], pulse_info["clock"])
            new_pulse_sum_end: dict[PortClock, SumEnd] = {
                port_clock: SumEnd(
                    sum=area_pulse(pulse_info, sampling_rate),
                    end=(time_offset + pulse_info["t0"] + pulse_info["duration"]),
                )
            }
            pulses_sum_end = _merge_sum_and_end(pulses_sum_end, new_pulse_sum_end)
        return pulses_sum_end
    else:
        return {}


@dataclass
class CompensationPulseParams:
    """Class to store start, duration and amp in floats."""

    start: float
    duration: float
    amp: float


def determine_compensating_pulse(
    schedule: Schedule,
    compensating_max_amp: dict[PortClock, float],
    time_grid: float,
    sampling_rate: float,
) -> dict[PortClock, CompensationPulseParams]:
    """
    Calculates the timing and the amplitude of a compensating pulse for each port clock.
    The `duration` and `amp` are calculated, with the requirements, that
    if a compensating square pulse is inserted in this schedule at `start` with duration `duration`,
    and amplitude `amp`, then
    * the integral of all pulses in the schedule would equal to 0,
    * the duration of the compensating pulse is divisible by `time_grid`,
    * the compensating pulse is the last pulse in the schedule, and
    * the compensating pulse starts just after the previous pulse.
    The function assumes there is no operation which needs to be pulse compensated inside.

    Parameters
    ----------
    schedule
        The original schedule to compensate for.
    compensating_max_amp
        The maximum amplitude of the compensating pulse.
    time_grid
        Time grid the compensating pulse needs to be on.
    sampling_rate
        Sampling rate of the pulses.

    Returns
    -------
    :
        The start, duration and amp of a compensating pulse
        with the given requirements as a `CompensationPulseParams` for each port clock.
    """
    pulses_start_duration_amp: dict[PortClock, CompensationPulseParams] = {}

    schedule_with_abs_times = _determine_absolute_timing(deepcopy(schedule), None)
    pulses_sum_end: dict[PortClock, SumEnd] = _determine_sum_and_end_of_all_pulses(
        schedule_with_abs_times, sampling_rate, 0
    )

    for port_clock, pulse_sum_end in pulses_sum_end.items():
        sum_abs: float = abs(pulse_sum_end.sum)

        if port_clock not in compensating_max_amp:
            raise ValueError(
                f"Error calculating compensating pulse amplitude for "
                f"portclock '{port_clock[0]}-{port_clock[1]}'. "
                f"No maximum amplitude for portclock found, make sure it's set."
            )

        duration: float = (
            math.ceil(sum_abs / time_grid / compensating_max_amp[port_clock])
            * time_grid
        )
        amp: float = -pulse_sum_end.sum / duration
        pulses_start_duration_amp[port_clock] = CompensationPulseParams(
            start=pulse_sum_end.end, duration=duration, amp=amp
        )

    return pulses_start_duration_amp
