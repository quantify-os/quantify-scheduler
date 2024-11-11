# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler for the quantify_scheduler."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.helpers.waveforms import area_pulse
from quantify_scheduler.operations.control_flow_library import (
    ControlFlowOperation,
    LoopOperation,
)
from quantify_scheduler.operations.pulse_compensation_library import (
    Port,
    PulseCompensation,
)
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import BasebandClockResource
from quantify_scheduler.schedules.schedule import Schedule, ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import (
        CompilationConfig,
    )
    from quantify_scheduler.operations.operation import Operation


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
    pulses_sum_end_1: dict[Port, SumEnd], pulses_sum_end_2: dict[Port, SumEnd]
) -> dict[Port, SumEnd]:
    merged_pulses_sum_end: dict[Port, SumEnd] = {}

    all_ports: set[Port] = pulses_sum_end_1.keys() | pulses_sum_end_2.keys()
    for port in all_ports:
        if (port in pulses_sum_end_1) and (port in pulses_sum_end_2):
            merged_pulses_sum_end[port] = pulses_sum_end_1[port].merge(pulses_sum_end_2[port])
        elif port in pulses_sum_end_1:
            merged_pulses_sum_end[port] = pulses_sum_end_1[port]
        elif port in pulses_sum_end_2:
            merged_pulses_sum_end[port] = pulses_sum_end_2[port]
    return merged_pulses_sum_end


def _determine_sum_and_end_of_all_pulses(
    operation: Schedule | Operation,
    sampling_rate: float,
    time_offset: float,
    ports: set[str],
) -> dict[Port, SumEnd]:
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
    ports
        Set of ports for which we need to calculate the pulse compensations.

    Returns
    -------
    :
        The sum and end time of all the pulses as a `SumEnd`.

    """
    assert not isinstance(operation, PulseCompensation)

    if isinstance(operation, ScheduleBase):
        pulses_sum_end: dict[Port, SumEnd] = {}
        for schedulable in operation.schedulables.values():
            abs_time = schedulable["abs_time"]
            inner_operation = operation.operations[schedulable["operation_id"]]
            new_pulses_sum_end: dict[Port, SumEnd] = _determine_sum_and_end_of_all_pulses(
                inner_operation, sampling_rate, time_offset + abs_time, ports
            )
            pulses_sum_end = _merge_sum_and_end(pulses_sum_end, new_pulses_sum_end)
        return pulses_sum_end
    elif isinstance(operation, ControlFlowOperation):
        if isinstance(operation, LoopOperation):
            body_pulses_sum_end: dict[Port, SumEnd] = _determine_sum_and_end_of_all_pulses(
                operation.body, sampling_rate, time_offset, ports
            )
            repetitions = operation.data["control_flow_info"]["repetitions"]
            assert repetitions != 0
            looped_pulses_sum_end: dict[Port, SumEnd] = {}
            for port, body_sum_end in body_pulses_sum_end.items():
                looped_pulses_sum_end[port] = SumEnd(
                    sum=(repetitions * body_sum_end.sum),
                    end=(repetitions - 1) * operation.body.duration + body_sum_end.end,
                )
            return looped_pulses_sum_end
        else:
            raise ValueError(
                f"Error calculating compensation pulse amplitude for '{operation}'. "
                f"This control flow operation type is not allowed "
                f"in a pulse compensation structure. "
            )
    elif operation.has_voltage_offset:
        raise ValueError(
            f"Error calculating compensation pulse amplitude for '{operation}'. "
            f"Voltage offset operation type is not allowed "
            f"in a pulse compensation structure. "
        )
    elif operation.valid_pulse:
        pulses_sum_end: dict[Port, SumEnd] = {}
        for pulse_info in operation["pulse_info"]:
            port: Port = pulse_info["port"]
            if port in ports:
                if pulse_info["clock"] != BasebandClockResource.IDENTITY:
                    raise ValueError(
                        f"Error calculating compensation pulse amplitude for '{operation}'. "
                        f"Clock must be the baseband clock. "
                    )
                new_pulse_sum_end: dict[Port, SumEnd] = {
                    port: SumEnd(
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


def _determine_compensation_pulse(
    operation: Schedule | Operation,
    max_compensation_amp: dict[Port, float],
    time_grid: float,
    sampling_rate: float,
) -> dict[Port, CompensationPulseParams]:
    """
    Calculates the timing and the amplitude of a compensation pulse for each port.
    The `duration` and `amp` are calculated, with the requirements, that
    if a compensation square pulse is inserted in the schedule at `start` with duration `duration`,
    and amplitude `amp`, then
    * the integral of all pulses in the operation would equal to 0,
    * the duration of the compensation pulse is divisible by `time_grid`,
    * the compensation pulse is the last pulse in the operation, and
    * the compensation pulse starts just after the previous pulse.
    The function assumes there is no operation which needs to be pulse compensated inside.
    The clock is assumed to be the baseband clock.

    Parameters
    ----------
    operation
        The original operation or schedule to compensate for.
    max_compensation_amp
        The maximum amplitude of the compensation pulse.
    time_grid
        Time grid the compensation pulse needs to be on.
    sampling_rate
        Sampling rate of the pulses.

    Returns
    -------
    :
        The start, duration and amp of a compensation pulse
        with the given requirements as a `CompensationPulseParams` for each port.

    """
    pulses_start_duration_amp: dict[Port, CompensationPulseParams] = {}

    operation_with_abs_times = (
        operation
        if not isinstance(operation, Schedule)
        else _determine_absolute_timing(deepcopy(operation), None)
    )
    ports: set[str] = set(max_compensation_amp.keys())
    pulses_sum_end: dict[Port, SumEnd] = _determine_sum_and_end_of_all_pulses(
        operation_with_abs_times, sampling_rate, 0, ports
    )

    for port, pulse_sum_end in pulses_sum_end.items():
        if pulse_sum_end.sum != 0 and port in max_compensation_amp:
            sum_abs: float = abs(pulse_sum_end.sum)

            duration: float = (
                math.ceil(sum_abs / time_grid / max_compensation_amp[port]) * time_grid
            )
            amp: float = -pulse_sum_end.sum / duration
            pulses_start_duration_amp[port] = CompensationPulseParams(
                start=pulse_sum_end.end, duration=duration, amp=amp
            )

    return pulses_start_duration_amp


@overload
def process_compensation_pulses(
    schedule: Schedule,
    config: CompilationConfig | None = None,
) -> Schedule: ...
@overload
def process_compensation_pulses(
    schedule: Operation,
    config: CompilationConfig | None = None,
) -> Schedule | Operation: ...
def process_compensation_pulses(
    schedule: Schedule | Operation,
    config: CompilationConfig | None = None,
) -> Schedule | Operation:
    """
    Replaces ``PulseCompensation`` with a subschedule with an additional compensation pulse.

    Parameters
    ----------
    schedule
        The schedule which contains potential ``PulseCompensation`` in it.
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`.


    Returns
    -------
    :
        The start, duration and amp of a compensation pulse
        with the given requirements as a `CompensationPulseParams` for each port.

    """
    operation: Operation | Schedule = schedule
    if isinstance(operation, ScheduleBase):
        for inner_op_key in operation.operations:
            # Here, we modify the subschedule, and every time the same
            # subschedule is reused in other parts of the whole schedule,
            # will also be modified. We assume those are compiled the same way every time.
            operation.operations[inner_op_key] = process_compensation_pulses(
                operation.operations[inner_op_key],
                config,
            )
        return operation
    elif isinstance(operation, ControlFlowOperation):
        operation.body = process_compensation_pulses(
            operation.body,
            config,
        )
        return operation
    elif isinstance(operation, PulseCompensation):
        # Inner pulse compensated blocks need to be resolved first.
        resolved_body = process_compensation_pulses(operation.body)
        all_compensation_pulse_params: dict[Port, CompensationPulseParams] = (
            _determine_compensation_pulse(
                resolved_body,
                operation.max_compensation_amp,
                operation.time_grid,
                operation.sampling_rate,
            )
        )

        pulse_compensated_schedule = Schedule("pulse_compensated_schedule")
        first_op_schedulable = pulse_compensated_schedule.add(resolved_body)

        for (
            port,
            compensation_pulse_params,
        ) in all_compensation_pulse_params.items():
            pulse_compensated_schedule.add(
                operation=SquarePulse(
                    amp=compensation_pulse_params.amp,
                    duration=compensation_pulse_params.duration,
                    port=port,
                    clock=BasebandClockResource.IDENTITY,
                ),
                rel_time=compensation_pulse_params.start,
                ref_op=first_op_schedulable,
                ref_pt="start",
                ref_pt_new="start",
            )

        return pulse_compensated_schedule
    return schedule
