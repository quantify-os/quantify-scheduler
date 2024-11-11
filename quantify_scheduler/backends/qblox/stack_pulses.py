# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pulse stacking algorithm for Qblox backend."""
from __future__ import annotations

import uuid
from collections import defaultdict, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantify_scheduler.backends.qblox import constants, helpers
from quantify_scheduler.backends.qblox.helpers import is_square_pulse, to_grid_time
from quantify_scheduler.backends.qblox.operations.pulse_library import (
    SimpleNumericalPulse,
)
from quantify_scheduler.schedules import Schedulable, Schedule

if TYPE_CHECKING:
    from quantify_scheduler.operations import Operation


@dataclass
class PulseInterval:
    """Represents an interval of (possibly) overlapping pulses."""

    start_time: float
    end_time: float
    pulse_keys: set[str]


@dataclass
class PulseParameters:
    """Represents information about a specific pulse. Used for calculating intervals."""

    time: float
    is_end: bool
    schedulable_key: str


PortClock = namedtuple("PortClock", ["port", "clock"])
""" Named tuple for port and clock information."""


def stack_pulses(schedule: Schedule, config) -> Schedule:  # noqa: D417, ARG001, ANN001
    """
    Processes a given schedule by identifying and stacking overlapping pulses.
    The function first defines intervals of overlapping pulses and then
    stacks the pulses within these intervals.

    Parameters
    ----------
    schedule
        The schedule containing the pulses to stack.

    Returns
    -------
    :
        The schedule with stacked pulses.

    """
    pulses_by_port_clock = _construct_pulses_by_port_clock(schedule)

    for pulses in pulses_by_port_clock.values():
        sorted_pulses = sorted(pulses, key=lambda pulse: (pulse.time, pulse.is_end))
        pulses_by_interval = _construct_pulses_by_interval(sorted_pulses)

        if any(len(overlap.pulse_keys) > 1 for overlap in pulses_by_interval):
            schedule = _stack_pulses_by_interval(schedule, pulses_by_interval)

    return schedule


def _construct_pulses_by_port_clock(
    schedule: Schedule,
) -> defaultdict[str, list[PulseParameters]]:
    """Construct a dictionary of pulses by port and clock."""
    pulses_by_port_clock = defaultdict(list)
    for schedulable_key, schedulable in schedule.schedulables.items():
        op = schedule.operations[schedulable["operation_id"]]
        if isinstance(op, Schedule):
            schedule.operations[schedulable["operation_id"]] = stack_pulses(op, {})
            continue
        elif not op.valid_pulse or not op.data.get("pulse_info"):
            continue

        pulse_info = next(iter(op.data["pulse_info"]))
        port_clock = PortClock(pulse_info["port"], pulse_info["clock"])
        start_time = schedulable["abs_time"] + pulse_info["t0"]
        end_time = start_time + pulse_info["duration"]

        pulses_by_port_clock[port_clock].extend(
            [
                PulseParameters(
                    time=round(to_grid_time(start_time) * (1 / constants.SAMPLING_RATE), 9),
                    is_end=False,
                    schedulable_key=schedulable_key,
                ),
                PulseParameters(
                    time=round(to_grid_time(end_time) * (1 / constants.SAMPLING_RATE), 9),
                    is_end=True,
                    schedulable_key=schedulable_key,
                ),
            ]
        )

    return pulses_by_port_clock


def _construct_pulses_by_interval(
    sorted_pulses: list[PulseParameters],
) -> list[PulseInterval]:
    """
    Constructs a list of `PulseInterval` objects representing time intervals and active pulses.

    Given a sorted list of `PulseParameters` objects, this function identifies distinct intervals
    where pulses are active. Each `PulseInterval` records the start time, end time, and the set of
    active pulses during that interval. Pulses are added to or removed from the set based on their
    `is_end` attribute, indicating whether the pulse is starting or ending at a given time.


    Example Input/Output:
    ---------------------
        If the input list has pulses with start and end times as:
            [PulseParameters(time=1, schedulable_key='A', is_end=False),
             PulseParameters(time=3, schedulable_key='A', is_end=True),
             PulseParameters(time=2, schedulable_key='B', is_end=False)]
        The output will be:
            [PulseInterval(start_time=1, end_time=2, active_pulses={'A'}),
             PulseInterval(start_time=2, end_time=3, active_pulses={'A', 'B'})]

    See https://softwareengineering.stackexchange.com/questions/363091 for algo.

    """
    pulses_by_interval = []
    active_pulses = set()
    last_time = None

    for pulse in sorted_pulses:
        time, key, is_end = pulse.time, pulse.schedulable_key, pulse.is_end

        if last_time is None:
            last_time = time
            active_pulses.add(key)
        else:
            if time > last_time:
                if len(active_pulses):
                    pulses_by_interval.append(PulseInterval(last_time, time, active_pulses.copy()))
                last_time = time
            if is_end:
                active_pulses.remove(key)
            else:
                active_pulses.add(key)

    return pulses_by_interval


def _stack_pulses_by_interval(
    schedule: Schedule, pulses_by_interval: list[PulseInterval]
) -> Schedule:
    old_schedulable_keys = set()

    for interval in pulses_by_interval:
        if not interval.pulse_keys:
            continue

        if all(
            is_square_pulse(schedule.operations[schedule.schedulables[key]["operation_id"]])
            for key in interval.pulse_keys
        ):
            _stack_square_pulses(interval, schedule, old_schedulable_keys)
        else:
            _stack_arbitrary_pulses(interval, schedule, old_schedulable_keys)

    # Delete old schedulables
    for key in old_schedulable_keys:
        del schedule.schedulables[key]

    # Dellete timing constraints
    for key, schedulable in schedule.schedulables.items():
        if "timing_constraints" in schedulable:
            del schedulable.data["timing_constraints"]

    return schedule


def _stack_arbitrary_pulses(
    interval: PulseInterval,
    schedule: Schedule,
    old_schedulable_keys: set[str],
) -> None:
    num_samples = round((interval.end_time - interval.start_time) * constants.SAMPLING_RATE)
    combined_waveform = np.zeros(num_samples)
    port, clock, pulse_info = None, None, None

    for key in interval.pulse_keys:
        pulse = schedule.operations[schedule.schedulables[key]["operation_id"]]
        for pulse_info in pulse.data["pulse_info"]:
            schedulable = schedule.schedulables[key]
            old_schedulable_keys.add(key)
            waveform = helpers.generate_waveform_data(
                pulse_info, sampling_rate=constants.SAMPLING_RATE
            )
            start_idx = round(
                (interval.start_time - schedulable["abs_time"] - pulse_info["t0"])
                * constants.SAMPLING_RATE
            )
            end_idx = start_idx + num_samples

            if end_idx == len(waveform) - 1:
                # Including the last sample makes waveform slice longer than combined_waveform;
                # append zero to combined_waveform to match lengths and prevent shape mismatch.
                end_idx += 1
                combined_waveform = np.append(combined_waveform, 0)

            combined_waveform = np.add(combined_waveform, waveform[start_idx:end_idx])
            if port is None and clock is None:
                port, clock = pulse_info.get("port"), pulse_info.get("clock")

    if port is None or clock is None:
        raise ValueError(
            f"pulse_info must contain non-None 'port' and 'clock' values. Pulse Info: {pulse_info},"
            f" Port: {port}, Clock: {clock}"
        )
    numerical_pulse = SimpleNumericalPulse(
        samples=combined_waveform,
        port=port,
        clock=clock,
    )

    _create_schedulable(schedule, interval.start_time, numerical_pulse)


def _stack_square_pulses(
    interval: PulseInterval,
    schedule: Schedule,
    old_schedulable_keys: set[str],
) -> None:
    combined_pulse = None

    for key in interval.pulse_keys:
        pulse = schedule.operations[schedule.schedulables[key]["operation_id"]]
        old_schedulable_keys.add(key)
        for pulse_info in pulse.data["pulse_info"]:
            if combined_pulse is None:
                combined_pulse = deepcopy(pulse)
                combined_pulse.data["pulse_info"][0]["t0"] = 0
                combined_pulse.data["pulse_info"][0]["duration"] = round(
                    to_grid_time(interval.end_time - interval.start_time)
                    * (1 / constants.SAMPLING_RATE),
                    9,
                )
            else:
                combined_pulse.data["pulse_info"][0]["amp"] += pulse_info["amp"]
    _create_schedulable(schedule, interval.start_time, combined_pulse)


def _create_schedulable(
    schedule: Schedule, start_time: float, pulse: Operation | Schedule | None
) -> None:
    if pulse is not None:
        new_schedulable_key = str(uuid.uuid4())
        new_schedulable = Schedulable(name=new_schedulable_key, operation_id=pulse.hash)
        new_schedulable["abs_time"] = start_time

        schedule.schedulables[new_schedulable_key] = new_schedulable
        schedule.operations[pulse.hash] = pulse
