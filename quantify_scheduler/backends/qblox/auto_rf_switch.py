# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing the auto RF switch dressing compilation pass."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import CompilationConfig
    from quantify_scheduler.schedules.schedule import Schedule


@dataclass
class VoltageOffsetEvent:
    """
    Event from a VoltageOffset operation for RF switch scheduling.

    A non-zero offset on a microwave port signals the start of a stitched pulse
    window; a zero offset signals the end of that window.
    """

    abs_time: float
    """Absolute time of the event."""
    port: str
    """Port of the event."""
    clock: str
    """Clock of the event."""
    is_nonzero: bool
    """True if the offset is non-zero (RF on), False if both paths are zero (RF off)."""


@dataclass
class MicrowavePulseInfo:
    """Information about a microwave pulse for RF switch grouping."""

    schedulable_key: str
    """Key of the schedulable in the schedule."""
    abs_time: float
    """Absolute start time of the pulse."""
    duration: float
    """Duration of the pulse."""
    port: str
    """Port of the pulse."""
    clock: str
    """Clock of the pulse."""

    @property
    def end_time(self) -> float:
        """Return the end time of the pulse."""
        return self.abs_time + self.duration


def auto_rf_switch_dressing(
    schedule: Schedule,
    config: CompilationConfig,
) -> Schedule:
    """
    Automatically insert RFSwitchToggle operations around microwave pulses.

    This compilation pass identifies microwave operations (pulses on ports ending with
    ``:mw``) and inserts :class:`~.RFSwitchToggle` operations to control RF switches.
    Adjacent microwave pulses within ``2 * ramp_buffer`` of each other are merged into
    a single RF switch window to minimize switching.

    Stitched pulses composed of :class:`~.VoltageOffset` operations are also handled:
    the RF switch is turned on at the first non-zero VoltageOffset on a port and turned
    off when a subsequent VoltageOffset sets both paths back to zero.

    All microwave pulses are collected from the full schedule tree (including nested
    sub-schedules) using global absolute times. RF switch operations are inserted
    exclusively on the root schedule so that sub-schedule durations are not modified,
    preventing timing issues for small schedules. Pulse windows from consecutive
    sub-schedules are also merged correctly because the merging operates on the global
    timeline.

    Parameters
    ----------
    schedule
        The schedule to process.
    config
        The compilation configuration containing hardware options.

    Returns
    -------
    Schedule
        The schedule with RFSwitchToggle operations inserted.

    Note
    ----
    This pass expects the schedule to already have absolute timing determined
    (i.e., ``abs_time`` must be set for all schedulables). The RF switch operations
    are inserted with their ``abs_time`` set directly, so no timing recalculation
    is needed.

    """
    # Import here to avoid circular imports
    from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig

    hardware_cfg = config.hardware_compilation_config
    if not isinstance(hardware_cfg, QbloxHardwareCompilationConfig):
        return schedule

    compiler_options = getattr(hardware_cfg, "compiler_options", None)
    if compiler_options is None or not compiler_options.auto_rf_switch:
        return schedule

    ramp_buffer = compiler_options.auto_rf_switch_ramp_buffer

    # Collect all MW pulses and VoltageOffset events from the full schedule tree.
    # Using global absolute times ensures pulses across sub-schedule boundaries are
    # merged correctly, and inserting only on the root schedule avoids extending
    # sub-schedule durations.
    mw_pulses, vo_events = _collect_all_mw_events_recursive(schedule, time_offset=0.0)

    # Convert VoltageOffset on/off pairs to pulse windows and combine with regular pulses
    all_pulses = mw_pulses + _voltage_offset_events_to_pulses(vo_events)

    if not all_pulses:
        return schedule

    # Group pulses by port-clock combination
    pulses_by_port_clock: dict[tuple[str, str], list[MicrowavePulseInfo]] = defaultdict(list)
    for pulse in all_pulses:
        pulses_by_port_clock[(pulse.port, pulse.clock)].append(pulse)

    # For each port-clock, merge adjacent pulses and insert RF switch operations
    # on the root schedule only.
    for (port, clock), pulses in pulses_by_port_clock.items():
        pulses_sorted = sorted(pulses, key=lambda p: p.abs_time)
        windows = _merge_pulses_into_windows(pulses_sorted, ramp_buffer)
        for window_start, window_end in windows:
            _insert_rf_switch_operation(
                schedule=schedule,
                port=port,
                clock=clock,
                start_time=window_start,
                end_time=window_end,
                ramp_buffer=ramp_buffer,
            )

    return schedule


def _collect_all_mw_events_recursive(
    schedule: Schedule,
    time_offset: float,
) -> tuple[list[MicrowavePulseInfo], list[VoltageOffsetEvent]]:
    """
    Recursively collect microwave pulses and VoltageOffset events from a schedule tree.

    Traverses the full schedule tree (including nested sub-schedules and
    control-flow bodies) and returns all events with *global* absolute times, i.e.
    times relative to the root schedule start.

    Regular microwave pulses (non-``None`` ``wf_func``, positive duration) are returned
    as :class:`MicrowavePulseInfo` objects.  :class:`~.VoltageOffset` operations
    (``wf_func=None``, ``duration=0``) on ``:mw`` ports are collected as
    :class:`VoltageOffsetEvent` objects so that stitched pulse windows can be derived
    by :func:`_voltage_offset_events_to_pulses`.

    Parameters
    ----------
    schedule
        The schedule to collect from.
    time_offset
        The global absolute time offset for this schedule's schedulables.
        For the root schedule this is ``0.0``; for a nested schedule it is the
        global start time of the sub-schedule schedulable in the parent.

    Returns
    -------
    tuple[list[MicrowavePulseInfo], list[VoltageOffsetEvent]]
        MW pulse info objects and VoltageOffset events, all with global absolute times.

    """
    # Import here to avoid circular imports
    from quantify_scheduler.operations.control_flow_library import ControlFlowOperation
    from quantify_scheduler.schedules.schedule import ScheduleBase

    mw_pulses: list[MicrowavePulseInfo] = []
    vo_events: list[VoltageOffsetEvent] = []

    for schedulable_key, schedulable in schedule.schedulables.items():
        operation = schedule.operations.get(schedulable["operation_id"])
        if operation is None:
            continue

        # Global absolute start time of this schedulable
        schedulable_abs_time = time_offset + schedulable.data.get("abs_time", 0.0)

        if isinstance(operation, ScheduleBase):
            # Recurse into nested schedule; its internal abs_times are relative to it
            sub_pulses, sub_events = _collect_all_mw_events_recursive(
                operation, schedulable_abs_time
            )
            mw_pulses.extend(sub_pulses)
            vo_events.extend(sub_events)
        elif isinstance(operation, ControlFlowOperation):
            if isinstance(operation.body, ScheduleBase):
                sub_pulses, sub_events = _collect_all_mw_events_recursive(
                    operation.body, schedulable_abs_time
                )
                mw_pulses.extend(sub_pulses)
                vo_events.extend(sub_events)
        else:
            for pulse_info in operation.data.get("pulse_info", []):
                port = pulse_info.get("port", "")
                if not port or not port.endswith(":mw"):
                    continue

                # Skip RF switch marker pulses
                if pulse_info.get("marker_pulse"):
                    continue

                abs_time = schedulable_abs_time + pulse_info.get("t0", 0.0)
                duration = pulse_info.get("duration", 0.0)
                clock = pulse_info.get("clock", "")

                if pulse_info.get("wf_func") is None:
                    # Could be a VoltageOffset (zero duration, no waveform function)
                    if duration == 0.0:
                        offset_I = pulse_info.get("offset_path_I") or 0.0  # noqa: N806
                        offset_Q = pulse_info.get("offset_path_Q") or 0.0  # noqa: N806
                        vo_events.append(
                            VoltageOffsetEvent(
                                abs_time=abs_time,
                                port=port,
                                clock=clock,
                                is_nonzero=(offset_I != 0.0 or offset_Q != 0.0),
                            )
                        )
                elif duration > 0.0:
                    mw_pulses.append(
                        MicrowavePulseInfo(
                            schedulable_key=schedulable_key,
                            abs_time=abs_time,
                            duration=duration,
                            port=port,
                            clock=clock,
                        )
                    )

    return mw_pulses, vo_events


def _voltage_offset_events_to_pulses(
    events: list[VoltageOffsetEvent],
) -> list[MicrowavePulseInfo]:
    """
    Convert VoltageOffset events into MicrowavePulseInfo windows.

    For each port-clock combination, scans the sorted events to find on/off pairs:
    a non-zero VoltageOffset starts a window, the subsequent zero VoltageOffset
    ends it.  If a window is opened but never closed (no zero event), it is
    silently ignored because the schedule is incomplete for RF switch purposes.

    Parameters
    ----------
    events
        VoltageOffset events collected from the schedule tree.

    Returns
    -------
    list[MicrowavePulseInfo]
        Pulse windows derived from VoltageOffset on/off pairs, suitable for merging
        with regular microwave pulses.

    """
    if not events:
        return []

    events_by_port_clock: dict[tuple[str, str], list[VoltageOffsetEvent]] = defaultdict(list)
    for event in events:
        events_by_port_clock[(event.port, event.clock)].append(event)

    pulses: list[MicrowavePulseInfo] = []
    for (port, clock), port_events in events_by_port_clock.items():
        sorted_events = sorted(port_events, key=lambda e: e.abs_time)

        window_start: float | None = None
        for event in sorted_events:
            if event.is_nonzero and window_start is None:
                window_start = event.abs_time
            elif not event.is_nonzero and window_start is not None:
                # Zero offset closes the current window
                pulses.append(
                    MicrowavePulseInfo(
                        schedulable_key="voltage_offset_window",
                        abs_time=window_start,
                        duration=event.abs_time - window_start,
                        port=port,
                        clock=clock,
                    )
                )
                window_start = None
        # An open window with no closing zero-offset event is ignored

    return pulses


def _collect_all_mw_pulses_recursive(
    schedule: Schedule,
    time_offset: float,
) -> list[MicrowavePulseInfo]:
    """
    Recursively collect all microwave pulses from a schedule tree (no VoltageOffset).

    This is a convenience wrapper around :func:`_collect_all_mw_events_recursive` that
    discards the VoltageOffset events and returns only regular pulse info objects.

    Parameters
    ----------
    schedule
        The schedule to collect from.
    time_offset
        The global absolute time offset for this schedule's schedulables.

    Returns
    -------
    list[MicrowavePulseInfo]
        MW pulse info objects with global absolute times.

    """
    pulses, _ = _collect_all_mw_events_recursive(schedule, time_offset)
    return pulses


def _collect_microwave_pulses(schedule: Schedule) -> list[MicrowavePulseInfo]:
    """
    Collect all microwave pulses from a single schedule level (non-recursive).

    Microwave pulses are identified by having a port ending with ``:mw`` and a
    non-``None`` waveform function (``wf_func``).  VoltageOffset operations
    (``wf_func=None``) and RF switch marker pulses are excluded.

    Parameters
    ----------
    schedule
        The schedule to collect pulses from.

    Returns
    -------
    list[MicrowavePulseInfo]
        List of microwave pulse information.

    """
    # Import here to avoid circular imports
    from quantify_scheduler.operations.control_flow_library import ControlFlowOperation
    from quantify_scheduler.schedules.schedule import ScheduleBase

    mw_pulses = []

    for schedulable_key, schedulable in schedule.schedulables.items():
        operation = schedule.operations.get(schedulable["operation_id"])
        if operation is None:
            continue

        # Skip nested schedules (they are processed recursively)
        if isinstance(operation, ScheduleBase):
            continue

        # Skip control flow operations (they are processed recursively)
        if isinstance(operation, ControlFlowOperation):
            continue

        # Check pulse_info for microwave operations
        pulse_infos = operation.data.get("pulse_info", [])
        for pulse_info in pulse_infos:
            port = pulse_info.get("port", "")
            if port and port.endswith(":mw"):
                # Skip if this is already an RF switch toggle (marker_pulse)
                if pulse_info.get("marker_pulse"):
                    continue

                # Skip if no waveform function (not a real pulse)
                if pulse_info.get("wf_func") is None:
                    continue

                abs_time = schedulable.data.get("abs_time", 0.0) + pulse_info.get("t0", 0.0)
                duration = pulse_info.get("duration", 0.0)
                clock = pulse_info.get("clock", "")

                mw_pulses.append(
                    MicrowavePulseInfo(
                        schedulable_key=schedulable_key,
                        abs_time=abs_time,
                        duration=duration,
                        port=port,
                        clock=clock,
                    )
                )

    return mw_pulses


def _merge_pulses_into_windows(
    pulses: list[MicrowavePulseInfo], ramp_buffer: float
) -> list[tuple[float, float]]:
    """
    Merge adjacent pulses into RF switch windows.

    Pulses within ``2 * ramp_buffer`` of each other are merged into a single window.

    Parameters
    ----------
    pulses
        List of microwave pulses, sorted by start time.
    ramp_buffer
        Buffer time for RF switch.

    Returns
    -------
    list[tuple[float, float]]
        List of (start_time, end_time) tuples for each RF switch window.
        These times are the actual pulse times, not including the ramp buffer.

    """
    if not pulses:
        return []

    windows = []
    current_start = pulses[0].abs_time
    current_end = pulses[0].end_time

    atol = 1e-15
    for pulse in pulses[1:]:
        gap = pulse.abs_time - current_end
        if gap <= 2 * ramp_buffer + atol:
            # Extend the current window
            current_end = max(current_end, pulse.end_time)
        else:
            # Save the current window and start a new one
            windows.append((current_start, current_end))
            current_start = pulse.abs_time
            current_end = pulse.end_time

    # Add the last window
    windows.append((current_start, current_end))

    return windows


def _insert_rf_switch_operation(
    schedule: Schedule,
    port: str,
    clock: str,
    start_time: float,
    end_time: float,
    ramp_buffer: float,
) -> None:
    """
    Insert an RFSwitchToggle operation into the schedule.

    The RF switch is turned on ``ramp_buffer`` before ``start_time`` and remains on
    until ``ramp_buffer`` after ``end_time``.

    Parameters
    ----------
    schedule
        The schedule to insert the operation into.
    port
        The port for the RF switch.
    clock
        The clock for the RF switch.
    start_time
        Start time of the first pulse in the window.
    end_time
        End time of the last pulse in the window.
    ramp_buffer
        Buffer time before/after pulses.

    """
    # Import here to avoid circular imports
    from quantify_scheduler.backends.qblox.operations.rf_switch_toggle import (
        RFSwitchToggle,
    )
    from quantify_scheduler.schedules.schedule import Schedulable

    # Calculate RF switch duration (includes ramp buffers on both sides)
    rf_switch_start = max(0.0, start_time - ramp_buffer)
    rf_switch_duration = (end_time + ramp_buffer) - rf_switch_start

    # Create the RF switch toggle operation
    rf_switch_op = RFSwitchToggle(
        duration=rf_switch_duration,
        port=port,
        clock=clock,
    )

    # Add operation to the schedule
    schedule.operations[rf_switch_op.hash] = rf_switch_op

    # Create a schedulable for the operation
    schedulable_key = str(uuid.uuid4())
    schedulable = Schedulable(name=schedulable_key, operation_id=rf_switch_op.hash)
    schedulable.data["abs_time"] = rf_switch_start

    # Insert the schedulable into the schedule
    schedule.schedulables[schedulable_key] = schedulable
