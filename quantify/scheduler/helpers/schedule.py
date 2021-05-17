# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Schedule helper functions."""
from __future__ import annotations

from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import quantify.utilities.general as general
from quantify.scheduler import types
from quantify.scheduler.helpers import waveforms as waveform_helpers


class CachedSchedule:
    """
    The CachedSchedule class wraps around the types.Schedule
    class and populates the lookup dictionaries that are
    used for compilation of the backends.
    """

    _start_offset_in_seconds: Optional[float] = None
    _total_duration_in_seconds: Optional[float] = None

    def __init__(self, schedule: types.Schedule):
        self._schedule = schedule

        self._pulseid_pulseinfo_dict = get_pulse_info_by_uuid(schedule)
        self._pulseid_waveformfn_dict = waveform_helpers.get_waveform_by_pulseid(
            schedule
        )
        self._port_timeline_dict = get_port_timeline(schedule)
        self._acqid_acqinfo_dict = get_acq_info_by_uuid(schedule)

    @property
    def schedule(self) -> types.Schedule:
        """
        Returns schedule.
        """
        return self._schedule

    @property
    def pulseid_pulseinfo_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Returns the pulse info lookup table.
        """
        return self._pulseid_pulseinfo_dict

    @property
    def pulseid_waveformfn_dict(self) -> Dict[int, waveform_helpers.GetWaveformPartial]:
        """
        Returns waveform function lookup table.
        """
        return self._pulseid_waveformfn_dict

    @property
    def acqid_acqinfo_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Returns the acquisition info lookup table.
        """
        return self._acqid_acqinfo_dict

    @property
    def port_timeline_dict(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Returns the timeline per port lookup dictionary.
        """
        return self._port_timeline_dict

    @property
    def start_offset_in_seconds(self) -> float:
        """
        Returns the schedule start offset in seconds.
        The start offset is determined by a Reset operation
        at the start of one of the ports.
        """
        if self._start_offset_in_seconds is None:
            self._start_offset_in_seconds = get_schedule_time_offset(
                self.schedule, self.port_timeline_dict
            )

        return self._start_offset_in_seconds

    @property
    def total_duration_in_seconds(self) -> float:
        """
        Returns the schedule total duration in seconds.
        """
        if self._total_duration_in_seconds is None:
            self._total_duration_in_seconds = get_total_duration(self.schedule)

        return self._total_duration_in_seconds


def get_pulse_uuid(pulse_info: Dict[str, Any], excludes: List[str] = None) -> int:
    """
    Returns an unique identifier for a pulse.

    Parameters
    ----------
    pulse_info
        The pulse information dictionary.

    Returns
    -------
    :
        The uuid hash.
    """
    if excludes is None:
        excludes = ["t0"]

    return general.make_hash(general.without(pulse_info, excludes))


def get_acq_uuid(acq_info: Dict[str, Any]) -> int:
    """
    Returns an unique identifier for a acquisition protocol.

    Parameters
    ----------
    acq_info
        The acquisition information dictionary.

    Returns
    -------
    :
        The uuid hash.
    """
    return general.make_hash(general.without(acq_info, ["t0", "waveforms"]))


def get_total_duration(schedule: types.Schedule) -> float:
    """
    Returns the total schedule duration in seconds.

    Parameters
    ----------
    schedule
        The schedule.

    Returns
    -------
    :
        Duration in seconds.
    """
    if len(schedule.timing_constraints) == 0:
        return 0.0

    def _get_operation_end(pair: Tuple[int, dict]) -> float:
        """Returns the operations end time in seconds."""
        (timeslot_index, _) = pair
        return get_operation_end(
            schedule,
            timeslot_index,
        )

    operations_ends = map(
        _get_operation_end,
        enumerate(schedule.timing_constraints),
    )

    return max(
        operations_ends,
        default=0,
    )


def get_operation_start(
    schedule: types.Schedule,
    timeslot_index: int,
) -> float:
    """
    Returns the start of an operation in seconds.

    Parameters
    ----------
    schedule
    timeslot_index

    Returns
    -------
    :
        The Operation start time in Seconds.
    """
    if len(schedule.timing_constraints) == 0:
        return 0.0

    t_constr = schedule.timing_constraints[timeslot_index]
    operation = schedule.operations[t_constr["operation_hash"]]

    t0: float = t_constr["abs_time"]

    pulse_info: dict = (
        operation["pulse_info"][0]
        if len(operation["pulse_info"]) > 0
        else {"t0": -1, "duration": 0}
    )
    acq_info: dict = (
        operation["acquisition_info"][0]
        if len(operation["acquisition_info"]) > 0
        else {"t0": -1, "duration": 0}
    )

    if acq_info["t0"] != -1 and acq_info["t0"] < pulse_info["t0"]:
        t0 += acq_info["t0"]
    elif pulse_info["t0"] >= 0:
        t0 += pulse_info["t0"]

    return t0


def get_operation_end(
    schedule: types.Schedule,
    timeslot_index: int,
) -> float:
    """
    Returns the end of an operation in seconds.

    Parameters
    ----------
    schedule
    timeslot_index

    Returns
    -------
    :
        The Operation start time in Seconds.
    """
    if len(schedule.timing_constraints) == 0:
        return 0.0

    t_constr = schedule.timing_constraints[timeslot_index]
    operation: types.Operation = schedule.operations[t_constr["operation_hash"]]
    t0: float = t_constr["abs_time"]

    return t0 + operation.duration


def get_port_timeline(
    schedule: types.Schedule,
) -> Dict[str, Dict[int, List[int]]]:
    """
    Returns a new dictionary containing the timeline of
    pulses, readout- and acquisition pulses of a port.

    Using iterators on this collection enables sorting.

    .. code-block::

        print(port_timeline_dict)
        # { {'q0:mw', {0, [123456789]}},
        # ... }

        # Sorted items.
        print(port_timeline_dict.items())

    Parameters
    ----------
    schedule
        The schedule.
    """
    port_timeline_dict: Dict[str, Dict[int, List[int]]] = dict()

    # Sort timing containts based on abs_time and keep the original index.
    timing_constrains_map = dict(
        sorted(
            map(
                lambda pair: (pair[0], pair[1]), enumerate(schedule.timing_constraints)
            ),
            key=lambda pair: pair[1]["abs_time"],
        )
    )

    for timeslot_index, t_constr in timing_constrains_map.items():
        operation = schedule.operations[t_constr["operation_hash"]]
        abs_time = t_constr["abs_time"]

        pulse_info_iter = map(
            lambda pulse_info: (get_pulse_uuid(pulse_info), pulse_info),
            operation["pulse_info"],
        )
        acq_info_iter = map(
            lambda acq_info: (get_acq_uuid(acq_info), acq_info),
            operation["acquisition_info"],
        )

        # Sort pulses and acquisitions within an operation.
        for uuid, info in sorted(
            chain(pulse_info_iter, acq_info_iter),
            key=lambda pair: abs_time  # pylint: disable=cell-var-from-loop
            + pair[1]["t0"],
        ):
            port = str(info["port"])
            if port not in port_timeline_dict:
                port_timeline_dict[port] = dict()

            if timeslot_index not in port_timeline_dict[port]:
                port_timeline_dict[port][timeslot_index] = list()

            port_timeline_dict[port][timeslot_index].append(uuid)

    return port_timeline_dict


def get_schedule_time_offset(
    schedule: types.Schedule, port_timeline_dict: Dict[str, Dict[int, List[int]]]
) -> float:
    """
    Returns the start time in seconds of the first pulse
    in the Schedule. The "None" port containing the Reset
    Operation will be ignored.

    Parameters
    ----------
    schedule
    port_timeline_dict

    Returns
    -------
    :
        The operation t0 in seconds.
    """
    return min(
        map(
            lambda port: get_operation_start(
                schedule,
                timeslot_index=next(iter(port_timeline_dict[port])),
            )
            if port != "None"
            else np.inf,
            port_timeline_dict.keys(),
        ),
        default=0,
    )


def get_pulse_info_by_uuid(schedule: types.Schedule) -> Dict[int, Dict[str, Any]]:
    """
    Returns a lookup dictionary of pulses with its
    hash as unique identifiers.

    Parameters
    ----------
    schedule
        The schedule.
    """
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]] = dict()
    for t_constr in schedule.timing_constraints:
        operation = schedule.operations[t_constr["operation_hash"]]
        for pulse_info in operation["pulse_info"]:
            pulse_id = get_pulse_uuid(pulse_info)
            if pulse_id in pulseid_pulseinfo_dict:
                # Unique pulse info already populated in the dictionary.
                continue

            pulseid_pulseinfo_dict[pulse_id] = pulse_info

        for acq_info in operation["acquisition_info"]:
            for pulse_info in acq_info["waveforms"]:
                pulse_id = get_pulse_uuid(pulse_info)
                if pulse_id in pulseid_pulseinfo_dict:
                    # Unique pulse info already populated in the dictionary.
                    continue

                pulseid_pulseinfo_dict[pulse_id] = pulse_info

    return pulseid_pulseinfo_dict


def get_acq_info_by_uuid(schedule: types.Schedule) -> Dict[int, Dict[str, Any]]:
    """
    Returns a lookup dictionary of unique identifiers
    of acquisition information.

    Parameters
    ----------
    schedule
        The schedule.
    """
    acqid_acqinfo_dict: Dict[int, Dict[str, Any]] = dict()
    for t_constr in schedule.timing_constraints:
        operation = schedule.operations[t_constr["operation_hash"]]

        for acq_info in operation["acquisition_info"]:
            acq_id = get_acq_uuid(acq_info)
            if acq_id in acqid_acqinfo_dict:
                # Unique acquisition info already populated in the dictionary.
                continue

            acqid_acqinfo_dict[acq_id] = acq_info

    return acqid_acqinfo_dict
