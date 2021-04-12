# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Schedule helper functions."""
from __future__ import annotations

from itertools import chain
from typing import Any, Dict, List

import numpy as np
import quantify.utilities.general as general

from quantify.scheduler import types


def get_pulse_uuid(pulse_info: Dict[str, Any]) -> int:
    """
    Returns an unique identifier for a pulse.

    Parameters
    ----------
    pulse_info :
        The pulse information dictionary.

    Returns
    -------
    int
        The uuid hash.
    """
    return general.make_hash(general.without(pulse_info, ["t0"]))


def get_acq_uuid(acq_info: Dict[str, Any]) -> int:
    """
    Returns an unique identifier for a acquisition protocol.

    Parameters
    ----------
    acq_info :
        The acquisition information dictionary.

    Returns
    -------
    int
        The uuid hash.
    """
    return general.make_hash(general.without(acq_info, ["t0", "waveforms"]))


def get_total_duration(schedule: types.Schedule) -> float:
    """
    Returns the total schedule duration in seconds.

    Parameters
    ----------
    schedule :
        The schedule.

    Returns
    -------
    float
        Duration in seconds.
    """
    if len(schedule.timing_constraints) == 0:
        return 0.0

    t_constr = schedule.timing_constraints[-1]
    operation = schedule.operations[t_constr["operation_hash"]]

    t0 = t_constr["abs_time"]
    duration = 0

    pulse_info: dict = (
        operation["pulse_info"][-1]
        if len(operation["pulse_info"]) > 0
        else {"t0": -1, "duration": 0}
    )
    acq_info: dict = (
        operation["acquisition_info"][-1]
        if len(operation["acquisition_info"]) > 0
        else {"t0": -1, "duration": 0}
    )

    if acq_info["t0"] != -1 and acq_info["t0"] > pulse_info["t0"]:
        t0 += acq_info["t0"]
        duration = acq_info["duration"]
    elif pulse_info["t0"] >= 0:
        t0 += pulse_info["t0"]
        duration = pulse_info["duration"]
    else:
        raise ValueError("Undefined 't0' in pulse_info or acquisition_info!")

    return t0 + duration


def get_operation_start(
    schedule: types.Schedule,
    timeslot_index: int,
) -> float:
    """
    Returns the start of an operation in seconds.

    Parameters
    ----------
    schedule :
    timeslot_index :

    Returns
    -------
    float
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
    schedule :
    timeslot_index :

    Returns
    -------
    float
        The Operation start time in Seconds.
    """
    if len(schedule.timing_constraints) == 0:
        return 0.0

    t_constr = schedule.timing_constraints[timeslot_index]
    operation = schedule.operations[t_constr["operation_hash"]]

    t0: float = get_operation_start(schedule, timeslot_index)

    pulse_info: dict = (
        operation["pulse_info"][-1]
        if len(operation["pulse_info"]) > 0
        else {"t0": -1, "duration": 0}
    )
    acq_info: dict = (
        operation["acquisition_info"][-1]
        if len(operation["acquisition_info"]) > 0
        else {"t0": -1, "duration": 0}
    )

    if acq_info["t0"] != -1 and acq_info["t0"] > pulse_info["t0"]:
        t0 += acq_info["duration"]
    elif pulse_info["t0"] >= 0:
        t0 += pulse_info["duration"]

    return t0


def get_port_timeline(
    schedule: types.Schedule,
) -> Dict[str, Dict[int, List[int]]]:
    """
    Returns a new dictionary containing the timeline of
    pulses, readout- and acquisition pulses of a port.

    Example:
    ```
    print(port_timeline_dict)
    # { {'q0:mw', {0, [123456789]}},
    # ... }
    ```

    Parameters
    ----------
    schedule :
        The schedule.

    Returns
    -------
    Dict[str, Dict[int, List[int]]]
    """
    port_timeline_dict: Dict[str, Dict[int, List[int]]] = dict()

    for timeslot_index, t_constr in enumerate(schedule.timing_constraints):
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

        # Sort pulses and acquisitions on time.
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
    schedule :
    port_timeline_dict :

    Returns
    -------
    float
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
    schedule :
        The schedule.

    Returns
    -------
    Dict[int, Dict[str, Any]]
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
    schedule :
        The schedule.

    Returns
    -------
    Dict[int, Dict[str, Any]]
    """
    acqid_acqinfo_dict: Dict[int, Dict[str, Any]] = dict()
    for t_constr in schedule.timing_constraints:
        operation = schedule.operations[t_constr["operation_hash"]]

        for acq_info in operation["acquisition_info"]:
            acq_id = get_acq_uuid(acq_info)
            if acq_id in acqid_acqinfo_dict:
                # Unique acquition info already populated in the dictionary.
                continue

            acqid_acqinfo_dict[acq_id] = acq_info

    return acqid_acqinfo_dict
