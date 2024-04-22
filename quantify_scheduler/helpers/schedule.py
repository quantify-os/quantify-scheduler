# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Schedule helper functions."""
from __future__ import annotations

from itertools import chain, count
from typing import TYPE_CHECKING, Any, Hashable

import numpy as np

from quantify_scheduler.helpers.collections import make_hash, without
from quantify_scheduler.schedules.schedule import (
    AcquisitionChannelMetadata,
    AcquisitionMetadata,
    CompiledSchedule,
    Schedule,
    ScheduleBase,
)

if TYPE_CHECKING:
    from quantify_scheduler.operations.operation import Operation


def get_pulse_uuid(pulse_info: dict[str, Any], excludes: list[str] = None) -> int:
    """
    Return an unique identifier for a pulse.

    Parameters
    ----------
    pulse_info
        The pulse information dictionary.
    excludes
        A list of keys to exclude.

    Returns
    -------
    :
        The uuid hash.
    """
    if excludes is None:
        excludes = ["t0"]

    return make_hash(without(pulse_info, excludes))


def get_acq_uuid(acq_info: dict[str, Any]) -> int:
    """
    Return an unique identifier for a acquisition protocol.

    Parameters
    ----------
    acq_info
        The acquisition information dictionary.

    Returns
    -------
    :
        The uuid hash.
    """
    return make_hash(without(acq_info, ["t0", "waveforms"]))


def get_total_duration(schedule: ScheduleBase) -> float:
    """
    Return the total schedule duration in seconds.

    Parameters
    ----------
    schedule
        The schedule.

    Returns
    -------
    :
        Duration in seconds.
    """
    if len(schedule.schedulables) == 0:
        return 0.0

    def _get_operation_end(pair: tuple[int, dict]) -> float:
        """Return the operations end time in seconds."""
        (timeslot_index, _) = pair
        return get_operation_end(
            schedule,
            timeslot_index,
        )

    operations_ends = map(
        _get_operation_end,
        enumerate(schedule.schedulables.values()),
    )

    return max(
        operations_ends,
        default=0,
    )


def get_operation_start(
    schedule: CompiledSchedule,
    timeslot_index: int,
) -> float:
    """
    Return the start of an operation in seconds.

    Parameters
    ----------
    schedule
        The schedule
    timeslot_index
        The index of the operation in the schedule.

    Returns
    -------
    :
        The Operation start time in Seconds.
    """
    if len(schedule.schedulables) == 0:
        return 0.0

    schedulable = list(schedule.schedulables.values())[timeslot_index]
    operation = schedule.operations[schedulable["operation_id"]]

    t0: float = schedulable["abs_time"]

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
    schedule: ScheduleBase,
    timeslot_index: int,
) -> float:
    """
    Return the end of an operation in seconds.

    Parameters
    ----------
    schedule
        The schedule
    timeslot_index
        The index of the operation in the schedule.

    Returns
    -------
    :
        The Operation end time in Seconds.
    """
    if len(schedule.schedulables) == 0:
        return 0.0

    schedulable = list(schedule.schedulables.values())[timeslot_index]
    operation: Operation = schedule.operations[schedulable["operation_id"]]
    t0: float = schedulable["abs_time"]

    return t0 + operation.duration


def get_port_timeline(
    schedule: CompiledSchedule,
) -> dict[str, dict[int, list[int]]]:
    """
    Return a new dictionary containing the port timeline.

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
    port_timeline_dict: dict[str, dict[int, list[int]]] = {}

    # Sort timing constraints based on abs_time and keep the original index.
    schedulables_map = dict(
        sorted(
            map(
                lambda pair: (pair[0], pair[1]),
                enumerate(schedule.schedulables.values()),
            ),
            key=lambda pair: pair[1]["abs_time"],
        )
    )

    for timeslot_index, schedulable in schedulables_map.items():
        operation = schedule.operations[schedulable["operation_id"]]
        abs_time = schedulable["abs_time"]

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
            key=lambda pair: abs_time + pair[1]["t0"],
        ):
            port = str(info["port"])
            if port not in port_timeline_dict:
                port_timeline_dict[port] = {}

            if timeslot_index not in port_timeline_dict[port]:
                port_timeline_dict[port][timeslot_index] = []

            port_timeline_dict[port][timeslot_index].append(uuid)

    return port_timeline_dict


def get_schedule_time_offset(
    schedule: CompiledSchedule,
    port_timeline_dict: dict[str, dict[int, list[int]]],
) -> float:
    """
    Return the start time in seconds of the first pulse in the CompiledSchedule.

    The "None" port containing the Reset Operation will be ignored.

    Parameters
    ----------
    schedule
        The schedule.
    port_timeline_dict
        Dictionary containing port timelines.

    Returns
    -------
    :
        The operation t0 in seconds.
    """
    return min(
        map(
            lambda port: (
                get_operation_start(
                    schedule,
                    timeslot_index=next(iter(port_timeline_dict[port])),
                )
                if port != "None"
                else np.inf
            ),
            port_timeline_dict.keys(),
        ),
        default=0,
    )


def get_pulse_info_by_uuid(
    schedule: CompiledSchedule,
) -> dict[int, dict[str, Any]]:
    """
    Return a lookup dictionary of pulses with its hash as unique identifiers.

    Parameters
    ----------
    schedule
        The schedule.
    """
    pulseid_pulseinfo_dict: dict[int, dict[str, Any]] = {}
    for schedulable in schedule.schedulables.values():
        operation = schedule.operations[schedulable["operation_id"]]
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


def get_acq_info_by_uuid(schedule: CompiledSchedule) -> dict[int, dict[str, Any]]:
    """
    Return a lookup dictionary of unique identifiers of acquisition information.

    Parameters
    ----------
    schedule
        The schedule.
    """
    acqid_acqinfo_dict: dict[int, dict[str, Any]] = {}
    for schedulable in schedule.schedulables.values():
        operation = schedule.operations[schedulable["operation_id"]]

        for acq_info in operation["acquisition_info"]:
            acq_id = get_acq_uuid(acq_info)
            if acq_id in acqid_acqinfo_dict:
                # Unique acquisition info already populated in the dictionary.
                continue

            acqid_acqinfo_dict[acq_id] = acq_info

    return acqid_acqinfo_dict


def extract_acquisition_metadata_from_schedule(
    schedule: Schedule,
) -> AcquisitionMetadata:
    """
    Extract acquisition metadata from a schedule.

    This function operates under certain assumptions with respect to the schedule.

    - The acquisition_metadata should be sufficient to initialize the xarray dataset
      (described in quantify-core !212) that executing the schedule will result in.
    - All measurements in the schedule use the same acquisition protocol.
    - The used acquisition index channel combinations for each measurement are unique.
    - The used acquisition indices for each channel are the same.
    - When :class:`~quantify_scheduler.enums.BinMode` is :code:`APPEND` The number of
      data points per acquisition index assumed to be given by the
      schedule's repetition property. This implies no support for feedback (conditional
      measurements).

    Parameters
    ----------
    schedule
        schedule containing measurements from which acquisition metadata can be
        extracted.

    Returns
    -------
    :
        The acquisition metadata provides a summary of the
        acquisition protocol, bin-mode, return-type and acquisition indices
        of the acquisitions in the schedule.

    Raises
    ------
    AssertionError

        If not all acquisition protocols in a schedule are the same.
        If not all acquisitions use the same bin_mode.
        If the return type of the acquisitions is different.


    """  # FIXME update when quantify-core!212 spec is ready
    # a dictionary containing the acquisition indices used for each channel
    acqid_acqinfo_dict = get_acq_info_by_uuid(schedule)

    return extract_acquisition_metadata_from_acquisition_protocols(
        acquisition_protocols=list(acqid_acqinfo_dict.values()),
        repetitions=schedule.repetitions,
    )


def extract_acquisition_metadata_from_acquisition_protocols(
    acquisition_protocols: list[dict[str, Any]], repetitions: int
) -> AcquisitionMetadata:
    """
    Private function containing the logic of extract_acquisition_metadata_from_schedule.

    The logic is factored out as to work around limitations of the different interfaces
    required.

    Parameters
    ----------
    acquisition_protocols
        A list of acquisition protocols.
    repetitions
        How many times the acquisition was repeated.
    """
    acq_channels_metadata: dict[int, AcquisitionChannelMetadata] = {}

    # Generating hardware indices this way is intended as a temporary solution.
    # Proper solution: SE-298.
    acq_channel_to_numeric_key: dict[Hashable, int] = {}
    numeric_key_counter = count()

    def _to_numeric_key(acq_channel: Hashable) -> int:
        nonlocal numeric_key_counter
        if acq_channel not in acq_channel_to_numeric_key:
            acq_channel_to_numeric_key[acq_channel] = next(numeric_key_counter)
        return acq_channel_to_numeric_key[acq_channel]

    for i, acq_protocol in enumerate(acquisition_protocols):
        if i == 0:
            # the protocol and bin mode of the first
            protocol = acq_protocol["protocol"]
            bin_mode = acq_protocol["bin_mode"]
            acq_return_type = acq_protocol["acq_return_type"]

        # test limitation: all acquisition protocols in a schedule must be of
        # the same kind
        if (
            acq_protocol["protocol"] != protocol
            or acq_protocol["bin_mode"] != bin_mode
            or acq_protocol["acq_return_type"] != acq_return_type
        ):
            raise RuntimeError(
                "Acquisition protocols or bin mode or acquisition return type are not"
                " of the same kind. "
                f"Expected protocol: {acquisition_protocols[0]}. "
                f"Offending: {i}, {acq_protocol} \n"
            )

        # add the individual channel
        acq_channel = acq_protocol["acq_channel"]
        numeric_key = _to_numeric_key(acq_channel)
        if numeric_key not in acq_channels_metadata:
            acq_channels_metadata[numeric_key] = AcquisitionChannelMetadata(
                acq_channel=acq_channel, acq_indices=[]
            )
        acq_indices = acq_protocol["acq_index"]
        acq_channels_metadata[numeric_key].acq_indices.append(acq_indices)

    # combine the information in the acq metadata dataclass.
    acq_metadata = AcquisitionMetadata(
        acq_protocol=protocol,
        bin_mode=bin_mode,
        acq_channels_metadata=acq_channels_metadata,
        acq_return_type=acq_return_type,
        repetitions=repetitions,
    )
    return acq_metadata


def _extract_port_clocks_used(operation: Operation | Schedule) -> set[tuple]:
    """Extracts which port-clock combinations are used in an operation or schedule."""
    if isinstance(operation, ScheduleBase):
        port_clocks_used = set()
        for op_data in operation.operations.values():
            port_clocks_used |= _extract_port_clocks_used(op_data)
        return port_clocks_used
    elif operation.valid_pulse or operation.valid_acquisition:
        port_clocks_used = set()
        for op_info in operation["pulse_info"] + operation["acquisition_info"]:
            if (port := op_info["port"]) is None or (clock := op_info["clock"]) is None:
                continue
            port_clocks_used.add((port, clock))
        return port_clocks_used
    else:
        raise RuntimeError(
            f"Operation {operation.name} is not a valid pulse or acquisition."
            f" Please check whether the device compilation has been performed successfully."
            f" Operation data: {repr(operation)}"
        )
