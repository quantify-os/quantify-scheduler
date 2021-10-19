# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Schedule helper functions."""
from __future__ import annotations

from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from quantify_core.utilities import general
from quantify_scheduler import types
from quantify_scheduler.helpers import waveforms as waveform_helpers
from quantify_scheduler.types import ScheduleBase, AcquisitionMetadata

if TYPE_CHECKING:
    from quantify_scheduler.backends.types import qblox


class CachedSchedule:
    """
    The CachedSchedule class wraps around the CompiledSchedule
    class and populates the lookup dictionaries that are
    used for compilation of the backends.
    """

    _start_offset_in_seconds: Optional[float] = None
    _total_duration_in_seconds: Optional[float] = None

    def __init__(self, schedule: types.CompiledSchedule):
        self._schedule = schedule

        self._pulseid_pulseinfo_dict = get_pulse_info_by_uuid(schedule)
        self._pulseid_waveformfn_dict = waveform_helpers.get_waveform_by_pulseid(
            schedule
        )
        self._port_timeline_dict = get_port_timeline(schedule)
        self._acqid_acqinfo_dict = get_acq_info_by_uuid(schedule)

    @property
    def schedule(self) -> types.CompiledSchedule:
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


def get_total_duration(schedule: types.CompiledSchedule) -> float:
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
    schedule: types.CompiledSchedule,
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
    operation = schedule.operations[t_constr["operation_repr"]]

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
    schedule: types.CompiledSchedule,
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
    operation: types.Operation = schedule.operations[t_constr["operation_repr"]]
    t0: float = t_constr["abs_time"]

    return t0 + operation.duration


def get_port_timeline(
    schedule: types.CompiledSchedule,
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

    # Sort timing constraints based on abs_time and keep the original index.
    timing_constrains_map = dict(
        sorted(
            map(
                lambda pair: (pair[0], pair[1]), enumerate(schedule.timing_constraints)
            ),
            key=lambda pair: pair[1]["abs_time"],
        )
    )

    for timeslot_index, t_constr in timing_constrains_map.items():
        operation = schedule.operations[t_constr["operation_repr"]]
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
    schedule: types.CompiledSchedule,
    port_timeline_dict: Dict[str, Dict[int, List[int]]],
) -> float:
    """
    Returns the start time in seconds of the first pulse
    in the CompiledSchedule. The "None" port containing the Reset
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


def get_pulse_info_by_uuid(
    schedule: types.CompiledSchedule,
) -> Dict[int, Dict[str, Any]]:
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
        operation = schedule.operations[t_constr["operation_repr"]]
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


def get_acq_info_by_uuid(schedule: types.CompiledSchedule) -> Dict[int, Dict[str, Any]]:
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
        operation = schedule.operations[t_constr["operation_repr"]]

        for acq_info in operation["acquisition_info"]:
            acq_id = get_acq_uuid(acq_info)
            if acq_id in acqid_acqinfo_dict:
                # Unique acquisition info already populated in the dictionary.
                continue

            acqid_acqinfo_dict[acq_id] = acq_info

    return acqid_acqinfo_dict


def extract_acquisition_metadata_from_schedule(
    schedule: ScheduleBase,
) -> AcquisitionMetadata:
    """
    Extracts acquisition metadata from a schedule.

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


    """  # FIXME update when quantify-core!212 spec is ready # pylint: disable=fixme
    # convert to a cached schedule to have useful metadata available.
    cached_sched = CachedSchedule(schedule)

    # a dictionary containing the acquisition indices used for each channel

    return _extract_acquisition_metadata_from_acquisition_protocols(
        list(cached_sched.acqid_acqinfo_dict.values())
    )


def _extract_acquisition_metadata_from_acquisition_protocols(
    acquisition_protocols: List[Dict[str, Any]],
) -> AcquisitionMetadata:
    """
    Private function containing the logic of extract_acquisition_metadata_from_schedule.
    The logic is factored out as to work around limitations of the different interfaces
    required.

    Parameters
    ----------
    acquisition_protocols
        A list of acquisition protocols.
    """
    acq_indices: Dict[int, List[int]] = {}

    for i, acq_protocol in enumerate(acquisition_protocols):
        if i == 0:
            # the protocol and bin mode of the first
            protocol = acq_protocol["protocol"]
            bin_mode = acq_protocol["bin_mode"]
            acq_return_type = acq_protocol["acq_return_type"]

        # test limitation: all acquisition protocols in a schedule must be of
        # the same kind
        assert acq_protocol["protocol"] == protocol
        assert acq_protocol["bin_mode"] == bin_mode
        assert acq_protocol["acq_return_type"] == acq_return_type

        # add the individual channel
        if acq_protocol["acq_channel"] not in acq_indices.keys():
            acq_indices[acq_protocol["acq_channel"]] = []

        acq_indices[acq_protocol["acq_channel"]].append(acq_protocol["acq_index"])

    # combine the information in the acq metada dataclass.
    acq_metadata = AcquisitionMetadata(
        acq_protocol=protocol,
        bin_mode=bin_mode,
        acq_indices=acq_indices,
        acq_return_type=acq_return_type,
    )
    return acq_metadata


def _extract_acquisition_metadata_from_acquisitions(
    acquisitions: List[qblox.OpInfo],
) -> AcquisitionMetadata:
    """
    Private variant of extract_acquisition_metadata_from_schedule explicitly for use
    with the qblox assembler backend.
    """
    acquisition_protocols = [acq.data for acq in acquisitions]
    return _extract_acquisition_metadata_from_acquisition_protocols(
        acquisition_protocols
    )