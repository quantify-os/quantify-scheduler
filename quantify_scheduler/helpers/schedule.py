# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Schedule helper functions."""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Any

from quantify_core.utilities import deprecated
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.collections import make_hash, without
from quantify_scheduler.operations.control_flow_library import ControlFlowOperation
from quantify_scheduler.schedules.schedule import (
    AcquisitionChannelMetadata,
    AcquisitionMetadata,
    Schedule,
    ScheduleBase,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from quantify_scheduler.operations.operation import Operation


def get_pulse_uuid(pulse_info: dict[str, Any], excludes: list[str] | None = None) -> int:
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


def _generate_acq_info_by_uuid(
    operation: Operation | ScheduleBase, acqid_acqinfo_dict: dict
) -> None:
    if isinstance(operation, ScheduleBase):
        for schedulable in operation.schedulables.values():
            inner_operation = operation.operations[schedulable["operation_id"]]
            _generate_acq_info_by_uuid(inner_operation, acqid_acqinfo_dict)
    elif isinstance(operation, ControlFlowOperation):
        _generate_acq_info_by_uuid(operation.body, acqid_acqinfo_dict)
    else:
        for acq_info in operation["acquisition_info"]:
            acq_id = get_acq_uuid(acq_info)
            if acq_id in acqid_acqinfo_dict:
                # Unique acquisition info already populated in the dictionary.
                continue

            acqid_acqinfo_dict[acq_id] = acq_info


def get_acq_info_by_uuid(schedule: Schedule) -> dict[int, dict[str, Any]]:
    """
    Return a lookup dictionary of unique identifiers of acquisition information.

    Parameters
    ----------
    schedule
        The schedule.

    """
    acqid_acqinfo_dict: dict[int, dict[str, Any]] = {}
    _generate_acq_info_by_uuid(schedule, acqid_acqinfo_dict)

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
    # TODO: https://gitlab.com/groups/quantify-os/-/epics/5 for proper solution.
    acq_channel_to_numeric_key: dict[Hashable, int] = {}
    numeric_key_counter = count()

    def _to_numeric_key(acq_channel: Hashable) -> int:
        nonlocal numeric_key_counter
        if acq_channel not in acq_channel_to_numeric_key:
            acq_channel_to_numeric_key[acq_channel] = next(numeric_key_counter)
        return acq_channel_to_numeric_key[acq_channel]

    # Extract information from first protocol
    protocol = acquisition_protocols[0]["protocol"]
    bin_mode = acquisition_protocols[0]["bin_mode"]
    acq_return_type = acquisition_protocols[0]["acq_return_type"]

    for acq_protocol in acquisition_protocols:
        # test limitation: all acquisition protocols in a schedule must be of
        # the same kind
        conflicts = []

        if acq_protocol["protocol"] != protocol:
            conflicts.append(
                f"acquisition protocol: found {protocol} and {acq_protocol['protocol']}"
            )
        if acq_protocol["bin_mode"] != bin_mode:
            conflicts.append(
                f"bin mode: found {bin_mode.__class__.__name__}.{bin_mode.name} and "
                f"{acq_protocol['bin_mode'].__class__.__name__}.{acq_protocol['bin_mode'].name}"
            )

        if conflicts:
            raise RuntimeError(
                "All acquisitions in a Schedule must be of the same kind:\n" + "\n".join(conflicts)
            )

        # add the individual channel
        acq_channel = acq_protocol["acq_channel"]
        numeric_key = _to_numeric_key(acq_channel)
        if numeric_key not in acq_channels_metadata:
            acq_channels_metadata[numeric_key] = AcquisitionChannelMetadata(
                acq_channel=acq_channel, acq_indices=[]
            )
        acq_indices = acq_protocol.get("acq_index_legacy", acq_protocol["acq_index"])
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


@deprecated(
    "0.25",
    "_extract_port_clocks_used has been moved to "
    "Operation.get_used_port_clocks and ScheduleBase.get_used_port_clocks",
)
def _extract_port_clocks_used(operation: Operation | Schedule) -> set[tuple]:
    """Extracts which port-clock combinations are used in an operation or schedule."""
    return operation.get_used_port_clocks()


def _is_acquisition_binned_average(protocol: str, bin_mode: BinMode) -> bool:
    return (
        protocol
        in (
            "SSBIntegrationComplex",
            "WeightedIntegratedSeparated",
            "NumericalSeparatedWeightedIntegration",
            "NumericalWeightedIntegration",
            "ThresholdedAcquisition",
            "WeightedThresholdedAcquisition",
            "Timetag",
        )
        and bin_mode == BinMode.AVERAGE
    ) or (
        protocol
        in (
            "TriggerCount",
            "ThresholdedTriggerCount",
            "DualThresholdedTriggerCount",
        )
        and bin_mode == BinMode.SUM
    )


def _is_acquisition_binned_append(protocol: str, bin_mode: BinMode) -> bool:
    return (
        protocol
        in (
            "SSBIntegrationComplex",
            "WeightedIntegratedSeparated",
            "NumericalSeparatedWeightedIntegration",
            "NumericalWeightedIntegration",
            "ThresholdedAcquisition",
            "WeightedThresholdedAcquisition",
            "Timetag",
            "TriggerCount",
            "ThresholdedTriggerCount",
            "DualThresholdedTriggerCount",
        )
        and bin_mode == BinMode.APPEND
    )
