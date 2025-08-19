# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Schedule helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantify_core.utilities import deprecated
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.collections import make_hash, without
from quantify_scheduler.operations.control_flow_library import ControlFlowOperation
from quantify_scheduler.schedules.schedule import ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler.operations.operation import Operation
    from quantify_scheduler.schedules.schedule import Schedule


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
