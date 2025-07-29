# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions to generate acq_indices."""

from __future__ import annotations

import warnings
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Union

from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.schedule import (
    _is_acquisition_binned_append,
    _is_acquisition_binned_average,
)
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    LoopOperation,
)
from quantify_scheduler.schedules.schedule import (
    AcquisitionChannelData,
    AcquisitionChannelsData,
    ScheduleBase,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from quantify_scheduler.operations.operation import Operation

SchedulableLabel = Union[str, None]
FullSchedulableLabel = tuple[SchedulableLabel, ...]
AcquisitionIndices = Union[int, list[int]]
SchedulableLabelToAcquisitionIndex = dict[tuple[FullSchedulableLabel, int], AcquisitionIndices]
"""
A mapping from schedulables to an acquisition index.

This mapping helps the backend to figure out which
binned acquisition corresponds to which acquisition index.
Note, it maps the (full schedulable label, acq_num_in_operation) to the bin,
where acq_num_in_operation is the i.th acquisition in the operation;
in other words: the i.th element in the acquisition_info.
Only defined for binned acquisitions, and backend independent.

For control flows, the `None` in the schedulable label refers to the `body`
of the control flow. This is for future proofing, if control flows were extended
to include maybe multiple suboperations.
"""


def _prod(iterable: Iterable[int]) -> int:
    return reduce(mul, iterable, 1)


def _generate_acq_channels_data_binned_average(
    acq_channel_data: AcquisitionChannelData,
    schedulable_label_to_acq_index: SchedulableLabelToAcquisitionIndex,
    full_schedulable_label: FullSchedulableLabel,
    coords: dict,
    acq_num_in_operation: int,
    acq_channel: Hashable,
    acq_index: int | None,
) -> None:
    """
    Generates the acquisition channel data, and updates acq_channel_data,
    and updates schedulable_label_to_acq_index for average bin mode.
    """
    assert isinstance(acq_channel_data.coords, list)

    if acq_index is not None and acq_index != len(acq_channel_data.coords):
        raise ValueError(
            f"Found invalid {acq_index=} for {acq_channel=}. "
            f"Make sure that each explicitly defined acq_index "
            f"starts at 0, and increments by 1 for each new acquisition "
            f"within the same acquisition channel, ordered by time.",
        )
    new_acq_index = len(acq_channel_data.coords)
    schedulable_label_to_acq_index[(full_schedulable_label, acq_num_in_operation)] = new_acq_index
    acq_channel_data.coords.append(coords)


def _generate_acq_channels_data_binned_append(
    acq_channel_data: AcquisitionChannelData,
    schedulable_label_to_acq_index: SchedulableLabelToAcquisitionIndex,
    full_schedulable_label: FullSchedulableLabel,
    nested_loop_repetitions: list[int],
    coords: dict,
    acq_num_in_operation: int,
    acq_channel: Hashable,
    acq_index: int | None,
) -> None:
    """
    Generates the acquisition channel data, and updates acq_channel_data,
    and updates schedulable_label_to_acq_index for average bin mode.
    """
    # Bear in mind: contrary to the average case,
    # we do not test whether the `acq_index`
    # are defined in order (starting from 0 and incremented by one for each new acquisition),
    # because that's very complicated in case there are loops inside the schedule.
    # We just assume that they are.

    assert isinstance(acq_channel_data.coords, list)
    multiple_coords: list[dict] = []
    if len(nested_loop_repetitions) == 0:
        multiple_coords = [{"acq_index_legacy": acq_index, **coords}]
    else:
        if acq_index is not None:
            warnings.warn(
                (
                    f"Explicitly defined acquisition index for an append mode acquisition "
                    f"within a loop will not be supported in the future. "
                    f"Ignoring {acq_index=} for {acq_channel=}."
                ),
                FutureWarning,
            )
        overall_nested_loop_repetitions = _prod(nested_loop_repetitions)
        multiple_coords = [
            {"loop_repetition": lr, "acq_index_legacy": acq_index, **coords}
            for lr in range(overall_nested_loop_repetitions)
        ]
    first_acq_index = len(acq_channel_data.coords)
    number_of_new_acq_indices = len(multiple_coords)
    schedulable_label_to_acq_index[(full_schedulable_label, acq_num_in_operation)] = list(
        range(
            first_acq_index,
            first_acq_index + number_of_new_acq_indices,
        )
    )
    acq_channel_data.coords.extend(multiple_coords)


def _validate_trace_protocol(
    acq_channel: Hashable,
    acq_channels_data: AcquisitionChannelsData,
    nested_loop_repetitions: list[int],  # noqa: ARG001
) -> None:
    if acq_channel in acq_channels_data:
        raise ValueError(
            f"Multiple acquisitions found for acq_channel '{acq_channel}' "
            f"which has a trace acquisition. "
            f"Only one trace acquisition is allowed for each acq_channel.",
        )


def _generate_acq_channels_data_for_protocol(
    acquisitions_info: list[dict],
    acq_channels_data: AcquisitionChannelsData,
    schedulable_label_to_acq_index: SchedulableLabelToAcquisitionIndex,
    full_schedulable_label: FullSchedulableLabel,
    nested_loop_repetitions: list[int],
    is_explicit_acq_index: bool,
) -> None:
    """
    Generates the acquisition channel data, and updates acq_channel_data,
    and updates schedulable_label_to_acq_index.
    """
    for acq_num_in_operation, acq_info in enumerate(acquisitions_info):
        acq_channel: Hashable = acq_info["acq_channel"]
        protocol: str = acq_info["protocol"]
        bin_mode: BinMode = acq_info["bin_mode"]

        # Coords is intended to be introduced later to the operation.
        # coords: dict = acq_info["coords"]
        coords: dict = {}

        acq_index: int | None = acq_info["acq_index"]
        # If is_explicit_acq_index, then only acquisitions where acq_index
        # is explicitly defined will be taken into account;
        # otherwise only the acquisitions where it's not defined.
        if is_explicit_acq_index is (acq_index is None):
            continue

        if (acq_channel_data := acq_channels_data.get(acq_channel, None)) is not None:
            if acq_channel_data.protocol != protocol:
                raise ValueError(
                    f"Found different acquisition protocols "
                    f"('{acq_channel_data.protocol}' and '{protocol}') "
                    f"for acq_channel '{acq_channel}'. "
                    f"Make sure there is only one protocol for each acq_channel.",
                )
            if acq_channel_data.bin_mode != bin_mode:
                raise ValueError(
                    f"Found different bin modes "
                    f"('{acq_channel_data.bin_mode}' and '{bin_mode}') "
                    f"for acq_channel '{acq_channel}'. "
                    f"Make sure there is only one bin mode for each acq_channel.",
                )

        if _is_acquisition_binned_average(protocol, bin_mode):
            if acq_channel not in acq_channels_data:
                acq_channels_data[acq_channel] = AcquisitionChannelData(
                    acq_index_dim_name=("acq_index_" + str(acq_channel)),
                    protocol=protocol,
                    bin_mode=bin_mode,
                    coords=[],
                )
            _generate_acq_channels_data_binned_average(
                acq_channel_data=acq_channels_data[acq_channel],
                schedulable_label_to_acq_index=schedulable_label_to_acq_index,
                full_schedulable_label=full_schedulable_label,
                coords=coords,
                acq_num_in_operation=acq_num_in_operation,
                acq_channel=acq_channel,
                acq_index=acq_index,
            )
        elif _is_acquisition_binned_append(protocol, bin_mode) or (
            protocol == "TimetagTrace" and bin_mode == BinMode.APPEND
        ):
            if acq_channel not in acq_channels_data:
                acq_channels_data[acq_channel] = AcquisitionChannelData(
                    acq_index_dim_name=("acq_index_" + str(acq_channel)),
                    protocol=protocol,
                    bin_mode=bin_mode,
                    coords=[],
                )
            _generate_acq_channels_data_binned_append(
                acq_channel_data=acq_channels_data[acq_channel],
                schedulable_label_to_acq_index=schedulable_label_to_acq_index,
                full_schedulable_label=full_schedulable_label,
                nested_loop_repetitions=nested_loop_repetitions,
                coords=coords,
                acq_num_in_operation=acq_num_in_operation,
                acq_channel=acq_channel,
                acq_index=acq_index,
            )
        elif protocol == "Trace" and bin_mode in (BinMode.AVERAGE, BinMode.FIRST):
            _validate_trace_protocol(
                acq_channel=acq_channel,
                acq_channels_data=acq_channels_data,
                nested_loop_repetitions=nested_loop_repetitions,
            )
            acq_channels_data[acq_channel] = AcquisitionChannelData(
                acq_index_dim_name=("acq_index_" + str(acq_channel)),
                protocol=protocol,
                bin_mode=bin_mode,
                coords=coords,
            )
        elif protocol == "TriggerCount" and bin_mode == BinMode.DISTRIBUTION:
            acq_channels_data[acq_channel] = AcquisitionChannelData(
                acq_index_dim_name=("acq_index_" + str(acq_channel)),
                protocol=protocol,
                bin_mode=bin_mode,
                coords=coords,
            )
        else:
            raise ValueError(
                f"Unsupported acquisition protocol '{protocol}' with bin mode '{bin_mode}' "
                f"on acq_channel '{acq_channel}'.",
            )


def _generate_acq_channels_data(
    operation: ScheduleBase | Operation,
    acq_channels_data: AcquisitionChannelsData,
    schedulable_label_to_acq_index: SchedulableLabelToAcquisitionIndex,
    is_explicit_acq_index: bool,
    full_schedulable_label: FullSchedulableLabel,
    nested_loop_repetitions: list[int],
) -> None:
    """
    Adds mappings to acq_channels_data and schedulable_label_to_acq_index;
    these are the output arguments; the others are input arguments.
    If explicit_acq_indices is True,
    then it only adds Schedulables where acq_index is not None,
    otherwise only adds Schedulables where acq_index is None.
    In this latter case, it will generate the acq_index.
    """
    if isinstance(operation, ScheduleBase):
        sorted_schedulables = sorted(operation.schedulables.values(), key=lambda s: s["abs_time"])
        for schedulable in sorted_schedulables:
            schedulable_label = schedulable["name"]
            new_full_schedulable_label = full_schedulable_label + (schedulable_label,)
            inner_operation = operation.operations[schedulable["operation_id"]]
            _generate_acq_channels_data(
                operation=inner_operation,
                acq_channels_data=acq_channels_data,
                schedulable_label_to_acq_index=schedulable_label_to_acq_index,
                is_explicit_acq_index=is_explicit_acq_index,
                full_schedulable_label=new_full_schedulable_label,
                nested_loop_repetitions=nested_loop_repetitions,
            )
    elif isinstance(operation, LoopOperation):
        # For control flows, `None` signifies we refer to the `body` of the control flow.
        new_full_schedulable_label: FullSchedulableLabel = full_schedulable_label + (None,)
        repetitions: int = operation.data["control_flow_info"]["repetitions"]
        new_nested_loop_repetitions: list[int] = nested_loop_repetitions + [repetitions]
        _generate_acq_channels_data(
            operation=operation.body,
            acq_channels_data=acq_channels_data,
            schedulable_label_to_acq_index=schedulable_label_to_acq_index,
            is_explicit_acq_index=is_explicit_acq_index,
            full_schedulable_label=new_full_schedulable_label,
            nested_loop_repetitions=new_nested_loop_repetitions,
        )
    elif isinstance(operation, ConditionalOperation):
        # For control flows, `None` signifies we refer to the `body` of the control flow.
        new_full_schedulable_label = full_schedulable_label + (None,)
        _generate_acq_channels_data(
            operation=operation.body,
            acq_channels_data=acq_channels_data,
            schedulable_label_to_acq_index=schedulable_label_to_acq_index,
            is_explicit_acq_index=is_explicit_acq_index,
            full_schedulable_label=new_full_schedulable_label,
            nested_loop_repetitions=nested_loop_repetitions,
        )
    elif operation.valid_acquisition:
        _generate_acq_channels_data_for_protocol(
            acquisitions_info=operation.data["acquisition_info"],
            acq_channels_data=acq_channels_data,
            schedulable_label_to_acq_index=schedulable_label_to_acq_index,
            full_schedulable_label=full_schedulable_label,
            nested_loop_repetitions=nested_loop_repetitions,
            is_explicit_acq_index=is_explicit_acq_index,
        )


def generate_acq_channels_data(
    schedule: ScheduleBase,
) -> tuple[AcquisitionChannelsData, SchedulableLabelToAcquisitionIndex]:
    """
    Generate acq_index for every schedulable,
    and validate schedule regarding the acquisitions.

    This function generates the ``AcquisitionChannelData`` for every ``acq_channel``,
    and the ``SchedulableLabelToAcquisitionIndex``. It assumes the schedule is device-level.
    """
    acq_channels_data: AcquisitionChannelsData = dict()
    schedulable_label_to_acq_index: SchedulableLabelToAcquisitionIndex = dict()

    # First we generate all mappings for Schedulables
    # where acq_index is explicitly given.
    # In the next step we generate new acq_indices
    # and mapping for Schedulables where acq_index is None.
    #
    # The reason for this is that
    # *   for compatibility reasons, temporarily we'd like to allow users to explicitly specify
    #     acquisition index on the operation (the long-term intention is not to allow this,
    #     and only allow the compiler to generate an acquisition index itself), and
    # *   the acquisition mapping data stores each acquisition index in a list, and the list
    #     index is not stored explicitly (to store memory), only implicitly in the `coords` list.
    # Imagine the schedule: `Acq(acq_index=0); Acq(acq_index(acq_index=None); Acq(acq_index=1);`.
    # We choose the following convention: the acquisition indices start from 0, increment by 1, this
    # is a restriction **only** where the acquisition index is explicitly set by the user.
    # (We could have chosen a different convention, but probably for the user this is easier than
    # the other convention that the acquisition indices are incremented by one for all acquisitions,
    # even when the acquisition index is not explicitly specified by the user.)
    # Then, the only way to generate the acquisition mapping is by first iterating through the
    # acquisition operations where the acquisition index has been explicitly defined.
    _generate_acq_channels_data(
        schedule,
        acq_channels_data,
        schedulable_label_to_acq_index,
        is_explicit_acq_index=True,
        full_schedulable_label=(),
        nested_loop_repetitions=[],
    )
    _generate_acq_channels_data(
        schedule,
        acq_channels_data,
        schedulable_label_to_acq_index,
        is_explicit_acq_index=False,
        full_schedulable_label=(),
        nested_loop_repetitions=[],
    )

    return acq_channels_data, schedulable_label_to_acq_index
