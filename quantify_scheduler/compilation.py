# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler for the quantify_scheduler."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

from quantify_scheduler.json_utils import load_json_schema, validate_json
from quantify_scheduler.operations.control_flow_library import (
    ControlFlowOperation,
)
from quantify_scheduler.schedules.schedule import Schedule, ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import (
        CompilationConfig,
    )
    from quantify_scheduler.operations.operation import Operation

logger = logging.getLogger(__name__)


@overload
def _determine_absolute_timing(
    schedule: Schedule,
    time_unit: Literal["physical", "ideal", None] = "physical",
    config: CompilationConfig | None = None,
) -> Schedule: ...
@overload
def _determine_absolute_timing(
    schedule: Operation,
    time_unit: Literal["physical", "ideal", None] = "physical",
    config: CompilationConfig | None = None,
) -> Operation | Schedule: ...
def _determine_absolute_timing(  # noqa: PLR0912
    schedule: Operation | Schedule,
    time_unit: Literal[
        "physical", "ideal", None
    ] = "physical",  # should be included in CompilationConfig
    config: CompilationConfig | None = None,
):
    """
    Determine the absolute timing of a schedule based on the timing constraints.

    This function determines absolute timings for every operation in the
    :attr:`~.ScheduleBase.schedulables`. It does this by:

        1. iterating over all and elements in the :attr:`~.ScheduleBase.schedulables`.
        2. determining the absolute time of the reference operation
           - reference point :code:`"ref_pt"` of the reference operation defaults to
           :code:`"end"` in case it is not set (i.e., is :code:`None`).
        3. determining the start of the operation based on the :code:`rel_time` and
           :code:`duration` of operations
           - reference point :code:`"ref_pt_new"` of the added operation defaults to
           :code:`"start"` in case it is not set.


    Parameters
    ----------
    schedule
        The schedule for which to determine timings.
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`.
    time_unit
        Whether to use physical units to determine the absolute time or ideal time.
        When :code:`time_unit == "physical"` the duration attribute is used.
        When :code:`time_unit == "ideal"` the duration attribute is ignored and treated
        as if it is :code:`1`.
        When :code:`time_unit == None` it will revert to :code:`"physical"`.

    Returns
    -------
    :
        The modified ``schedule`` where the absolute time for each operation has been
        determined.

    Raises
    ------
    NotImplementedError
        If the scheduling strategy is not "asap"

    """
    time_unit = time_unit or "physical"
    if time_unit not in (valid_time_units := ("physical", "ideal")):
        raise ValueError(f"Undefined time_unit '{time_unit}'! Must be one of {valid_time_units}")

    if isinstance(schedule, ScheduleBase):
        return _determine_absolute_timing_schedule(schedule, time_unit, config)
    elif isinstance(schedule, ControlFlowOperation):
        schedule.body = _determine_absolute_timing(schedule.body, time_unit, config)
        return schedule
    elif schedule.duration is None:
        raise RuntimeError(
            f"Cannot determine timing for operation {schedule.name}."
            f" Operation data: {repr(schedule)}"
        )
    else:
        return schedule


def _determine_absolute_timing_schedule(  # noqa: PLR0912
    schedule: Schedule,
    time_unit: Literal["physical", "ideal", None],
    config: CompilationConfig | None,
) -> Schedule:
    for op_key in schedule.operations:
        if isinstance(schedule.operations[op_key], Schedule):
            if schedule.operations[op_key].get("duration", None) is None:
                schedule.operations[op_key] = _determine_absolute_timing(
                    schedule=schedule.operations[op_key],
                    time_unit=time_unit,
                    config=config,
                )

        elif isinstance(schedule.operations[op_key], ControlFlowOperation):
            schedule.operations[op_key] = _determine_absolute_timing(
                schedule=schedule.operations[op_key],
                time_unit=time_unit,
                config=config,
            )

        # Note: type checker cannot reason that schedule.operations[op_key] can only be
        # an Operation after the `or`.
        elif isinstance(schedule.operations[op_key], Schedule) or (
            time_unit == "physical"
            and not schedule.operations[op_key].valid_pulse  # type: ignore
            and not schedule.operations[op_key].valid_acquisition  # type: ignore
        ):
            # Gates do not have a defined duration, so only ideal timing is defined
            raise RuntimeError(
                f"Operation {schedule.operations[op_key].name} is not a valid pulse or acquisition."
                f" Please check whether the device compilation has been performed."
                f" Operation data: {repr(schedule.operations[op_key])}"
            )

    scheduling_strategy = "asap"
    if config is not None and config.device_compilation_config is not None:
        scheduling_strategy = config.device_compilation_config.scheduling_strategy

    if scheduling_strategy != "asap":
        raise NotImplementedError(
            f"{_determine_absolute_timing.__name__} does not currently support "
            f"{scheduling_strategy=}. Please change to 'asap' scheduling strategy "
            "in the `DeviceCompilationConfig`."
        )

    if not schedule.schedulables:
        raise ValueError(f"schedule '{schedule.name}' contains no schedulables.")

    schedulable_iterator = iter(schedule.schedulables.values())

    # The first schedulable by starts at time 0, and cannot have relative timings
    last_schedulable = next(schedulable_iterator)
    last_schedulable["abs_time"] = 0

    for schedulable in schedulable_iterator:
        curr_op = schedule.operations[schedulable["operation_id"]]

        for t_constr in schedulable["timing_constraints"]:
            t_constr["ref_schedulable"] = t_constr["ref_schedulable"] or str(last_schedulable)
            abs_time = _get_start_time(schedule, t_constr, curr_op, time_unit)

            if "abs_time" not in schedulable or abs_time > schedulable["abs_time"]:
                schedulable["abs_time"] = abs_time
        last_schedulable = schedulable

    schedule["duration"] = schedule.get_schedule_duration()
    if time_unit == "ideal":
        schedule["depth"] = schedule["duration"] + 1
    return schedule


def _get_start_time(
    schedule: Schedule,
    t_constr: dict[str, str | float],
    curr_op: Operation | Schedule,
    time_unit: Literal["physical", "ideal", None],
) -> float:
    # this assumes the reference op exists. This is ensured in schedule.add
    ref_schedulable = schedule.schedulables[str(t_constr["ref_schedulable"])]
    ref_op = schedule.operations[ref_schedulable["operation_id"]]

    # duration = 1 is useful when e.g., drawing a circuit diagram.
    if time_unit == "physical":
        duration_ref_op = ref_op.duration
    else:
        duration_ref_op = (
            ref_op.body.get("depth", 1)
            if isinstance(ref_op, ControlFlowOperation)
            else ref_op.get("depth", 1)
        )
    # Type checker does not know that ref_op.duration is not None if time_unit ==
    # "physical"
    assert duration_ref_op is not None

    ref_pt = t_constr["ref_pt"] or "end"
    if ref_pt == "start":
        t0 = ref_schedulable["abs_time"]
    elif ref_pt == "center":
        t0 = ref_schedulable["abs_time"] + duration_ref_op / 2
    elif ref_pt == "end":
        t0 = ref_schedulable["abs_time"] + duration_ref_op
    else:
        raise NotImplementedError(f'Timing "{ref_pt=}" not supported by backend.')

    if time_unit == "physical":
        duration_new_op = curr_op.duration
    else:
        duration_new_op = (
            curr_op.body.get("depth", 1)
            if isinstance(curr_op, ControlFlowOperation)
            else curr_op.get("depth", 1)
        )
    assert duration_new_op is not None

    ref_pt_new = t_constr["ref_pt_new"] or "start"
    if ref_pt_new == "start":
        abs_time = t0 + t_constr["rel_time"]
    elif ref_pt_new == "center":
        abs_time = t0 + t_constr["rel_time"] - duration_new_op / 2
    elif ref_pt_new == "end":
        abs_time = t0 + t_constr["rel_time"] - duration_new_op
    else:
        raise NotImplementedError(f'Timing "{ref_pt_new=}" not supported by backend.')
    return abs_time


def validate_config(config: dict, scheme_fn: str) -> bool:
    """
    Validate a configuration using a schema.

    Parameters
    ----------
    config
        The configuration to validate
    scheme_fn
        The name of a json schema in the quantify_scheduler.schemas folder.

    Returns
    -------
    :
        True if valid

    """
    scheme = load_json_schema(__file__, scheme_fn)
    validate_json(config, scheme)
    return True
