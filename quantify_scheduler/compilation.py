# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler for the quantify_scheduler."""
from __future__ import annotations

import logging
from typing import Literal, Optional
from copy import deepcopy

from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
)
from quantify_scheduler.json_utils import load_json_schema, validate_json
from quantify_scheduler.schedules.schedule import Schedule

logger = logging.getLogger(__name__)


def determine_absolute_timing(
    schedule: Schedule,
    time_unit: Literal[
        "physical", "ideal", None
    ] = "physical",  # should be included in CompilationConfig
    config: Optional[CompilationConfig] = None,
    keep_original_schedule=True,
) -> Schedule:
    """
    Determines the absolute timing of a schedule based on the timing constraints.

    This function determines absolute timings for every operation in the
    :attr:`~.ScheduleBase.schedulables`. It does this by:

        1. iterating over all and elements in the :attr:`~.ScheduleBase.schedulables`.
        2. determining the absolute time of the reference operation
           - reference point :code:`"ref_pt"` of the reference operation defaults to :code:`"end"` in case it is not set (i.e., is :code:`None`).
        3. determining the start of the operation based on the :code:`rel_time` and :code:`duration`
           of operations
           - reference point :code:`"ref_pt_new"` of the added operation defaults to :code:`"start"` in case it is not set.


    Parameters
    ----------
    schedule
        The schedule for which to determine timings.
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`,
        which is currently not used in this compilation step.
    time_unit
        Whether to use physical units to determine the absolute time or ideal time.
        When :code:`time_unit == "physical"` the duration attribute is used.
        When :code:`time_unit == "ideal"` the duration attribute is ignored and treated
        as if it is :code:`1`.
        When :code:`time_unit == None` it will revert to :code:`"physical"`.
    keep_original_schedule
        If `True`, this function will not modify the schedule argument.
        If `False`, the compilation modifies the schedule, thereby
        making the original schedule unusable for further usage; this
        improves compilation time. Warning: if `False`, the returned schedule
        references objects from the original schedule, please refrain from modifying
        the original schedule after compilation in this case!

    Returns
    -------
    :
        The modified `schedule` where the absolute time for each operation has been
        determined.

    Raises
    ------
    NotImplementedError
        If the scheduling strategy is not "asap"
    """
    if keep_original_schedule:
        schedule = deepcopy(schedule)

    scheduling_strategy = "asap"
    if config is not None:
        scheduling_strategy = config.device_compilation_config.scheduling_strategy
    if scheduling_strategy != "asap":
        raise NotImplementedError(
            f"{determine_absolute_timing.__name__} does not currently support {scheduling_strategy=}."
            "Please change to 'asap' scheduling strategy in the `DeviceCompilationConfig`."
        )

    if len(schedule.schedulables) == 0:
        raise ValueError(f"schedule '{schedule.name}' contains no schedulables.")

    if time_unit is None:
        time_unit = "physical"
    valid_time_units = ("physical", "ideal")
    if time_unit not in valid_time_units:
        raise ValueError(
            f"Undefined time_unit '{time_unit}'! Must be one of {valid_time_units}"
        )

    # iterate over the objects in the schedule.
    last_schedulable = next(iter(schedule.schedulables.values()))
    last_op = schedule.operations[last_schedulable["operation_repr"]]

    last_schedulable["abs_time"] = 0

    for schedulable in list(schedule.schedulables.values())[1:]:
        curr_op = schedule.operations[schedulable["operation_repr"]]

        if len(schedulable["timing_constraints"]) == 0:
            schedulable.add_timing_constraint(ref_schedulable=last_schedulable)
        for t_constr in schedulable["timing_constraints"]:
            if t_constr["ref_schedulable"] is None:
                ref_schedulable = last_schedulable
                ref_op = last_op
            else:
                # this assumes the reference op exists. This is ensured in schedule.add
                ref_schedulable = schedule.schedulables[
                    str(t_constr["ref_schedulable"])
                ]
                ref_op = schedule.operations[ref_schedulable["operation_repr"]]

            # duration = 1 is useful when e.g., drawing a circuit diagram.
            duration_ref_op = ref_op.duration if time_unit == "physical" else 1

            ref_pt = t_constr["ref_pt"] or "end"
            if ref_pt == "start":
                t0 = ref_schedulable["abs_time"]
            elif ref_pt == "center":
                t0 = ref_schedulable["abs_time"] + duration_ref_op / 2
            elif ref_pt == "end":
                t0 = ref_schedulable["abs_time"] + duration_ref_op
            else:
                raise NotImplementedError(
                    f'Timing "{ref_schedulable["abs_time"]}" not supported by backend.'
                )

            duration_new_op = curr_op.duration if time_unit == "physical" else 1

            ref_pt_new = t_constr["ref_pt_new"] or "start"
            if ref_pt_new == "start":
                abs_time = t0 + t_constr["rel_time"]
            elif ref_pt_new == "center":
                abs_time = t0 + t_constr["rel_time"] - duration_new_op / 2
            elif ref_pt_new == "end":
                abs_time = t0 + t_constr["rel_time"] - duration_new_op
            if "abs_time" not in schedulable or abs_time > schedulable["abs_time"]:
                schedulable["abs_time"] = abs_time

        # update last_constraint and operation for next iteration of the loop
        last_schedulable = schedulable
        last_op = curr_op

    return schedule


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
    ----------
    :
        True if valid
    """
    scheme = load_json_schema(__file__, scheme_fn)
    validate_json(config, scheme)
    return True
