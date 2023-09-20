# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler for the quantify_scheduler."""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from quantify_scheduler.json_utils import load_json_schema, validate_json
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.schedules.schedule import Schedulable, Schedule

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import (
        CompilationConfig,
    )

logger = logging.getLogger(__name__)


class _ControlFlowReturn(Operation):
    """
    An operation that signals the end of the current control flow statement.

    Cannot be added to Schedule manually.

    Parameters
    ----------
    t0 : float, optional
        time offset, by default 0
    """

    def __init__(self, t0: float = 0) -> None:
        super().__init__(name="ControlFlowReturn")
        self.data.update(
            {
                "name": "ControlFlowReturn ",
                "control_flow_info": {
                    "t0": t0,
                    "return_stack": True,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        return self._get_signature(self.data["control_flow_info"])


def determine_absolute_timing(
    schedule: Schedule,
    time_unit: Literal[
        "physical", "ideal", None
    ] = "physical",  # should be included in CompilationConfig
    config: CompilationConfig | None = None,
    keep_original_schedule: bool = True,
) -> Schedule:
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

    for op in schedule.operations.values():
        if isinstance(op, Schedule) and op.get("duration", None) is None:
            determine_absolute_timing(
                op, time_unit, config, keep_original_schedule=False
            )

    # If called directly and not by the compiler, ensure control flow is resolved
    if config is None:
        resolve_control_flow(schedule)

    scheduling_strategy = "asap"
    if config is not None:
        scheduling_strategy = config.device_compilation_config.scheduling_strategy
    if scheduling_strategy != "asap":
        raise NotImplementedError(
            f"{determine_absolute_timing.__name__} does not currently support "
            f"{scheduling_strategy=}. Please change to 'asap' scheduling strategy "
            "in the `DeviceCompilationConfig`."
        )

    if not schedule.schedulables:
        raise ValueError(f"schedule '{schedule.name}' contains no schedulables.")

    time_unit = time_unit or "physical"
    valid_time_units = ("physical", "ideal")
    if time_unit not in valid_time_units:
        raise ValueError(
            f"Undefined time_unit '{time_unit}'! Must be one of {valid_time_units}"
        )

    schedulable_iterator = iter(schedule.schedulables.values())

    # The first schedulable by starts at time 0, and cannot have relative timings
    last_schedulable = next(schedulable_iterator)
    last_schedulable["abs_time"] = 0

    for schedulable in schedulable_iterator:
        curr_op = schedule.operations[schedulable["operation_repr"]]

        for t_constr in schedulable["timing_constraints"]:
            t_constr["ref_schedulable"] = t_constr["ref_schedulable"] or str(
                last_schedulable
            )
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
    time_unit: str,
) -> float:
    # this assumes the reference op exists. This is ensured in schedule.add
    ref_schedulable = schedule.schedulables[str(t_constr["ref_schedulable"])]
    ref_op = schedule.operations[ref_schedulable["operation_repr"]]

    # duration = 1 is useful when e.g., drawing a circuit diagram.
    duration_ref_op = (
        ref_op.duration if time_unit == "physical" else ref_op.get("depth", 1)
    )

    ref_pt = t_constr["ref_pt"] or "end"
    if ref_pt == "start":
        t0 = ref_schedulable["abs_time"]
    elif ref_pt == "center":
        t0 = ref_schedulable["abs_time"] + duration_ref_op / 2
    elif ref_pt == "end":
        t0 = ref_schedulable["abs_time"] + duration_ref_op
    else:
        raise NotImplementedError(f'Timing "{ref_pt=}" not supported by backend.')

    duration_new_op = (
        curr_op.duration if time_unit == "physical" else curr_op.get("depth", 1)
    )

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


def _get_port_clocks(schedule: Schedule) -> set:
    """Recursively collect all port-clock combinations in a Schedule."""
    port_clocks: set = set()
    for op in schedule.operations.values():
        if isinstance(op, Schedule):
            port_clocks |= _get_port_clocks(op)
        elif isinstance(op, Operation):
            for pulse in op["pulse_info"]:
                port = pulse.get("port", None)
                clock = pulse.get("clock", None)
                if port and clock:
                    port_clocks.add((port, clock))
    return port_clocks


def resolve_control_flow(
    schedule: Schedule,
    config: CompilationConfig | None = None,
    port_clocks: set | None = None,
) -> Schedule:
    """
    If control flow is used, insert virtual operations before and after the schedulable.

    Parameters
    ----------
    schedule
        The schedule for which to fill relative timings.
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`,
        which is currently not used in this compilation step.
    port_clocks
        Port-clock combinations to be used for control flow. Determined automatically
        for the outermost schedule.

    Returns
    -------
    :
        a new schedule object where the timing constraints for each operation have
        been determined.
    """
    if not port_clocks:
        port_clocks = _get_port_clocks(schedule)
    for op in schedule.operations.values():
        if isinstance(op, Schedule):
            resolve_control_flow(op, config, port_clocks)

    if not schedule.schedulables:
        raise ValueError(f"schedule '{schedule.name}' contains no schedulables.")

    schedulables = tuple(schedule.schedulables.values())

    for schedulable in schedulables:
        cf = schedulable.get("control_flow", None)
        if cf is not None:
            op = schedule.operations[schedulable["operation_repr"]]
            cf["pulse_info"] = [
                {
                    "wf_func": None,
                    "clock": clock,
                    "port": port,
                    "duration": 0,
                    **cf["control_flow_info"],
                }
                for port, clock in port_clocks
            ]

            rst_op = _ControlFlowReturn()
            rst_op["pulse_info"] = [
                {
                    "wf_func": None,
                    "clock": clock,
                    "port": port,
                    "duration": 0,
                    **rst_op["control_flow_info"],
                }
                for port, clock in port_clocks
            ]

            schedule.add(
                cf,
                rel_time=-0.001e-9,
                ref_op=str(schedulable),
                ref_pt="start",
                ref_pt_new="start",
                validate=False,
            )

            # insert return stack op after the current operation
            schedule.add(
                rst_op,
                rel_time=0,
                ref_op=str(schedulable),
                ref_pt="end",
                ref_pt_new="start",
                validate=False,
            )
    return schedule


def flatten_schedule(
    schedule: Schedule, config: CompilationConfig | None = None
) -> Schedule:
    """
    Recursively flatten subschedules based on the absolute timing.

    Parameters
    ----------
    schedule : Schedule
        schedule to be flattened
    config : CompilationConfig | None, optional
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`,
        which is currently not only used to detect if the function is called directly.
        by default None

    Returns
    -------
    Schedule
        Equivalent schedule without subschedules
    """
    # If called directly and not by the compiler, ensure timings are filled
    if config is None:
        resolve_control_flow(schedule)

    for op in schedule.operations.values():
        if isinstance(op, Schedule):
            flatten_schedule(op, config)

    op_keys_to_pop = set()
    schedulable_keys_to_pop = set()
    # we cannot use .items() directly since we modify schedule.schedulables in the loop
    schedulable_iter = tuple(schedule.schedulables.items())
    for schedulable_key, schedulable in schedulable_iter:
        op_key = schedulable["operation_repr"]
        op = schedule.operations[op_key]
        if isinstance(op, Schedule):
            offset = schedulable["abs_time"]

            # insert new schedulables shifted by the correct offset
            for inner_schedulable in op.schedulables.values():
                inner_op = op.operations[inner_schedulable["operation_repr"]]
                _insert_op_at_time(
                    schedule, inner_op, inner_schedulable["abs_time"] + offset
                )

            # mark the inner schedule for removal from the parent
            op_keys_to_pop.add(op_key)
            schedulable_keys_to_pop.add(schedulable_key)

    for key in op_keys_to_pop:
        schedule["operation_dict"].pop(key)
    for key in schedulable_keys_to_pop:
        schedule["schedulables"].pop(key)

    return schedule


def _insert_op_at_time(
    schedule: Schedule, operation: Operation, abs_time: float
) -> None:
    new_key = str(uuid4())
    new_schedulable = Schedulable(
        name=new_key,
        operation_repr=operation.hash,
    )
    # Timing constraints in the new schedulable are meaningless, so remove the list
    new_schedulable["timing_constraints"] = None
    new_schedulable["abs_time"] = abs_time
    schedule["operation_dict"][operation.hash] = operation
    schedule["schedulables"][new_key] = new_schedulable


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
