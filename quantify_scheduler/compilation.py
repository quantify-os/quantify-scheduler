# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler for the quantify_scheduler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

import networkx as nx

from quantify_scheduler.backends.qblox.operations.inline_q1asm import InlineQ1ASM
from quantify_scheduler.enums import SchedulingStrategy
from quantify_scheduler.json_utils import load_json_schema, validate_json
from quantify_scheduler.operations.control_flow_library import (
    ControlFlowOperation,
)
from quantify_scheduler.operations.gate_library import Rz
from quantify_scheduler.operations.pulse_library import IdlePulse
from quantify_scheduler.schedules.schedule import (
    Schedulable,
    Schedule,
    ScheduleBase,
    TimingConstraint,
)

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
def _determine_absolute_timing(
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
        If the scheduling strategy is not SchedulingStrategy.ASAP

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
            f"Cannot determine timing for operation {schedule.name}. Operation data: {schedule!r}"
        )
    else:
        return schedule


def _determine_absolute_timing_schedule(
    schedule: Schedule,
    time_unit: Literal["physical", "ideal", None],
    config: CompilationConfig | None,
) -> Schedule:
    scheduling_strategy = _determine_scheduling_strategy(config)

    if not schedule.schedulables:
        raise ValueError(f"schedule '{schedule.name}' contains no schedulables.")

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
            # TODO (SE-650): move to qblox backend.
            and not isinstance(schedule.operations[op_key], InlineQ1ASM)
        ):
            # Gates do not have a defined duration, so only ideal timing is defined
            raise RuntimeError(
                f"Operation {schedule.operations[op_key].name} is not a valid pulse or acquisition."
                f" Please check whether the device compilation has been performed."
                f" Operation data: {schedule.operations[op_key]!r}"
            )

    _make_timing_constraints_explicit(schedule, scheduling_strategy)
    references_graph = _populate_references_graph(schedule)
    _validate_schedulable_references(schedule, references_graph)

    schedulables_sorted_by_reference = nx.topological_sort(references_graph)
    for i, schedulable_name in enumerate(schedulables_sorted_by_reference):
        i: int
        schedulable_name: str
        schedulable: Schedulable = schedule.schedulables[schedulable_name]
        timing_constraints: list[TimingConstraint] = schedulable.data["timing_constraints"]
        operation: Operation | Schedule = schedule.operations[schedulable.data["operation_id"]]

        if i == 0:
            schedulable.data["abs_time"] = 0.0
        elif scheduling_strategy == SchedulingStrategy.ASAP:
            for timing_constraint in timing_constraints:
                abs_time = _get_start_time(schedule, timing_constraint, operation, time_unit)
                if "abs_time" not in schedulable or abs_time > schedulable["abs_time"]:
                    schedulable.data["abs_time"] = abs_time
        else:
            schedulable.data["abs_time"] = _get_start_time(
                schedule, timing_constraints[0], operation, time_unit
            )

    schedule = _normalize_absolute_timing(schedule)
    schedule["duration"] = schedule.get_schedule_duration()
    if time_unit == "ideal":
        schedule["depth"] = schedule["duration"] + 1
    return schedule


def _determine_scheduling_strategy(config: CompilationConfig | None = None) -> SchedulingStrategy:
    if config is not None and config.device_compilation_config is not None:
        return config.device_compilation_config.scheduling_strategy

    return SchedulingStrategy.ASAP


def _validate_schedulable_references(schedule: Schedule, references_graph: nx.DiGraph) -> None:
    """Check the schedulable references for circular references."""
    for node in references_graph.nodes:
        if node not in schedule.schedulables:
            raise ValueError(f"Node {node} not found in schedulables.")

    if not nx.is_directed_acyclic_graph(references_graph):
        raise TypeError(
            "`schedulable_references` is not a Directed Acyclic Graph. This is most likely "
            "caused by a circular reference in the Timing Constraints."
        )


def _populate_references_graph(schedule: Schedule) -> nx.DiGraph:
    """Add nodes and edges to the graph containing schedulable references."""
    graph = nx.DiGraph()

    # Add nodes
    graph.add_nodes_from(schedule.schedulables.keys())

    # Add edges
    for schedulable_name, schedulable in schedule.schedulables.items():
        schedulable_name: str
        schedulable: Schedulable

        graph.add_edges_from(
            (timing_constraint.ref_schedulable, schedulable_name)
            for timing_constraint in schedulable.data["timing_constraints"]
            if timing_constraint.ref_schedulable is not None
        )

    return graph


def _make_timing_constraints_explicit(schedule: Schedule, strategy: SchedulingStrategy) -> None:
    default_schedulable_by_schedulable: list[tuple[str, str | None]] = (
        _determine_default_ref_schedulables_by_schedulable(schedule, strategy)
    )

    for (
        schedulable_name,
        default_reference_schedulable_name,
    ) in default_schedulable_by_schedulable:
        schedulable_name: str
        default_reference_schedulable_name: str | None

        _make_timing_constraints_explicit_for_schedulable(
            schedule=schedule,
            schedulable_name=schedulable_name,
            default_reference_schedulable_name=default_reference_schedulable_name,
            strategy=strategy,
        )


def _make_timing_constraints_explicit_for_schedulable(
    schedule: Schedule,
    schedulable_name: str,
    default_reference_schedulable_name: str | None,
    strategy: SchedulingStrategy,
) -> None:
    schedulable: Schedulable = schedule.schedulables[schedulable_name]
    given_timing_constraints: list[TimingConstraint] = schedulable.data["timing_constraints"]

    # Support only one timing constraint for now
    if strategy == SchedulingStrategy.ALAP and len(given_timing_constraints) != 1:
        raise NotImplementedError("Only exactly one timing constraint per Schedulable supported.")

    timing_constraint: TimingConstraint = given_timing_constraints[0]

    if timing_constraint.ref_schedulable is None:
        timing_constraint.ref_schedulable = default_reference_schedulable_name

    if timing_constraint.ref_pt is None:
        timing_constraint.ref_pt = _determine_default_ref_pt(strategy)

    if timing_constraint.ref_pt_new is None:
        timing_constraint.ref_pt_new = _determine_default_ref_pt_new(strategy)

    if timing_constraint.rel_time is None:
        timing_constraint.rel_time = 0.0


def _determine_default_ref_pt(strategy: SchedulingStrategy) -> Literal["start", "end"]:
    if strategy == SchedulingStrategy.ASAP:
        return "end"

    if strategy == SchedulingStrategy.ALAP:
        return "start"

    raise ValueError(f"Cannot determine default `ref_pt`. Unknown scheduling strategy: {strategy}")


def _determine_default_ref_pt_new(strategy: SchedulingStrategy) -> Literal["start", "end"]:
    if strategy == SchedulingStrategy.ASAP:
        return "start"

    if strategy == SchedulingStrategy.ALAP:
        return "end"

    raise ValueError(
        f"Cannot determine default `ref_pt_new`. Unknown scheduling strategy: {strategy}"
    )


def _determine_default_ref_schedulables_by_schedulable(
    schedule: Schedule, strategy: SchedulingStrategy
) -> list[tuple[str, str | None]]:
    schedulable_names: list[str] = list(schedule.schedulables)

    if strategy == SchedulingStrategy.ASAP:
        default_schedulable_names: list[str | None] = [None] + list(schedule.schedulables)[:-1]
    elif strategy == SchedulingStrategy.ALAP:
        default_schedulable_names: list[str | None] = list(schedule.schedulables)[1:] + [None]
    else:
        raise ValueError(f"Scheduling strategy {strategy} not one of `ASAP` or `ALAP`.")

    return [
        (schedulable_name, default_schedulable_name)
        for schedulable_name, default_schedulable_name in zip(
            schedulable_names, default_schedulable_names
        )
    ]


def _get_start_time(
    schedule: Schedule,
    t_constr: TimingConstraint,
    curr_op: Operation | Schedule,
    time_unit: Literal["physical", "ideal", None],
) -> float:
    assert t_constr.ref_schedulable is not None

    ref_schedulable: Schedulable = schedule.schedulables[t_constr.ref_schedulable]
    ref_op: Operation | Schedule = schedule.operations[ref_schedulable["operation_id"]]

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

    ref_pt = t_constr.ref_pt or "end"
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

    ref_pt_new = t_constr.ref_pt_new or "start"
    if ref_pt_new == "start":
        abs_time = t0 + t_constr.rel_time
    elif ref_pt_new == "center":
        abs_time = t0 + t_constr.rel_time - duration_new_op / 2
    elif ref_pt_new == "end":
        abs_time = t0 + t_constr.rel_time - duration_new_op
    else:
        raise NotImplementedError(f'Timing "{ref_pt_new=}" not supported by backend.')
    return abs_time


def _normalize_absolute_timing(
    schedule: Schedule,
    config: CompilationConfig | None = None,  # noqa: ARG001
) -> Schedule:
    # TODO: Support normalization of absolute timing in subschedules
    # See test_negative_absolute_timing_is_normalized_with_subschedule in test_compilation.py
    # and https://gitlab.com/quantify-os/quantify-scheduler/-/issues/489
    min_time = min(schedulable["abs_time"] for schedulable in schedule.schedulables.values())
    if min_time < 0:
        for schedulable in schedule.schedulables.values():
            schedulable["abs_time"] -= min_time
    return schedule


@overload
def _merge_rz_gates(
    schedule: Schedule,
    config: CompilationConfig | None = None,
) -> Schedule: ...
@overload
def _merge_rz_gates(
    schedule: Operation,
    config: CompilationConfig | None = None,
) -> Operation | Schedule: ...
def _merge_rz_gates(
    schedule: Schedule | Operation,
    config: CompilationConfig | None = None,
):
    # This is a recursive function, the argument `schedule` is not always a `Schedule` type,
    # so we rename it at the beginning to not cause confusion.
    op = schedule

    if isinstance(op, Schedule):
        # First, process operations
        for inner_op_key, inner_op in op.operations.items():
            op.operations[inner_op_key] = _merge_rz_gates(
                schedule=inner_op,
                config=config,
            )

        last_op = None
        # Copy list so we can modify it while iterating
        for inner_sched_key in list(op.schedulables):
            inner_sched = op.schedulables[inner_sched_key]
            inner_op = op.operations[inner_sched["operation_id"]]

            # Merge Rz gates if possible
            if (
                isinstance(inner_op, Rz)
                and isinstance(last_op, Rz)
                and inner_op.qubit == last_op.qubit
            ):
                last_op.theta += inner_op.theta
                noop = IdlePulse(0)
                noop_id = noop.hash
                op.operations[noop_id] = noop
                inner_sched["operation_id"] = noop_id
                continue

            # Record last
            last_op = inner_op

        return op
    elif isinstance(op, ControlFlowOperation):
        op.body = _merge_rz_gates(
            schedule=op.body,
            config=config,
        )
        return op
    else:
        return op


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


def plot_schedulable_references_graph(schedule: Schedule) -> None:
    """
    Show the schedulable reference graph.

    Can be used as a debugging tool to spot any circular references.
    """
    graph = _populate_references_graph(schedule)
    nx.draw(graph, with_labels=True)
