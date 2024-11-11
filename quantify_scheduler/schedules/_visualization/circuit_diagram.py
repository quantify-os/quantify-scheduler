# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Plotting functions used in the visualization backend of the sequencer."""
from __future__ import annotations

from copy import deepcopy
from enum import Enum, auto
from itertools import chain
from typing import TYPE_CHECKING, Iterable, Iterator

import matplotlib

import quantify_scheduler.schedules._visualization.pulse_scheme as ps
from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    LoopOperation,
)
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.schedules._visualization import constants
from quantify_scheduler.schedules.schedule import Schedule, ScheduleBase

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from quantify_scheduler.resources import Resource


def gate_box(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:
    """
    A box for a single gate containing a label.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the gate.
    qubit_idxs :
        The qubit indices.
    text :
        The gate name.
    kw :
        Additional keyword arguments to be passed to drawing the gate box.

    """
    for qubit_idx in qubit_idxs:
        ps.box_text(
            ax,
            x0=time,
            y0=qubit_idx,
            text=text,
            fillcolor=constants.COLOR_LAZURE,
            width=0.8,
            height=0.5,
            **kw,
        )


def pulse_baseband(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:
    """
    Adds a visual indicator for a Baseband pulse to the `matplotlib.axes.Axis`
    instance.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the pulse.
    qubit_idxs :
        The qubit indices.
    text :
        The pulse name.
    kw :
        Additional keyword arguments to be passed to drawing the pulse.

    """
    cartoon_width = 0.6
    for qubit_idx in qubit_idxs:
        ps.flux_pulse(
            ax,
            pos=time - cartoon_width / 2,
            y_offs=qubit_idx,
            width=cartoon_width,
            s=0.0025,
            amp=0.33,
            **kw,
        )
        ax.text(time, qubit_idx + 0.45, text, ha="center", va="center", zorder=6)


def pulse_modulated(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:
    """
    Adds a visual indicator for a Modulated pulse to the `matplotlib.axes.Axis`
    instance.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the pulse.
    qubit_idxs :
        The qubit indices.
    text :
        The pulse name.
    kw :
        Additional keyword arguments to be passed to drawing the pulse.

    """
    cartoon_width = 0.6
    for qubit_idx in qubit_idxs:
        ps.mw_pulse(
            ax,
            pos=time - cartoon_width / 2,
            y_offs=qubit_idx,
            width=cartoon_width,
            amp=0.33,
            **kw,
        )
        ax.text(time, qubit_idx + 0.45, text, ha="center", va="center", zorder=6)


def meter(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:  # Noqa: ARG001
    """
    A simple meter to depict a measurement.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the measurement.
    qubit_idxs :
        The qubit indices.
    text :
        The measurement name.
    kw :
        Additional keyword arguments to be passed to drawing the meter.

    """
    for qubit_idx in qubit_idxs:
        ps.meter(
            ax,
            x0=time,
            y0=qubit_idx,
            fillcolor=constants.COLOR_GREY,
            y_offs=0,
            width=0.8,
            height=0.5,
            **kw,
        )


def acq_meter(
    ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw  # Noqa: ARG001
) -> None:
    """
    Variation of the meter to depict a acquisition.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the measurement.
    qubit_idxs :
        The qubit indices.
    text :
        The measurement name.
    kw :
        Additional keyword arguments to be passed to drawing the acq meter.

    """
    for qubit_idx in qubit_idxs:
        ps.meter(
            ax,
            x0=time,
            y0=qubit_idx,
            fillcolor="white",
            y_offs=0.0,
            width=0.8,
            height=0.5,
            framewidth=constants.ACQ_METER_LINEWIDTH,
            **kw,
        )


def acq_meter_text(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:
    """
    Same as acq_meter, but also displays text.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the measurement.
    qubit_idxs :
        The qubit indices.
    text :
        The measurement name.
    kw :
        Additional keyword arguments to be passed to drawing the acq meter.

    """
    acq_meter(ax, time, qubit_idxs, text, **kw)
    ax.text(time, max(qubit_idxs) + 0.45, text, ha="center", va="center", zorder=6)


def cnot(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:  # Noqa: ARG001
    """
    Markers to denote a CNOT gate between two qubits.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the CNOT.
    qubit_idxs :
        The qubit indices.
    text :
        The CNOT name.
    kw :
        Additional keyword arguments to be passed to drawing the CNOT.

    """
    ax.plot([time, time], qubit_idxs, marker="o", markersize=15, color=constants.COLOR_BLUE)
    ax.plot([time], qubit_idxs[1], marker="+", markersize=12, color="white")


def cz(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:  # Noqa: ARG001
    """
    Markers to denote a CZ gate between two qubits.

    Parameters
    ----------
    ax :
        The matplotlib Axes.
    time :
        The time of the CZ.
    qubit_idxs :
        The qubit indices.
    text :
        The CZ name.
    kw :
        Additional keyword arguments to be passed to drawing the CZ.

    """
    ax.plot([time, time], qubit_idxs, marker="o", markersize=15, color=constants.COLOR_BLUE)


def reset(ax: Axes, time: float, qubit_idxs: list[int], text: str, **kw) -> None:
    """
    A broken line to denote qubit initialization.

    Parameters
    ----------
    ax
        matplotlib axis object.
    time
        x position to draw the reset on
    qubit_idxs
        indices of the qubits that the reset is performed on.
    text :
        The reset name.
    kw :
        Additional keyword arguments to be passed to drawing the reset.

    """
    for qubit_idx in qubit_idxs:
        ps.box_text(
            ax,
            x0=time,
            y0=qubit_idx,
            text=text,
            color="white",
            fillcolor="white",
            width=0.4,
            height=0.5,
            **kw,
        )


class _ControlFlowEnd(Enum):
    """Identifer for end of a control-flow scope."""

    LOOP_END = auto()
    CONDI_END = auto()


def _walk_schedule(
    sched_or_op: Schedule | Operation, time_offset: int = 0
) -> Iterator[tuple[int, Operation | Schedule | _ControlFlowEnd]]:
    if isinstance(sched_or_op, ScheduleBase):
        yield time_offset, sched_or_op
        for schedulable in sched_or_op.schedulables.values():
            operation = sched_or_op.operations[schedulable["operation_id"]]
            yield from _walk_schedule(
                sched_or_op=operation, time_offset=time_offset + schedulable["abs_time"]
            )
    elif isinstance(sched_or_op, LoopOperation):
        yield time_offset, sched_or_op
        if isinstance(sched_or_op.body, ScheduleBase):
            yield from _walk_schedule(sched_or_op.body, time_offset)
        else:
            yield time_offset, sched_or_op.body
        yield time_offset + int(sched_or_op.duration), _ControlFlowEnd.LOOP_END
    elif isinstance(sched_or_op, ConditionalOperation):
        yield time_offset, sched_or_op
        if isinstance(sched_or_op.body, ScheduleBase):
            yield from _walk_schedule(sched_or_op.body, time_offset)
        else:
            yield time_offset, sched_or_op.body
        yield time_offset + int(sched_or_op.duration), _ControlFlowEnd.CONDI_END
    elif isinstance(sched_or_op, Operation):
        yield time_offset, sched_or_op
    else:
        raise ValueError(f"Unknown operation type {type(sched_or_op)}.")


def _walk_schedule_only_operations(
    sched_or_op: Schedule | Operation,
) -> Iterator[Operation]:
    if isinstance(sched_or_op, ScheduleBase):
        for operation in sched_or_op.operations.values():
            yield from _walk_schedule_only_operations(operation)
    elif isinstance(sched_or_op, (LoopOperation, ConditionalOperation)):
        if isinstance(sched_or_op.body, ScheduleBase):
            yield from _walk_schedule_only_operations(sched_or_op.body)
        else:
            yield sched_or_op.body
    elif isinstance(sched_or_op, Operation):
        yield sched_or_op
    else:
        raise ValueError(f"Unknown operation type {type(sched_or_op)}.")


def _draw_operation(
    operation: Operation,
    qubit_map: dict[str, int],
    port_map: dict[str, int],
    ax: Axes,
    time: int,
    schedule_resources: dict[str, Resource],
) -> None:
    if operation.valid_gate:
        plot_func = import_python_object_from_string(operation["gate_info"]["plot_func"])
        idxs = [qubit_map[qubit] for qubit in operation["gate_info"]["qubits"]]
        plot_func(ax, time=time, qubit_idxs=idxs, text=operation["gate_info"]["tex"])
    elif operation.valid_pulse:
        idxs = list(
            {
                port_map[pulse_info["port"]]
                for pulse_info in operation["pulse_info"]
                if pulse_info["port"] is not None
            }
        )

        for pulse_info in operation["pulse_info"]:
            clock_id: str = pulse_info["clock"]
            clock_resource = schedule_resources[clock_id]
            if clock_resource["freq"] == 0:
                pulse_baseband(ax, time=time, qubit_idxs=idxs, text=operation.name)
            else:
                pulse_modulated(ax, time=time, qubit_idxs=idxs, text=operation.name)
    elif operation.valid_acquisition:
        idxs = list({port_map[acq_info["port"]] for acq_info in operation["acquisition_info"]})

        for _ in operation["acquisition_info"]:
            acq_meter(ax, time=time, qubit_idxs=idxs, text=operation.name)
    else:
        raise ValueError("Unknown operation")


def _get_indices(
    sched_or_op: Schedule | Operation,
    qubit_map: dict[str, int],
    port_map: dict[str, int],
) -> set[int]:
    def add_index_from_operation(operation: Operation, index_set: set[int]) -> None:
        if operation.valid_gate:
            index_set.update(qubit_map[qubit] for qubit in operation["gate_info"]["qubits"])
        index_set.update(
            port_map[info["port"]]
            for info in chain(operation["pulse_info"], operation["acquisition_info"])
        )

    indices: set[int] = set()
    if isinstance(sched_or_op, Operation):
        add_index_from_operation(sched_or_op, indices)
        return indices

    for operation in _walk_schedule_only_operations(sched_or_op):
        add_index_from_operation(operation, indices)

    return indices


def _draw_loop(
    ax: Axes,
    qubit_map: dict[str, int],
    port_map: dict[str, int],
    operation: LoopOperation,
    start_time: int,
    end_time: int,
    x_offset: float = 0.35,
    y_offset: float = 0.3,
    fraction: float = 0.2,
) -> None:
    reps = operation["control_flow_info"]["repetitions"]

    def draw_brackets(bottom_qubit: int, top_qubit: int) -> None:
        x_start = start_time - x_offset
        x_end = end_time + x_offset
        y_top = top_qubit + y_offset
        y_bottom = bottom_qubit - y_offset
        ax.annotate(
            "",
            xy=(x_start, y_bottom),
            xytext=(x_start, y_top),
            arrowprops=dict(
                arrowstyle="-",
                linewidth=constants.CTRL_FLOW_ARROW_LINEWIDTH,
                facecolor=constants.COLOR_DARK_MODE_LINE,
                connectionstyle=f"bar,fraction={fraction/(top_qubit-bottom_qubit+1)}",
            ),
        )
        ax.annotate(
            "",
            xy=(x_end, y_bottom),
            xytext=(x_end, y_top),
            arrowprops=dict(
                arrowstyle="-",
                linewidth=constants.CTRL_FLOW_ARROW_LINEWIDTH,
                facecolor=constants.COLOR_DARK_MODE_LINE,
                connectionstyle=f"bar,fraction=-{fraction/(top_qubit-bottom_qubit+1)}",
            ),
        )
        ax.text(x_end + 0.1, y_top + 0.05, f"x{reps}")

    involved_indices = _get_indices(operation.body, qubit_map, port_map)
    if len(involved_indices) == len(qubit_map):
        draw_brackets(0, len(qubit_map) - 1)
    else:
        for idx in involved_indices:
            draw_brackets(idx, idx)


def _draw_conditional(
    ax: Axes,
    measure_time: int,
    measure_qubit_idx: int,
    body: Operation | Schedule,
    body_start: int,
    body_end: int,
    qubit_map: dict[str, int],
    port_map: dict[str, int],
) -> None:
    def draw_for_single_operation(index: int) -> None:
        ax.annotate(
            "",
            xy=(body_start, index + 0.25),
            xytext=(measure_time, measure_qubit_idx + 0.25),
            arrowprops=dict(
                arrowstyle="->",
                facecolor=constants.COLOR_DARK_MODE_LINE,
                linewidth=constants.CTRL_FLOW_ARROW_LINEWIDTH,
                connectionstyle="bar,angle=180,fraction=-0.25",
            ),
        )
        ax.text(
            measure_time + 0.5,
            measure_qubit_idx + 0.5,
            "m=1",
            ha="center",
            va="center",
            backgroundcolor="white",
        )

    def draw_rectangle_with_arrow(bottom_qubit: int, top_qubit: int) -> None:
        p1 = matplotlib.patches.Rectangle(  # type: ignore
            (body_start - 0.45, bottom_qubit - 0.45),
            body_end - body_start + 0.9,
            top_qubit - bottom_qubit + 0.95,
            edgecolor=constants.COLOR_DARK_MODE_LINE,
            fill=False,
        )
        ax.add_patch(p1)
        ax.annotate(
            "",
            xy=(body_start - 0.45, top_qubit + 0.4),
            xytext=(measure_time, measure_qubit_idx + 0.25),
            arrowprops=dict(
                arrowstyle="->",
                facecolor=constants.COLOR_DARK_MODE_LINE,
                linewidth=constants.CTRL_FLOW_ARROW_LINEWIDTH,
                connectionstyle="angle,angleA=90,angleB=180,rad=0",
            ),
        )

    def draw_for_schedule_all_qubits() -> None:
        draw_rectangle_with_arrow(0, len(qubit_map) - 1)
        ax.text(
            measure_time - 0.1,
            measure_qubit_idx + 0.5,
            "m=1",
            ha="center",
            va="center",
            backgroundcolor="white",
        )

    def draw_for_schedule_single_qubits(involved_indices: Iterable[int]) -> None:
        for index in involved_indices:
            draw_rectangle_with_arrow(index, index)
        ax.text(
            measure_time - 0.1,
            measure_qubit_idx + 0.5,
            "m=1",
            ha="center",
            va="center",
            backgroundcolor="white",
        )

    involved_indices = _get_indices(body, qubit_map, port_map)
    if isinstance(body, Operation):
        for idx in involved_indices:
            draw_for_single_operation(idx)
    elif len(involved_indices) == len(qubit_map):
        draw_for_schedule_all_qubits()
    else:
        draw_for_schedule_single_qubits(involved_indices)


def _get_qubit_and_port_map_from_schedule(
    schedule: Schedule,
) -> tuple[dict[str, int], dict[str, int]]:
    ports: set[str] = set()
    qubits: set[str] = set()
    for operation in _walk_schedule_only_operations(schedule):
        if operation.valid_gate:
            qubits.update(operation["gate_info"]["qubits"])
            continue
        for info in chain(operation["pulse_info"], operation["acquisition_info"]):
            if (port := info["port"]) is not None:
                # Can be None e.g. in case of NCO operations.
                ports.add(port)

    qubit_map = {qubit: idx for idx, qubit in enumerate(sorted(qubits))}
    port_map: dict[str, int] = {}
    added_other = False
    for port in ports:
        maybe_qubit = port.split(":")[0]
        if maybe_qubit in qubits:
            port_map[port] = qubit_map[maybe_qubit]
        elif not added_other:
            for qubit, idx in qubit_map.items():
                qubit_map[qubit] = idx + 1
            qubit_map["other"] = 0
            port_map[port] = 0
            added_other = True
        else:
            port_map[port] = qubit_map["other"]
    return qubit_map, port_map


def _get_feedback_label_and_qubit_idx(
    operation: Operation, port_map: dict[str, int], qubit_map: dict[str, int]
) -> tuple[str, int] | None:
    """Check if the operation is an acquisition/measure gate with a feedback trigger label."""
    if (
        len(operation["acquisition_info"])
        and (feedback_label := operation["acquisition_info"][0].get("feedback_trigger_label", None))
        is not None
    ):
        return feedback_label, port_map[operation["acquisition_info"][0]["port"]]
    if (
        operation.valid_gate
        and (feedback_label := operation["gate_info"].get("feedback_trigger_label", None))
        is not None
    ):
        return feedback_label, qubit_map[operation["gate_info"]["qubits"][0]]
    return None


def circuit_diagram_matplotlib(
    schedule: Schedule,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    # to prevent the original input schedule from being modified.
    schedule = _determine_absolute_timing(deepcopy(schedule), "ideal")

    qubit_map, port_map = _get_qubit_and_port_map_from_schedule(schedule)

    if figsize is None:
        figsize = (10, len(qubit_map))
    fig, ax = ps.new_pulse_fig(figsize=figsize, ax=ax)
    ax.set_title(schedule.data["name"])
    ax.set_aspect("equal")

    ax.set_ylim(-0.5, len(qubit_map) - 0.5)
    for y in qubit_map.values():
        ax.axhline(y, color=constants.COLOR_DARK_MODE_LINE, linewidth=0.9)

    # plot the qubit names on the y-axis
    ax.set_yticks(list(qubit_map.values()))
    ax.set_yticklabels(qubit_map.keys())

    current_diagram_time = 0
    last_operation_time = 0

    # Stack of (loop start time, loop operation) tuples
    loop_scopes: list[tuple[int, LoopOperation]] = []
    # Stack of (conditional start time, conditional operation) tuples
    conditional_scopes: list[tuple[int, ConditionalOperation]] = []
    # Map from feedback_trigger_label to (thresholded acq time, qubit) tuple
    feedback_acq_map: dict[str, tuple[int, int]] = {}
    for abs_time, operation in _walk_schedule(schedule):
        if isinstance(operation, LoopOperation):
            loop_scopes.append((current_diagram_time + 1, operation))
        elif isinstance(operation, ConditionalOperation):
            conditional_scopes.append((current_diagram_time + 1, operation))
        elif isinstance(operation, Operation):
            if abs_time > last_operation_time:
                current_diagram_time += 1
                last_operation_time = abs_time
                # draw_time is a quick fix for displaying simultaneity consistently if a
                # single operation is simultaneous with a sub-schedule with multiple
                # operations.
                draw_time = current_diagram_time
            else:
                draw_time = abs_time
            if (
                feedback_label_and_qubit_idx := _get_feedback_label_and_qubit_idx(
                    operation, port_map, qubit_map
                )
            ) is not None:
                feedback_label, qubit_idx = feedback_label_and_qubit_idx
                feedback_acq_map[feedback_label] = current_diagram_time, qubit_idx
            _draw_operation(
                operation=operation,
                qubit_map=qubit_map,
                port_map=port_map,
                ax=ax,
                time=draw_time,
                schedule_resources=schedule.resources,
            )
        elif isinstance(operation, ScheduleBase):
            pass
        elif operation == _ControlFlowEnd.LOOP_END:
            start_time, loop_op = loop_scopes.pop()
            _draw_loop(
                ax=ax,
                qubit_map=qubit_map,
                port_map=port_map,
                operation=loop_op,
                start_time=start_time,
                end_time=current_diagram_time,
            )
        elif operation == _ControlFlowEnd.CONDI_END:
            body_start, conditional_op = conditional_scopes.pop()
            feedback_trigger_label = conditional_op["control_flow_info"]["feedback_trigger_label"]
            try:
                measure_time, measure_qubit_idx = feedback_acq_map[feedback_trigger_label]
            except KeyError as err:
                raise KeyError(
                    f"Feedback trigger label '{feedback_trigger_label}' not found in "
                    "any preceding Measure or acquisition operation."
                ) from err
            _draw_conditional(
                ax=ax,
                measure_time=measure_time,
                measure_qubit_idx=measure_qubit_idx,
                body=conditional_op.body,
                body_start=body_start,
                body_end=current_diagram_time,
                qubit_map=qubit_map,
                port_map=port_map,
            )

    ax.set_xlim(-1, current_diagram_time + 1)

    return fig, ax
