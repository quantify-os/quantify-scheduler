# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Plotting functions used in the visualization backend of the sequencer."""
from __future__ import annotations
from typing import Tuple, Union, List, Dict
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import quantify.scheduler.visualization.pulse_scheme as ps
from quantify.scheduler.types import Schedule
from quantify.scheduler.compilation import determine_absolute_timing

# pylint: disable=no-name-in-module
from quantify.utilities.general import import_func_from_string


def gate_box(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    A box for a single gate containing a label.

    Parameters
    ----------
    ax :

    time :

    qubit_idxs :

    text :

    """
    for qubit_idx in qubit_idxs:
        ps.box_text(
            ax,
            x0=time,
            y0=qubit_idx,
            text=text,
            fillcolor="C0",
            width=0.8,
            height=0.5,
            **kw
        )


def pulse_baseband(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Adds a visual indicator for a Baseband pulse to the `mathplotlib.axes.Axis` instance.

    Parameters
    ----------
    ax :

    time :

    qubit_idxs :

    text :

    """
    for qubit_idx in qubit_idxs:
        ps.fluxPulse(
            ax, pos=time, y_offs=qubit_idx, width=0.4, s=0.0025, amp=0.33, **kw
        )
        ax.text(time + 0.2, qubit_idx + 0.45, text, ha="center", va="center", zorder=6)


def pulse_modulated(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Adds a visual indicator for a Modulated pulse to the `mathplotlib.axes.Axis` instance.

    Parameters
    ----------
    ax :

    time :

    qubit_idxs :

    text :

    """
    for qubit_idx in qubit_idxs:
        ps.mwPulse(ax, pos=time, y_offs=qubit_idx, width=0.4, amp=0.33, **kw)
        ax.text(time + 0.2, qubit_idx + 0.45, text, ha="center", va="center", zorder=6)


def meter(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    A simple meter to depict a measurement.

    Parameters
    ----------
    ax :

    time :

    qubit_idxs :

    text :

    """
    for qubit_idx in qubit_idxs:
        ps.meter(
            ax,
            x0=time,
            y0=qubit_idx,
            fillcolor="C4",
            y_offs=0,
            width=0.8,
            height=0.5,
            **kw
        )


def cnot(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Markers to denote a CNOT gate between two qubits.

    Parameters
    ----------
    ax :

    time :

    qubit_idxs :

    text :

    """
    ax.plot([time, time], qubit_idxs, marker="o", markersize=15, color="C1")
    ax.plot([time], qubit_idxs[1], marker="+", markersize=12, color="white")


def cz(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Markers to denote a CZ gate between two qubits.

    Parameters
    ----------
    ax :

    time :

    qubit_idxs :

    text :

    """
    ax.plot([time, time], qubit_idxs, marker="o", markersize=15, color="C1")


def reset(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    A broken line to denote qubit initialization.

    Parameters
    ----------
    ax :

    time :

    qubit_idxs :

    text :

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
            **kw
        )


def _locate_qubit_in_address(qubit_map, address):
    """
    Returns the name of a qubit in  a pulse address.
    """
    for sub_addr in address.split(":"):
        if sub_addr in qubit_map:
            return sub_addr
    raise ValueError("Could not resolve address '{}'".format(address))


def circuit_diagram_matplotlib(
    schedule: Schedule, figsize: Tuple[int, int] = None
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Creates a circuit diagram visualization of a schedule using matplotlib.

    For this visualization backend to work, the schedule must contain a value for `abs_time` for each element in the
    timing_constraints.

    Parameters
    ----------
    schedule :
        the schedule to render.
    figsize :
        matplotlib figsize.

    Returns
    -------
    fig :
        matplotlib figure object.
    ax :
        matplotlib axis object.
    """
    schedule = determine_absolute_timing(schedule, "ideal")

    qubit_map: Dict[str, int] = dict()
    qubits: List[str] = set()
    for _, op in schedule.operations.items():
        if op.valid_gate:
            qubits.update(op.data["gate_info"]["qubits"])

    for index, qubit in enumerate(sorted(qubits)):
        qubit_map[qubit] = index

    # Validate pulses
    # If the pulse's port address was not found then the pulse
    # will be plotted on the 'other' timeline.
    # Note: needs to be done be done before creating figure and axhline
    # in order to avoid unnecessary redraws.
    for t_constr in schedule.timing_constraints:
        op = schedule.operations[t_constr["operation_hash"]]
        if op.valid_pulse:
            try:
                for pulse_info in op["pulse_info"]:
                    _locate_qubit_in_address(qubit_map, pulse_info["port"])
            except ValueError:
                for key in qubit_map:
                    qubit_map[key] += 1
                qubit_map["other"] = 0
                break

    if figsize is None:
        figsize = (10, len(qubit_map))
    fig, ax = ps.new_pulse_fig(figsize=figsize)
    ax.set_title(schedule.data["name"])
    ax.set_aspect("equal")

    ax.set_ylim(-0.5, len(qubit_map) - 0.5)
    for q in qubits:
        ax.axhline(qubit_map[q], color=".75")

    # plot the qubit names on the y-axis
    ax.set_yticks(list(qubit_map.values()))
    ax.set_yticklabels(qubit_map.keys())

    total_duration = 0
    for t_constr in schedule.timing_constraints:
        op = schedule.operations[t_constr["operation_hash"]]

        time = t_constr["abs_time"]
        total_duration = total_duration if total_duration > time else time

        if op.valid_gate:
            plot_func = import_func_from_string(op["gate_info"]["plot_func"])
            idxs = [qubit_map[q] for q in op["gate_info"]["qubits"]]
            plot_func(ax, time=time, qubit_idxs=idxs, text=op["gate_info"]["tex"])
        elif op.valid_pulse:
            idxs: List[int]
            try:
                idxs = [
                    qubit_map[_locate_qubit_in_address(qubit_map, pulse_info["port"])]
                    for pulse_info in op["pulse_info"]
                ]
            except ValueError:
                # The pulse port was not found in the qubit_map
                # move this pulse to the 'other' timeline
                idxs = [0]

            for pulse_info in op["pulse_info"]:
                clock_id: str = pulse_info["clock"]
                clock_resource: dict = schedule.data["resource_dict"][clock_id]
                if clock_resource["freq"] == 0:
                    pulse_baseband(ax, time=time, qubit_idxs=idxs, text=op.name)
                else:
                    pulse_modulated(ax, time=time, qubit_idxs=idxs, text=op.name)
        else:
            raise ValueError("Unknown operation")

    ax.set_xlim(-1, total_duration + 1)

    return fig, ax
