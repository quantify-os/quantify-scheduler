# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Plotting functions used in the visualization backend of the sequencer."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

import quantify_scheduler.visualization.pulse_scheme as ps
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.visualization import constants

if TYPE_CHECKING:
    from quantify_scheduler import Schedule


def gate_box(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    A box for a single gate containing a label.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
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


def pulse_baseband(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Adds a visual indicator for a Baseband pulse to the `mathplotlib.axes.Axis`
    instance.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
    """
    for qubit_idx in qubit_idxs:
        ps.fluxPulse(
            ax, pos=time, y_offs=qubit_idx, width=0.4, s=0.0025, amp=0.33, **kw
        )
        ax.text(time + 0.2, qubit_idx + 0.45, text, ha="center", va="center", zorder=6)


def pulse_modulated(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Adds a visual indicator for a Modulated pulse to the `mathplotlib.axes.Axis`
    instance.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
    """
    for qubit_idx in qubit_idxs:
        ps.mwPulse(ax, pos=time, y_offs=qubit_idx, width=0.4, amp=0.33, **kw)
        ax.text(time + 0.2, qubit_idx + 0.45, text, ha="center", va="center", zorder=6)


# pylint: disable=unused-argument
def meter(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    A simple meter to depict a measurement.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
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


# pylint: disable=unused-argument
def acq_meter(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Variation of the meter to depict a acquisition.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
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


def acq_meter_text(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Same as acq_meter, but also displays text.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
    """
    acq_meter(ax, time, qubit_idxs, text, **kw)
    ax.text(time, max(qubit_idxs) + 0.45, text, ha="center", va="center", zorder=6)


# pylint: disable=unused-argument
def cnot(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Markers to denote a CNOT gate between two qubits.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
    """
    ax.plot(
        [time, time], qubit_idxs, marker="o", markersize=15, color=constants.COLOR_BLUE
    )
    ax.plot([time], qubit_idxs[1], marker="+", markersize=12, color="white")


# pylint: disable=unused-argument, invalid-name
def cz(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
    """
    Markers to denote a CZ gate between two qubits.

    Parameters
    ----------
    ax
    time
    qubit_idxs
    text
    """
    ax.plot(
        [time, time], qubit_idxs, marker="o", markersize=15, color=constants.COLOR_BLUE
    )


def reset(ax: Axes, time: float, qubit_idxs: List[int], text: str, **kw):
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
    text
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


def _locate_qubit_in_address(qubit_map, address):
    """
    Returns the name of a qubit in  a pulse address.
    """
    if address is None:
        raise ValueError(f"Could not resolve address '{address}'")
    for sub_addr in address.split(":"):
        if sub_addr in qubit_map:
            return sub_addr
    raise ValueError(f"Could not resolve address '{address}'")


# pylint disabled because func was implemented before pylint was adopted
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def circuit_diagram_matplotlib(
    schedule: Schedule,
    figsize: Tuple[int, int] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Creates a circuit diagram visualization of a schedule using matplotlib.

    Parameters
    ----------
    schedule
        the schedule to render.
    figsize
        matplotlib figsize.
    ax
        Axis handle to use for plotting.

    Returns
    -------
    fig
        matplotlib figure object.
    ax
        matplotlib axis object.
    """
    # to prevent the original input schedule from being modified.
    schedule = deepcopy(schedule)

    # pylint: disable=import-outside-toplevel
    # importing inside function scope to prevent circular import
    from quantify_scheduler.compilation import determine_absolute_timing

    schedule = determine_absolute_timing(schedule, "ideal")

    qubit_map: Dict[str, int] = {}
    qubits: List[str] = set()
    for _, operation in schedule.operations.items():
        if operation.valid_gate:
            qubits.update(operation.data["gate_info"]["qubits"])

    for index, qubit in enumerate(sorted(qubits)):
        qubit_map[qubit] = index

    # Validate pulses
    # If the pulse's port address was not found then the pulse
    # will be plotted on the 'other' timeline.
    # Note: needs to be done before creating figure and axhline
    # in order to avoid unnecessary redraws.
    for schedulable in schedule.schedulables.values():
        operation = schedule.operations[schedulable["operation_repr"]]
        if operation.valid_pulse:
            try:
                for pulse_info in operation["pulse_info"]:
                    _locate_qubit_in_address(qubit_map, pulse_info["port"])
            except ValueError:
                for key in qubit_map:
                    qubit_map[key] += 1
                qubit_map["other"] = 0
                break
        if operation.valid_acquisition:
            try:
                for acq_info in operation["acquisition_info"]:
                    _locate_qubit_in_address(qubit_map, acq_info["port"])
            except ValueError:
                for key in qubit_map:
                    qubit_map[key] += 1
                qubit_map["other"] = 0
                break

    if figsize is None:
        figsize = (10, len(qubit_map))
    fig, ax = ps.new_pulse_fig(figsize=figsize, ax=ax)
    ax.set_title(schedule.data["name"])
    ax.set_aspect("equal")

    ax.set_ylim(-0.5, len(qubit_map) - 0.5)
    ax.axhline(0, color="0.1", linewidth=0.9)
    for qubit in qubits:
        ax.axhline(qubit_map[qubit], color="0.1", linewidth=0.9)

    # plot the qubit names on the y-axis
    ax.set_yticks(list(qubit_map.values()))
    ax.set_yticklabels(qubit_map.keys())

    total_duration = 0
    for schedulable in schedule.schedulables.values():
        operation = schedule.operations[schedulable["operation_repr"]]

        time = schedulable["abs_time"]
        total_duration = total_duration if total_duration > time else time

        if operation.valid_gate:
            plot_func = import_python_object_from_string(
                operation["gate_info"]["plot_func"]
            )
            idxs = [qubit_map[qubit] for qubit in operation["gate_info"]["qubits"]]
            plot_func(
                ax, time=time, qubit_idxs=idxs, text=operation["gate_info"]["tex"]
            )
        elif operation.valid_pulse:
            idxs: List[int]
            try:
                idxs = [
                    qubit_map[_locate_qubit_in_address(qubit_map, pulse_info["port"])]
                    for pulse_info in operation["pulse_info"]
                ]
            except ValueError:
                # The pulse port was not found in the qubit_map
                # move this pulse to the 'other' timeline
                idxs = [0]

            for pulse_info in operation["pulse_info"]:
                clock_id: str = pulse_info["clock"]
                clock_resource: dict = schedule.data["resource_dict"][clock_id]
                if clock_resource["freq"] == 0:
                    pulse_baseband(ax, time=time, qubit_idxs=idxs, text=operation.name)
                else:
                    pulse_modulated(ax, time=time, qubit_idxs=idxs, text=operation.name)
        elif operation.valid_acquisition:
            idxs: List[int]
            try:
                idxs = [
                    qubit_map[_locate_qubit_in_address(qubit_map, acq_info["port"])]
                    for acq_info in operation["acquisition_info"]
                ]
            except ValueError:
                # The pulse port was not found in the qubit_map
                # move this pulse to the 'other' timeline
                idxs = [0]

            for _ in operation["acquisition_info"]:
                acq_meter(ax, time=time, qubit_idxs=idxs, text=operation.name)
        else:
            raise ValueError("Unknown operation")

    ax.set_xlim(-1, total_duration + 1)

    return fig, ax
