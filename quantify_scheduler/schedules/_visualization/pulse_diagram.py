# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for drawing pulse diagrams."""
from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quantify_core.visualization.SI_utilities import set_xlabel, set_ylabel

import quantify_scheduler.operations.pulse_library as pl
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.helpers.waveforms import modulate_waveform
from quantify_scheduler.operations.acquisition_library import AcquisitionOperation
from quantify_scheduler.backends.qblox.operations.stitched_pulse import (
    convert_to_numerical_pulse,
)

if TYPE_CHECKING:
    from quantify_scheduler import CompiledSchedule, Operation, Schedule

logger = logging.getLogger(__name__)


def _populate_port_mapping(schedule, portmap: Dict[str, int], ports_length) -> None:
    """Dynamically add up to 8 ports to the port_map dictionary."""
    offset_idx: int = 0

    for schedulable in schedule.schedulables.values():
        operation = schedule.operations[schedulable["operation_id"]]
        for operation_info in operation["pulse_info"] + operation["acquisition_info"]:
            if offset_idx == ports_length:
                return

            port = operation_info["port"]
            if port is None:
                continue

            if port not in portmap:
                portmap[port] = offset_idx
                offset_idx += 1


def validate_operation_data(operation_data, port_map, schedulable, operation):
    """Validates if the pulse/acquisition information is valid for visualization."""
    if operation_data["port"] not in port_map:
        # Do not draw pulses for this port
        return False

    if operation_data["port"] is None:
        logger.warning(
            "Unable to sample waveform for operation_data due to missing 'port' for "
            f"operation name={operation['name']} "
            f"id={schedulable['operation_id']} operation_data={operation_data}"
        )
        return False

    if "acq_index" not in operation_data:  # This will be skipped for acquisitions
        if operation_data["wf_func"] is None:
            logger.warning(
                "Unable to sample pulse for pulse_info due to missing 'wf_func' for "
                f"operation name={operation['name']} "
                f"id={schedulable['operation_id']} operation_data={operation_data}"
            )
            return False
    return True


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def pulse_diagram_plotly(
    schedule: Schedule | CompiledSchedule,
    port_list: Optional[List[str]] = None,
    fig_ch_height: float = 300,
    fig_width: float = 1000,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    sampling_rate: float = 1e9,
) -> go.Figure:
    """
    Produce a plotly visualization of the pulses used in the schedule.

    Parameters
    ----------
    schedule :
        The schedule to render.
    port_list :
        A list of ports to show. if set to ``None`` will use the first
        8 ports it encounters in the sequence.
    fig_ch_height :
        Height for each channel subplot in px.
    fig_width :
        Width for the figure in px.
    modulation :
        Determines if modulation is included in the visualization.
    modulation_if :
        Modulation frequency used when modulation is set to "if".
    sampling_rate :
        The time resolution used to sample the schedule in Hz.

    Returns
    -------
    :class:`plotly.graph_objects.Figure` :
        the plot
    """
    port_map: Dict[str, int] = {}
    ports_length: int = 8

    if port_list is not None:
        ports_length = len(port_list)
        port_map = dict(zip(port_list, range(len(port_list))))
    else:
        _populate_port_mapping(schedule, port_map, ports_length)
        ports_length = len(port_map)

    nrows = ports_length
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.update_layout(
        height=fig_ch_height * nrows,
        width=fig_width,
        title=schedule.data["name"],
        showlegend=False,
    )

    colors = px.colors.qualitative.Plotly
    col_idx: int = 0

    for pulse_idx, schedulable in enumerate(schedule.schedulables.values()):
        operation = schedule.operations[schedulable["operation_id"]]

        if operation.has_voltage_offset:
            operation = convert_to_numerical_pulse(
                operation, scheduled_at=schedulable["abs_time"]
            )

        for pulse_info in operation["pulse_info"]:
            if not validate_operation_data(
                pulse_info, port_map, schedulable, operation
            ):
                continue

            port: str = pulse_info["port"]

            wf_func: Callable = import_python_object_from_string(pulse_info["wf_func"])

            col_idx = (col_idx + 1) % len(colors)

            t0 = schedulable["abs_time"] + pulse_info["t0"]
            t = np.arange(t0, t0 + pulse_info["duration"], 1 / sampling_rate)
            par_map = inspect.signature(wf_func).parameters
            wf_kwargs = {}
            for kwargs in par_map.keys():
                if kwargs in pulse_info.keys():
                    wf_kwargs[kwargs] = pulse_info[kwargs]

            # check for reference equality in case import alias is used
            if wf_func == import_python_object_from_string(
                "quantify_scheduler.waveforms.interpolated_complex_waveform"
            ):
                wf_kwargs["t_samples"] = (
                    np.asarray(wf_kwargs["t_samples"]) - wf_kwargs["t_samples"][0]
                )

            waveform = wf_func(t=t - t[0], **wf_kwargs)

            if modulation == "clock":
                waveform = modulate_waveform(
                    t, waveform, schedule.resources[pulse_info["clock"]]["freq"]
                )

            if modulation == "if":
                waveform = modulate_waveform(t, waveform, modulation_if)

            row: int = port_map[port] + 1

            label = operation["name"]
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=waveform.real,
                    mode="lines",
                    name=f"{label}, clock: {pulse_info['clock']}",
                    legendgroup=pulse_idx,
                    showlegend=True,
                    line_color=colors[col_idx],
                    fill="tozeroy",
                    hoverinfo="x+y+name",
                    hoverlabel={"namelength": -1},
                ),
                row=row,
                col=1,
            )

            if waveform.dtype.kind == "c":
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=waveform.imag,
                        mode="lines",
                        name=f"Im[{label}], clock: {pulse_info['clock']}",
                        legendgroup=pulse_idx,
                        showlegend=True,
                        line_color="darkgrey",
                        fill="tozeroy",
                        hoverinfo="x+y+name",
                        hoverlabel={"namelength": -1},
                    ),
                    row=row,
                    col=1,
                )

            fig.update_xaxes(
                row=row,
                col=1,
                tickformat=".2s",
                hoverformat=".3s",
                ticksuffix="s",
                showgrid=True,
            )
            fig.update_yaxes(
                row=row,
                col=1,
                tickformat=".2s",
                hoverformat=".3s",
                ticksuffix="V",
                title=port,
                autorange=True,
            )

        for acq_info in operation["acquisition_info"]:
            if not validate_operation_data(acq_info, port_map, schedulable, operation):
                continue
            acq_port: str = acq_info["port"]
            label = operation["name"]

            row = port_map[acq_port] + 1
            t = schedulable["abs_time"] + acq_info["t0"]
            yref: str = f"y{row} domain" if row != 1 else "y domain"
            fig.add_trace(
                go.Scatter(
                    x=[t, t + acq_info["duration"]],
                    y=[0, 0],
                    name=label,
                    mode="markers",
                    marker=dict(
                        size=15,
                        color="rgba(0,0,0,.25)",
                        symbol=["arrow-bar-left", "arrow-bar-right"],
                    ),
                ),
                row=row,
                col=1,
            )
            fig.add_shape(
                type="rect",
                xref="x",
                yref=yref,
                x0=t,
                y0=0,
                x1=t + acq_info["duration"],
                y1=1,
                name=label,
                line=dict(
                    color="rgba(0,0,0,0)",
                    width=3,
                ),
                fillcolor="rgba(255,0,0,0.1)",
                layer="below",
            )
            fig.update_xaxes(
                row=row,
                col=1,
                tickformat=".2s",
                hoverformat=".3s",
                ticksuffix="s",
                showgrid=True,
            )
            fig.update_yaxes(
                row=row,
                col=1,
                tickformat=".2s",
                hoverformat=".3s",
                ticksuffix="V",
                title=acq_port,
                autorange=True,
            )

    fig.update_xaxes(
        row=ports_length,
        col=1,
        title="Time",
        tickformatstops=[
            dict(dtickrange=[None, 1e-9], value=".10s"),
            dict(dtickrange=[1e-9, 1e-6], value=".7s"),
            dict(dtickrange=[1e-6, 1e-3], value=".4s"),
        ],
        ticksuffix="s",
    )

    return fig


@dataclass
class SampledPulse:
    time: np.ndarray
    signal: np.ndarray
    label: str


def sample_schedule(
    schedule: Schedule | CompiledSchedule,
    port_list: Optional[List[str]] = None,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    sampling_rate: float = 1e9,
    x_range: Tuple[float, float] = (-np.inf, np.inf),
) -> Dict[str, List[SampledPulse]]:
    """
    Sample a schedule at discrete points in time.

    Parameters
    ----------
    schedule :
        The schedule to render.
    port_list :
        A list of ports to show. if set to ``None`` will use the first
        8 ports it encounters in the sequence.
    modulation :
        Determines if modulation is included in the visualization.
    modulation_if :
        Modulation frequency used when modulation is set to "if".
    sampling_rate :
        The time resolution used to sample the schedule in Hz.
    x_range :
        The minimum and maximum time values at which to sample the waveforms.

    Returns
    -------
    :
        Dictionary that maps each used port to the sampled pulses played on that port.
    """
    if x_range[0] > x_range[1]:
        raise ValueError(
            f"Expected the left limit of x_range to be smaller than the right limit, "
            f"but got (left, right) = {x_range}"
        )

    port_map: Dict[str, int] = {}
    ports_length: int = 8

    if port_list is not None:
        ports_length = len(port_list)
        port_map = dict(zip(port_list, range(len(port_list))))
    else:
        _populate_port_mapping(schedule, port_map, ports_length)
        ports_length = len(port_map)

    waveforms: Dict[str, List[SampledPulse]] = {}

    min_x, max_x = x_range
    for schedulable in schedule.schedulables.values():
        operation = schedule.operations[schedulable["operation_id"]]

        if operation.has_voltage_offset:
            operation = convert_to_numerical_pulse(
                operation, scheduled_at=schedulable["abs_time"]
            )

        for pulse_info in operation["pulse_info"]:
            if not validate_operation_data(
                pulse_info, port_map, schedulable, operation
            ):
                logging.info(f"Operation {operation} is not valid for plotting.")
                continue

            t0 = schedulable["abs_time"] + pulse_info["t0"]
            t1 = t0 + pulse_info["duration"]

            if t1 < min_x or t0 > max_x:
                continue

            t0 = max(min_x, t0)
            t1 = min(max_x, t1)

            port: str = pulse_info["port"]

            wf_func: Callable = import_python_object_from_string(pulse_info["wf_func"])

            t = np.arange(t0, t1 + 0.5 / sampling_rate, 1 / sampling_rate)
            if len(t) == 0:
                continue

            par_map = inspect.signature(wf_func).parameters
            wf_kwargs = {}
            for kwargs in par_map.keys():
                if kwargs in pulse_info.keys():
                    wf_kwargs[kwargs] = pulse_info[kwargs]

            # check for reference equality in case import alias is used
            if wf_func == import_python_object_from_string(
                "quantify_scheduler.waveforms.interpolated_complex_waveform"
            ):
                wf_kwargs["t_samples"] = (
                    np.asarray(wf_kwargs["t_samples"]) - wf_kwargs["t_samples"][0]
                )

            waveform = wf_func(t=t - t[0], **wf_kwargs)

            if modulation == "clock":
                waveform = modulate_waveform(
                    t, waveform, schedule.resources[pulse_info["clock"]]["freq"]
                )

            if modulation == "if":
                waveform = modulate_waveform(t, waveform, modulation_if)

            waveform = np.real_if_close(waveform)
            label = f"{operation['name']}, clock {pulse_info['clock']}"
            if port in waveforms:
                waveforms[port].append(SampledPulse(t, waveform, label))
            else:
                waveforms[port] = [SampledPulse(t, waveform, label)]

    return waveforms


def deduplicate_legend_handles_labels(ax: mpl.axes.Axes) -> None:
    """
    Remove duplicate legend entries.

    See also: https://stackoverflow.com/a/13589144
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def plot_single_subplot_mpl(
    sampled_schedule: Dict[str, List[SampledPulse]],
    ax: Optional[mpl.axes.Axes] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot all pulses for all ports in the schedule in the same subplot.

    Pulses in the same port have the same color and legend entry, and each port
    has its own legend entry.

    Parameters
    ----------
    sampled_schedule :
        Dictionary that maps each used port to the sampled pulses played on that port.
    ax :
        A pre-existing Axes object to plot the pulses in. If ``None`` (default), this object is
        created within the function.

    Returns
    -------
    fig :
        A matplotlib :class:`matplotlib.figure.Figure` containing the subplot.

    ax :
        The Axes of the subplot belonging to the Figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for i, (port, data) in enumerate(sampled_schedule.items()):
        for pulse in data:
            ax.plot(pulse.time, pulse.signal.real, color=f"C{i}", label=f"port {port}")
            ax.fill_between(pulse.time, pulse.signal.real, color=f"C{i}", alpha=0.2)

            if np.iscomplexobj(pulse.signal):
                ax.plot(
                    pulse.time,
                    pulse.signal.imag,
                    color=f"C{i}",
                    linestyle="--",
                    label=f"port {port} (imag)",
                )
                ax.fill_between(pulse.time, pulse.signal.imag, color=f"C{i}", alpha=0.4)

    deduplicate_legend_handles_labels(ax)
    set_xlabel(label="Time", unit="s", axis=ax)
    set_ylabel(label="Amplitude", unit="V", axis=ax)
    return fig, ax


def plot_multiple_subplots_mpl(
    sampled_schedule: Dict[str, List[SampledPulse]]
) -> Tuple[mpl.figure.Figure, List[mpl.axes.Axes]]:
    """
    Plot pulses in a different subplot for each port in the sampled schedule.

    For each subplot, each different type of pulse gets its own color and legend
    entry.

    Parameters
    ----------
    sampled_schedule :
        Dictionary that maps each used port to the sampled pulses played on that port.

    Returns
    -------
    fig :
        A matplotlib :class:`matplotlib.figure.Figure` containing the subplots.

    axs :
        An array of Axes objects belonging to the Figure.
    """
    fig, axs = plt.subplots(len(sampled_schedule), 1, sharex=True)

    for i, (port, data) in enumerate(sampled_schedule.items()):
        # This automatically creates a label-to-color map as the plots get created.
        color: Dict[str, str] = defaultdict(lambda: f"C{len(color)}")

        for pulse in data:
            axs[i].plot(
                pulse.time,
                pulse.signal.real,
                color=color[pulse.label],
                label=pulse.label,
            )
            axs[i].fill_between(
                pulse.time, pulse.signal.real, color=color[pulse.label], alpha=0.2
            )

            if np.iscomplexobj(pulse.signal):
                axs[i].plot(
                    pulse.time,
                    pulse.signal.imag,
                    color=color[pulse.label],
                    linestyle="--",
                    label=f"{pulse.label} (imag)",
                )
                axs[i].fill_between(
                    pulse.time, pulse.signal.imag, color=color[pulse.label], alpha=0.4
                )

        deduplicate_legend_handles_labels(axs[i])
        set_ylabel(label=f"port {port}\nAmplitude", unit="V", axis=axs[i])

    set_xlabel(label="Time", unit="s", axis=axs[-1])

    # Make the figure taller if y-labels overlap.
    fig.set_figheight(max(4.8 * len(axs) / 3, 4.8))
    return fig, axs


def pulse_diagram_matplotlib(
    schedule: Schedule | CompiledSchedule,
    port_list: Optional[List[str]] = None,
    sampling_rate: float = 1e9,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    x_range: Tuple[float, float] = (-np.inf, np.inf),
    multiple_subplots: bool = False,
    ax: Optional[mpl.axes.Axes] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes | List[mpl.axes.Axes]]:
    """
    Plots a schedule using matplotlib.

    Parameters
    ----------
    schedule :
        The schedule to plot.
    port_list :
        A list of ports to show. If ``None`` (default) the first 8 ports
        encountered in the sequence are used.
    sampling_rate :
        The time resolution used to sample the schedule in Hz. By default 1e9.
    modulation :
        Determines if modulation is included in the visualization. By default "off".
    modulation_if :
        Modulation frequency used when modulation is set to "if". By default 0.0.
    x_range :
        The range of the x-axis that is plotted, given as a tuple (left limit, right
        limit). This can be used to reduce memory usage when plotting a small section of
        a long pulse sequence. By default (-np.inf, np.inf).
    multiple_subplots :
        Plot the pulses for each port on a different subplot if True, else plot
        everything in one subplot. By default False. When using just one
        subplot, the pulses are colored according to the port on which they
        play. For multiple subplots, each pulse has its own
        color and legend entry.
    ax :
        Axis onto which to plot. If `None`, this is created within the function. By
        default None.

    Returns
    -------
    fig :
        A matplotlib :class:`matplotlib.figure.Figure` containing the subplot(s).

    ax :
        The Axes object belonging to the Figure, or an array of Axes if
        ``multiple_subplots=True``.
    """
    pulses = sample_schedule(
        schedule,
        sampling_rate=sampling_rate,
        port_list=port_list,
        modulation=modulation,
        modulation_if=modulation_if,
        x_range=x_range,
    )

    if len(pulses) == 0:
        raise RuntimeError(
            f"Attempting to sample schedule {schedule.name}, "
            "but the schedule does not contain any `pulse_info`. "
            "Please verify that the schedule has been populated and "
            "device compilation has been performed."
        )

    if not multiple_subplots or len(pulses) == 1:
        return plot_single_subplot_mpl(pulses, ax)
    return plot_multiple_subplots_mpl(pulses)


def get_window_operations(
    schedule: Schedule,
) -> List[Tuple[float, float, Operation]]:
    r"""
    Return a list of all :class:`.WindowOperation`\s with start and end time.

    Parameters
    ----------
    schedule:
        Schedule to use.

    Returns
    -------
    :
        List of all window operations in the schedule.
    """
    window_operations = []
    for _, schedulable in enumerate(schedule.schedulables.values()):
        operation = schedule.operations[schedulable["operation_id"]]
        if isinstance(operation, pl.WindowOperation):
            for pulse_info in operation["pulse_info"]:
                t0 = schedulable["abs_time"] + pulse_info["t0"]
                t1 = t0 + pulse_info["duration"]

            window_operations.append((t0, t1, operation))
    return window_operations


def plot_window_operations(
    schedule: Schedule,
    ax: Optional[mpl.axes.Axes] = None,
    time_scale_factor: float = 1,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot the window operations in a schedule.

    Parameters
    ----------
    schedule:
        Schedule from which to plot window operations.
    ax:
        Axis handle to use for plotting.
    time_scale_factor:
        Used to scale the independent data before using as data for the
        x-axis of the plot.

    Returns
    -------
    fig
        The matplotlib figure.
    ax
        The matplotlib ax.
    """
    if ax is None:
        ax = plt.gca()

    window_operations = get_window_operations(schedule)

    cmap = mpl.cm.get_cmap("jet")

    for idx, (t0, t1, operation) in enumerate(window_operations):
        window_name = operation.window_name
        logging.debug(f"plot_window_operations: window {window_name}: {t0}, {t1}")
        colormap = cmap(idx / (1 + len(window_operations)))
        label = window_name
        ax.axvspan(
            time_scale_factor * t0,
            time_scale_factor * (t1),
            alpha=0.2,
            color=colormap,
            label=label,
        )

    return ax.get_figure(), ax


def plot_acquisition_operations(
    schedule: Schedule, ax: Optional[mpl.axes.Axes] = None, **kwargs
) -> List[Any]:
    """
    Plot the acquisition operations in a schedule.

    Parameters
    ----------
    schedule:
        Schedule from which to plot window operations.
    ax:
        Axis handle to use for plotting.
    kwargs:
        Passed to matplotlib plotting routine

    Returns
    -------
    :
        List of handles
    """
    if ax is None:
        ax = plt.gca()

    handles_list = []
    for idx, schedulable in enumerate(schedule.schedulables.values()):
        _ = idx  # unused variable
        operation = schedule.operations[schedulable["operation_id"]]
        if isinstance(operation, AcquisitionOperation):
            t0 = schedulable["abs_time"] + operation.data["acquisition_info"][0]["t0"]
            t1 = t0 + operation.duration
            handle = ax.axvspan(t0, t1, **kwargs)
            handles_list.append(handle)
    return handles_list
