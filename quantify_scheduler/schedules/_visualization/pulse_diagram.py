# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for drawing pulse diagrams."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import quantify_scheduler.operations.pulse_library as pl
from quantify_core.visualization.SI_utilities import set_xlabel, set_ylabel
from quantify_scheduler.helpers.waveforms import (
    exec_waveform_function,
    modulate_waveform,
)
from quantify_scheduler.operations.acquisition_library import Acquisition
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    LoopOperation,
)
from quantify_scheduler.schedules.schedule import ScheduleBase
from quantify_scheduler.waveforms import interpolated_complex_waveform

if TYPE_CHECKING:
    from quantify_scheduler import CompiledSchedule, Operation, Schedule

logger = logging.getLogger(__name__)


@dataclass
class SampledPulse:
    """Class containing the necessary information to display pulses in a plot."""

    time: np.ndarray
    signal: np.ndarray
    label: str


@dataclass
class SampledAcquisition:
    """Class containing the necessary information to display acquisitions in a plot."""

    t0: float
    duration: float
    label: str


@dataclass
class ScheduledInfo:
    """
    Class containing pulse or acquisition info, with some additional information.

    This class is used in the schedule sampling process to temporarily hold pulse info
    or acquisition info dictionaries, together with some useful information from the
    operation and schedulable that they are a part of.
    """

    op_info: dict[str, Any]
    """Pulse info or acquisition info."""
    time: float
    """The sum of the ``Schedulable["abs_time"]`` and the ``info["t0"]``."""
    op_name: str
    """The name of the operation containing the pulse or acquisition info."""


def get_sampled_pulses_from_voltage_offsets(
    schedule: Schedule | CompiledSchedule,
    offset_infos: dict[str, dict[str, list[ScheduledInfo]]],
    x_min: float,
    x_max: float,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    sampling_rate: float = 1e9,
    sampled_pulses: dict[str, list[SampledPulse]] | None = None,
) -> dict[str, list[SampledPulse]]:
    """
    Generate :class:`.SampledPulse` objects from :class:`.VoltageOffset` pulse_info dicts.

    This function groups all VoltageOffset operations by port-clock combination and
    turns each of those groups of operations into a single SampledPulse. The returned
    dictionary contains these SampledPulse objects grouped by port.

    Parameters
    ----------
    schedule :
        The schedule to render.
    offset_infos :
        A nested dictionary containing lists of pulse_info dictionaries. The outer
        dictionary's keys are ports, and the inner dictionary's keys are clocks.
    x_min :
        The left limit of the x-axis of the intended plot.
    x_max :
        The right limit of the x-axis of the intended plot.
    modulation :
        Determines if modulation is included in the visualization.
    modulation_if :
        Modulation frequency used when modulation is set to "if".
    sampling_rate :
        Number of samples per second to draw when drawing modulated pulses.
    sampled_pulses :
        An already existing dictionary (same type as the return value). If provided,
        this dictionary will be extended with the SampledPulse objects created in this
        function.

    Returns
    -------
    dict[str, list[SampledPulse]] :
        SampledPulse objects grouped by port.

    """
    if sampled_pulses is None:
        sampled_pulses = defaultdict(list)
    for port in offset_infos:
        for clock, info_list in offset_infos[port].items():
            time: list[float] = []
            signal: list[float] = []
            for info in info_list:
                if len(time) > 0:
                    # Each offset is a point, so the previous offset is extended to a
                    # line before adding the next.
                    # Subtract a small number from the time so that interpolation (in
                    # sum_waveforms) looks correct visually.
                    time.append(info.time - 0.01 / sampling_rate)
                    signal.append(signal[-1])
                time.append(info.time)
                signal.append(info.op_info["offset_path_I"] + 1j * info.op_info["offset_path_Q"])
            time.append(schedule.duration)
            signal.append(signal[-1])

            # Filter in time: Keep one point before and one point after the limit (if
            # possible).
            start_idx = next(i for i, v in enumerate(time) if v > x_min)
            if start_idx > 0:
                start_idx -= 1
            try:
                stop_idx = next(i for i, v in enumerate(time) if v > x_max) + 1
            except StopIteration:
                stop_idx = len(time)
            time = time[start_idx:stop_idx]
            signal = signal[start_idx:stop_idx]

            time = np.array(time)
            signal = np.array(signal)
            if modulation != "off":
                new_time = np.linspace(
                    time[0], time[-1], round((time[-1] - time[0]) * sampling_rate) + 1
                )
                signal = interpolated_complex_waveform(t=new_time, samples=signal, t_samples=time)
                time = new_time
            if modulation == "clock":
                signal = modulate_waveform(time, signal, schedule.resources[clock]["freq"])

            elif modulation == "if":
                signal = modulate_waveform(time, signal, modulation_if)

            signal = np.real_if_close(signal)
            sampled_pulses[port].append(
                SampledPulse(
                    time=np.array(time),
                    signal=np.array(signal),
                    label=f"VoltageOffset, clock {clock}",
                )
            )
    return sampled_pulses


def get_sampled_pulses(
    schedule: Schedule | CompiledSchedule,
    pulse_infos: dict[str, list[ScheduledInfo]],
    x_min: float,
    x_max: float,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    sampling_rate: float = 1e9,
    sampled_pulses: dict[str, list[SampledPulse]] | None = None,
) -> dict[str, list[SampledPulse]]:
    """
    Generate :class:`.SampledPulse` objects from pulse_info dicts.

    This function creates a SampledPulse for each pulse_info dict. The pulse_info must
    contain a valid ``"wf_func"``.

    Parameters
    ----------
    schedule :
        The schedule to render.
    pulse_infos :
        A dictionary from ports to lists of pulse_info dictionaries.
    x_min :
        The left limit of the x-axis of the intended plot.
    x_max :
        The right limit of the x-axis of the intended plot.
    modulation :
        Determines if modulation is included in the visualization.
    modulation_if :
        Modulation frequency used when modulation is set to "if".
    sampling_rate :
        The time resolution used to sample the schedule in Hz.
    sampled_pulses :
        An already existing dictionary (same type as the return value). If provided,
        this dictionary will be extended with the SampledPulse objects created in this
        function.

    Returns
    -------
    dict[str, list[SampledPulse]] :
        SampledPulse objects grouped by port.

    """
    if sampled_pulses is None:
        sampled_pulses = defaultdict(list)
    for port, info_list in pulse_infos.items():
        for info in info_list:
            t0 = info.time
            t1 = t0 + info.op_info["duration"]

            if t1 < x_min or t0 > x_max:
                continue
            t0 = max(x_min, t0)
            t1 = min(x_max, t1)

            t = np.arange(t0, t1 - 0.5 / sampling_rate, 1 / sampling_rate)
            if len(t) == 0:
                continue

            if (
                info.op_info["wf_func"]
                == "quantify_scheduler.waveforms.interpolated_complex_waveform"
            ):
                info.op_info["t_samples"] = (
                    np.asarray(info.op_info["t_samples"]) - info.op_info["t_samples"][0]
                )

            # Add the final datapoint for nicer plots
            t = np.append(t, t[-1] + 0.99 / sampling_rate)

            waveform = exec_waveform_function(
                wf_func=info.op_info["wf_func"],
                t=t - t[0],
                pulse_info=info.op_info,
            )

            # Add 0 amplitude points before and after the pulse such that interpolation
            # in sum_waveforms looks correct visually.
            t = np.concatenate(
                (
                    [t[0] - 0.01 / sampling_rate],
                    t,
                    [t[-1] + 0.01 / sampling_rate],
                )
            )
            waveform = np.concatenate(([0], waveform, [0]))

            if modulation == "clock":
                waveform = modulate_waveform(
                    t, waveform, schedule.resources[info.op_info["clock"]]["freq"]
                )

            if modulation == "if":
                waveform = modulate_waveform(t, waveform, modulation_if)

            waveform = np.real_if_close(waveform)
            label = f"{info.op_name}, clock {info.op_info['clock']}"
            sampled_pulses[port].append(SampledPulse(time=t, signal=waveform, label=label))
    return sampled_pulses


def get_sampled_acquisitions(
    acq_infos: dict[str, list[ScheduledInfo]],
) -> dict[str, list[SampledAcquisition]]:
    """
    Generate :class:`.SampledAcquisition` objects from acquisition_info dicts.

    Parameters
    ----------
    acq_infos :
        A dictionary from ports to lists of acquisition_info dictionaries.

    Returns
    -------
    dict[str, list[SampledAcquisition]] :
        SampledAcquisition objects grouped by port.

    """
    sampled_acqs: dict[str, list[SampledAcquisition]] = defaultdict(list)
    for port, info_list in acq_infos.items():
        for info in info_list:
            sampled_acqs[port].append(
                SampledAcquisition(
                    t0=info.time, duration=info.op_info["duration"], label=info.op_name
                )
            )
    return sampled_acqs


def merge_pulses_and_offsets(operations: list[SampledPulse]) -> SampledPulse:
    """
    Combine multiple ``SampledPulse`` objects by interpolating the ``signal`` at the
    ``time`` points used by all pulses together, and then summing the result.
    Interpolation outside of a ``SampledPulse.time`` array results in 0 for that pulse.
    """
    result_time = np.sort(np.concatenate([op.time for op in operations]))
    if len(operations) > 3:
        # If the label would become too large, opt for this short form:
        label = f"{len(operations)} operations"
    else:
        label = "+\n".join(op.label for op in operations)

    return SampledPulse(
        time=result_time,
        signal=sum(
            np.interp(result_time, op.time, op.signal, left=0.0, right=0.0) for op in operations
        ),  # type: ignore
        label=label,
    )


def _extract_schedule_infos(
    operation: Operation | ScheduleBase,
    port_list: list[str],
    time_offset: float,
    offset_infos: dict[str, dict[str, list[ScheduledInfo]]],
    pulse_infos: dict[str, list[ScheduledInfo]],
    acq_infos: dict[str, list[ScheduledInfo]],
) -> None:
    if isinstance(operation, ScheduleBase):
        for schedulable in operation.schedulables.values():
            inner_operation = operation.operations[schedulable["operation_id"]]
            abs_time = schedulable["abs_time"]
            _extract_schedule_infos(
                inner_operation,
                port_list,
                time_offset + abs_time,
                offset_infos,
                pulse_infos,
                acq_infos,
            )
    elif isinstance(operation, ConditionalOperation):
        _extract_schedule_infos(
            operation.body, port_list, time_offset, offset_infos, pulse_infos, acq_infos
        )
    elif isinstance(operation, LoopOperation):
        for i in range(operation.repetitions):
            _extract_schedule_infos(
                operation.body,
                port_list,
                time_offset + i * operation.body.duration,
                offset_infos,
                pulse_infos,
                acq_infos,
            )
    else:
        for acq_info in operation["acquisition_info"]:
            if port_list is not None and acq_info["port"] not in port_list:
                continue
            acq_info_cpy = ScheduledInfo(
                op_info=acq_info,
                time=time_offset + acq_info["t0"],
                op_name=operation["name"],
            )
            acq_infos[acq_info["port"]].append(acq_info_cpy)

        for pulse_info in operation["pulse_info"]:
            if port_list is not None and pulse_info["port"] not in port_list:
                continue
            if pulse_info.get("wf_func") is not None:
                pulse_info_cpy = ScheduledInfo(
                    op_info=pulse_info,
                    time=time_offset + pulse_info["t0"],
                    op_name=operation["name"],
                )
                pulse_infos[pulse_info["port"]].append(pulse_info_cpy)
            elif "offset_path_I" in pulse_info:
                pulse_info_cpy = ScheduledInfo(
                    op_info=pulse_info,
                    time=time_offset + pulse_info["t0"],
                    op_name=operation["name"],
                )
                offset_infos[pulse_info["port"]][pulse_info["clock"]].append(pulse_info_cpy)


def sample_schedule(
    schedule: Schedule | CompiledSchedule,
    port_list: list[str] | None = None,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    sampling_rate: float = 1e9,
    x_range: tuple[float, float] = (-np.inf, np.inf),
    combine_waveforms_on_same_port: bool = False,
) -> dict[str, tuple[list[SampledPulse], list[SampledAcquisition]]]:
    """
    Generate :class:`.SampledPulse` and :class:`.SampledAcquisition` objects grouped by
    port.

    This function generates SampledPulse objects for all pulses and voltage offsets
    defined in the Schedule, and SampledAcquisition for all acquisitions defined in the
    Schedule.

    Parameters
    ----------
    schedule :
        The schedule to render.
    port_list :
        A list of ports to show. if set to ``None`` (default), it will use all ports in
        the Schedule.
    modulation :
        Determines if modulation is included in the visualization.
    modulation_if :
        Modulation frequency used when modulation is set to "if".
    sampling_rate :
        The time resolution used to sample the schedule in Hz.
    x_range :
        The range of the x-axis that is plotted, given as a tuple (left limit, right
        limit). This can be used to reduce memory usage when plotting a small section of
        a long pulse sequence. By default (-np.inf, np.inf).
    combine_waveforms_on_same_port :
        By default False. If True, combines all waveforms on the same port into one
        single waveform. The resulting waveform is the sum of all waveforms on that
        port (small inaccuracies may occur due to floating point approximation). If
        False, the waveforms are shown individually.

    Returns
    -------
    dict[str, tuple[list[SampledPulse], list[SampledAcquisition]]] :
        SampledPulse and SampledAcquisition objects grouped by port.

    """
    offset_infos: dict[str, dict[str, list[ScheduledInfo]]] = defaultdict(lambda: defaultdict(list))
    pulse_infos: dict[str, list[ScheduledInfo]] = defaultdict(list)
    acq_infos: dict[str, list[ScheduledInfo]] = defaultdict(list)

    _extract_schedule_infos(
        schedule,
        port_list,
        0,
        offset_infos,
        pulse_infos,
        acq_infos,
    )

    x_min, x_max = x_range

    sampled_pulses = get_sampled_pulses_from_voltage_offsets(
        schedule=schedule,
        offset_infos=offset_infos,
        x_min=x_min,
        x_max=x_max,
        modulation=modulation,
        modulation_if=modulation_if,
    )

    sampled_pulses = get_sampled_pulses(
        schedule=schedule,
        pulse_infos=pulse_infos,
        x_min=x_min,
        x_max=x_max,
        modulation=modulation,
        modulation_if=modulation_if,
        sampling_rate=sampling_rate,
        sampled_pulses=sampled_pulses,
    )

    if combine_waveforms_on_same_port:
        for port, pulses in sampled_pulses.copy().items():
            sampled_pulses[port] = [merge_pulses_and_offsets(pulses)]

    sampled_acqs = get_sampled_acquisitions(acq_infos)

    sampled_all: dict[str, tuple[list[SampledPulse], list[SampledAcquisition]]] = {}
    for port in chain(sampled_pulses, sampled_acqs):
        sampled_all[port] = (sampled_pulses[port], sampled_acqs[port])
    return sampled_all


def pulse_diagram_plotly(
    sampled_pulses_and_acqs: dict[str, tuple[list[SampledPulse], list[SampledAcquisition]]],
    title: str = "Pulse diagram",
    fig_ch_height: float = 300,
    fig_width: float = 1000,
) -> go.Figure:
    """
    Produce a plotly visualization of the pulses used in the schedule.

    Parameters
    ----------
    sampled_pulses_and_acqs :
        SampledPulse and SampledAcquisition objects grouped by port.
    title :
        Plot title.
    fig_ch_height :
        Height for each channel subplot in px.
    fig_width :
        Width for the figure in px.

    Returns
    -------
    :class:`plotly.graph_objects.Figure` :
        the plot

    """
    n_rows = len(sampled_pulses_and_acqs)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.update_layout(
        height=fig_ch_height * n_rows,
        width=fig_width,
        title=title,
        showlegend=False,
    )

    colors = px.colors.qualitative.Plotly
    col_idx = 0

    legendgroup = -1
    for i, (port, (pulses, acqs)) in enumerate(sampled_pulses_and_acqs.items()):
        row = i + 1
        for pulse in pulses:
            legendgroup += 1
            fig.add_trace(
                go.Scatter(
                    x=pulse.time,
                    y=pulse.signal.real,
                    mode="lines",
                    name=pulse.label,
                    legendgroup=legendgroup,
                    showlegend=True,
                    line_color=colors[col_idx],
                    fill="tozeroy",
                    hoverinfo="x+y+name",
                    hoverlabel={"namelength": -1},
                ),
                row=row,
                col=1,
            )
            col_idx = (col_idx + 1) % len(colors)

            if np.iscomplexobj(pulse.signal):
                fig.add_trace(
                    go.Scatter(
                        x=pulse.time,
                        y=pulse.signal.imag,
                        mode="lines",
                        name=f"{pulse.label} (imag)",
                        legendgroup=legendgroup,
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

        for acq in acqs:
            yref = f"y{row} domain" if row != 1 else "y domain"
            fig.add_trace(
                go.Scatter(
                    x=[acq.t0, acq.t0 + acq.duration],
                    y=[0, 0],
                    name=acq.label,
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
                x0=acq.t0,
                y0=0,
                x1=acq.t0 + acq.duration,
                y1=1,
                name=acq.label,
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
                title=port,
                autorange=True,
            )

    fig.update_xaxes(
        row=n_rows,
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


def deduplicate_legend_handles_labels(ax: mpl.axes.Axes) -> None:
    """
    Remove duplicate legend entries.

    See also: https://stackoverflow.com/a/13589144
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def plot_single_subplot_mpl(
    sampled_schedule: dict[str, list[SampledPulse]],
    ax: mpl.axes.Axes | None = None,
    title: str = "Pulse diagram",
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot all pulses for all ports in the schedule in the same subplot.

    Pulses in the same port have the same color and legend entry, and each port
    has its own legend entry.

    Parameters
    ----------
    sampled_schedule :
        Dictionary that maps each used port to the sampled pulses played on that port.
    ax :
        A pre-existing Axes object to plot the pulses in. If ``None`` (default), this
        object is created within the function.
    title :
        Plot title.

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

    ax.set_title(title)
    return fig, ax


def plot_multiple_subplots_mpl(
    sampled_schedule: dict[str, list[SampledPulse]],
    title: str = "Pulse diagram",
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
    """
    Plot pulses in a different subplot for each port in the sampled schedule.

    For each subplot, each different type of pulse gets its own color and legend
    entry.

    Parameters
    ----------
    sampled_schedule :
        Dictionary that maps each used port to the sampled pulses played on that port.
    title :
        Plot title.

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
        color: dict[str, str] = defaultdict(lambda: f"C{len(color)}")

        for pulse in data:
            axs[i].plot(
                pulse.time,
                pulse.signal.real,
                color=color[pulse.label],
                label=pulse.label,
            )
            axs[i].fill_between(pulse.time, pulse.signal.real, color=color[pulse.label], alpha=0.2)

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

    axs[0].set_title(title)
    return fig, axs


def pulse_diagram_matplotlib(
    sampled_pulses_and_acqs: dict[str, tuple[list[SampledPulse], list[SampledAcquisition]]],
    multiple_subplots: bool = False,
    ax: mpl.axes.Axes | None = None,
    title: str = "Pulse diagram",
) -> tuple[mpl.figure.Figure, mpl.axes.Axes | list[mpl.axes.Axes]]:
    """
    Plots a schedule using matplotlib.

    Parameters
    ----------
    sampled_pulses_and_acqs :
        SampledPulse and SampledAcquisition objects grouped by port.
    multiple_subplots :
        Plot the pulses for each port on a different subplot if True, else plot
        everything in one subplot. By default False. When using just one
        subplot, the pulses are colored according to the port on which they
        play. For multiple subplots, each pulse has its own
        color and legend entry.
    ax :
        Axis onto which to plot. If ``None`` (default), this is created within the
        function. By default None.
    title :
        Plot title.

    Returns
    -------
    fig :
        A matplotlib :class:`matplotlib.figure.Figure` containing the subplot(s).

    ax :
        The Axes object belonging to the Figure, or an array of Axes if
        ``multiple_subplots=True``.

    """
    pulses = {port: pulses for port, (pulses, _) in sampled_pulses_and_acqs.items()}

    if len(pulses) == 0:
        raise RuntimeError(
            "Attempting to sample schedule, "
            "but the schedule does not contain any `pulse_info`. "
            "Please verify that the schedule has been populated and "
            "device compilation has been performed."
        )

    if not multiple_subplots or len(pulses) == 1:
        return plot_single_subplot_mpl(sampled_schedule=pulses, ax=ax, title=title)
    return plot_multiple_subplots_mpl(sampled_schedule=pulses, title=title)


def get_window_operations(
    schedule: Schedule,
) -> list[tuple[float, float, Operation]]:
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
    ax: mpl.axes.Axes | None = None,
    time_scale_factor: float = 1,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
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

    cmap = mpl.colormaps.get_cmap("jet")

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
    schedule: Schedule, ax: mpl.axes.Axes | None = None, **kwargs
) -> list[Any]:
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
        if isinstance(operation, Acquisition):
            t0 = schedulable["abs_time"] + operation.data["acquisition_info"][0]["t0"]
            t1 = t0 + operation.duration
            handle = ax.axvspan(t0, t1, **kwargs)
            handles_list.append(handle)
    return handles_list
