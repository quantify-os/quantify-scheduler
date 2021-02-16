# -----------------------------------------------------------------------------
# Description:    Module containing functions for drawing pulse schemes and circuit diagrams using matplotlib.
# Repository:     https://gitlab.com/qblox/packages/software/quantify/
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, List, Dict, Optional
import logging
import inspect
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from quantify.scheduler.waveforms import modulate_wave
from quantify.utilities.general import import_func_from_string

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from quantify.scheduler.types import Schedule


def new_pulse_fig(
    figsize: Optional[Tuple[int, int]] = None
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Open a new figure and configure it to plot pulse schemes.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.patch.set_alpha(0)
    ax.axhline(0, color="0.75")

    return fig, ax


def new_pulse_subplot(fig: Figure, *args, **kwargs) -> Axes:
    """
    Add a new subplot configured for plotting pulse schemes to a figure.

    All `*args` and `**kwargs` are passed to fig.add_subplot.
    """
    ax = fig.add_subplot(*args, **kwargs)
    ax.axis("off")
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    ax.axhline(0, color="0.75")

    return ax


def mwPulse(
    ax: Axes,
    pos: float,
    y_offs: float = 0.0,
    width: float = 1.5,
    amp: float = 1,
    label: Optional[str] = None,
    phase=0,
    label_height: float = 1.3,
    color: str = "C0",
    modulation: str = "normal",
    **plot_kws,
) -> float:
    """
    Draw a microwave pulse: Gaussian envelope with modulation.
    """
    x = np.linspace(pos, pos + width, 100)
    envPos = amp * np.exp(-((x - (pos + width / 2)) ** 2) / (width / 4) ** 2)
    envNeg = -amp * np.exp(-((x - (pos + width / 2)) ** 2) / (width / 4) ** 2)

    if modulation == "normal":
        mod = envPos * np.sin(2 * np.pi * 3 / width * x + phase)
    elif modulation == "high":
        mod = envPos * np.sin(5 * np.pi * 3 / width * x + phase)
    else:
        raise ValueError()

    ax.plot(x, envPos + y_offs, "--", color=color, **plot_kws)
    ax.plot(x, envNeg + y_offs, "--", color=color, **plot_kws)
    ax.plot(x, mod + y_offs, "-", color=color, **plot_kws)

    if label is not None:
        ax.text(
            pos + width / 2,
            label_height,
            label,
            horizontalalignment="right",
            color=color,
        ).set_clip_on(True)

    return pos + width


def fluxPulse(
    ax: Axes,
    pos: float,
    y_offs: float = 0.0,
    width: float = 2.5,
    s: float = 0.1,
    amp: float = 1.5,
    label: Optional[str] = None,
    label_height: float = 1.7,
    color: str = "C1",
    **plot_kws,
) -> float:
    """
    Draw a smooth flux pulse, where the rising and falling edges are given by
    Fermi-Dirac functions.
    s: smoothness of edge
    """
    x = np.linspace(pos, pos + width, 100)
    y = amp / (
        (np.exp(-(x - (pos + 5.5 * s)) / s) + 1)
        * (np.exp((x - (pos + width - 5.5 * s)) / s) + 1)
    )

    ax.fill_between(x, y + y_offs, color=color, alpha=0.3)
    ax.plot(x, y + y_offs, color=color, **plot_kws)

    if label is not None:
        ax.text(
            pos + width / 2,
            label_height,
            label,
            horizontalalignment="center",
            color=color,
        ).set_clip_on(True)

    return pos + width


def ramZPulse(
    ax: Axes,
    pos: float,
    y_offs: float = 0.0,
    width: float = 2.5,
    s: float = 0.1,
    amp: float = 1.5,
    sep: float = 1.5,
    color: str = "C1",
) -> float:
    """
    Draw a Ram-Z flux pulse, i.e. only part of the pulse is shaded, to indicate
    cutting off the pulse at some time.
    """
    xLeft = np.linspace(pos, pos + sep, 100)
    xRight = np.linspace(pos + sep, pos + width, 100)
    xFull = np.concatenate((xLeft, xRight))
    y = amp / (
        (np.exp(-(xFull - (pos + 5.5 * s)) / s) + 1)
        * (np.exp((xFull - (pos + width - 5.5 * s)) / s) + 1)
    )
    yLeft = y[: len(xLeft)]

    ax.fill_between(xLeft, yLeft + y_offs, alpha=0.3, color=color, linewidth=0.0)
    ax.plot(xFull, y + y_offs, color=color)

    return pos + width


def interval(
    ax: Axes,
    start: float,
    stop: float,
    y_offs: float = 0.0,
    height: float = 1.5,
    label: Optional[str] = None,
    label_height: Optional[str] = None,
    vlines: bool = True,
    color: str = "k",
    arrowstyle: str = "<|-|>",
    **plot_kws,
) -> None:
    """
    Draw an arrow to indicate an interval.
    """
    if label_height is None:
        label_height = height + 0.2

    arrow = matplotlib.patches.FancyArrowPatch(
        posA=(start, height + y_offs),
        posB=(stop, height + y_offs),
        arrowstyle=arrowstyle,
        color=color,
        mutation_scale=7,
        **plot_kws,
    )
    ax.add_patch(arrow)

    if vlines:
        ax.plot(
            [start, start], [0 + y_offs, height + y_offs], "--", color=color, **plot_kws
        )
        ax.plot(
            [stop, stop], [0 + y_offs, height + y_offs], "--", color=color, **plot_kws
        )

    if label is not None:
        ax.text(
            (start + stop) / 2, label_height + y_offs, label, color=color, ha="center"
        ).set_clip_on(True)


def meter(
    ax: Axes,
    x0: float,
    y0: float,
    y_offs: float = 0.0,
    w: float = 1.1,
    h: float = 0.8,
    color: str = "black",
    fillcolor: Optional[str] = None,
) -> None:
    """
    Draws a measurement meter on the specified position.
    """
    if fillcolor is None:
        fill = False
    else:
        fill = True
    p1 = matplotlib.patches.Rectangle(
        (x0 - w / 2, y0 - h / 2 + y_offs),
        w,
        h,
        facecolor=fillcolor,
        edgecolor=color,
        fill=fill,
        zorder=5,
    )
    ax.add_patch(p1)
    p0 = matplotlib.patches.Wedge(
        (x0, y0 - h / 1.75 + y_offs),
        0.4,
        theta1=40,
        theta2=180 - 40,
        color=color,
        lw=2,
        width=0.01,
        zorder=5,
    )
    ax.add_patch(p0)
    r0 = h / 2.2
    ax.arrow(
        x0,
        y0 - h / 5 + y_offs,
        dx=r0 * np.cos(np.deg2rad(70)),
        dy=r0 * np.sin(np.deg2rad(70)),
        width=0.03,
        color=color,
        zorder=5,
    )


def box_text(
    ax: Axes,
    x0: float,
    y0: float,
    text: str = "",
    w: float = 1.1,
    h: float = 0.8,
    color: str = "black",
    fillcolor: Optional[str] = None,
    textcolor: str = "black",
    fontsize: Optional[int] = None,
) -> None:
    """
    Draws a box filled with text at the specified position.
    """
    if fillcolor is None:
        fill = False
    else:
        fill = True
    p1 = matplotlib.patches.Rectangle(
        (x0 - w / 2, y0 - h / 2),
        w,
        h,
        facecolor=fillcolor,
        edgecolor=color,
        fill=fill,
        zorder=5,
    )
    ax.add_patch(p1)

    ax.text(
        x0, y0, text, ha="center", va="center", zorder=6, size=fontsize, color=textcolor
    ).set_clip_on(True)


def pulse_diagram_plotly(
    schedule: Schedule,
    port_list: Optional[List[str]] = None,
    fig_ch_height: float = 150,
    fig_width: float = 1000,
    modulation: str = "off",
    modulation_if: float = 0,
    sampling_rate: int = 1e9,
) -> Figure:
    """
    Produce a plotly visualization of the pulses used in the schedule.

    Parameters
    ------------
    schedule : :class:`~quantify.scheduler.types.Schedule`
        The schedule to render.
    port_list : list
        A list of ports to show. if set to `None` will use the first
        8 ports it encounters in the sequence.
    fig_ch_height: float
        Height for each channel subplot in px.
    fig_width: float
        Width for the figure in px.
    modulation: str
        Determines if modulation is included in the visualization. Options are "off", "if", "clock".
    modulation_if: float
        Modulation frequency used when modulation is set to "if".
    sampling_rate : float
        The time resolution used in the visualization.
    Returns
    -------
    :class:`plotly.graph_objects.Figure`
        the plot
    """

    port_map: Dict[str, int] = dict()
    ports_length: int = 8
    auto_map: bool = True if port_list is None else False

    def _populate_port_mapping(map: Dict[str, int]) -> None:
        """
        Dynammically add up to 8 ports to the port_map dictionary.
        """
        offset_idx: int = 0

        for t_constr in schedule.timing_constraints:
            operation = schedule.operations[t_constr["operation_hash"]]
            for pulse_info in operation["pulse_info"]:
                if offset_idx == ports_length:
                    return

                port = pulse_info["port"]
                if port is None:
                    continue

                if port not in port_map:
                    port_map[port] = offset_idx
                    offset_idx += 1

    if auto_map is False:
        ports_length = len(port_list)
        port_map = dict(zip(port_list, range(len(port_list))))
    else:
        _populate_port_mapping(port_map)
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

    for pls_idx, t_constr in enumerate(schedule.timing_constraints):
        operation = schedule.operations[t_constr["operation_hash"]]

        for pulse_info in operation["pulse_info"]:
            if pulse_info["port"] not in port_map:
                # Do not draw pulses for this port
                continue

            if pulse_info["port"] is None:
                logger.warning(
                    f"Unable to draw pulse for pulse_info due to missing 'port' for \
                        operation name={operation['name']} \
                        id={t_constr['operation_hash']} pulse_info={pulse_info}"
                )
                continue

            if pulse_info["wf_func"] is None:
                logger.warning(
                    f"Unable to draw pulse for pulse_info due to missing 'wf_func' for \
                        operation name={operation['name']} \
                        id={t_constr['operation_hash']} pulse_info={pulse_info}"
                )
                continue

            # port to map the waveform too
            port: Optional[str] = pulse_info["port"]

            # function to generate waveform
            wf_func: Optional[str] = import_func_from_string(pulse_info["wf_func"])

            # iterate through the colors in the color map
            col_idx = (col_idx + 1) % len(colors)

            # times at which to evaluate waveform
            t0 = t_constr["abs_time"] + pulse_info["t0"]
            t = np.arange(t0, t0 + pulse_info["duration"], 1 / sampling_rate)

            # select the arguments for the waveform function that are present in pulse info
            par_map = inspect.signature(wf_func).parameters
            wf_kwargs = {}
            for kw in par_map.keys():
                if kw in pulse_info.keys():
                    wf_kwargs[kw] = pulse_info[kw]

            # Calculate the numerical waveform using the wf_func
            wf = wf_func(t=t, **wf_kwargs)

            # optionally adds some modulation
            if modulation == "clock":
                # apply modulation to the waveforms
                wf = modulate_wave(
                    t, wf, schedule.resources[pulse_info["clock"]]["freq"]
                )

            if modulation == "if":
                # apply modulation to the waveforms
                wf = modulate_wave(t, wf, modulation_if)

            row: int = port_map[port] + 1
            # FIXME properly deal with complex waveforms.
            for i in range(2):
                showlegend = i == 0
                label = operation["name"]
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=wf.imag,
                        mode="lines",
                        name=label,
                        legendgroup=pls_idx,
                        showlegend=showlegend,
                        line_color="lightgrey",
                    ),
                    row=row,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=wf.real,
                        mode="lines",
                        name=label,
                        legendgroup=pls_idx,
                        showlegend=showlegend,
                        line_color=colors[col_idx],
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
                range=[-1.1, 1.1],
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
        rangeslider_visible=True,
    )

    return fig
