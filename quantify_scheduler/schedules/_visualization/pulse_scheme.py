# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing functions for drawing pulse schemes and circuit diagrams
using matplotlib.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from quantify_core.utilities import deprecated
from quantify_scheduler.schedules._visualization import constants

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def new_pulse_fig(
    figsize: tuple[int, int] | None = None, ax: Axes | None = None
) -> tuple[Figure | None, Axes]:
    """
    Open a new figure and configure it to plot pulse schemes.

    Parameters
    ----------
    figsize :
        Size of the figure.
    ax
        Axis to use for plotting. If ``None``, then creates a new one.

    Returns
    -------
    :
        Tuple of figure handle and axis handle.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, frameon=False)
    else:
        fig = None
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if fig is not None:
        fig.patch.set_alpha(0)

    return fig, ax


def new_pulse_subplot(fig: Figure, *args, **kwargs) -> Axes:
    """
    Add a new subplot configured for plotting pulse schemes to a figure.

    All `*args` and `**kwargs` are passed to fig.add_subplot.

    Parameters
    ----------
    fig :
        Figure to add the subplot to.
    *args
        Positional arguments to pass to fig.add_subplot.
    **kwargs
        Keyword arguments to pass to fig.add_subplot.

    Returns
    -------
    :

    """
    ax = fig.add_subplot(*args, **kwargs)
    ax.axis("off")
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    ax.axhline(0, color="0.75")

    return ax


def mw_pulse(
    ax: Axes,
    pos: float,
    y_offs: float = 0.0,
    width: float = 1.5,
    amp: float = 1,
    label: str | None = None,
    phase: float = 0,
    label_height: float = 1.3,
    color: str = constants.COLOR_ORANGE,
    modulation: str = "normal",
    **plot_kws,
) -> float:
    """
    Draw a microwave pulse: Gaussian envelope with modulation.

    Parameters
    ----------
    ax :
        Axis to plot on.
    pos :
        Position of the pulse.
    y_offs :
        Vertical offset of the pulse.
    width :
        Width of the pulse.
    amp :
        Amplitude
    label :
        Label to add to the pulse.
    label_height :
        Height of the label.
    color :
        Color of the pulse.

    modulation :
        Modulation

    Returns
    -------
    :

    """
    x = np.linspace(pos, pos + width, 100)
    env_pos = amp * np.exp(-((x - (pos + width / 2)) ** 2) / (width / 4) ** 2)
    env_neg = -amp * np.exp(-((x - (pos + width / 2)) ** 2) / (width / 4) ** 2)

    if modulation == "normal":
        mod = env_pos * np.sin(2 * np.pi * 3 / width * x + phase)
    elif modulation == "high":
        mod = env_pos * np.sin(5 * np.pi * 3 / width * x + phase)
    else:
        raise ValueError()

    ax.plot(x, env_pos + y_offs, "--", color=color, **plot_kws)
    ax.plot(x, env_neg + y_offs, "--", color=color, **plot_kws)
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


def flux_pulse(
    ax: Axes,
    pos: float,
    y_offs: float = 0.0,
    width: float = 2.5,
    s: float = 0.1,
    amp: float = 1.5,
    label: str | None = None,
    label_height: float = 1.7,
    color: str = constants.COLOR_ORANGE,
    **plot_kws,
) -> float:
    """
    Draw a smooth flux pulse, where the rising and falling edges are given by
    Fermi-Dirac functions.

    Parameters
    ----------
    ax :
        Axis to plot on.

    pos :
        Position of the pulse.

    y_offs :
        Vertical offset of the pulse.

    width :
        Width of the pulse.

    s :
        smoothness of edge
    amp :
        Amplitude

    label :
        Label to add to the pulse.

    label_height :
        Height of the label.

    color :
        Color of the pulse.

    Returns
    -------
    :

    """
    x = np.linspace(pos, pos + width, 100)
    y = amp / (
        (np.exp(-(x - (pos + 5.5 * s)) / s) + 1) * (np.exp((x - (pos + width - 5.5 * s)) / s) + 1)
    )

    ax.fill_between(x, y + y_offs, y_offs, color=color, alpha=0.3)
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


def ram_Z_pulse(  # noqa N802 uppercase Z is allowed here
    ax: Axes,
    pos: float,
    y_offs: float = 0.0,
    width: float = 2.5,
    s: float = 0.1,
    amp: float = 1.5,
    sep: float = 1.5,
    color: str = constants.COLOR_ORANGE,
) -> float:
    """
    Draw a Ram-Z flux pulse, i.e. only part of the pulse is shaded, to indicate
    cutting off the pulse at some time.

    Parameters
    ----------
    ax :
        Axis to plot on.
    pos :
        Position of the pulse.

    y_offs :
        Vertical offset of the pulse.

    width :
        Width of the pulse.

    s :
        smoothness of edge
    amp :
        Amplitude

    sep :
        Separation between pulses.

    color :
        Color of the pulse.

    Returns
    -------
    :

    """
    x_left = np.linspace(pos, pos + sep, 100)
    x_right = np.linspace(pos + sep, pos + width, 100)
    x_full = np.concatenate((x_left, x_right))
    y = amp / (
        (np.exp(-(x_full - (pos + 5.5 * s)) / s) + 1)
        * (np.exp((x_full - (pos + width - 5.5 * s)) / s) + 1)
    )
    y_left = y[: len(x_left)]

    ax.fill_between(x_left, y_left + y_offs, y_offs, alpha=0.3, color=color, linewidth=0.0)
    ax.plot(x_full, y + y_offs, color=color)

    return pos + width


def interval(
    ax: Axes,
    start: float,
    stop: float,
    y_offs: float = 0.0,
    height: float = 1.5,
    label: str | None = None,
    label_height: str | None = None,
    vlines: bool = True,
    color: str = "k",
    arrowstyle: str = "<|-|>",
    **plot_kws,
) -> None:
    """
    Draw an arrow to indicate an interval.

    Parameters
    ----------
    ax :
        Axis to plot on.
    pos :
        Position of the pulse.
    y_offs :
        Vertical offset of the pulse.
    width :
        Width of the pulse.
    s :
        smoothness of edge
    amp :
        Amplitude
    sep :
        Separation between pulses.
    color :
        Color of the pulse.
    arrow_style :


    Returns
    -------
    :

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
        ax.plot([start, start], [0 + y_offs, height + y_offs], "--", color=color, **plot_kws)
        ax.plot([stop, stop], [0 + y_offs, height + y_offs], "--", color=color, **plot_kws)

    if label is not None:
        ax.text(
            (start + stop) / 2, label_height + y_offs, label, color=color, ha="center"
        ).set_clip_on(True)


def meter(
    ax: Axes,
    x0: float,
    y0: float,
    y_offs: float = 0.0,
    width: float = 1.1,
    height: float = 0.8,
    color: str = "black",
    framewidth: float = 0.0,
    fillcolor: str | None = None,
) -> None:
    """
    Draws a measurement meter on the specified position.

    Parameters
    ----------
    ax :

    x0 :

    y0 :

    y_offs :

    width :

    height :

    color :

    framewidth:

    fillcolor :


    Returns
    -------
    :

    """
    fill = fillcolor is not None
    p1 = matplotlib.patches.Rectangle(
        (x0 - width / 2, y0 - height / 2 + y_offs),
        width,
        height,
        facecolor=fillcolor,
        edgecolor=color,
        fill=fill,
        zorder=5,
        linewidth=framewidth,
    )
    ax.add_patch(p1)
    p0 = matplotlib.patches.Wedge(
        (x0, y0 - height / constants.METER_WEDGE_HEIGHT_SCALING + y_offs),
        constants.METER_WEDGE_RADIUS,
        theta1=constants.METER_WEDGE_ANGLE,
        theta2=180 - constants.METER_WEDGE_ANGLE,
        color=color,
        lw=1.5,
        width=0.01,
        zorder=5,
    )
    ax.add_patch(p0)
    arrow_len = height / 2.0
    ax.arrow(
        x0,
        y0 - height / constants.METER_ARROW_HEIGHT_SCALING + y_offs,
        dx=arrow_len * np.cos(np.deg2rad(constants.METER_ARROW_ANGLE)),
        dy=arrow_len * np.sin(np.deg2rad(constants.METER_ARROW_ANGLE)),
        width=0.025,
        color=color,
        zorder=5,
    )


def box_text(
    ax: Axes,
    x0: float,
    y0: float,
    text: str = "",
    width: float = 1.1,
    height: float = 0.8,
    color: str = "black",
    fillcolor: str | None = None,
    textcolor: str = "black",
    fontsize: int | None = None,
) -> None:
    """
    Draws a box filled with text at the specified position.

    Parameters
    ----------
    ax :

    x0 :

    y0 :

    text :

    width :

    height :

    color :

    fillcolor :

    textcolor :

    fontsize :


    Returns
    -------
    :

    """
    fill = fillcolor is not None
    p1 = matplotlib.patches.Rectangle(
        (x0 - width / 2, y0 - height / 2),
        width,
        height,
        facecolor=fillcolor,
        edgecolor=color,
        fill=fill,
        linewidth=0,
        zorder=5,
    )
    ax.add_patch(p1)

    ax.text(
        x0,
        y0,
        text,
        ha="center",
        va="center",
        zorder=6,
        size=fontsize,
        color=textcolor,
    ).set_clip_on(True)


@deprecated("0.25.0", mw_pulse)
def mwPulse(*args, **kwargs):  # noqa ANN202, N802 deprecated

    return mw_pulse(*args, **kwargs)


@deprecated("0.25.0", flux_pulse)
def fluxPulse(*args, **kwargs):  # noqa ANN202, N802 deprecated

    return flux_pulse(*args, **kwargs)


@deprecated("0.25.0", ram_Z_pulse)
def ramZPulse(*args, **kwargs):  # noqa ANN202, N802 deprecated

    return ram_Z_pulse(*args, **kwargs)
