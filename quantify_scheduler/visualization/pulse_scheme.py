# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Module containing functions for drawing pulse schemes and circuit diagrams
using matplotlib.
"""
# pylint: disable=too-many-arguments
from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# For type hints, import modules to avoid circular dependencies
from matplotlib.figure import Figure

from quantify_scheduler.visualization import constants, pulse_diagram

logger = logging.getLogger(__name__)


def new_pulse_fig(
    figsize: Optional[Tuple[int, int]] = None, ax: Optional[Axes] = None
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Open a new figure and configure it to plot pulse schemes.

    Parameters
    ----------
    figsize
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
    ax.axhline(0, color="0.75")

    return fig, ax


def new_pulse_subplot(fig: Figure, *args, **kwargs) -> Axes:
    """
    Add a new subplot configured for plotting pulse schemes to a figure.

    All `*args` and `**kwargs` are passed to fig.add_subplot.

    Parameters
    ----------
    fig :

    Returns
    -------
    :

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
    color: str = constants.COLOR_ORANGE,
    modulation: str = "normal",
    **plot_kws,
) -> float:
    """
    Draw a microwave pulse: Gaussian envelope with modulation.

    Parameters
    ----------

    ax :

    pos :

    y_offs :

    width :

    amp :

    label :

    label_height :

    color :

    modulation :

    Returns
    -------
    :

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
    color: str = constants.COLOR_ORANGE,
    **plot_kws,
) -> float:
    """
    Draw a smooth flux pulse, where the rising and falling edges are given by
    Fermi-Dirac functions.

    Parameters
    ----------
    ax :

    pos :

    y_offs :

    width :

    s :
        smoothness of edge
    amp :

    label :

    label_height :

    color :

    Returns
    -------
    :

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
    color: str = constants.COLOR_ORANGE,
) -> float:
    """
    Draw a Ram-Z flux pulse, i.e. only part of the pulse is shaded, to indicate
    cutting off the pulse at some time.

    Parameters
    ----------
    ax :

    pos :

    y_offs :

    width :

    s :

    amp :

    sep :

    color :

    Returns
    -------
    :

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

    Parameters
    ----------
    ax :

    start :

    stop :

    y_offs :

    height :

    label :

    label_height :

    vlines :

    color :

    arrowstyle :

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
    width: float = 1.1,
    height: float = 0.8,
    color: str = "black",
    framewidth: float = 0.0,
    fillcolor: Optional[str] = None,
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
    if fillcolor is None:
        fill = False
    else:
        fill = True
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
    fillcolor: Optional[str] = None,
    textcolor: str = "black",
    fontsize: Optional[int] = None,
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
    if fillcolor is None:
        fill = False
    else:
        fill = True
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


def pulse_diagram_plotly(*args, **kwargs):
    """
    Deprecated import use
    :func:`~quantify_scheduler.visualization.pulse_diagram.pulse_diagram_plotly`
    """
    warnings.warn(
        "`pulse_diagram_plotly` will be removed from this module in quantify 1.0.0.\n"
        "Import as follows instead:\n"
        "",
        ImportWarning,
    )

    return pulse_diagram.pulse_diagram_plotly(*args, **kwargs)
