# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for drawing pulse diagrams"""
from __future__ import annotations

import inspect
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quantify_core.visualization.SI_utilities import set_xlabel, set_ylabel

from quantify_core.utilities import deprecated
import quantify_scheduler.operations.pulse_library as pl
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.operations.acquisition_library import AcquisitionOperation
from quantify_scheduler.waveforms import modulate_wave

if TYPE_CHECKING:
    from quantify_scheduler import CompiledSchedule, Operation, Schedule

logger = logging.getLogger(__name__)


def _populate_port_mapping(schedule, portmap: Dict[str, int], ports_length) -> None:
    """
    Dynamically add up to 8 ports to the port_map dictionary.
    """
    offset_idx: int = 0

    for schedulable in schedule.schedulables.values():
        operation = schedule.operations[schedulable["operation_repr"]]
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
            f"id={schedulable['operation_repr']} operation_data={operation_data}"
        )
        return False

    if "acq_index" not in operation_data:  # This will be skipped for acquisitions
        if operation_data["wf_func"] is None:
            logger.warning(
                "Unable to sample pulse for pulse_info due to missing 'wf_func' for "
                f"operation name={operation['name']} "
                f"id={schedulable['operation_repr']} operation_data={operation_data}"
            )
            return False
    return True


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
@deprecated(
    "0.14.0",
    "To plot a pulse diagram, please call `ScheduleBase.plot_pulse_diagram()` from"
    "`quantify_scheduler.schedules.schedule.py` instead.",
)
def pulse_diagram_plotly(
    schedule: Union[Schedule, CompiledSchedule],
    port_list: Optional[List[str]] = None,
    fig_ch_height: float = 300,
    fig_width: float = 1000,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    sampling_rate: float = 1e9,
) -> go.Figure:
    """
    Produce a plotly visualization of the pulses used in the schedule.

    .. warning:
        This function is deprecated and will be removed after `quantify-scheduler>=0.14`.
        To plot a circuit diagram, please call :func:`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram()`
        from :class:`~quantify_scheduler.schedules.schedule.ScheduleBase` instead. Make sure to include the argument `plot_backend="plotly"`.

    Parameters
    ------------
    schedule :
        The schedule to render.
    port_list :
        A list of ports to show. if set to `None` will use the first
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
    :class:`!plotly.graph_objects.Figure` :
        the plot
    """

    port_map: Dict[str, int] = {}
    ports_length: int = 8
    auto_map: bool = port_list is None

    if auto_map is False:
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
        operation = schedule.operations[schedulable["operation_repr"]]

        for pulse_info in operation["pulse_info"]:
            if not validate_operation_data(
                pulse_info, port_map, schedulable, operation
            ):
                continue

            # port to map the waveform to
            port: str = pulse_info["port"]

            # function to generate waveform
            wf_func: Callable = import_python_object_from_string(pulse_info["wf_func"])

            # iterate through the colors in the color map
            col_idx = (col_idx + 1) % len(colors)

            # times at which to evaluate waveform
            t0 = schedulable["abs_time"] + pulse_info["t0"]
            t = np.arange(t0, t0 + pulse_info["duration"], 1 / sampling_rate)
            # select the arguments for the waveform function
            # that are present in pulse info
            par_map = inspect.signature(wf_func).parameters
            wf_kwargs = {}
            for kwargs in par_map.keys():
                if kwargs in pulse_info.keys():
                    wf_kwargs[kwargs] = pulse_info[kwargs]

            # Calculate the numerical waveform using the wf_func
            waveform = wf_func(t=t, **wf_kwargs)

            # optionally adds some modulation
            if modulation == "clock":
                # apply modulation to the waveforms
                waveform = modulate_wave(
                    t, waveform, schedule.resources[pulse_info["clock"]]["freq"]
                )

            if modulation == "if":
                # apply modulation to the waveforms
                waveform = modulate_wave(t, waveform, modulation_if)

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
                    hoverinfo="x+y+name",
                    hoverlabel={"namelength": -1},
                ),
                row=row,
                col=1,
            )

            if waveform.dtype.kind == "c":
                # Only plot if the array is a complex numpy dtype
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=waveform.imag,
                        mode="lines",
                        name=f"Im[{label}], clock: {pulse_info['clock']}",
                        legendgroup=pulse_idx,
                        showlegend=True,
                        line_color="darkgrey",
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
                range=[-1.1, 1.1],
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


# pylint: disable=too-many-branches
def sample_schedule(
    schedule: Schedule,
    port_list: Optional[List[str]] = None,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    sampling_rate: float = 1e9,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Sample a schedule at discrete points in time.

    Parameters
    ----------
    schedule :
        The schedule to render.
    port_list :
        A list of ports to show. if set to `None` will use the first
        8 ports it encounters in the sequence.
    modulation :
        Determines if modulation is included in the visualization.
    modulation_if :
        Modulation frequency used when modulation is set to "if".
    sampling_rate :
        The time resolution used to sample the schedule in Hz.

    Returns
    -------
    timestamps
        Sample times.
    waveforms
        Dictionary with the data samples for each port.
    """

    port_map: Dict[str, int] = {}
    ports_length: int = 8
    auto_map: bool = port_list is None

    if auto_map is False:
        ports_length = len(port_list)
        port_map = dict(zip(port_list, range(len(port_list))))
    else:
        _populate_port_mapping(schedule, port_map, ports_length)
        ports_length = len(port_map)

    time_window: list = None
    for pls_idx, schedulable in enumerate(schedule.schedulables.values()):
        operation = schedule.operations[schedulable["operation_repr"]]

        for pulse_info in operation["pulse_info"]:
            if not validate_operation_data(
                pulse_info, port_map, schedulable, operation
            ):
                logging.info(f"Operation {operation} is not valid for plotting.")

            # times at which to evaluate waveform
            t0 = schedulable["abs_time"] + pulse_info["t0"]
            if time_window is None:
                time_window = [t0, t0 + pulse_info["duration"]]
            else:
                time_window = [
                    min(t0, time_window[0]),
                    max(t0 + pulse_info["duration"], time_window[1]),
                ]

    logger.debug(f"time_window {time_window}, port_map {port_map}")

    if time_window is None:
        raise RuntimeError(
            f"Attempting to sample schedule {schedule.name}, "
            "but the schedule does not contain any `pulse_info`. "
            "Please verify that the schedule has been populated and "
            "device compilation has been performed."
        )

    timestamps = np.arange(time_window[0], time_window[1], 1 / sampling_rate)
    waveforms = {key: np.zeros_like(timestamps) for key in port_map}

    for pls_idx, schedulable in enumerate(schedule.schedulables.values()):
        operation = schedule.operations[schedulable["operation_repr"]]
        logger.debug(f"{pls_idx}: {operation}")

        for pulse_info in operation["pulse_info"]:
            if not validate_operation_data(
                pulse_info, port_map, schedulable, operation
            ):
                continue

            # port to map the waveform too
            port: str = pulse_info["port"]

            # function to generate waveform
            wf_func: Callable = import_python_object_from_string(pulse_info["wf_func"])

            # times at which to evaluate waveform
            t0 = schedulable["abs_time"] + pulse_info["t0"]
            t1 = t0 + pulse_info["duration"]

            time_indices = np.where(np.logical_and(timestamps >= t0, timestamps < t1))
            t = timestamps[time_indices]
            logging.debug(f"t0 {t0} t1 {t1} indices {time_indices} t {t}")
            if len(t) == 0:
                continue

            par_map = inspect.signature(wf_func).parameters
            wf_kwargs = {}
            for kwargs in par_map.keys():
                if kwargs in pulse_info.keys():
                    wf_kwargs[kwargs] = pulse_info[kwargs]

            # Calculate the numerical waveform using the wf_func
            waveform = wf_func(t=t, **wf_kwargs)

            # optionally adds some modulation
            if modulation == "clock":
                # apply modulation to the waveforms
                waveform = modulate_wave(
                    t, waveform, schedule.resources[pulse_info["clock"]]["freq"]
                )
                waveform = np.real_if_close(waveform)

            if modulation == "if":
                # apply modulation to the waveforms
                waveform = modulate_wave(t, waveform, modulation_if)

            if np.iscomplexobj(waveform):
                waveforms[port] = waveforms[port].astype(complex)
            waveforms[port][time_indices] += waveform

    return timestamps, waveforms


@deprecated(
    "0.14.0",
    "To plot a pulse diagram, please call `ScheduleBase.plot_pulse_diagram()` from"
    "`quantify_scheduler.schedules.schedule.py` instead.",
)
def pulse_diagram_matplotlib(
    schedule: Union[Schedule, CompiledSchedule],
    port_list: Optional[List[str]] = None,
    sampling_rate: float = 1e9,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0.0,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plots a schedule using matplotlib.

    .. warning:
        This function is deprecated and will be removed after `quantify-scheduler>=0.14`.
        To plot a circuit diagram, please call :func:`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram()`
        from :class:`~quantify_scheduler.schedules.schedule.ScheduleBase` instead.

    Parameters
    ----------
    schedule:
        The schedule to plot.
    port_list :
        A list of ports to show. if set to `None` will use the first
        8 ports it encounters in the sequence.
    modulation :
        Determines if modulation is included in the visualization.
    modulation_if :
        Modulation frequency used when modulation is set to "if".
    sampling_rate :
        The time resolution used to sample the schedule in Hz.
    ax:
        Axis onto which to plot.

    Returns
    -------
    fig
        The matplotlib figure.
    ax
        The matplotlib ax.
    """
    times, pulses = sample_schedule(
        schedule,
        sampling_rate=sampling_rate,
        port_list=port_list,
        modulation=modulation,
        modulation_if=modulation_if,
    )
    if ax is None:
        _, ax = plt.subplots()
    for gate, data in pulses.items():
        ax.plot(times, data.real, label=gate)
    set_xlabel(label="Time", unit="s", axis=ax)
    # N.B. we currently use unity gain in the hardware backends so strictly
    # speaking this is not the amplitude on the device, but the amplitude on the output.
    set_ylabel(label="Amplitude", unit="V", axis=ax)
    ax.legend()

    return ax.get_figure(), ax


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
        operation = schedule.operations[schedulable["operation_repr"]]
        if isinstance(operation, pl.WindowOperation):
            for pulse_info in operation["pulse_info"]:
                t0 = schedulable["abs_time"] + pulse_info["t0"]
                t1 = t0 + pulse_info["duration"]

            window_operations.append((t0, t1, operation))
    return window_operations


def plot_window_operations(
    schedule: Schedule,
    ax: Optional[matplotlib.axes.Axes] = None,
    time_scale_factor: float = 1,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
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

    cmap = matplotlib.cm.get_cmap("jet")

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
    schedule: Schedule, ax: Optional[matplotlib.axes.Axes] = None, **kwargs
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
        operation = schedule.operations[schedulable["operation_repr"]]
        if isinstance(operation, AcquisitionOperation):
            t0 = schedulable["abs_time"] + operation.data["acquisition_info"][0]["t0"]
            t1 = t0 + operation.duration
            handle = ax.axvspan(t0, t1, **kwargs)
            handles_list.append(handle)
    return handles_list
