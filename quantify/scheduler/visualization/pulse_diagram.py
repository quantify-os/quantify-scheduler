# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Functions for drawing pulse diagrams"""
from __future__ import annotations
import inspect
import logging
from typing import List, Dict, Optional
from typing_extensions import Literal

import numpy as np

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from quantify.scheduler.types import Schedule
from quantify.scheduler.waveforms import modulate_wave

# pylint: disable=no-name-in-module
from quantify.utilities.general import import_func_from_string

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def pulse_diagram_plotly(
    schedule: Schedule,
    port_list: Optional[List[str]] = None,
    fig_ch_height: float = 300,
    fig_width: float = 1000,
    modulation: Literal["off", "if", "clock"] = "off",
    modulation_if: float = 0,
    sampling_rate: int = 1e9,
) -> go.Figure:
    """
    Produce a plotly visualization of the pulses used in the schedule.

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
        The time resolution used in the visualization.

    Returns
    -------
    :class:`!plotly.graph_objects.Figure` :
        the plot
    """

    port_map: Dict[str, int] = dict()
    ports_length: int = 8
    auto_map: bool = port_list is None

    def _populate_port_mapping(portmap: Dict[str, int]) -> None:
        """
        Dynamically add up to 8 ports to the port_map dictionary.
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

                if port not in portmap:
                    portmap[port] = offset_idx
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
                    f"Unable to draw pulse for pulse_info due to missing 'port' for "
                    f"operation name={operation['name']} "
                    f"id={t_constr['operation_hash']} pulse_info={pulse_info}"
                )
                continue

            if pulse_info["wf_func"] is None:
                logger.warning(
                    f"Unable to draw pulse for pulse_info due to missing 'wf_func' for "
                    f"operation name={operation['name']} "
                    f"id={t_constr['operation_hash']} pulse_info={pulse_info}"
                )
                continue

            # port to map the waveform too
            port: str = pulse_info["port"]

            # function to generate waveform
            wf_func: str = import_func_from_string(pulse_info["wf_func"])

            # iterate through the colors in the color map
            col_idx = (col_idx + 1) % len(colors)

            # times at which to evaluate waveform
            t0 = t_constr["abs_time"] + pulse_info["t0"]
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
                    name=label,
                    legendgroup=pls_idx,
                    showlegend=True,
                    line_color=colors[col_idx],
                    hoverinfo="x+y+name",
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
                        name=f"Im[{label}]",
                        legendgroup=pls_idx,
                        showlegend=True,
                        line_color="darkgrey",
                        hoverinfo="x+y+name",
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
