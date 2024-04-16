# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

import matplotlib
import plotly
import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.backends.qblox.operations.stitched_pulse import (
    StitchedPulseBuilder,
)
from quantify_scheduler.resources import ClockResource


def test_schedule_plotting() -> None:
    sched = Schedule("test")
    sched.add(SquarePulse(amp=0.2, duration=4e-6, port="SDP"))
    sched = _determine_absolute_timing(schedule=sched)

    circuit_fig_mpl, _ = sched.plot_circuit_diagram()
    pulse_fig_mpl, _ = sched.plot_pulse_diagram()
    pulse_fig_plt = sched.plot_pulse_diagram(plot_backend="plotly")

    assert isinstance(circuit_fig_mpl, matplotlib.figure.Figure)
    assert isinstance(pulse_fig_mpl, matplotlib.figure.Figure)
    assert isinstance(pulse_fig_plt, plotly.graph_objects.Figure)


@pytest.mark.parametrize("plot_backend", ["plotly", "mpl"])
def test_plot_stitched_pulse(plot_backend, mock_setup_basic_nv_qblox_hardware):
    sched = Schedule("Test schedule")
    port = "qe0:optical_readout"
    clock = "qe0.ge0"
    sched.add_resource(ClockResource(name=clock, freq=470.4e12))

    sched.add(SquarePulse(amp=0.1, duration=1e-6, port=port, clock=clock))

    builder = StitchedPulseBuilder(port=port, clock=clock)
    stitched_pulse = (
        builder.add_pulse(SquarePulse(amp=0.16, duration=5e-6, port=port, clock=clock))
        .add_voltage_offset(0.4, 0.0, duration=5e-6)
        .build()
    )
    sched.add(stitched_pulse)

    sched.add(SquarePulse(amp=0.2, duration=1e-6, port=port, clock=clock))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    if plot_backend == "plotly":
        fig = compiled_schedule.plot_pulse_diagram(plot_backend=plot_backend)
        assert isinstance(fig, plotly.graph_objects.Figure)
    else:
        fig, ax = compiled_schedule.plot_pulse_diagram(plot_backend=plot_backend)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
