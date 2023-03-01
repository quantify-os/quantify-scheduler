# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=missing-function-docstring
import matplotlib
import plotly

from quantify_scheduler import Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.operations.pulse_library import SquarePulse


def test_schedule_plotting() -> None:
    sched = Schedule("test")
    sched.add(SquarePulse(amp=0.2, duration=4e-6, port="SDP"))
    determine_absolute_timing(schedule=sched)

    circuit_fig_mpl, _ = sched.plot_circuit_diagram()
    pulse_fig_mpl, _ = sched.plot_pulse_diagram()
    pulse_fig_plt = sched.plot_pulse_diagram(plot_backend="plotly")

    assert isinstance(circuit_fig_mpl, matplotlib.figure.Figure)
    assert isinstance(pulse_fig_mpl, matplotlib.figure.Figure)
    assert isinstance(pulse_fig_plt, plotly.graph_objects.Figure)
