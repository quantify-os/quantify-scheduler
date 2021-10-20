# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-function-docstring

import pytest

import matplotlib.pyplot as plt
from quantify_scheduler.types import Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.pulse_library import SquarePulse, WindowOperation
from quantify_scheduler.visualization.pulse_diagram import (
    pulse_diagram_matplotlib,
    get_window_operations,
    plot_window_operations,
)


def test_pulse_diagram_matplotlib() -> None:

    schedule = Schedule("test")
    schedule.add(SquarePulse(amp=0.2, duration=4e-6, port="SDP"))
    schedule.add(SquarePulse(amp=0.3, duration=6e-6, port="SDP"))
    schedule.add(
        WindowOperation(window_name="second pulse", duration=6e-6), ref_pt="start"
    )
    schedule.add(SquarePulse(amp=0.25, duration=6e-6, port="SDP"))
    determine_absolute_timing(schedule=schedule)

    plt.figure(1)
    plt.clf()
    pulse_diagram_matplotlib(schedule, sampling_rate=20e6)

    window_operations = get_window_operations(schedule)
    plot_window_operations(schedule)

    assert len(window_operations) == 1
    window = window_operations[0]
    assert window[0] == pytest.approx(4e-6)
    assert window[1] == pytest.approx(10e-6)
    assert isinstance(window[2], WindowOperation)

    plt.close(1)
