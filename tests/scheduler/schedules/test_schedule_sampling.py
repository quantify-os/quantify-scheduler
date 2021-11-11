# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import BasebandClockResource
from quantify_scheduler.visualization.pulse_diagram import sample_schedule


def test_sample_schedule() -> None:
    schedule = Schedule("test")
    r = SquarePulse(amp=0.2, duration=4e-9, port="SDP")
    schedule.add(r)
    rm = SquarePulse(amp=-0.2, duration=6e-9, port="T")
    schedule.add(rm, ref_pt="start")
    r = SquarePulse(amp=0.3, duration=6e-9, port="SDP")
    schedule.add(r)
    schedule.add(r)
    determine_absolute_timing(schedule=schedule)

    timestamps, waveforms = sample_schedule(schedule, sampling_rate=0.5e9)

    np.testing.assert_array_almost_equal(
        timestamps,
        np.array(
            [
                0.0e00,
                2.0e-09,
                4.0e-09,
                6.0e-09,
                8.0e-09,
                1.0e-08,
                1.2e-08,
                1.4e-08,
                1.6e-08,
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        waveforms["SDP"], np.array([0.2, 0.2, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    )
    np.testing.assert_array_almost_equal(
        waveforms["T"], np.array([-0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_sample_modulated_waveform() -> None:
    schedule = Schedule("test")
    clock0 = BasebandClockResource(
        name="clock0",
        data={
            "name": "clock0",
            "type": "BasebandClockResource",
            "freq": 0.15e9,
            "phase": 0,
        },
    )
    schedule.add_resource(clock0)

    square_pulse_op = SquarePulse(amp=0.2, duration=3e-9, port="SDP", clock="clock0")
    schedule.add(square_pulse_op)
    square_pulse_op = SquarePulse(amp=0.2, duration=3e-9, port="T")
    schedule.add(square_pulse_op, ref_pt="start")
    determine_absolute_timing(schedule=schedule)

    timestamps, waveforms = sample_schedule(
        schedule,
        sampling_rate=1e9,
        modulation="clock",
    )

    assert waveforms["SDP"].dtype.kind == "c"
    np.testing.assert_array_almost_equal(
        waveforms["SDP"],
        np.array([0.2 + 0.0j, 0.117557 - 0.161803j, -0.061803 - 0.190211j]),
    )
    np.testing.assert_array_almost_equal(waveforms["T"], np.array([0.2, 0.2, 0.2]))


test_sample_modulated_waveform()


def test_sample_custom_port_list() -> None:
    schedule = Schedule("test")
    r = SquarePulse(amp=0.2, duration=4e-9, port="SDP")
    schedule.add(r)
    determine_absolute_timing(schedule=schedule)

    timestamps, waveforms = sample_schedule(
        schedule, sampling_rate=0.5e9, port_list=["SDP"]
    )
    assert list(waveforms.keys()) == ["SDP"]


def test_sample_empty_schedule() -> None:
    schedule = Schedule("test")

    with pytest.raises(RuntimeError):
        timestamps, waveforms = sample_schedule(schedule, sampling_rate=1e9)
