"""Tests for pulse factory functions."""

from functools import partial

import numpy as np
import pytest

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.operations import (
    long_ramp_pulse,
    long_square_pulse,
    staircase_pulse,
)
from quantify_scheduler.operations.pulse_factories import (
    rxy_drag_pulse,
    rxy_gauss_pulse,
    rxy_hermite_pulse,
)
from quantify_scheduler.operations.pulse_library import (
    ReferenceMagnitude,
    SquarePulse,
    VoltageOffset,
)


def test_rxy_drag_pulse():
    """Test a long_ramp_pulse that is composed of one part."""
    pulse = rxy_drag_pulse(
        amp180=0.6,
        motzoi=0.2,
        theta=200,
        phi=19,
        port="q0:res",
        duration=1e-7,
        clock="q0.ro",
    )
    assert pulse.data["pulse_info"] == [
        {
            "wf_func": "quantify_scheduler.waveforms.drag",
            "G_amp": 0.6 * 200 / 180,
            "D_amp": 0.2,
            "reference_magnitude": None,
            "duration": 1e-7,
            "phase": 19,
            "nr_sigma": 4,
            "sigma": None,
            "clock": "q0.ro",
            "port": "q0:res",
            "t0": 0,
        }
    ]


def test_rxy_gauss_pulse():
    """Test a long_ramp_pulse that is composed of one part."""
    pulse = rxy_gauss_pulse(
        amp180=0.8, theta=180, phi=10, port="q0:res", duration=1e-7, clock="q0.ro"
    )
    assert pulse.data["pulse_info"] == [
        {
            "wf_func": "quantify_scheduler.waveforms.drag",
            "G_amp": 0.8,
            "D_amp": 0,
            "reference_magnitude": None,
            "duration": 1e-7,
            "phase": 10,
            "nr_sigma": 4,
            "sigma": None,
            "clock": "q0.ro",
            "port": "q0:res",
            "t0": 0,
        }
    ]


def test_rxy_hermite_pulse():
    """Test the rxy_hermite_pulse"""
    pulse = rxy_hermite_pulse(
        amp180=0.8,
        theta=180,
        phi=10,
        port="q0:res",
        duration=100e-9,
        clock="q0.ro",
        skewness=0.0,
    )
    assert pulse.data["pulse_info"] == [
        {
            "wf_func": "quantify_scheduler.waveforms.skewed_hermite",
            "duration": 100e-9,
            "amplitude": 0.8,
            "skewness": 0.0,
            "phase": 10,
            "port": "q0:res",
            "clock": "q0.ro",
            "reference_magnitude": None,
            "t0": 0.0,
        }
    ]


def test_short_long_ramp_pulse():
    """Test a long_ramp_pulse that is composed of one part."""
    pulse = long_ramp_pulse(amp=0.8, duration=1e-7, port="q0:res")
    assert pulse.data["pulse_info"] == [
        {
            "wf_func": "quantify_scheduler.waveforms.ramp",
            "amp": 0.8,
            "reference_magnitude": None,
            "duration": pytest.approx(1e-07),
            "offset": 0,
            "t0": 0.0,
            "clock": "cl0.baseband",
            "port": "q0:res",
        }
    ]


def test_long_long_ramp_pulse():
    """Test a long_ramp_pulse that is composed of multiple parts."""
    pulse = long_ramp_pulse(amp=0.5, duration=2.5e-6, offset=-0.2, port="q0:res")

    ramp_parts = []
    offsets = []
    for pulse_info in pulse.data["pulse_info"]:
        if "offset_path_I" in pulse_info:
            offsets.append(pulse_info)
        else:
            ramp_parts.append(pulse_info)

    assert offsets[0]["offset_path_I"] == pytest.approx(-0.2)
    assert sum(pul_inf["amp"] for pul_inf in ramp_parts) == pytest.approx(0.5)
    assert sum(pul_inf["duration"] for pul_inf in ramp_parts) == pytest.approx(2.5e-6)
    assert ramp_parts[-1]["offset"] + ramp_parts[-1]["amp"] == pytest.approx(0.3)
    assert offsets[-1]["offset_path_I"] == pytest.approx(0.0)


def test_long_square_pulse():
    """Test a long square pulse."""
    port = "q0:res"
    clock = "q0.ro"
    pulse = long_square_pulse(amp=0.8, duration=1e-3, port=port, clock=clock)
    assert len(pulse["pulse_info"]) == 3
    assert (
        pulse["pulse_info"][0]
        == VoltageOffset(offset_path_I=0.8, offset_path_Q=0.0, port=port, clock=clock)[
            "pulse_info"
        ][0]
    )
    assert (
        pulse["pulse_info"][1]
        == VoltageOffset(
            offset_path_I=0.0,
            offset_path_Q=0.0,
            port=port,
            clock=clock,
            t0=1e-3 - 4e-9,
        )["pulse_info"][0]
    )
    assert (
        pulse["pulse_info"][2]
        == SquarePulse(amp=0.8, duration=4e-9, port=port, clock=clock, t0=1e-3 - 4e-9)[
            "pulse_info"
        ][0]
    )


def test_long_square_pulse_that_is_too_short():
    """A square pulse less than `constants.MIN_TIME_BETWEEN_OPERATIONS` should error"""
    with pytest.raises(ValueError, match=f"{constants.MIN_TIME_BETWEEN_OPERATIONS}"):
        long_square_pulse(amp=0.8, duration=1e-9, port="", clock="")


def test_long_square_pulse_that_is_exactly_long_enough():
    """A square pulse less than `constants.MIN_TIME_BETWEEN_OPERATIONS` should error"""
    pulse = long_square_pulse(amp=0.8, duration=4e-9, port="", clock="")
    assert len(pulse["pulse_info"]) == 3
    p0, p1, p2 = pulse["pulse_info"]
    assert p0["t0"] == 0
    assert p1["t0"] == 0
    assert p2["t0"] == 0
    assert p2["duration"] == 4e-9


def test_long_square_pulse_with_t0():
    port = "q0:res"
    clock = "q0.ro"
    pulse = long_square_pulse(amp=0.8, duration=1e-3, port=port, clock=clock, t0=100e-9)
    assert len(pulse["pulse_info"]) == 3
    p0, p1, p2 = pulse["pulse_info"]
    assert p0["t0"] == 100e-9
    assert p1["t0"] == 1e-3 - 4e-9 + 100e-9
    assert p2["t0"] == 1e-3 - 4e-9 + 100e-9
    assert p2["duration"] == 4e-9


def test_staircase():
    """Test a staircase pulse."""
    pulse = staircase_pulse(
        start_amp=0.1,
        final_amp=0.9,
        num_steps=20,
        duration=1e-3,
        port="q0:res",
        clock="q0.ro",
    )
    amps = np.linspace(0.1, 0.9, 20)
    t0s = np.linspace(0, 0.95e-3, 20)
    assert pulse["pulse_info"][-1]["amp"] == 0.9
    assert pulse["pulse_info"][-1]["duration"] == 4e-9
    assert pulse["pulse_info"][-1]["t0"] == pytest.approx(1e-3 - 4e-9)
    for amp, t0, pulse_inf in zip(amps, t0s, pulse["pulse_info"][0:-3]):
        assert pulse_inf["offset_path_I"] == pytest.approx(amp)
        assert pulse_inf["t0"] == pytest.approx(t0)
    assert pulse["pulse_info"][-3]["offset_path_I"] == 0.9
    assert pulse["pulse_info"][-2]["offset_path_I"] == 0.0


def test_staircase_raises_not_multiple_of_grid_time():
    """Test that an error is raised if step duration is not a multiple of grid time."""
    with pytest.raises(ValueError) as err:
        _ = staircase_pulse(
            start_amp=0.1,
            final_amp=0.9,
            num_steps=20,
            duration=20 * 9e-9,
            min_operation_time_ns=4,
            port="q0:res",
            clock="q0.ro",
        )
    # Exact phrasing is not important, but should be about staircase
    assert "step" in str(err.value) and "staircase" in str(err.value)


def test_staircase_raises_step_duration_too_short():
    """Test that an error is raised if step duration is shorter than the grid time."""
    with pytest.raises(ValueError) as err:
        _ = staircase_pulse(
            start_amp=0.1,
            final_amp=0.9,
            num_steps=20,
            duration=20 * 4e-9,
            min_operation_time_ns=8,
            port="q0:res",
            clock="q0.ro",
        )
    # Exact phrasing is not important, but should be about staircase
    assert "step" in str(err.value) and "staircase" in str(err.value)


@pytest.mark.parametrize(
    "pulse",
    [
        partial(long_square_pulse, amp=0.8, duration=1e-3, port="q0:res", clock="q0.ro"),
        partial(
            staircase_pulse,
            start_amp=0.1,
            final_amp=0.9,
            num_steps=20,
            duration=1e-3,
            port="q0:res",
            clock="q0.ro",
        ),
        partial(long_ramp_pulse, amp=0.8, duration=1e-7, port="q0:res"),
    ],
)
def test_voltage_offset_operations_reference_magnitude(pulse):
    reference_magnitude = ReferenceMagnitude(20, "dBm")

    pulse = pulse(reference_magnitude=reference_magnitude)

    assert pulse["pulse_info"][0]["reference_magnitude"] == reference_magnitude
