"""Tests for pulse factory functions."""
import numpy as np
import pytest

from quantify_scheduler.operations.pulse_factories import (
    long_ramp_pulse,
    long_square_pulse,
    staircase_pulse,
)


def test_short_long_ramp_pulse():
    """Test a long_ramp_pulse that is composed of one part."""
    pulse = long_ramp_pulse(amp=0.8, duration=1e-7, port="q0:res")
    assert pulse.data["pulse_info"] == [
        {
            "wf_func": "quantify_scheduler.waveforms.ramp",
            "amp": 0.8,
            "duration": pytest.approx(1e-07),
            "offset": 0,
            "t0": 0,
            "clock": "cl0.baseband",
            "port": "q0:res",
        }
    ]


def test_long_long_ramp_pulse():
    """Test a long_ramp_pulse that is composed of multiple parts."""
    pulse = long_ramp_pulse(amp=0.8, duration=3e-5, offset=0.2, port="q0:res")

    ramp_parts = []
    offsets = []
    for pulse_info in pulse.data["pulse_info"]:
        if "offset_path_0" in pulse_info:
            offsets.append(pulse_info)
        else:
            ramp_parts.append(pulse_info)

    assert offsets[0]["offset_path_0"] == pytest.approx(0.2)
    assert sum(pul_inf["amp"] for pul_inf in ramp_parts) == pytest.approx(0.8)
    assert sum(pul_inf["duration"] for pul_inf in ramp_parts) == pytest.approx(3e-5)
    assert offsets[-2]["offset_path_0"] + ramp_parts[-1]["amp"] == pytest.approx(1.0)
    assert offsets[-1]["offset_path_0"] == pytest.approx(0.0)


def test_long_square_pulse():
    """Test a long square pulse."""
    pulse = long_square_pulse(amp=0.8, duration=1e-3, port="q0:res", clock="q0.ro")
    assert pulse["pulse_info"][0]["offset_path_0"] == 0.8
    assert pulse["pulse_info"][0]["duration"] == 1e-3
    assert pulse["pulse_info"][1]["offset_path_0"] == 0.0
    assert pulse["pulse_info"][1]["t0"] == 1e-3


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
    for amp, pulse_inf in zip(amps, pulse["pulse_info"][:-1]):
        assert pulse_inf["offset_path_0"] == pytest.approx(amp)
        assert pulse_inf["duration"] == pytest.approx(5e-5)
    assert pulse["pulse_info"][-1]["offset_path_0"] == 0.0


def test_staircase_raises():
    """Test that an error is raised if step duration is not a multiple of grid time."""
    with pytest.raises(ValueError) as err:
        _ = staircase_pulse(
            start_amp=0.1,
            final_amp=0.9,
            num_steps=19,
            duration=1e-4,
            port="q0:res",
            clock="q0.ro",
        )
    # Exact phrasing is not important, but should be about staircase
    assert "step" in str(err.value) and "staircase" in str(err.value)
