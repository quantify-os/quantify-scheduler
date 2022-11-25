from typing import Dict
import pytest

import numpy as np

from quantify_scheduler.operations.measurement_factories import (
    optical_measurement,
    optical_measurement_multiple_pulses,
)

# pylint: disable=redefined-outer-name


@pytest.fixture
def typical_kwargs_optical_measurement():
    """Default keyword arguments for optical_measurement function.

    Missing are the optional arguments.
    """
    return {
        "pulse_amplitude": 0.23,
        "pulse_duration": 100e-6,
        "pulse_port": "pulse_port_name",
        "pulse_clock": "pulse_clock_name",
        "acq_duration": 200e-6,
        "acq_delay": 0,
        "acq_port": "acq_port_name",
        "acq_clock": "acq_clock_name",
        "acq_channel": 1,
        "acq_index": 0,
        "acq_protocol": None,
        "bin_mode": None,
        "acq_protocol_default": "TriggerCount",
        "pulse_type": "SquarePulse",
    }


def assert_pulse_equal(pulse_info: Dict, optical_meas_kwargs: Dict):
    """Assert that info of pulse is equal to arguments used as input to
    optical_meas_kwargs."""
    assert isinstance(pulse_info, dict)
    assert pulse_info["amp"] == optical_meas_kwargs["pulse_amplitude"]
    assert pulse_info["duration"] == optical_meas_kwargs["pulse_duration"]
    assert pulse_info["phase"] == 0
    assert pulse_info["port"] == optical_meas_kwargs["pulse_port"]
    assert pulse_info["clock"] == optical_meas_kwargs["pulse_clock"]
    if optical_meas_kwargs["acq_delay"] >= 0:
        assert pulse_info["t0"] == 0
    else:
        assert pulse_info["t0"] == -optical_meas_kwargs["acq_delay"]


def assert_acquisition_equal(acq_info: Dict, optical_meas_kwargs: Dict):
    """Assert that info of acquisition is equal to arguments used as input to
    optical_meas_kwargs."""
    assert acq_info["duration"] == optical_meas_kwargs["acq_duration"]
    assert acq_info["port"] == optical_meas_kwargs["acq_port"]
    assert acq_info["clock"] == optical_meas_kwargs["acq_clock"]
    assert acq_info["acq_channel"] == optical_meas_kwargs["acq_channel"]
    assert acq_info["acq_index"] == optical_meas_kwargs["acq_index"]
    if optical_meas_kwargs["acq_delay"] >= 0:
        assert acq_info["t0"] == optical_meas_kwargs["acq_delay"]
    else:
        assert acq_info["t0"] == 0


def test_optical_measurement(typical_kwargs_optical_measurement):
    # Arrange
    kwargs = typical_kwargs_optical_measurement

    # Act
    operation = optical_measurement(**kwargs)

    # Assert
    assert operation.valid_acquisition
    assert operation.valid_pulse
    assert len(operation.data["pulse_info"]) == 1
    assert len(operation.data["acquisition_info"]) == 1
    square_pulse_info = operation.data["pulse_info"][0]
    assert_pulse_equal(square_pulse_info, kwargs)


@pytest.mark.parametrize("acq_delay", [-50e-6, 0, 52e-5])
def test_optical_measurement_trigger_count(
    typical_kwargs_optical_measurement, acq_delay
):
    """optical_measurement factory works well with TriggerCount protocol"""
    # Arrange
    kwargs = typical_kwargs_optical_measurement
    kwargs["acq_delay"] = acq_delay
    kwargs["acq_protocol"] = "TriggerCount"

    # Act
    operation = optical_measurement(**kwargs)

    # Assert
    assert isinstance(operation.data["acquisition_info"][0], dict)
    acq_info = operation.data["acquisition_info"][0]
    assert acq_info["protocol"] == "trigger_count"
    assert acq_info["acq_return_type"] == int
    assert_acquisition_equal(acq_info, kwargs)


@pytest.mark.parametrize("acq_delay", [-50e-6, 0, 52e-5])
def test_optical_measurement_trace(typical_kwargs_optical_measurement, acq_delay):
    """optical_measurement factory works well with Trace protocol"""
    # Arrange
    kwargs = typical_kwargs_optical_measurement
    kwargs["acq_delay"] = acq_delay
    kwargs["acq_protocol"] = "Trace"

    # Act
    operation = optical_measurement(**kwargs)

    # Assert
    assert isinstance(operation.data["acquisition_info"][0], dict)
    acq_info = operation.data["acquisition_info"][0]
    assert acq_info["protocol"] == "trace"
    assert acq_info["acq_return_type"] == np.ndarray
    assert_acquisition_equal(acq_info, kwargs)


@pytest.fixture
def typical_kwargs_optical_measurement_multiple_pulses():
    return {
        "pulse_amplitudes": [1, 2],
        "pulse_durations": [50e-9, 70e-9],
        "pulse_ports": ["pulse1_port", "pulse2_port"],
        "pulse_clocks": ["pulse1_clock", "pulse2_clock"],
        "acq_duration": 200e-6,
        "acq_delay": 0,
        "acq_port": "acq_port_name",
        "acq_clock": "acq_clock_name",
        "acq_channel": 1,
        "acq_index": 0,
        "acq_protocol": None,
        "bin_mode": None,
        "acq_protocol_default": "TriggerCount",
        "pulse_type": "SquarePulse",
    }


@pytest.mark.parametrize("acq_delay", [-50e-6, 0, 52e-5])
def test_optical_measurement_multiple_pulses(
    typical_kwargs_optical_measurement_multiple_pulses,
    acq_delay,
):
    """``optical_measurement_multiple_pulses`` returns correct pulses."""
    # Arrange
    kwargs = typical_kwargs_optical_measurement_multiple_pulses
    kwargs["acq_delay"] = acq_delay

    # Act
    operation = optical_measurement_multiple_pulses(**kwargs)

    # Assert
    assert operation.valid_acquisition
    assert operation.valid_pulse

    assert len(operation.data["pulse_info"]) == len(kwargs["pulse_amplitudes"])
    for index, pulse_info in enumerate(operation.data["pulse_info"]):
        assert isinstance(pulse_info, dict)
        # Convert dict into a dict compatible with optical_measurement
        kwargs_single_pulse = {
            (key[:-1] if key.startswith("pulse_") else key): (
                val[index] if key.startswith("pulse_") else val
            )
            for key, val in kwargs.items()
        }
        assert_pulse_equal(pulse_info, kwargs_single_pulse)

    assert len(operation.data["acquisition_info"]) == 1
    acq_info = operation.data["acquisition_info"][0]
    assert isinstance(acq_info, dict)
    assert_acquisition_equal(acq_info, kwargs)
