from typing import Dict
import pytest

import numpy as np

from quantify_scheduler.operations.measurement_factories import optical_measurement

# pylint: disable=redefined-outer-name


@pytest.fixture
def default_kwargs_optical_measurement():
    """Default keyword arguments for optical_measurement function.

    Missing are ``'acq_delay'`` and ``'acq_protocol'``.
    """
    return {
        "pulse_amplitude": 0.23,
        "pulse_duration": 100e-6,
        "pulse_port": "pulse_port_name",
        "pulse_clock": "pulse_clock_name",
        "acq_duration": 200e-6,
        "acq_port": "acq_port_name",
        "acq_clock": "acq_clock_name",
        "acq_channel": 1,
        "acq_index": 0,
        "acq_protocol": "TriggerCount",
    }


def assert_square_pulse_equal(pulse_info: Dict, optical_meas_args: Dict):
    """Assert that info of square pulse is equal to arguments used as input to
    optical_meas_args."""
    assert isinstance(pulse_info, dict)
    assert pulse_info["amp"] == optical_meas_args["pulse_amplitude"]
    assert pulse_info["duration"] == optical_meas_args["pulse_duration"]
    assert pulse_info["phase"] == 0
    assert pulse_info["port"] == optical_meas_args["pulse_port"]
    assert pulse_info["clock"] == optical_meas_args["pulse_clock"]
    if optical_meas_args["acq_delay"] >= 0:
        assert pulse_info["t0"] == 0
    else:
        assert pulse_info["t0"] == -optical_meas_args["acq_delay"]


@pytest.mark.parametrize("acq_delay", [-50e-6, 0, 52e-5])
def test_optical_measurement_trigger_count(
    default_kwargs_optical_measurement, acq_delay
):
    """optical_measurement factory works well with TriggerCount protocol"""
    # Arrange
    kwargs = default_kwargs_optical_measurement
    kwargs["acq_delay"] = acq_delay
    kwargs["acq_protocol"] = "TriggerCount"

    # Act
    operation = optical_measurement(**kwargs)

    # Assert
    assert operation.valid_acquisition
    assert operation.valid_pulse
    assert len(operation.data["pulse_info"]) == 1
    assert len(operation.data["acquisition_info"]) == 1

    square_pulse_info = operation.data["pulse_info"][0]
    assert_square_pulse_equal(square_pulse_info, kwargs)

    assert isinstance(operation.data["acquisition_info"][0], dict)
    trigger_count_info = operation.data["acquisition_info"][0]
    assert trigger_count_info["duration"] == kwargs["acq_duration"]
    assert trigger_count_info["port"] == kwargs["acq_port"]
    assert trigger_count_info["clock"] == kwargs["acq_clock"]
    assert trigger_count_info["acq_channel"] == kwargs["acq_channel"]
    assert trigger_count_info["acq_index"] == kwargs["acq_index"]
    assert trigger_count_info["protocol"] == "trigger_count"
    assert trigger_count_info["acq_return_type"] == int
    if acq_delay >= 0:
        assert trigger_count_info["t0"] == acq_delay
    else:
        assert trigger_count_info["t0"] == 0


@pytest.mark.parametrize("acq_delay", [-50e-6, 0, 52e-5])
def test_optical_measurement_trace(default_kwargs_optical_measurement, acq_delay):
    """optical_measurement factory works well with Trace protocol"""
    # Arrange
    kwargs = default_kwargs_optical_measurement
    kwargs["acq_delay"] = acq_delay
    kwargs["acq_protocol"] = "Trace"

    # Act
    operation = optical_measurement(**kwargs)

    # Assert
    assert operation.valid_acquisition
    assert operation.valid_pulse
    assert len(operation.data["pulse_info"]) == 1
    assert len(operation.data["acquisition_info"]) == 1

    square_pulse_info = operation.data["pulse_info"][0]
    assert_square_pulse_equal(square_pulse_info, kwargs)

    assert isinstance(operation.data["acquisition_info"][0], dict)
    trace_info = operation.data["acquisition_info"][0]
    assert trace_info["duration"] == kwargs["acq_duration"]
    assert trace_info["port"] == kwargs["acq_port"]
    assert trace_info["clock"] == kwargs["acq_clock"]
    assert trace_info["acq_channel"] == kwargs["acq_channel"]
    assert trace_info["acq_index"] == kwargs["acq_index"]
    assert trace_info["protocol"] == "trace"
    assert trace_info["acq_return_type"] == np.ndarray
    if acq_delay >= 0:
        assert trace_info["t0"] == acq_delay
    else:
        assert trace_info["t0"] == 0
