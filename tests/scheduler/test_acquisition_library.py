# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Unit tests acquisition protocols for use with the quantify_scheduler."""

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
from unittest import TestCase

import numpy as np
import pytest

from quantify_scheduler import Operation
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.acquisition_library import (
    NumericalWeightedIntegrationComplex,
    SSBIntegrationComplex,
    Trace,
)
from quantify_scheduler.operations.gate_library import X90
from quantify_scheduler.operations.pulse_library import DRAGPulse


def test_ssb_integration_complex():
    ssb_acq = SSBIntegrationComplex(
        duration=100e-9,
        port="q0.res",
        clock="q0.01",
        acq_channel=-1337,
        acq_index=1234,
        bin_mode=BinMode.APPEND,
        phase=0,
        t0=20e-9,
    )
    assert Operation.is_valid(ssb_acq)
    assert ssb_acq.data["acquisition_info"][0]["acq_index"] == 1234
    assert ssb_acq.data["acquisition_info"][0]["acq_channel"] == -1337


def test_valid_acquisition():
    ssb_acq = SSBIntegrationComplex(
        duration=100e-9,
        port="q0.res",
        clock="q0.01",
        acq_channel=-1337,
        acq_index=1234,
        bin_mode=BinMode.APPEND,
        phase=0,
        t0=20e-9,
    )
    assert ssb_acq.valid_acquisition
    assert not ssb_acq.valid_pulse

    dgp = DRAGPulse(
        G_amp=0.8,
        D_amp=-0.3,
        phase=24.3,
        duration=10e-9,
        clock="cl:01",
        port="p.01",
        t0=3.4e-9,
    )
    assert not dgp.valid_acquisition

    dgp.add_acquisition(ssb_acq)
    assert dgp.valid_acquisition

    x90 = X90("q1")
    assert len(x90["acquisition_info"]) == 0
    assert not x90.valid_acquisition

    x90.add_acquisition(ssb_acq)
    assert x90.valid_acquisition
    assert len(x90["acquisition_info"]) == 1

    x90.add_acquisition(ssb_acq)
    assert x90.valid_acquisition
    assert len(x90["acquisition_info"]) == 2


def test_trace():
    trace = Trace(
        1234e-9,
        port="q0:res",
        clock="q0.ro",
        acq_channel=4815162342,
        acq_index=4815162342,
        bin_mode=BinMode.AVERAGE,
        t0=12e-9,
    )
    assert Operation.is_valid(trace)
    assert trace.data["acquisition_info"][0]["acq_index"] == 4815162342
    assert trace.data["acquisition_info"][0]["acq_channel"] == 4815162342


@pytest.mark.parametrize(
    "operation",
    [
        Trace(duration=16e-9, port="q0:res", clock="q0.ro"),
        SSBIntegrationComplex(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        NumericalWeightedIntegrationComplex(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            t=np.linspace(0, 3, 1),
            port="q0.res",
            clock="q0.ro",
        ),
    ],
)
def test__repr__(operation: Operation):
    assert eval(repr(operation)) == operation


@pytest.mark.parametrize(
    "operation",
    [
        Trace(
            duration=16e-9,
            port="q0:res",
            clock="q0.ro",
        ),
        SSBIntegrationComplex(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        NumericalWeightedIntegrationComplex(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            t=np.linspace(0, 3, 1),
            port="q0:res",
            clock="q0.ro",
        ),
    ],
)
def test__str__(operation: Operation):
    assert isinstance(eval(str(operation)), type(operation))


@pytest.mark.parametrize(
    "operation",
    [
        Trace(
            duration=16e-9,
            port="q0:res",
            clock="q0.ro",
        ),
        SSBIntegrationComplex(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        NumericalWeightedIntegrationComplex(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            t=np.linspace(0, 3, 1),
            port="q0:res",
            clock="q0.ro",
        ),
    ],
)
def test_deserialize(operation: Operation):
    # Arrange
    operation_repr: str = repr(operation)

    # Act
    obj = eval(operation_repr)

    # Assert
    if isinstance(operation, NumericalWeightedIntegrationComplex):
        waveforms = operation.data["acquisition_info"][0]["waveforms"]
        for i, waveform in enumerate(waveforms):
            assert isinstance(waveform["t"], (np.generic, np.ndarray))
            assert isinstance(waveform["weights"], (np.generic, np.ndarray))
            np.testing.assert_array_almost_equal(
                obj.data["acquisition_info"][0]["waveforms"][i]["t"],
                waveform["t"],
                decimal=9,
            )
            np.testing.assert_array_almost_equal(
                obj.data["acquisition_info"][0]["waveforms"][i]["weights"],
                waveform["weights"],
                decimal=9,
            )

            # TestCase().assertDictEqual cannot compare numpy arrays for equality
            # therefore "unitary" is removed
            del obj.data["acquisition_info"][0]["waveforms"][i]["t"]
            del waveform["t"]
            del obj.data["acquisition_info"][0]["waveforms"][i]["weights"]
            del waveform["weights"]

    TestCase().assertDictEqual(obj.data, operation.data)


@pytest.mark.parametrize(
    "operation",
    [
        Trace(
            duration=16e-9,
            port="q0:res",
            clock="q0.ro",
        ),
        SSBIntegrationComplex(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        NumericalWeightedIntegrationComplex(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            t=np.linspace(0, 3, 1),
            port="q0:res",
            clock="q0.ro",
        ),
    ],
)
def test__repr__modify_not_equal(operation: Operation):
    # Arrange
    obj = eval(repr(operation))
    assert obj == operation

    # Act
    obj.data["acquisition_info"][0]["foo"] = "bar"

    # Assert
    assert obj != operation
