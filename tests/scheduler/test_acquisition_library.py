# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Unit tests acquisition protocols for use with the quantify_scheduler."""


import json
from unittest import TestCase

import numpy as np
import pytest

from quantify_scheduler import Operation
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.acquisition_library import (
    NumericalSeparatedWeightedIntegration,
    NumericalWeightedIntegration,
    NumericalWeightedIntegrationComplex,
    SSBIntegrationComplex,
    ThresholdedAcquisition,
    Trace,
    TriggerCount,
)
from quantify_scheduler.operations.gate_library import X90
from quantify_scheduler.operations.pulse_library import DRAGPulse
from quantify_scheduler.json_utils import SchedulerJSONEncoder, SchedulerJSONDecoder


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


def test_trigger_count():
    trigger_count = TriggerCount(
        port="q0:res",
        clock="q0.ro",
        duration=0.001,
        acq_channel=4815162342,
        acq_index=0,
        bin_mode=BinMode.AVERAGE,
        t0=12e-9,
    )
    assert Operation.is_valid(trigger_count)
    assert trigger_count.data["acquisition_info"][0]["port"] == "q0:res"
    assert trigger_count.data["acquisition_info"][0]["clock"] == "q0.ro"
    assert trigger_count.data["acquisition_info"][0]["duration"] == 0.001
    assert trigger_count.data["acquisition_info"][0]["acq_index"] == 0
    assert trigger_count.data["acquisition_info"][0]["acq_channel"] == 4815162342
    assert trigger_count.data["acquisition_info"][0]["bin_mode"] == BinMode.AVERAGE
    assert trigger_count.data["acquisition_info"][0]["t0"] == 12e-9


def test_trigger_count_invalid_index_average_mode():
    with pytest.raises(NotImplementedError) as error:
        _ = TriggerCount(
            port="q0:res",
            clock="q0.ro",
            duration=0.001,
            acq_channel=0,
            acq_index=1,
            bin_mode=BinMode.AVERAGE,
            t0=12e-9,
        )

    assert (
        error.value.args[0]
        == "Using nonzero acq_index is not yet implemented for AVERAGE bin mode for "
        "the trigger count protocol"
    )


def test_deprecated_weighted_acquisition():
    with pytest.warns(
        FutureWarning,
        match="0.20.0",
    ):
        weighted = NumericalWeightedIntegrationComplex(
            port="q0:res",
            clock="q0.ro",
            weights_a=[0.25, 0.5, 0.25, 0.25],
            weights_b=[0.25, 0.5, 0.5, 0.25],
            interpolation="linear",
            acq_channel=1,
            acq_index=2,
            bin_mode=BinMode.APPEND,
            t0=16e-9,
        )
    expected = {
        "t0": 1.6e-08,
        "clock": "q0.ro",
        "port": "q0:res",
        "duration": pytest.approx(4e-9),
        "phase": 0,
        "acq_channel": 1,
        "acq_index": 2,
        "bin_mode": BinMode.APPEND,
        "protocol": "NumericalSeparatedWeightedIntegration",
        "acq_return_type": complex,
    }
    for k, v in expected.items():
        assert weighted.data["acquisition_info"][0][k] == v
    wf_a, wf_b = weighted.data["acquisition_info"][0]["waveforms"]
    assert list(wf_a["t_samples"]) == [0.0e00, 1.0e-09, 2.0e-09, 3.0e-09]
    assert list(wf_b["t_samples"]) == [0.0e00, 1.0e-09, 2.0e-09, 3.0e-09]


def test_weighted_acquisition():
    weighted = NumericalSeparatedWeightedIntegration(
        port="q0:res",
        clock="q0.ro",
        weights_a=[0.25, 0.5, 0.25, 0.25],
        weights_b=[0.25, 0.5, 0.5, 0.25],
        interpolation="linear",
        acq_channel=1,
        acq_index=2,
        bin_mode=BinMode.APPEND,
        t0=16e-9,
    )
    expected = {
        "t0": 1.6e-08,
        "clock": "q0.ro",
        "port": "q0:res",
        "duration": pytest.approx(4e-9),
        "phase": 0,
        "acq_channel": 1,
        "acq_index": 2,
        "bin_mode": BinMode.APPEND,
        "protocol": "NumericalSeparatedWeightedIntegration",
        "acq_return_type": complex,
    }
    for k, v in expected.items():
        assert weighted.data["acquisition_info"][0][k] == v
    wf_a, wf_b = weighted.data["acquisition_info"][0]["waveforms"]
    assert list(wf_a["t_samples"]) == [0.0e00, 1.0e-09, 2.0e-09, 3.0e-09]
    assert list(wf_b["t_samples"]) == [0.0e00, 1.0e-09, 2.0e-09, 3.0e-09]

    weighted = NumericalWeightedIntegration(
        port="q0:res",
        clock="q0.ro",
        weights_a=[0.25, 0.5, 0.25, 0.25],
        weights_b=[0.25, 0.5, 0.5, 0.25],
        interpolation="linear",
        acq_channel=1,
        acq_index=2,
        bin_mode=BinMode.APPEND,
        t0=16e-9,
    )
    expected = {
        "t0": 1.6e-08,
        "clock": "q0.ro",
        "port": "q0:res",
        "duration": pytest.approx(4e-9),
        "phase": 0,
        "acq_channel": 1,
        "acq_index": 2,
        "bin_mode": BinMode.APPEND,
        "protocol": "NumericalWeightedIntegration",
        "acq_return_type": complex,
    }
    for k, v in expected.items():
        assert weighted.data["acquisition_info"][0][k] == v
    wf_a, wf_b = weighted.data["acquisition_info"][0]["waveforms"]
    assert list(wf_a["t_samples"]) == [0.0e00, 1.0e-09, 2.0e-09, 3.0e-09]
    assert list(wf_b["t_samples"]) == [0.0e00, 1.0e-09, 2.0e-09, 3.0e-09]


@pytest.mark.parametrize(
    "operation",
    [
        Trace(duration=16e-9, port="q0:res", clock="q0.ro"),
        SSBIntegrationComplex(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        TriggerCount(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        ThresholdedAcquisition(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
    ],
)
def test__repr__(operation: Operation):
    # Arrange
    operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=SchedulerJSONDecoder)
    assert obj == operation


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
        NumericalSeparatedWeightedIntegration(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            port="q0:res",
            clock="q0.ro",
        ),
        NumericalWeightedIntegration(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            port="q0:res",
            clock="q0.ro",
        ),
        TriggerCount(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        ThresholdedAcquisition(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
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
        NumericalSeparatedWeightedIntegration(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            port="q0:res",
            clock="q0.ro",
        ),
        NumericalWeightedIntegration(
            weights_a=np.zeros(3, dtype=complex),
            weights_b=np.ones(3, dtype=complex),
            port="q0:res",
            clock="q0.ro",
        ),
        TriggerCount(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        ThresholdedAcquisition(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
    ],
)
def test_deserialize(operation: Operation):
    # Arrange
    operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=SchedulerJSONDecoder)

    # Assert
    if isinstance(
        operation, (NumericalSeparatedWeightedIntegration, NumericalWeightedIntegration)
    ):
        waveforms = operation.data["acquisition_info"][0]["waveforms"]
        for i, waveform in enumerate(waveforms):
            assert isinstance(waveform["t_samples"], (np.generic, np.ndarray))
            assert isinstance(waveform["samples"], (np.generic, np.ndarray))
            np.testing.assert_array_almost_equal(
                obj.data["acquisition_info"][0]["waveforms"][i]["t_samples"],
                waveform["t_samples"],
                decimal=9,
            )
            np.testing.assert_array_almost_equal(
                obj.data["acquisition_info"][0]["waveforms"][i]["samples"],
                waveform["samples"],
                decimal=9,
            )

            # TestCase().assertDictEqual cannot compare numpy arrays for equality
            # therefore "unitary" is removed
            del obj.data["acquisition_info"][0]["waveforms"][i]["t_samples"]
            del waveform["t_samples"]
            del obj.data["acquisition_info"][0]["waveforms"][i]["samples"]
            del waveform["samples"]

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
        TriggerCount(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
        ThresholdedAcquisition(
            port="q0:res",
            clock="q0.ro",
            duration=100e-9,
        ),
    ],
)
def test__repr__modify_not_equal(operation: Operation):
    # Arrange
    operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=SchedulerJSONDecoder)
    assert obj == operation

    # Act
    obj.data["acquisition_info"][0]["foo"] = "bar"

    # Assert
    assert obj != operation
