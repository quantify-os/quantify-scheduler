from quantify.scheduler.types import Operation
from quantify.scheduler.enums import BinMode
from quantify.scheduler.acquisition_library import (
    SSBIntegrationComplex,
    Trace,
)
from quantify.scheduler.gate_library import X90
from quantify.scheduler.pulse_library import DRAGPulse


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
    tr = Trace(
        1234e-9,
        port="q0.res",
        acq_channel=4815162342,
        acq_index=4815162342,
        bin_mode=BinMode.AVERAGE,
        t0=12e-9,
    )
    assert Operation.is_valid(tr)
    assert tr.data["acquisition_info"][0]["acq_index"] == 4815162342
    assert tr.data["acquisition_info"][0]["acq_channel"] == 4815162342
