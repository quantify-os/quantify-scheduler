from quantify.scheduler.types import Operation
from quantify.scheduler.enums import BinMode
from quantify.scheduler.acquisition_library import (
    SSBIntegrationComplex,
    Trace,
)


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


def test_add_acquisition():
    pass
