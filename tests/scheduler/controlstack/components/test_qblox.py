# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
from __future__ import annotations
import inspect
import json
import tempfile

from pathlib import Path
import numpy as np

import pytest

from pulsar_qcm import pulsar_qcm
from pulsar_qrm import pulsar_qrm

from quantify.data.handling import set_datadir  # pylint: disable=no-name-in-module

from quantify.scheduler.compilation import qcompile
from quantify.scheduler.helpers import waveforms
import quantify.scheduler.schemas.examples as es
from quantify.scheduler.controlstack.components import qblox

esp = inspect.getfile(es)

cfg_f = Path(esp).parent / "transmon_test_config.json"
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = Path(esp).parent / "qblox_test_mapping.json"
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)


@pytest.fixture
def dummy_qcm():
    _qcm = qblox.PulsarQCMComponent("qcm0", "dummy")
    yield _qcm

    _qcm.close()


@pytest.fixture
def dummy_qrm():
    _qrm = qblox.PulsarQRMComponent("qrm0", "dummy")
    yield _qrm

    _qrm.close()


def mock_ip_transport(
    host: str, port: int = 5025, timeout: float = 60.0, snd_buf_size: int = 512 * 1024
):
    del host, port, timeout, snd_buf_size  # unused by mock

    transport_inst = pulsar_qcm.pulsar_dummy_transport(
        pulsar_qcm.pulsar_qcm_ifc._get_sequencer_cfg_format()
    )
    return transport_inst


def test_initialize_pulsar_qcm_component(monkeypatch):
    monkeypatch.setattr(pulsar_qcm, "ip_transport", mock_ip_transport)
    instr = qblox.PulsarQCMComponent("qblox_qcm0", "1234", debug=1)
    instr.close()


def test_initialize_pulsar_qrm_component(monkeypatch):
    monkeypatch.setattr(pulsar_qrm, "ip_transport", mock_ip_transport)
    instr = qblox.PulsarQRMComponent("qblox_qrm0", "1234", debug=1)
    instr.close()


def test_prepare(schedule_with_measurement, dummy_qcm, dummy_qrm, mocker):
    spy_qcm = mocker.spy(dummy_qcm, "arm_sequencer")
    spy_qrm = mocker.spy(dummy_qrm, "arm_sequencer")
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        prog = qcompile(schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING)
        for instr_name, params in prog.items():
            for cs_comp in [dummy_qcm, dummy_qrm]:
                if instr_name == cs_comp.name:
                    cs_comp.prepare(params)
        assert spy_qcm.call_count == 1
        assert spy_qrm.call_count == 1


def test_prepare_exception_qcm(dummy_qcm):
    prep_with_invalid_key = {"idontexist": "this is not used"}
    with pytest.raises(KeyError):
        dummy_qcm.prepare(prep_with_invalid_key)


def test_prepare_exception_qrm(dummy_qrm):
    prep_with_invalid_key = {"idontexist": "this is not used"}
    with pytest.raises(KeyError):
        dummy_qrm.prepare(prep_with_invalid_key)


def test_retrieve_acquisition_qcm(dummy_qcm):
    assert dummy_qcm.retrieve_acquisition() is None


def test_retrieve_acquisition_qrm(schedule_with_measurement, dummy_qrm):
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)
        prog = qcompile(schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING)
        prog = dict(prog)

        dummy_qrm.prepare(prog[dummy_qrm.name])
        dummy_qrm.start()
        acq = dummy_qrm.retrieve_acquisition()

        assert len(acq) == 2


def test_start_qcm_qrm(schedule_with_measurement, dummy_qcm, dummy_qrm, mocker):
    spy_qcm = mocker.spy(dummy_qcm, "start_sequencer")
    spy_qrm = mocker.spy(dummy_qrm, "start_sequencer")

    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        prog = qcompile(schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING)
        prog = dict(prog)
        for pulsar in (dummy_qcm, dummy_qrm):
            pulsar.prepare(prog[pulsar.name])
            pulsar.start()

        assert spy_qcm.call_count == 1
        assert spy_qrm.call_count == 1


def test_stop_qcm(dummy_qcm, mocker):
    spy = mocker.spy(dummy_qcm, "stop_sequencer")
    dummy_qcm.stop()
    assert spy.call_count == 1


def test_stop_qrm(dummy_qrm, mocker):
    spy = mocker.spy(dummy_qrm, "stop_sequencer")
    dummy_qrm.stop()
    assert spy.call_count == 1


def test_demodulate_trace():
    data = np.ones(1_000_000)
    freq = 10e6
    sampling_rate = 1e9

    t = np.arange(0, len(data) / sampling_rate, 1 / sampling_rate)
    mod_data = waveforms.modulate_waveform(t, data, freq)
    demod_data = qblox._demodulate_trace(
        freq, mod_data.real, mod_data.imag, sampling_rate
    )
    np.testing.assert_allclose(data, demod_data)
