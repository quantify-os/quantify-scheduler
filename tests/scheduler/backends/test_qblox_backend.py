import os
import inspect
import json
import tempfile
import pytest

from qcodes.instrument.base import Instrument

from quantify.data.handling import set_datadir

from quantify.scheduler.types import Schedule
from quantify.scheduler.gate_library import Reset, Measure, Rxy, X
from quantify.scheduler.pulse_library import SquarePulse, DRAGPulse, RampPulse
from quantify.scheduler.resources import ClockResource
from quantify.scheduler.compilation import qcompile, determine_absolute_timing

from quantify.scheduler.backends import qblox_backend
from quantify.scheduler.backends.qblox_backend import *

import quantify.scheduler.schemas.examples as es

esp = inspect.getfile(es)

cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)


try:
    from pulsar_qcm.pulsar_qcm import pulsar_qcm_dummy
    from pulsar_qrm.pulsar_qrm import pulsar_qrm_dummy

    PULSAR_ASSEMBLER = True
except ImportError:
    PULSAR_ASSEMBLER = False


@pytest.fixture
def dummy_pulsars():
    if PULSAR_ASSEMBLER:
        _pulsars = []
        for qcm in ["qcm0", "qcm1"]:
            _pulsars.append(pulsar_qcm_dummy(qcm))
        for qrm in ["qrm0", "qrm1"]:
            _pulsars.append(pulsar_qrm_dummy(qrm))
    else:
        _pulsars = []

    # ensures a temporary datadir is used which is excluded from git
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    yield _pulsars

    # teardown
    for instr_name in list(Instrument._all_instruments):
        try:
            inst = Instrument.find_instrument(instr_name)
            inst.close()
        except KeyError:
            pass


@pytest.fixture
def pulse_only_schedule():
    sched = Schedule("pulse_only_experiment")
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    return sched


@pytest.fixture
def mixed_schedule_with_acquisition():
    sched = Schedule("mixed_schedule_with_acquisition")
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(Measure("q0"))
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    return sched


def test_contruct_sequencer():
    class Test_Pulsar(Pulsar_base):
        SEQ_TYPE = QCM_sequencer
        MAX_SEQUENCERS = 10

        def __init__(self):
            super(Test_Pulsar, self).__init__(
                name="tester", total_play_time=1, hw_mapping=HARDWARE_MAPPING["qcm0"]
            )

        def hardware_compile(self) -> Dict[str, Any]:
            return dict()

    tp = Test_Pulsar()
    tp.sequencers = tp._construct_sequencers()
    seq_keys = list(tp.sequencers.keys())
    assert len(seq_keys) == 2
    assert type(tp.sequencers[seq_keys[0]]) == QCM_sequencer


def test_simple_compile(dummy_pulsars, pulse_only_schedule):
    """Tests if compilation with only pulses finishes without exceptions"""
    qcompile(pulse_only_schedule, DEVICE_CFG, HARDWARE_MAPPING)


def test_simple_compile_with_acq(dummy_pulsars, mixed_schedule_with_acquisition):
    qcompile(mixed_schedule_with_acquisition, DEVICE_CFG, HARDWARE_MAPPING)


def test_sanitize_fn():
    filename = "this.isavalid=filename.exe.jpeg"
    new_filename = qblox_backend._sanitize_file_name(filename)
    assert new_filename == "this.isavalid=filename.exe.jpeg"

    filename = "this.isan:in>,valid=filename***!.exe.jpeg"
    new_filename = qblox_backend._sanitize_file_name(filename)
    assert new_filename == "this.isan_in__valid=filename____.exe.jpeg"


def test_modulate_waveform():
    number_of_points = 1000
    freq = 10e6
    t0 = 50e-9
    t = np.linspace(0, 1e-6, number_of_points)
    envelope = np.ones(number_of_points)
    mod_wf = modulate_waveform(t, envelope, freq, t0)
    test_re = np.cos(2 * np.pi * freq * (t + t0))
    test_imag = np.sin(2 * np.pi * freq * (t + t0))
    assert np.allclose(mod_wf.real, test_re)
    assert np.allclose(test_re, mod_wf.real)

    assert np.allclose(mod_wf.imag, test_imag)
    assert np.allclose(test_imag, mod_wf.imag)


def test_apply_mixer_corrections():
    number_of_points = 1000
    freq = 10e6
    t = np.linspace(0, 1e-6, number_of_points)
    amp_ratio = 2.1234

    test_re = np.cos(2 * np.pi * freq * t)
    test_imag = np.sin(2 * np.pi * freq * t)
    corrected_wf = apply_mixer_skewness_corrections(
        test_re + 1.0j * test_imag, amp_ratio, 90
    )

    amp_ratio_after = np.max(np.abs(corrected_wf.real)) / np.max(
        np.abs(corrected_wf.imag)
    )
    assert pytest.approx(amp_ratio_after, amp_ratio)

    re_normalized = corrected_wf.real / np.max(np.abs(corrected_wf.real))
    im_normalized = corrected_wf.imag / np.max(np.abs(corrected_wf.imag))
    assert np.allclose(re_normalized, im_normalized)
