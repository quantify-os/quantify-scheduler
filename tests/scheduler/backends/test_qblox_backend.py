import os
import inspect
import json
import tempfile
import pytest
import numpy as np
from typing import Dict, Any

from qcodes.instrument.base import Instrument

from quantify.data.handling import set_datadir

from quantify.scheduler.types import Schedule
from quantify.scheduler.gate_library import Reset, Measure, Rxy, X
from quantify.scheduler.pulse_library import SquarePulse, DRAGPulse, RampPulse
from quantify.scheduler.resources import ClockResource
from quantify.scheduler.compilation import (
    qcompile,
    determine_absolute_timing,
    device_compile,
)

from quantify.scheduler.backends import qblox_backend as qb

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

# --------- Test fixtures ---------


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


# --------- Test classes and member methods ---------


def test_contruct_sequencer():
    class Test_Pulsar(qb.Pulsar_base):
        SEQ_TYPE = qb.QCM_sequencer
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
    assert type(tp.sequencers[seq_keys[0]]) == qb.QCM_sequencer


def test_simple_compile(dummy_pulsars, pulse_only_schedule):
    """Tests if compilation with only pulses finishes without exceptions"""
    qcompile(pulse_only_schedule, DEVICE_CFG, HARDWARE_MAPPING)


def test_simple_compile_with_acq(dummy_pulsars, mixed_schedule_with_acquisition):
    full_program = qcompile(
        mixed_schedule_with_acquisition, DEVICE_CFG, HARDWARE_MAPPING
    )
    qcm0_seq0_json = full_program["qcm0"]["seq0"]["seq_fn"]

    qcm0 = dummy_pulsars[0]
    qcm0.sequencer0_waveforms_and_program(qcm0_seq0_json)
    qcm0.arm_sequencer(0)
    uploaded_waveforms = qcm0.get_waveforms(0)
    assert uploaded_waveforms is not None


# --------- Test utility functions ---------


def test_sanitize_fn():
    filename = "this.isavalid=filename.exe.jpeg"
    new_filename = qb._sanitize_file_name(filename)
    assert new_filename == "this.isavalid=filename.exe.jpeg"

    filename = "this.isan:in>,valid=filename***!.exe.jpeg"
    new_filename = qb._sanitize_file_name(filename)
    assert new_filename == "this.isan_in__valid=filename____.exe.jpeg"


def test_modulate_waveform():
    number_of_points = 1000
    freq = 10e6
    t0 = 50e-9
    t = np.linspace(0, 1e-6, number_of_points)
    envelope = np.ones(number_of_points)
    mod_wf = qb.modulate_waveform(t, envelope, freq, t0)
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
    corrected_wf = qb.apply_mixer_skewness_corrections(
        test_re + 1.0j * test_imag, amp_ratio, 90
    )

    amp_ratio_after = np.max(np.abs(corrected_wf.real)) / np.max(
        np.abs(corrected_wf.imag)
    )
    assert pytest.approx(amp_ratio_after, amp_ratio)

    re_normalized = corrected_wf.real / np.max(np.abs(corrected_wf.real))
    im_normalized = corrected_wf.imag / np.max(np.abs(corrected_wf.imag))
    assert np.allclose(re_normalized, im_normalized)


def function_for_test_generate_waveform_data(t, foo, bar):
    return foo * t + bar


def test_generate_waveform_data():
    foo = 10
    bar = np.pi
    sampling_rate = 1e9
    duration = 1e-8
    t_verification = np.arange(0, 0 + duration, 1 / sampling_rate)
    verification_data = function_for_test_generate_waveform_data(
        t_verification, foo, bar
    )
    data_dict = {
        "wf_func": __name__ + ".function_for_test_generate_waveform_data",
        "foo": foo,
        "bar": bar,
        "duration": 1e-8,
    }
    gen_data = qb._generate_waveform_data(data_dict, sampling_rate)
    assert np.allclose(gen_data, verification_data)


def test_generate_ext_local_oscillators():
    lo_dict = qb.generate_ext_local_oscillators(10, HARDWARE_MAPPING)
    defined_los = {"lo0", "lo1"}
    assert lo_dict.keys() == defined_los

    lo1 = lo_dict["lo1"]
    lo1_freq = lo1.frequency
    assert lo1_freq == 7.2e9


def test_calculate_total_play_time(mixed_schedule_with_acquisition):
    sched = device_compile(mixed_schedule_with_acquisition, DEVICE_CFG)
    play_time = qb._calculate_total_play_time(sched)
    answer = 184e-9
    assert play_time == answer


def test_find_inner_dicts_containing_key():
    test_dict = {
        "foo": "bar",
        "list": [{"key": 1, "hello": "world", "other_key": "other_value"}, 4, "12"],
        "nested": {"hello": "world", "other_key": "other_value"},
    }
    dicts_found = qb.find_inner_dicts_containing_key(test_dict, "hello")
    assert len(dicts_found) == 2
    for d in dicts_found:
        assert d["hello"] == "world"
        assert d["other_key"] == "other_value"


def test_find_all_port_clock_combinations():
    portclocks = qb.find_all_port_clock_combinations(HARDWARE_MAPPING)
    portclocks = set(portclocks)
    portclocks.discard((None, None))
    assert portclocks == {
        ("q1:mw", "q1.12"),
        ("q1:mw", "q1.01"),
        ("q0:mw", "q0.01"),
        ("q0:mw", "q0.12"),
        ("q0:res", "q0.ro"),
    }


def test_find_abs_time_from_operation_hash(mixed_schedule_with_acquisition):
    sched = device_compile(mixed_schedule_with_acquisition, DEVICE_CFG)
    first_op_hash = sched.timing_constraints[0]["operation_hash"]
    first_abs_time = qb.find_abs_time_from_operation_hash(sched, first_op_hash)
    assert first_abs_time == 0

    second_op_hash = sched.timing_constraints[1]["operation_hash"]
    second_abs_time = qb.find_abs_time_from_operation_hash(sched, second_op_hash)
    assert second_abs_time == 24e-9
