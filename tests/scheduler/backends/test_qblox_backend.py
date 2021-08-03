# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Tests for Qblox backend."""

from typing import Dict, Any

import os
import inspect
import json
import tempfile
import pytest
import numpy as np

from qcodes.instrument.base import Instrument

# pylint: disable=no-name-in-module
from quantify_core.data.handling import set_datadir

from quantify_scheduler.types import Schedule
from quantify_scheduler.gate_library import Reset, Measure, X
from quantify_scheduler.pulse_library import (
    DRAGPulse,
    RampPulse,
    SquarePulse,
    StaircasePulse,
)
from quantify_scheduler.resources import ClockResource, BasebandClockResource
from quantify_scheduler.compilation import (
    qcompile,
    determine_absolute_timing,
    device_compile,
)

from quantify_scheduler.backends.qblox.helpers import (
    generate_waveform_data,
    find_inner_dicts_containing_key,
    find_all_port_clock_combinations,
)
from quantify_scheduler.backends import qblox_backend as qb
from quantify_scheduler.backends.types.qblox import (
    QASMRuntimeSettings,
)
from quantify_scheduler.backends.qblox.instrument_compilers import (
    Pulsar_QCM,
    Pulsar_QRM,
    QCMSequencer,
)
from quantify_scheduler.backends.qblox.compiler_abc import (
    PulsarBase,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import q1asm_instructions, compiler_container
from quantify_scheduler.backends.qblox import constants

import quantify_scheduler.schemas.examples as es

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
def hardware_cfg_baseband():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "int",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "lo0",
                "seq0": {
                    "port": "q0:mw",
                    "clock": "cl0.baseband",
                    "instruction_generated_pulses_enabled": True,
                    "interm_freq": 50e6,
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "seq1": {"port": "q1:mw", "clock": "q1.01"},
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "lo_freq": None, "power": 1},
    }


@pytest.fixture
def hardware_cfg_real_mode():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "int",
            "real_output_0": {
                "line_gain_db": 0,
                "seq0": {
                    "port": "LP",
                    "clock": "cl0.baseband",
                    "instruction_generated_pulses_enabled": True,
                },
            },
            "real_output_1": {
                "line_gain_db": 0,
                "seq1": {
                    "port": "RP",
                    "clock": "cl0.baseband",
                    "instruction_generated_pulses_enabled": True,
                },
            },
            "real_output_2": {
                "line_gain_db": 0,
                "seq2": {
                    "port": "TB",
                    "clock": "cl0.baseband",
                    "instruction_generated_pulses_enabled": True,
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_multiplexing():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "int",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "lo0",
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6,
                },
                "seq2": {
                    "port": "q1:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6,
                },
                "seq3": {
                    "port": "q2:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6,
                },
                "seq4": {
                    "port": "q3:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6,
                },
                "seq5": {
                    "port": "q4:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6,
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "seq1": {"port": "q1:mw", "clock": "q1.01"},
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "lo_freq": None, "power": 1},
    }


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
    sched.add(Reset("q0"))
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
def pulse_only_schedule_multiplexed():
    sched = Schedule("pulse_only_experiment")
    sched.add(Reset("q0"))
    operation = sched.add(
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
    for i in range(1, 4):
        sched.add(
            DRAGPulse(
                G_amp=0.7,
                D_amp=-0.2,
                phase=90,
                port=f"q{i}:mw",
                duration=20e-9,
                clock="q0.01",
                t0=8e-9,
            ),
            ref_op=operation,
            ref_pt="start",
        )

    sched.add(RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    return sched


@pytest.fixture
def pulse_only_schedule_no_lo():
    sched = Schedule("pulse_only_schedule_no_lo")
    sched.add(Reset("q1"))
    sched.add(
        SquarePulse(
            amp=0.5,
            phase=0,
            port="q1:res",
            duration=20e-9,
            clock="q1.ro",
            t0=4e-9,
        )
    )
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q1.ro", freq=100e6)])
    determine_absolute_timing(sched)
    return sched


@pytest.fixture
def identical_pulses_schedule():
    sched = Schedule("identical_pulses_schedule")
    sched.add(Reset("q0"))
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
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=0,
        )
    )
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    return sched


@pytest.fixture
def pulse_only_schedule_with_operation_timing():
    sched = Schedule("pulse_only_schedule_with_operation_timing")
    sched.add(Reset("q0"))
    first_op = sched.add(
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
    sched.add(
        RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"),
        ref_op=first_op,
        ref_pt="end",
        rel_time=1e-3,
    )
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    return sched


@pytest.fixture
def mixed_schedule_with_acquisition():
    sched = Schedule("mixed_schedule_with_acquisition")
    sched.add(Reset("q0"))
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


@pytest.fixture
def gate_only_schedule():
    sched = Schedule("gate_only_schedule")
    sched.add(Reset("q0"))
    x_gate = sched.add(X("q0"))
    sched.add(Measure("q0"), ref_op=x_gate, rel_time=1e-6, ref_pt="end")
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    return sched


@pytest.fixture
def baseband_square_pulse_schedule():
    sched = Schedule("baseband_square_pulse_schedule")
    sched.add(Reset("q0"))
    sched.add(
        SquarePulse(
            amp=2.0,
            duration=2.5e-6,
            port="q0:mw",
            clock=BasebandClockResource.IDENTITY,
            t0=1e-6,
        )
    )
    determine_absolute_timing(sched)
    return sched


@pytest.fixture
def real_square_pulse_schedule():
    sched = Schedule("real_square_pulse_schedule")
    sched.add(Reset("q0"))
    sched.add(
        SquarePulse(
            amp=2.0,
            duration=2.5e-6,
            port="LP",
            clock=BasebandClockResource.IDENTITY,
            t0=1e-6,
        )
    )
    sched.add(
        SquarePulse(
            amp=1.0,
            duration=2.0e-6,
            port="RP",
            clock=BasebandClockResource.IDENTITY,
            t0=0.5e-6,
        )
    )
    sched.add(
        SquarePulse(
            amp=1.2,
            duration=3.5e-6,
            port="TB",
            clock=BasebandClockResource.IDENTITY,
            t0=0,
        )
    )
    determine_absolute_timing(sched)
    return sched


# --------- Test utility functions ---------


def function_for_test_generate_waveform_data(t, x, y):
    return x * t + y


def test_generate_waveform_data():
    x = 10
    y = np.pi
    sampling_rate = 1e9
    duration = 1e-8
    t_verification = np.arange(0, 0 + duration, 1 / sampling_rate)
    verification_data = function_for_test_generate_waveform_data(t_verification, x, y)
    data_dict = {
        "wf_func": __name__ + ".function_for_test_generate_waveform_data",
        "x": x,
        "y": y,
        "duration": 1e-8,
    }
    gen_data = generate_waveform_data(data_dict, sampling_rate)
    assert np.allclose(gen_data, verification_data)


def test_find_inner_dicts_containing_key():
    test_dict = {
        "foo": "bar",
        "list": [{"key": 1, "hello": "world", "other_key": "other_value"}, 4, "12"],
        "nested": {"hello": "world", "other_key": "other_value"},
    }
    dicts_found = find_inner_dicts_containing_key(test_dict, "hello")
    assert len(dicts_found) == 2
    for inner_dict in dicts_found:
        assert inner_dict["hello"] == "world"
        assert inner_dict["other_key"] == "other_value"


def test_find_all_port_clock_combinations():
    portclocks = find_all_port_clock_combinations(HARDWARE_MAPPING)
    portclocks = set(portclocks)
    portclocks.discard((None, None))
    answer = {
        ("q1:mw", "q1.01"),
        ("q0:mw", "q0.01"),
        ("q0:res", "q0.ro"),
        ("q1:res", "q1.ro"),
    }
    assert portclocks == answer


def test_generate_port_clock_to_device_map():
    portclock_map = qb.generate_port_clock_to_device_map(HARDWARE_MAPPING)
    assert (None, None) not in portclock_map.keys()
    assert len(portclock_map.keys()) == 4


# --------- Test classes and member methods ---------
def test_contruct_sequencer():
    class TestPulsar(PulsarBase):
        sequencer_type = QCMSequencer
        max_sequencers = 10

        def __init__(self):
            super().__init__(
                parent=None,
                name="tester",
                total_play_time=1,
                hw_mapping=HARDWARE_MAPPING["qcm0"],
            )

        def compile(self, repetitions: int = 1) -> Dict[str, Any]:
            return dict()

    test_p = TestPulsar()
    test_p.sequencers = test_p._construct_sequencers()
    seq_keys = list(test_p.sequencers.keys())
    assert len(seq_keys) == 2
    assert isinstance(test_p.sequencers[seq_keys[0]], QCMSequencer)


def test_simple_compile(pulse_only_schedule):
    """Tests if compilation with only pulses finishes without exceptions"""
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    qcompile(pulse_only_schedule, DEVICE_CFG, HARDWARE_MAPPING)


def test_simple_compile_multiplexing(
    pulse_only_schedule_multiplexed, hardware_cfg_multiplexing
):
    """Tests if compilation with only pulses finishes without exceptions"""
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    qcompile(pulse_only_schedule_multiplexed, DEVICE_CFG, hardware_cfg_multiplexing)


def test_identical_pulses_compile(identical_pulses_schedule):
    """Tests if compilation with only pulses finishes without exceptions"""
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    qcompile(identical_pulses_schedule, DEVICE_CFG, HARDWARE_MAPPING)


def test_simple_compile_with_acq(dummy_pulsars, mixed_schedule_with_acquisition):
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    full_program = qcompile(
        mixed_schedule_with_acquisition, DEVICE_CFG, HARDWARE_MAPPING
    )

    qcm0_seq0_json = full_program["qcm0"]["seq0"]["seq_fn"]

    qcm0 = dummy_pulsars[0]
    qcm0.sequencer0_waveforms_and_program(qcm0_seq0_json)
    qcm0.arm_sequencer(0)
    uploaded_waveforms = qcm0.get_waveforms(0)
    assert uploaded_waveforms is not None


def test_compile_with_rel_time(
    dummy_pulsars, pulse_only_schedule_with_operation_timing
):
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    full_program = qcompile(
        pulse_only_schedule_with_operation_timing, DEVICE_CFG, HARDWARE_MAPPING
    )

    qcm0_seq0_json = full_program["qcm0"]["seq0"]["seq_fn"]

    qcm0 = dummy_pulsars[0]
    qcm0.sequencer0_waveforms_and_program(qcm0_seq0_json)


def test_compile_with_repetitions(mixed_schedule_with_acquisition):
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    mixed_schedule_with_acquisition.repetitions = 10
    full_program = qcompile(
        mixed_schedule_with_acquisition, DEVICE_CFG, HARDWARE_MAPPING
    )
    qcm0_seq0_json = full_program["qcm0"]["seq0"]["seq_fn"]

    with open(qcm0_seq0_json) as file:
        wf_and_prog = json.load(file)
    program_from_json = wf_and_prog["program"]
    move_line = program_from_json.split("\n")[3]
    move_items = move_line.split()  # splits on whitespace
    args = move_items[1]
    iterations = int(args.split(",")[0])
    assert iterations == 10


def test_compile_with_pulse_stitching(
    dummy_pulsars, hardware_cfg_baseband, baseband_square_pulse_schedule
):
    sched = baseband_square_pulse_schedule
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    sched.repetitions = 11
    full_program = qcompile(sched, DEVICE_CFG, hardware_cfg_baseband)
    qcm0_seq0_json = full_program["qcm0"]["seq0"]["seq_fn"]

    qcm0 = dummy_pulsars[0]
    qcm0.sequencer0_waveforms_and_program(qcm0_seq0_json)


def test_qcm_acquisition_error():
    qcm = Pulsar_QCM(
        None, "qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"]
    )
    qcm._acquisitions[0] = 0

    with pytest.raises(RuntimeError):
        qcm._distribute_data()


def test_real_mode_pulses(real_square_pulse_schedule, hardware_cfg_real_mode):
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    real_square_pulse_schedule.repetitions = 10
    full_program = qcompile(
        real_square_pulse_schedule, DEVICE_CFG, hardware_cfg_real_mode
    )
    for seq in (f"seq{i}" for i in range(3)):
        assert seq in full_program["qcm0"]


# --------- Test QASMProgram class ---------


def test_emit():
    qcm = Pulsar_QCM(
        None, "qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"]
    )
    qasm = QASMProgram(qcm)
    qasm.emit(q1asm_instructions.PLAY, 0, 1, 120)
    qasm.emit(q1asm_instructions.STOP, comment="This is a comment that is added")

    assert len(qasm.instructions) == 2
    with pytest.raises(SyntaxError):
        qasm.emit(q1asm_instructions.ACQUIRE, 0, 1, 120, "argument too many")


def test_auto_wait():
    qcm = Pulsar_QCM(
        None, "qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"]
    )
    qasm = QASMProgram(qcm.sequencers["seq0"])
    qasm.auto_wait(120)
    assert len(qasm.instructions) == 1
    qasm.auto_wait(70000)
    assert len(qasm.instructions) == 3  # since it should split the waits
    assert qasm.elapsed_time == 70120
    qasm.auto_wait(700000)
    assert qasm.elapsed_time == 770120
    assert len(qasm.instructions) == 8  # now loops are used
    with pytest.raises(ValueError):
        qasm.auto_wait(-120)


def test_wait_till_start_then_play():
    minimal_pulse_data = {"duration": 20e-9}
    runtime_settings = QASMRuntimeSettings(1, 1)
    pulse = qb.OpInfo(
        uuid="test_pulse",
        data=minimal_pulse_data,
        timing=4e-9,
        pulse_settings=runtime_settings,
    )
    qcm = Pulsar_QCM(
        None, "qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"]
    )
    qasm = QASMProgram(qcm.sequencers["seq0"])
    qasm.wait_till_start_then_play(pulse, 0, 1)
    assert len(qasm.instructions) == 3
    assert qasm.instructions[0][1] == q1asm_instructions.WAIT
    assert qasm.instructions[1][1] == q1asm_instructions.SET_AWG_GAIN
    assert qasm.instructions[2][1] == q1asm_instructions.PLAY

    pulse = qb.OpInfo(
        uuid="test_pulse",
        data=minimal_pulse_data,
        timing=1e-9,
        pulse_settings=runtime_settings,
    )
    with pytest.raises(ValueError):
        qasm.wait_till_start_then_play(pulse, 0, 1)


def test_wait_till_start_then_acquire():
    minimal_pulse_data = {"duration": 20e-9}
    acq = qb.OpInfo(uuid="test_acq", data=minimal_pulse_data, timing=4e-9)
    qrm = Pulsar_QRM(
        None, "qrm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qrm0"]
    )
    qasm = QASMProgram(qrm.sequencers["seq0"])
    qasm.wait_till_start_then_acquire(acq, 0, 1)
    assert len(qasm.instructions) == 2
    assert qasm.instructions[0][1] == q1asm_instructions.WAIT
    assert qasm.instructions[1][1] == q1asm_instructions.ACQUIRE


def test_expand_from_normalised_range():
    minimal_pulse_data = {"duration": 20e-9}
    acq = qb.OpInfo(uuid="test_acq", data=minimal_pulse_data, timing=4e-9)
    expanded_val = QASMProgram._expand_from_normalised_range(
        1, constants.IMMEDIATE_SZ_WAIT, "test_param", acq
    )
    assert expanded_val == constants.IMMEDIATE_SZ_WAIT // 2
    with pytest.raises(ValueError):
        QASMProgram._expand_from_normalised_range(
            10, constants.IMMEDIATE_SZ_WAIT, "test_param", acq
        )


def test_pulse_stitching_qasm_prog():
    minimal_pulse_data = {
        "wf_func": "quantify_scheduler.waveforms.square",
        "duration": 20.5e-6,
    }
    runtime_settings = QASMRuntimeSettings(1, 1)
    pulse = qb.OpInfo(
        uuid="stitched_square_pulse",
        data=minimal_pulse_data,
        timing=4e-9,
        pulse_settings=runtime_settings,
    )
    qcm = Pulsar_QCM(
        None, "qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"]
    )
    qasm = QASMProgram(qcm.sequencers["seq0"])
    qasm.wait_till_start_then_play(pulse, 0, 1)
    assert qasm.instructions[2][2] == "20,R2"


@pytest.mark.parametrize("start_amp, final_amp", [(-1.1, 2.1), (1.23456, -2)])
def test_staircase_qasm_prog(start_amp, final_amp):

    s_ramp_pulse = StaircasePulse(start_amp, final_amp, 10, 12.4e-6, "q0:mw")

    runtime_settings = QASMRuntimeSettings(1, 1)
    pulse = qb.OpInfo(
        uuid="staircase",
        data=s_ramp_pulse.data["pulse_info"][0],
        timing=4e-9,
        pulse_settings=runtime_settings,
    )
    qcm = Pulsar_QCM(
        None, "qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"]
    )
    qasm = QASMProgram(qcm.sequencers["seq0"])
    qasm.wait_till_start_then_play(pulse, 0, 1)

    amp_step_used = int(qasm.instructions[9][2].split(",")[1])
    if final_amp < start_amp:
        amp_step_used = -amp_step_used
    steps_taken = int(qasm.instructions[5][2].split(",")[0])
    init_amp = int(qasm.instructions[2][2].split(",")[0])
    if init_amp > constants.IMMEDIATE_SZ_OFFSET:
        init_amp = init_amp - constants.REGISTER_SIZE

    final_amp_imm = amp_step_used * (steps_taken - 1) + init_amp
    awg_output_volt = qcm.sequencers["seq0"].awg_output_volt

    final_amp_volt = 2 * final_amp_imm / constants.IMMEDIATE_SZ_OFFSET * awg_output_volt
    assert final_amp_volt == pytest.approx(final_amp, 1e-3)


def test_to_pulsar_time():
    time_ns = QASMProgram.to_pulsar_time(8e-9)
    assert time_ns == 8
    with pytest.raises(ValueError):
        QASMProgram.to_pulsar_time(7e-9)


def test_loop():
    num_rep = 10
    reg = "R0"

    qcm = Pulsar_QCM(
        None, "qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"]
    )
    qasm = QASMProgram(qcm.sequencers["seq0"])
    qasm.emit(q1asm_instructions.WAIT_SYNC, 4)
    with qasm.loop(reg, "this_loop", repetitions=num_rep):
        qasm.emit(q1asm_instructions.WAIT, 20)
    assert len(qasm.instructions) == 5
    assert qasm.instructions[1][1] == q1asm_instructions.MOVE
    num_rep_used, reg_used = qasm.instructions[1][2].split(",")
    assert int(num_rep_used) == num_rep
    assert reg_used == reg


# --------- Test compilation functions ---------
def test_assign_pulse_and_acq_info_to_devices(mixed_schedule_with_acquisition):
    sched_with_pulse_info = device_compile(mixed_schedule_with_acquisition, DEVICE_CFG)
    portclock_map = qb.generate_port_clock_to_device_map(HARDWARE_MAPPING)

    container = compiler_container.CompilerContainer.from_mapping(
        sched_with_pulse_info, HARDWARE_MAPPING
    )
    qb._assign_pulse_and_acq_info_to_devices(
        sched_with_pulse_info, container.instrument_compilers, portclock_map
    )
    qrm = container.instrument_compilers["qrm0"]
    assert len(qrm._pulses[list(qrm.portclocks_with_data)[0]]) == 1
    assert len(qrm._acquisitions[list(qrm.portclocks_with_data)[0]]) == 1


def test_container_prepare(pulse_only_schedule):
    container = compiler_container.CompilerContainer.from_mapping(
        pulse_only_schedule, HARDWARE_MAPPING
    )
    for instr in container.instrument_compilers.values():
        instr.prepare()

    assert (
        container.instrument_compilers["qcm0"].sequencers["seq0"].frequency is not None
    )
    assert container.instrument_compilers["lo0"].frequency is not None


def test_container_prepare_baseband(
    baseband_square_pulse_schedule, hardware_cfg_baseband
):
    container = compiler_container.CompilerContainer.from_mapping(
        baseband_square_pulse_schedule, hardware_cfg_baseband
    )
    for instr in container.instrument_compilers.values():
        instr.prepare()

    assert (
        container.instrument_compilers["qcm0"].sequencers["seq0"].frequency is not None
    )
    assert container.instrument_compilers["lo0"].frequency is not None


def test_container_prepare_no_lo(pulse_only_schedule_no_lo):
    container = compiler_container.CompilerContainer.from_mapping(
        pulse_only_schedule_no_lo, HARDWARE_MAPPING
    )
    container.compile(repetitions=10)

    assert container.instrument_compilers["qrm1"].sequencers["seq0"].frequency == 100e6


def test_container_add_from_type(pulse_only_schedule):
    container = compiler_container.CompilerContainer(pulse_only_schedule)
    container.add_instrument_compiler("qcm0", Pulsar_QCM, HARDWARE_MAPPING["qcm0"])
    assert "qcm0" in container.instrument_compilers
    assert isinstance(container.instrument_compilers["qcm0"], Pulsar_QCM)


def test_container_add_from_str(pulse_only_schedule):
    container = compiler_container.CompilerContainer(pulse_only_schedule)
    container.add_instrument_compiler("qcm0", "Pulsar_QCM", HARDWARE_MAPPING["qcm0"])
    assert "qcm0" in container.instrument_compilers
    assert isinstance(container.instrument_compilers["qcm0"], Pulsar_QCM)


def test_from_mapping(pulse_only_schedule):
    container = compiler_container.CompilerContainer.from_mapping(
        pulse_only_schedule, HARDWARE_MAPPING
    )
    for instr_name in HARDWARE_MAPPING.keys():
        if instr_name == "backend":
            continue
        assert instr_name in container.instrument_compilers


def test_real_mode_container(real_square_pulse_schedule, hardware_cfg_real_mode):
    container = compiler_container.CompilerContainer.from_mapping(
        real_square_pulse_schedule, hardware_cfg_real_mode
    )
    qcm0 = container.instrument_compilers["qcm0"]
    for output, seq_name in enumerate(f"seq{i}" for i in range(3)):
        seq_settings = qcm0.sequencers[seq_name].settings
        assert seq_settings.connected_outputs[0] == output
