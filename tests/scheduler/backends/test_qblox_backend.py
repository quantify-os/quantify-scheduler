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
from quantify.data.handling import set_datadir

from quantify.scheduler.types import Schedule
from quantify.scheduler.gate_library import Reset, Measure, X
from quantify.scheduler.pulse_library import DRAGPulse, RampPulse
from quantify.scheduler.resources import ClockResource
from quantify.scheduler.compilation import (
    qcompile,
    determine_absolute_timing,
    device_compile,
)
from quantify.scheduler.helpers.schedule import get_total_duration

from quantify.scheduler.backends.qblox.helpers import (
    generate_waveform_data,
    find_inner_dicts_containing_key,
    find_all_port_clock_combinations,
)
from quantify.scheduler.backends import qblox_backend as qb
from quantify.scheduler.backends.types.qblox import (
    QASMRuntimeSettings,
)
from quantify.scheduler.backends.qblox.instrument_compilers import (
    Pulsar_QCM,
    QCMSequencer,
)
from quantify.scheduler.backends.qblox.compiler_abc import (
    PulsarBase,
)
from quantify.scheduler.backends.qblox.qasm_program import QASMProgram
from quantify.scheduler.backends.qblox import q1asm_instructions
from quantify.scheduler.backends.qblox import constants

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


def test_generate_ext_local_oscillators():
    lo_dict = qb.generate_ext_local_oscillators(10, HARDWARE_MAPPING)
    defined_los = {"lo0", "lo1", "lo3"}
    assert lo_dict.keys() == defined_los

    lo1 = lo_dict["lo1"]
    lo1_freq = lo1.frequency
    assert lo1_freq == 7.2e9


def test_calculate_total_play_time_without_acq(pulse_only_schedule):
    sched = device_compile(pulse_only_schedule, DEVICE_CFG)
    init_duration = DEVICE_CFG["qubits"]["q0"]["params"]["init_duration"]
    play_time = get_total_duration(sched)
    answer = 24e-9 + 2e-3 + 28e-9 + init_duration
    assert play_time == answer


def test_calculate_total_play_time_with_op_timing(
    pulse_only_schedule_with_operation_timing,
):
    sched = device_compile(pulse_only_schedule_with_operation_timing, DEVICE_CFG)
    play_time = get_total_duration(sched)
    init_duration = DEVICE_CFG["qubits"]["q0"]["params"]["init_duration"]
    answer = 3e-3 + 28e-9 + 24e-9 + init_duration
    assert play_time == answer


def test_calculate_total_play_time_with_gates(
    gate_only_schedule,
):
    rel_time = 1e-6
    mw_duration = DEVICE_CFG["qubits"]["q0"]["params"]["mw_duration"]
    end_acq = (
        DEVICE_CFG["qubits"]["q0"]["params"]["ro_acq_delay"]
        + DEVICE_CFG["qubits"]["q0"]["params"]["ro_acq_integration_time"]
    )
    init_duration = DEVICE_CFG["qubits"]["q0"]["params"]["init_duration"]
    ro_pulse_duration = DEVICE_CFG["qubits"]["q0"]["params"]["ro_pulse_duration"]
    sched = device_compile(gate_only_schedule, DEVICE_CFG)
    play_time = get_total_duration(sched)
    answer = mw_duration + rel_time + max(end_acq, ro_pulse_duration) + init_duration
    assert play_time == answer


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
                name="tester", total_play_time=1, hw_mapping=HARDWARE_MAPPING["qcm0"]
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


def test_qcm_acquisition_error():
    qcm = Pulsar_QCM("qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"])
    qcm._acquisitions[0] = 0

    with pytest.raises(RuntimeError):
        qcm._distribute_data()


# --------- Test QASMProgram class ---------


def test_emit():
    qasm = QASMProgram()
    qasm.emit(q1asm_instructions.PLAY, 0, 1, 120)
    qasm.emit(q1asm_instructions.STOP, comment="This is a comment that is added")

    assert len(qasm.instructions) == 2
    with pytest.raises(SyntaxError):
        qasm.emit(q1asm_instructions.ACQUIRE, 0, 1, 120, "argument too many")


def test_auto_wait():
    qasm = QASMProgram()
    qasm.auto_wait(120)
    assert len(qasm.instructions) == 1
    qasm.auto_wait(70000)
    assert len(qasm.instructions) == 3  # since it should split the waits
    assert qasm.elapsed_time == 70120
    with pytest.raises(ValueError):
        qasm.auto_wait(-120)


def test_wait_till_start_then_play():
    minimal_pulse_data = {"duration": 20e-9}
    runtime_settings = QASMRuntimeSettings(1, 1)
    pulse = qb.OpInfo(
        uuid=0, data=minimal_pulse_data, timing=4e-9, pulse_settings=runtime_settings
    )
    qasm = QASMProgram()
    qasm.wait_till_start_then_play(pulse, 0, 1)
    assert len(qasm.instructions) == 3
    assert qasm.instructions[0][1] == q1asm_instructions.WAIT
    assert qasm.instructions[1][1] == q1asm_instructions.SET_AWG_GAIN
    assert qasm.instructions[2][1] == q1asm_instructions.PLAY

    pulse = qb.OpInfo(
        uuid=0, data=minimal_pulse_data, timing=1e-9, pulse_settings=runtime_settings
    )
    with pytest.raises(ValueError):
        qasm.wait_till_start_then_play(pulse, 0, 1)


def test_wait_till_start_then_acquire():
    minimal_pulse_data = {"duration": 20e-9}
    acq = qb.OpInfo(uuid=0, data=minimal_pulse_data, timing=4e-9)
    qasm = QASMProgram()
    qasm.wait_till_start_then_acquire(acq, 0, 1)
    assert len(qasm.instructions) == 2
    assert qasm.instructions[0][1] == q1asm_instructions.WAIT
    assert qasm.instructions[1][1] == q1asm_instructions.ACQUIRE


def test_expand_from_normalised_range():
    minimal_pulse_data = {"duration": 20e-9}
    acq = qb.OpInfo(uuid=0, data=minimal_pulse_data, timing=4e-9)
    expanded_val = QASMProgram._expand_from_normalised_range(
        1, constants.IMMEDIATE_SZ_WAIT, "test_param", acq
    )
    assert expanded_val == constants.IMMEDIATE_SZ_WAIT // 2
    with pytest.raises(ValueError):
        QASMProgram._expand_from_normalised_range(
            10, constants.IMMEDIATE_SZ_WAIT, "test_param", acq
        )


def test_to_pulsar_time():
    time_ns = QASMProgram.to_pulsar_time(8e-9)
    assert time_ns == 8
    with pytest.raises(ValueError):
        QASMProgram.to_pulsar_time(7e-9)


def test_loop():
    num_rep = 10
    reg = "R0"

    qasm = QASMProgram()
    qasm.emit(q1asm_instructions.WAIT_SYNC, 4)
    with qasm.loop(reg, "this_loop", repetitions=num_rep):
        qasm.emit(q1asm_instructions.WAIT, 20)
    assert len(qasm.instructions) == 5
    assert qasm.instructions[1][1] == q1asm_instructions.MOVE
    num_rep_used, reg_used = qasm.instructions[1][2].split(",")
    assert int(num_rep_used) == num_rep
    assert reg_used == reg


# --------- Test sequencer compilation ---------
def test_assign_frequency():
    qcm = Pulsar_QCM("qcm0", total_play_time=10, hw_mapping=HARDWARE_MAPPING["qcm0"])
    qcm_seq0 = qcm.sequencers["seq0"]
    qcm_seq0.assign_frequency(100e6)
    qcm_seq0.assign_frequency(100e6)

    assert qcm_seq0.settings.modulation_freq == 100e6

    with pytest.raises(ValueError):
        qcm_seq0.assign_frequency(110e6)


# --------- Test compilation functions ---------
def test_assign_pulse_and_acq_info_to_devices_exception(
    mixed_schedule_with_acquisition,
):
    total_play_time = get_total_duration(mixed_schedule_with_acquisition)
    portclock_map = qb.generate_port_clock_to_device_map(HARDWARE_MAPPING)

    device_compilers = qb._construct_compiler_objects(
        total_play_time=total_play_time,
        mapping=HARDWARE_MAPPING,
    )
    with pytest.raises(RuntimeError):
        qb._assign_pulse_and_acq_info_to_devices(
            mixed_schedule_with_acquisition, device_compilers, portclock_map
        )


def test_assign_pulse_and_acq_info_to_devices(mixed_schedule_with_acquisition):
    sched_with_pulse_info = device_compile(mixed_schedule_with_acquisition, DEVICE_CFG)
    total_play_time = get_total_duration(mixed_schedule_with_acquisition)
    portclock_map = qb.generate_port_clock_to_device_map(HARDWARE_MAPPING)

    device_compilers = qb._construct_compiler_objects(
        total_play_time=total_play_time,
        mapping=HARDWARE_MAPPING,
    )
    qb._assign_pulse_and_acq_info_to_devices(
        sched_with_pulse_info, device_compilers, portclock_map
    )
    qrm = device_compilers["qrm0"]
    assert len(qrm._pulses[list(qrm.portclocks_with_data)[0]]) == 1
    assert len(qrm._acquisitions[list(qrm.portclocks_with_data)[0]]) == 1


def test_assign_frequencies(mixed_schedule_with_acquisition):
    schedule = device_compile(mixed_schedule_with_acquisition, DEVICE_CFG)
    total_play_time = get_total_duration(schedule)

    portclock_map = qb.generate_port_clock_to_device_map(HARDWARE_MAPPING)

    device_compilers = qb._construct_compiler_objects(
        total_play_time=total_play_time,
        mapping=HARDWARE_MAPPING,
    )
    qb._assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=device_compilers,
        portclock_mapping=portclock_map,
    )

    lo_compilers = qb.generate_ext_local_oscillators(total_play_time, HARDWARE_MAPPING)
    qb._assign_frequencies(
        device_compilers,
        lo_compilers,
        hw_mapping=HARDWARE_MAPPING,
        portclock_mapping=portclock_map,
        schedule_resources=schedule.resources,
    )
    qcm = device_compilers["qcm0"]
    qrm = device_compilers["qrm0"]

    qcm_if = qcm.sequencers["seq0"].settings.modulation_freq
    qrm_if = qrm.sequencers["seq0"].settings.modulation_freq

    lo0_freq = lo_compilers["lo0"].frequency
    lo1_freq = lo_compilers["lo1"].frequency

    qcm_rf = schedule.resources["q0.01"].data["freq"]
    qrm_rf = schedule.resources["q0.ro"].data["freq"]
    assert qcm_rf == lo0_freq + qcm_if
    assert qrm_rf == lo1_freq + qrm_if


def test_assign_frequencies_unused_lo(pulse_only_schedule):
    schedule = device_compile(pulse_only_schedule, DEVICE_CFG)
    total_play_time = get_total_duration(schedule)

    portclock_map = qb.generate_port_clock_to_device_map(HARDWARE_MAPPING)

    device_compilers = qb._construct_compiler_objects(
        total_play_time=total_play_time,
        mapping=HARDWARE_MAPPING,
    )
    qb._assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=device_compilers,
        portclock_mapping=portclock_map,
    )

    lo_compilers = qb.generate_ext_local_oscillators(total_play_time, HARDWARE_MAPPING)
    assert len(lo_compilers) == 3
    qb._assign_frequencies(
        device_compilers,
        lo_compilers,
        hw_mapping=HARDWARE_MAPPING,
        portclock_mapping=portclock_map,
        schedule_resources=schedule.resources,
    )
    qcm = device_compilers["qcm0"]

    qcm_if = qcm.sequencers["seq0"].settings.modulation_freq

    lo0_freq = lo_compilers["lo0"].frequency

    qcm_rf = schedule.resources["q0.01"].data["freq"]
    assert qcm_rf == lo0_freq + qcm_if
    assert len(lo_compilers) == 1
