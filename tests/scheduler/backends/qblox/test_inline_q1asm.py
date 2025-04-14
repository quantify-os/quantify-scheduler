# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the inline Q1ASM operation."""

from __future__ import annotations

import numpy as np
import pytest

from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.backends.qblox import constants, helpers
from quantify_scheduler.backends.qblox.operation_handling import (
    q1asm_injection_strategy,
)
from quantify_scheduler.backends.qblox.operations.inline_q1asm import InlineQ1ASM
from quantify_scheduler.operations import SquarePulse
from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.schedules.schedule import Schedule


# ============== QUANTUM DEVICE FIXTURES =============== #
@pytest.fixture
def basic_quantum_device_inline_tests(
    mock_quantum_device_basic_transmon_qblox_hardware,
):
    quantum_device = mock_quantum_device_basic_transmon_qblox_hardware
    config_t = "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig"
    q0 = quantum_device.get_element("q0")
    q3 = quantum_device.get_element("q3")
    readout_lo_freq = (q0.clock_freqs.readout() + q3.clock_freqs.readout()) / 2
    quantum_device.hardware_config(
        {
            "config_type": config_t,
            "hardware_description": {
                "cluster0": {
                    "instrument_type": "Cluster",
                    "modules": {
                        2: {"instrument_type": "QCM"},
                        3: {"instrument_type": "QCM"},
                        4: {"instrument_type": "QCM_RF"},
                        5: {"instrument_type": "QCM_RF"},
                        6: {"instrument_type": "QCM_RF"},
                        7: {"instrument_type": "QRM_RF"},
                    },
                    "sequence_to_file": False,
                    "ref": "internal",
                }
            },
            "hardware_options": {
                "modulation_frequencies": {
                    "q0:res-q0.ro": {"lo_freq": readout_lo_freq},
                    "q3:res-q3.ro": {"lo_freq": readout_lo_freq},
                    "q1:mw-q1.01": {"interm_freq": 80e6},
                }
            },
            "connectivity": {
                "graph": [
                    ("cluster0.module2.real_output_0", "q0:fl"),
                    ("cluster0.module2.real_output_1", "q1:fl"),
                    ("cluster0.module2.real_output_2", "q2:fl"),
                    ("cluster0.module2.real_output_3", "q3:fl"),
                    ("cluster0.module3.real_output_0", "q4:fl"),
                    ("cluster0.module4.complex_output_0", "q0:mw"),
                    ("cluster0.module4.complex_output_1", "q1:mw"),
                    ("cluster0.module5.complex_output_0", "q2:mw"),
                    ("cluster0.module5.complex_output_1", "q3:mw"),
                    ("cluster0.module6.complex_output_0", "q4:mw"),
                    ("cluster0.module7.complex_output_0", "q0:res"),
                    ("cluster0.module7.complex_output_0", "q1:res"),
                    ("cluster0.module7.complex_output_0", "q2:res"),
                    ("cluster0.module7.complex_output_0", "q3:res"),
                    ("cluster0.module7.complex_output_0", "q4:res"),
                ]
            },
        }
    )
    return quantum_device


# ================ TESTS ================= #
def test___str___inline_q1asm():
    block_wf = [1] * 10
    inline_q1asm_operation = InlineQ1ASM(
        program=f"play 22, 22, {len(block_wf)}",
        port="q0:mw",
        clock="q0.12",
        duration=200 / constants.SAMPLING_RATE,
        waveforms={
            "block": {
                "data": block_wf,
                "index": 22,
            },
        },
    )
    assert str(inline_q1asm_operation) == (
        "InlineQ1ASM("
        f"program='play 22, 22, {len(block_wf)}', "
        "duration=2e-07, "
        "port='q0:mw', "
        "clock='q0.12', "
        "waveforms={'block': {'data': " + f"{block_wf}" + ", 'index': 22}}, "
        "safe_labels=True)"
    )


def test_non_zero_start_time(basic_quantum_device_inline_tests):
    schedule = Schedule("sched")
    op1 = SquarePulse(amp=0.5, port="q0:res", clock="q0.ro", duration=4000)
    op2 = InlineQ1ASM(program="wait 200", port="q0:res", clock="q0.ro", duration=200)
    schedule.add(op1, rel_time=50)
    schedule.add(op2, rel_time=20)
    compiler = SerialCompiler("compiler", basic_quantum_device_inline_tests)
    compiled_schedule = compiler.compile(schedule)
    assert compiled_schedule.duration == 4220


def test_waveform_with_same_name(basic_quantum_device_inline_tests):
    schedule = Schedule("same indices")
    op1 = InlineQ1ASM(
        program="play 42, 42, 200",
        port="q0:res",
        clock="q0.ro",
        duration=200,
        waveforms={"w1": {"data": [1] * 10, "index": 42}},
    )
    op2 = InlineQ1ASM(
        program="play 41, 41, 100",
        port="q0:res",
        clock="q0.ro",
        duration=200,
        waveforms={"w1": {"data": [2] * 10, "index": 41}},
    )
    schedule.add(op1)
    schedule.add(op2)
    compiler = SerialCompiler("compiler", basic_quantum_device_inline_tests)
    with pytest.raises(RuntimeError, match="Duplicate waveform name"):
        compiler.compile(schedule)


def test_inline_waveforms_do_not_overlap_indices_with_normal_waveforms(
    basic_quantum_device_inline_tests,
):
    schedule = Schedule("same indices")
    op1 = SquarePulse(amp=0.5, port="q0:res", clock="q0.ro", duration=200)
    op2 = InlineQ1ASM(
        program="play 0, 0, 200",
        port="q0:res",
        clock="q0.ro",
        duration=200,
        waveforms={"w1": {"data": [2] * 10, "index": 0}},
    )
    schedule.add(op1)
    schedule.add(op2)
    compiler = SerialCompiler("compiler", basic_quantum_device_inline_tests)
    c_sched = compiler.compile(schedule)
    sequencer_data = c_sched.compiled_instructions["cluster0"]["cluster0_module7"]["sequencers"]
    prog = sequencer_data["seq0"].sequence["program"]
    assert "play 0,0,200 # [inline]" in prog
    assert "play 1,1,4 # play SquarePulse (4 ns)" in prog


def test_waveform_with_same_indices(basic_quantum_device_inline_tests):
    schedule = Schedule("same indices")
    op1 = InlineQ1ASM(
        program="play 42, 42, 200",
        port="q0:res",
        clock="q0.ro",
        duration=200,
        waveforms={"w1": {"data": [1] * 10, "index": 42}},
    )
    op2 = InlineQ1ASM(
        program="play 42, 42, 100",
        port="q0:res",
        clock="q0.ro",
        duration=200,
        waveforms={"w2": {"data": [2] * 10, "index": 42}},
    )
    schedule.add(op1)
    schedule.add(op2)
    compiler = SerialCompiler("compiler", basic_quantum_device_inline_tests)
    with pytest.raises(RuntimeError, match="Duplicate index"):
        compiler.compile(schedule)


# for this test q0 and q3 are connected to the same readout line (two sequencers used), so
# q3:res will use the second sequencer of that module. All other
# programs will be assigned to the first sequencer of each respective module
@pytest.mark.parametrize("safe_labels", [True, False])
@pytest.mark.parametrize(
    "connectivity",
    [
        ("q0:fl", "cl0.baseband", "cluster0_module2", "seq0", 8),
        ("q1:mw", "q1.01", "cluster0_module4", "seq0", 8),
        ("q0:res", "q0.ro", "cluster0_module7", "seq0", 8),
        ("q3:res", "q3.ro", "cluster0_module7", "seq1", 8),
    ],
)
def test_inline_q1asm_transmon(
    safe_labels,
    connectivity,
    basic_quantum_device_inline_tests,
    assert_equal_q1asm,
):
    """Test that a basic inline example works on the transmon backend"""
    port, clock, module, seq, inj_start_line_idx = connectivity
    compiler = SerialCompiler("compiler", basic_quantum_device_inline_tests)

    sched = Schedule("basic_inline_example", 1)
    sched.add(
        InlineQ1ASM(
            program="""
            move 0, R200 # r200
            abc: move 0, R11 # r11
            add R11, R11, R11 # appel
            upd_param 10000
            # this is just a comment R11
            jmp @abc
        """,
            port=port,
            clock=clock,
            duration=10e-6,
            safe_labels=safe_labels,
        )
    )
    sched.add(Measure("q0"))
    c_sched = compiler.compile(sched)
    sequencer_data = c_sched.compiled_instructions["cluster0"][module]["sequencers"]
    prog = sequencer_data[seq].sequence["program"]

    # extract the subprogram where the injection should be
    prog = "\n".join(prog.splitlines()[inj_start_line_idx : inj_start_line_idx + 6])

    r200, r11 = "R1", "R2"
    abc = "inj8_abc" if safe_labels else "abc"

    assert_equal_q1asm(
        prog,
        f"""
                move 0,{r200} # [inline] r200
{abc}:          move 0,{r11} # [inline] r11
                add {r11},{r11},{r11} # [inline] appel
                upd_param 10000 # [inline]
                # [inline] this is just a comment R11
                jmp @{abc} # [inline]
        """,
    )


@pytest.mark.parametrize("safe_labels", [True, False])
@pytest.mark.parametrize(
    "connectivity",
    [
        ("qe0:optical_readout", "qe0.ge0", "cluster0_module4", "seq0", 9),
        ("qe1:switch", "digital", "cluster0_module5", "seq0", 6),
    ],
)
def test_inline_q1asm_nv(
    safe_labels,
    connectivity,
    mock_setup_basic_nv_qblox_hardware,
    assert_equal_q1asm,
):
    """Test that the basic inline example works on NV backend"""
    port, clock, module, seq, inj_start_line_idx = connectivity
    compiler = SerialCompiler("compiler", mock_setup_basic_nv_qblox_hardware["quantum_device"])

    sched = Schedule("basic_inline_example", 1)
    sched.add(
        InlineQ1ASM(
            program="""
            move 0, R200 #R200
            abc: move 0, R11 #R11
            add R11, R11, R11 # appel
            upd_param 10000
            # this is just a comment R11
            jmp @abc
        """,
            port=port,
            clock=clock,
            duration=10e-6,
            safe_labels=safe_labels,
        )
    )
    sched.add(Measure("qe0"))
    c_sched = compiler.compile(sched)
    sequencer_data = c_sched.compiled_instructions["cluster0"][module]["sequencers"]
    prog = sequencer_data[seq].sequence["program"]

    # extract the subprogram where the injection should be
    prog = "\n".join(prog.splitlines()[inj_start_line_idx : inj_start_line_idx + 6])

    # TODO, figure out a better way to distinguish between the two connectivities
    r200 = "R1" if inj_start_line_idx == 6 else "R2"
    r11 = "R2" if inj_start_line_idx == 6 else "R3"
    abc = f"inj{inj_start_line_idx}_abc" if safe_labels else "abc"

    assert_equal_q1asm(
        prog,
        f"""
                move 0,{r200} # [inline] R200
{abc}:          move 0,{r11} # [inline] R11
                add {r11},{r11},{r11} # [inline] appel
                upd_param 10000 # [inline]
                # [inline] this is just a comment R11
                jmp @{abc} # [inline]
        """,
    )


@pytest.mark.parametrize("waveform_length", [32, 2000, constants.MAX_SAMPLE_SIZE_WAVEFORMS])
@pytest.mark.parametrize("waveform_index", [0, 1])
@pytest.mark.parametrize(
    "connectivity",
    [
        ("q0:fl", "cl0.baseband", "cluster0_module2", "seq0", 8),
        ("q1:mw", "q1.01", "cluster0_module4", "seq0", 8),
        ("q3:res", "q3.ro", "cluster0_module7", "seq1", 8),
    ],
)
def test_inline_play_block_waveform(
    waveform_length,
    waveform_index,
    connectivity,
    basic_quantum_device_inline_tests,
    assert_equal_q1asm,
):
    """Test that we can play waveforms with inline q1asm operations."""
    port, clock, module, seq, inj_start_line_idx = connectivity
    compiler = SerialCompiler("compiler", basic_quantum_device_inline_tests)

    sched = Schedule("inline_play_block_wf")
    sched.add(
        InlineQ1ASM(
            program=f"play {waveform_index}, {waveform_index}, {waveform_length}",
            port=port,
            clock=clock,
            duration=waveform_length / constants.SAMPLING_RATE,
            waveforms={
                "block": {
                    "data": [1.0] * waveform_length,
                    "index": waveform_index,
                },
            },
        )
    )
    sched.add(Measure("q0"))
    c_sched = compiler.compile(sched)
    sequencer_data = c_sched.compiled_instructions["cluster0"][module]["sequencers"]
    waveforms = sequencer_data[seq].sequence["waveforms"]
    prog = sequencer_data[seq].sequence["program"]

    # extract the subprogram where the injection should be
    prog = "\n".join(prog.splitlines()[inj_start_line_idx : inj_start_line_idx + 1])
    assert_equal_q1asm(
        prog, f"play {waveform_index}, {waveform_index}, {waveform_length} # [inline]"
    )

    assert len(waveforms) == 1
    assert len(waveforms["block"]["data"]) == waveform_length
    assert np.all(np.asarray(waveforms["block"]["data"]) == 1.0)
    assert waveforms["block"]["index"] == waveform_index


@pytest.mark.parametrize("waveform_index", [0, 1])
def test_inline_large_play_block_waveform_raises(
    waveform_index,
    basic_quantum_device_inline_tests,
):
    """Test that error is raised when waveform memory is exceeded."""
    compiler = SerialCompiler("compiler", basic_quantum_device_inline_tests)
    waveform_length = constants.MAX_SAMPLE_SIZE_WAVEFORMS
    sched = Schedule("inline_play_block_wf")
    sched.add(
        InlineQ1ASM(
            program=f"play {waveform_index}, {waveform_index}, {waveform_length}",
            port="q0:res",
            clock="q0.ro",
            duration=waveform_length / constants.SAMPLING_RATE,
            waveforms={
                "block": {
                    "data": [1.0] * waveform_length,
                    "index": waveform_index,
                },
            },
        )
    )
    sched.add(Measure("q0"))

    q0 = basic_quantum_device_inline_tests.get_element("q0")
    ro_pulse_dur_ns = int(q0.measure.pulse_duration() * constants.SAMPLING_RATE)
    total_size = ro_pulse_dur_ns + waveform_length
    wf_overflow = (
        f"Total waveform size specified for port-clock q0:res-q0.ro is {total_size} samples,"
        f" which exceeds the sample limit of {constants.MAX_SAMPLE_SIZE_WAVEFORMS}."
    )
    with pytest.raises(RuntimeError, match=wf_overflow):
        _ = compiler.compile(sched)
