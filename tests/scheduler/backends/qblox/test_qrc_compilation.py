# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the QRC."""

import pytest

from quantify_scheduler.backends.graph_compilation import (
    SerialCompiler,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations.gate_library import Measure, Reset
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    SquarePulse,
)
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import Schedule


@pytest.mark.parametrize("element_names", [["q8"]])
def test_simple_qrc_schedule_compilation_end_to_end(
    qblox_hardware_config_transmon,
    element_names,
    mock_setup_basic_transmon_elements,
    assert_equal_q1asm,
):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(SquarePulse(amp=0.1, duration=40e-9, port="q8:mw", clock="q8.01"))
    schedule.add(Measure("q8", acq_index=0))

    schedule.add_resource(ClockResource("q8.01", 5e9))
    schedule.add_resource(ClockResource("q8.ro", 6e9))

    mock_setup = mock_setup_basic_transmon_elements
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_transmon)
    q8 = mock_setup["q8"]
    q8.measure.acq_delay(1e-6)
    q8.measure.acq_channel(1)
    q8.measure.integration_time(5e-6)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    compiled_seq0 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module14"][
        "sequencers"
    ]["seq0"]
    compiled_seq1 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module14"][
        "sequencers"
    ]["seq1"]

    assert_equal_q1asm(
        compiled_seq0.sequence["program"],
        """ set_mrk 0
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 set_awg_gain 3277,0 # setting gain for SquarePulse
 play 0,0,4 # play SquarePulse (40 ns)
 wait 6036 # auto generated wait (6036 ns)
 loop R0,@start
 stop
""",
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module14"]["sequencers"][
            "seq1"
        ].sequence["program"],
        """ set_mrk 0
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 wait 40 # auto generated wait (40 ns)
 reset_ph
 set_awg_gain 8192,0 # setting gain for SquarePulse
 play 0,0,4 # play SquarePulse (40 ns)
 wait 996 # auto generated wait (996 ns)
 acquire 0,0,4
 wait 4996 # auto generated wait (4996 ns)
 loop R0,@start
 stop
""",
    )

    assert compiled_seq0.channel_name == "complex_output_5"
    assert compiled_seq0.connected_output_indices == (10, 11)
    assert compiled_seq0.connected_input_indices == ()
    assert compiled_seq0.modulation_freq == 5e7

    assert compiled_seq1.channel_name == "complex_input_1"
    assert compiled_seq1.connected_output_indices == ()
    assert compiled_seq1.connected_input_indices == (2, 3)
    assert compiled_seq1.modulation_freq == 5e7


@pytest.mark.parametrize("element_names", [["q8"]])
def test_qrc_markers(
    qblox_hardware_config_transmon,
    element_names,
    mock_setup_basic_transmon_elements,
    assert_equal_q1asm,
):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(SquarePulse(amp=0.1, duration=40e-9, port="q8:mw", clock="q8.01"))
    schedule.add(MarkerPulse(duration=16e-9, port="q8:switch"))
    schedule.add(IdlePulse(16e-9))

    schedule.add_resource(ClockResource("q8.01", 5e9))

    mock_setup = mock_setup_basic_transmon_elements
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_transmon)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    compiled_seq0 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module14"][
        "sequencers"
    ]["seq0"]
    compiled_seq1 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module14"][
        "sequencers"
    ]["seq1"]

    assert_equal_q1asm(
        compiled_seq0.sequence["program"],
        """ set_mrk 0
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 set_awg_gain 3277,0 # setting gain for SquarePulse
 play 0,0,4 # play SquarePulse (40 ns)
 wait 68 # auto generated wait (68 ns)
 loop R0,@start
 stop
""",
    )

    assert_equal_q1asm(
        compiled_seq1.sequence["program"],
        """ set_mrk 0
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 wait 40 # auto generated wait (40 ns)
 set_mrk 1 # set markers to 1
 upd_param 4
 wait 12 # auto generated wait (12 ns)
 set_mrk 0 # set markers to 0
 upd_param 4
 wait 12 # auto generated wait (12 ns)
 loop R0,@start
 stop
""",
    )
