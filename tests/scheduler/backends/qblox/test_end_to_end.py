# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""End-to-end tests."""
import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    ResetClockPhase,
    SetClockFrequency,
    ShiftClockPhase,
    SquarePulse,
    VoltageOffset,
)


def test_zero_duration_parameter_operations(
    mock_setup_basic_transmon_with_standard_params,
    qblox_hardware_config_transmon,
    assert_equal_q1asm,
):
    sched = Schedule("test")
    sched.add(ResetClockPhase(clock="q0.01"))
    sched.add(SetClockFrequency(clock="q0.01", clock_freq_new=7250e6))
    sched.add(VoltageOffset(offset_path_I=0.5, offset_path_Q=0.5, port="q0:mw", clock="q0.01"))
    sched.add(
        VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="q0:mw", clock="q0.01"),
        rel_time=100e-9,
    )
    sched.add(ShiftClockPhase(phase_shift=120, clock="q0.01"), rel_time=100e-9)
    sq_pulse = sched.add(SquarePulse(amp=0.5, duration=100e-9, port="q0:mw", clock="q0.01"))
    sched.add(MarkerPulse(duration=80e-9, port="q0:switch"), ref_op=sq_pulse, ref_pt="start")

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    quantum_device.hardware_config(qblox_hardware_config_transmon)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module2"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
        """
 set_mrk 1 # set markers to 1
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 reset_ph
 set_freq 0 # set nco frequency to 0.000000e+00 Hz
 set_awg_offs 16384,16384 # setting offset for VoltageOffset
 upd_param 4
 wait 96 # auto generated wait (96 ns)
 set_awg_offs 0,0 # setting offset for VoltageOffset
 upd_param 4
 wait 96 # auto generated wait (96 ns)
 set_ph_delta 333333333 # increment nco phase by 120.00 deg
 set_awg_gain 16384,0 # setting gain for SquarePulse
 play 0,0,4 # play SquarePulse (100 ns)
 wait 96 # auto generated wait (96 ns)
 loop R0,@start
 stop
""",
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module2"]["sequencers"]["seq1"][
            "sequence"
        ]["program"],
        """
 set_mrk 3 # set markers to 3
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 wait 200 # auto generated wait (200 ns)
 set_mrk 11 # set markers to 11
 upd_param 4
 wait 76 # auto generated wait (76 ns)
 set_mrk 3 # set markers to 3
 upd_param 4
 wait 16 # auto generated wait (16 ns)
 loop R0,@start
 stop
""",
    )


def test_zero_duration_parameter_operations_with_loops(
    mock_setup_basic_transmon_with_standard_params,
    qblox_hardware_config_transmon,
    assert_equal_q1asm,
):
    sched = Schedule("test")
    sched.add(ResetClockPhase(clock="q0.01"))
    sched.add(SetClockFrequency(clock="q0.01", clock_freq_new=7250e6))

    inner = Schedule("inner")
    inner.add(ShiftClockPhase(phase_shift=120, clock="q0.01"))
    inner.add(SquarePulse(amp=0.5, duration=100e-9, port="q0:mw", clock="q0.01"))

    sched.add(LoopOperation(body=inner, repetitions=3), rel_time=4e-9)
    sched.add(VoltageOffset(offset_path_I=0.5, offset_path_Q=0.5, port="q0:mw", clock="q0.01"))
    sched.add(
        VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="q0:mw", clock="q0.01"),
        rel_time=100e-9,
    )
    sched.add(IdlePulse(4e-9))

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    quantum_device.hardware_config(qblox_hardware_config_transmon)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module2"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
        """
 set_mrk 1 # set markers to 1
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 reset_ph
 set_freq 0 # set nco frequency to 0.000000e+00 Hz
 upd_param 4
 move 3,R1 # iterator for loop with label loop11
loop11:
 set_ph_delta 333333333 # increment nco phase by 120.00 deg
 set_awg_gain 16384,0 # setting gain for SquarePulse
 play 0,0,4 # play SquarePulse (100 ns)
 wait 96 # auto generated wait (96 ns)
 loop R1,@loop11
 set_awg_offs 16384,16384 # setting offset for VoltageOffset
 upd_param 4
 wait 96 # auto generated wait (96 ns)
 set_awg_offs 0,0 # setting offset for VoltageOffset
 upd_param 4
 loop R0,@start
 stop
""",
    )


@pytest.mark.xfail(reason="Part of MarkerPulse is in the loop. Known issue #473.")
def test_marker_pulse_with_loop(
    mock_setup_basic_transmon_with_standard_params,
    qblox_hardware_config_transmon,
):
    inner = Schedule("inner")
    inner.add(SquarePulse(amp=0.5, duration=100e-9, port="q0:mw", clock="q0.01"))

    sched = Schedule("test")
    sched.add(MarkerPulse(duration=80e-9, port="q0:switch"))
    sched.add(LoopOperation(body=inner, repetitions=3), ref_pt="start", rel_time=40e-9)

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    quantum_device.hardware_config(qblox_hardware_config_transmon)

    compiler = SerialCompiler(name="compiler")
    # FIXME #473. When fixed, this exception type should be made more specific and the xfail removed
    with pytest.raises(Exception):
        _ = compiler.compile(
            sched,
            config=quantum_device.generate_compilation_config(),
        )
