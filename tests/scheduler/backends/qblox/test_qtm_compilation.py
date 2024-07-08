# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the QTM."""
from unittest.mock import Mock

import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.qblox.compiler_container import CompilerContainer
from quantify_scheduler.backends.qblox.instrument_compilers import QTMCompiler
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler
from quantify_scheduler.backends.types.qblox import TimetagSequencerSettings
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations.acquisition_library import (
    SSBIntegrationComplex,
    TriggerCount,
)
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    SquarePulse,
)
from quantify_scheduler.schedules.schedule import Schedule


def test_generate_qasm_empty_program_qtm(assert_equal_q1asm):
    mod = Mock()
    mod.configure_mock(max_number_of_instructions=100)
    settings = TimetagSequencerSettings.initialize_from_config_dict(
        {
            "port": "p_test",
            "clock": "c_test",
        },
        channel_name="digital_output_1",
        connected_input_indices=(),
        connected_output_indices=(0,),
    )
    component = TimetagSequencerCompiler(
        parent=mod,
        index=0,
        portclock=("p_test", "c_test"),
        static_hw_properties=Mock(),
        settings=settings,
        latency_corrections={},
    )

    assert_equal_q1asm(
        component.generate_qasm_program(
            ordered_op_strategies=[],
            total_sequence_time=120e-9,
            align_qasm_fields=False,
            acq_metadata=None,
            repetitions=1,
        ),
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 wait 4
 wait 120 # auto generated wait (120 ns)
 loop R0,@start
 stop
""",
    )


def test_get_compiler_container():
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QTM",
                "digital_output_0": {
                    "portclock_configs": [
                        {
                            "port": "port1",
                            "clock": "clock1",
                        }
                    ],
                },
            },
        },
    }

    schedule = Schedule("Test")

    container = CompilerContainer.from_hardware_cfg(schedule, hardware_cfg)

    assert isinstance(
        container.instrument_compilers["cluster0"].instrument_compilers[  # type: ignore
            "cluster0_module1"
        ],
        QTMCompiler,
    )
    assert container.instrument_compilers["cluster0"].instrument_compilers[  # type: ignore
        "cluster0_module1"
    ].instrument_cfg == {
        "instrument_type": "QTM",
        "digital_output_0": {
            "portclock_configs": [{"port": "port1", "clock": "clock1"}]
        },
    }


def test_construct_sequencer_compilers():
    test_module = QTMCompiler(
        parent=Mock(),
        name="cluster0_module1",
        total_play_time=100e-9,
        instrument_cfg={
            "instrument_type": "QTM",
            "digital_output_0": {
                "portclock_configs": [
                    {
                        "port": "q0:switch",
                        "clock": "digital",
                    }
                ]
            },
            "digital_input_1": {
                "portclock_configs": [
                    {
                        "port": "q0:readout",
                        "clock": "digital",
                    }
                ]
            },
        },
    )

    test_module._op_infos = {
        ("q0:switch", "digital"): [Mock()],
        ("q0:readout", "digital"): [Mock()],
    }

    test_module._construct_all_sequencer_compilers()
    seq_keys = list(test_module.sequencers.keys())

    assert len(seq_keys) == 2
    assert isinstance(test_module.sequencers[seq_keys[0]], TimetagSequencerCompiler)
    assert isinstance(test_module.sequencers[seq_keys[1]], TimetagSequencerCompiler)


def test_simple_qtm_schedule_compilation_end_to_end(assert_equal_q1asm):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(MarkerPulse(duration=40e-9, port="q0:foo"))
    schedule.add(IdlePulse(duration=4e-9))

    quantum_device = QuantumDevice(name="quantum_device")

    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": "QTM"},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_0", "q0:foo"),
            ]
        },
    }

    quantum_device.hardware_config(hw_config)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
            "sequencers"
        ]["seq0"]["sequence"]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 wait 4
 set_digital 1,1,0 # Set output high
 upd_param 4
 wait 36 # auto generated wait (36 ns)
 set_digital 0,1,0 # Set output low
 upd_param 4
 loop R0,@start
 stop
""",
    )


def test_qtm_loop_schedule_compilation_end_to_end(assert_equal_q1asm):
    schedule = Schedule(name="Test", repetitions=1)

    inner = Schedule(name="Inner", repetitions=1)
    inner.add(MarkerPulse(duration=40e-9, port="q0:foo"))
    inner.add(IdlePulse(duration=4e-9))
    schedule.add(LoopOperation(body=inner, repetitions=2))

    quantum_device = QuantumDevice(name="quantum_device")

    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": "QTM"},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_0", "q0:foo"),
            ]
        },
    }

    quantum_device.hardware_config(hw_config)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
            "sequencers"
        ]["seq0"]["sequence"]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 wait 4
 move 2,R1 # iterator for loop with label loop6
loop6:
 set_digital 1,1,0 # Set output high
 upd_param 4
 wait 36 # auto generated wait (36 ns)
 set_digital 0,1,0 # Set output low
 upd_param 4
 loop R1,@loop6
 loop R0,@start
 stop
""",
    )


@pytest.mark.parametrize(
    "operation",
    [
        SquarePulse(amp=0.5, duration=40e-9, port="q0:foo", clock="digital"),
        SSBIntegrationComplex(port="q0:foo", clock="digital", duration=1e-6),
    ],
)
def test_qtm_compile_unsupported_operations_raises(operation):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(operation)

    quantum_device = QuantumDevice(name="quantum_device")

    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": "QTM"},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_0", "q0:foo"),
            ]
        },
    }

    quantum_device.hardware_config(hw_config)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match=f"Operation info .*{operation.__class__.__name__}.* cannot be compiled for a QTM",
    ):
        _ = compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )


@pytest.mark.parametrize("repetitions", [1, 10])
def test_trigger_count_acq_qtm_compilation(repetitions, assert_equal_q1asm):
    schedule = Schedule(name="Test", repetitions=repetitions)

    tg = schedule.add(TriggerCount(port="q0:in", clock="digital", duration=100e-9))
    schedule.add(
        MarkerPulse(duration=4e-9, port="q0:out"), rel_time=0, ref_op=tg, ref_pt="start"
    )
    for _ in range(3):
        schedule.add(MarkerPulse(duration=4e-9, port="q0:out"), rel_time=16e-9)

    quantum_device = QuantumDevice(name="quantum_device")

    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": "QTM"},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_0", "q0:out"),
                ("cluster0.module1.digital_input_4", "q0:in"),
            ]
        },
    }

    quantum_device.hardware_config(hw_config)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
            "sequencers"
        ]["seq4"]["sequence"]["acquisitions"]["0"]["num_bins"]
        == repetitions
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
            "sequencers"
        ]["seq4"]["sequence"]["program"],
        f""" wait_sync 4
 upd_param 4
 move 0,R0 # Initialize acquisition bin_idx for ch0
 wait 4 # latency correction of 4 + 0 ns
 move {repetitions},R1 # iterator for loop with label start
start:
 wait 4
 move 0,R10
 acquire_timetags 0,R0,1,R10,4 # Enable TTL acquisition of acq_channel:0, store in bin:R0
 wait 92 # auto generated wait (92 ns)
 acquire_timetags 0,R0,0,R10,4 # Disable TTL acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 loop R1,@start
 stop
""",
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
            "sequencers"
        ]["seq0"]["sequence"]["program"],
        f""" wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move {repetitions},R0 # iterator for loop with label start
start:
 wait 4
 set_digital 1,1,0
 upd_param 4
 set_digital 0,1,0
 upd_param 4
 wait 12 # auto generated wait (12 ns)
 set_digital 1,1,0
 upd_param 4
 set_digital 0,1,0
 upd_param 4
 wait 12 # auto generated wait (12 ns)
 set_digital 1,1,0
 upd_param 4
 set_digital 0,1,0
 upd_param 4
 wait 12 # auto generated wait (12 ns)
 set_digital 1,1,0
 upd_param 4
 set_digital 0,1,0
 upd_param 4
 wait 32 # auto generated wait (32 ns)
 loop R0,@start
 stop
""",
    )
