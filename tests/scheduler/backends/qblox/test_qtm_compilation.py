# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the QTM."""
import re
from unittest.mock import Mock

import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.qblox.compiler_container import CompilerContainer
from quantify_scheduler.backends.qblox.enums import TimetagTraceType
from quantify_scheduler.backends.qblox.instrument_compilers import QTMCompiler
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler
from quantify_scheduler.backends.qblox_backend import (
    ChannelPath,
    QbloxHardwareCompilationConfig,
    _SequencerCompilationConfig,
)
from quantify_scheduler.backends.types.common import ModulationFrequencies
from quantify_scheduler.backends.types.qblox import (
    AnalogSequencerSettings,
    BoundedParameter,
    DigitalChannelDescription,
    OpInfo,
    SequencerOptions,
    SequencerSettings,
    StaticAnalogModuleProperties,
    StaticTimetagModuleProperties,
    TimetagSequencerSettings,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.enums import BinMode, TimeRef, TimeSource
from quantify_scheduler.operations.acquisition_library import (
    SSBIntegrationComplex,
    Timetag,
    TimetagTrace,
    Trace,
    TriggerCount,
)
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    SquarePulse,
    Timestamp,
)
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.schemas.examples import utils

EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER = utils.load_json_example_scheme(
    "qblox_hardware_config_nv_center.json"
)


def test_generate_qasm_empty_program_qtm(assert_equal_q1asm):
    mod = Mock()
    mod.configure_mock(max_number_of_instructions=100)
    sequencer_cfg = _SequencerCompilationConfig(
        sequencer_options=SequencerOptions(),
        hardware_description=DigitalChannelDescription(),
        portclock="q1:mw-q1.01",
        channel_name="digital_output_1",
        channel_name_measure=None,
        latency_correction=0,
        distortion_correction=None,
        lo_name=None,
        modulation_frequencies=ModulationFrequencies.model_validate(
            {"lo_freq": None, "interm_freq": 50e6}
        ),
        mixer_corrections=None,
    )
    static_hw_properties = StaticTimetagModuleProperties(
        instrument_type="QTM",
        max_sequencers=8,
    )
    component = TimetagSequencerCompiler(
        parent=mod,
        index=0,
        static_hw_properties=static_hw_properties,
        sequencer_cfg=sequencer_cfg,
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


def test_get_compiler_container(create_schedule_with_pulse_info):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    5: {
                        "instrument_type": "QTM",
                    }
                },
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module5.digital_output_0", "qe1:switch"),
                ("cluster0.module5.digital_input_4", "qe1:optical_readout"),
            ]
        },
    }

    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg)

    schedule = Schedule("Test")
    schedule.add(SquarePulse(amp=0.5, duration=40e-9, port="qe1:switch", clock="digital"))
    schedule.add(SSBIntegrationComplex(port="qe1:optical_readout", clock="qe1.ge0", duration=5e-6))
    schedule = create_schedule_with_pulse_info(schedule)

    container = CompilerContainer.from_hardware_cfg(
        schedule=schedule,
        hardware_cfg=hardware_cfg,
    )

    assert isinstance(
        container.instrument_compilers["cluster0"].instrument_compilers[  # type: ignore
            "cluster0_module5"
        ],
        QTMCompiler,
    )
    module_compilation_config = (
        container.instrument_compilers["cluster0"]
        .instrument_compilers["cluster0_module5"]  # type: ignore
        .instrument_cfg
    )

    assert module_compilation_config.hardware_description.instrument_type == "QTM"
    assert module_compilation_config.hardware_description.sequence_to_file is False
    assert module_compilation_config.hardware_options.model_dump(exclude_unset=True) == {}
    assert module_compilation_config.portclock_to_path == {
        ("qe1:switch-digital"): ChannelPath.from_path("cluster0.module5.digital_output_0"),
        ("qe1:optical_readout-qe1.ge0"): ChannelPath.from_path("cluster0.module5.digital_input_4"),
    }


def test_construct_sequencer_compilers():
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {1: {"instrument_type": "QTM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_0", "q0:switch"),
                ("cluster0.module1.digital_input_1", "q0:readout"),
            ]
        },
    }
    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg)
    compilation_configs = hardware_cfg._extract_instrument_compilation_configs(
        {("q0:switch", "digital"), ("q0:readout", "digital")}
    )
    module_configs = compilation_configs["cluster0"]._extract_module_compilation_configs()

    test_module = QTMCompiler(
        name="cluster0_module1",
        total_play_time=100e-9,
        instrument_cfg=module_configs[1],
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

    schedule.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    schedule.add(IdlePulse(duration=4e-9))

    quantum_device = QuantumDevice(name="quantum_device")

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
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
    inner.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    inner.add(IdlePulse(duration=4e-9))
    schedule.add(LoopOperation(body=inner, repetitions=2))

    quantum_device = QuantumDevice(name="quantum_device")

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
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
        SquarePulse(amp=0.5, duration=40e-9, port="qe1:switch", clock="digital"),
        SSBIntegrationComplex(port="qe1:optical_readout", clock="qe1.ge0", duration=1e-6),
    ],
)
def test_qtm_compile_unsupported_operations_raises(mock_setup_basic_nv, operation):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(operation)

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match=f"Operation info .*{operation.__class__.__name__}.* cannot be compiled for a QTM",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


@pytest.mark.parametrize("repetitions", [1, 10])
def test_trigger_count_acq_qtm_compilation(mock_setup_basic_nv, repetitions, assert_equal_q1asm):
    schedule = Schedule(name="Test", repetitions=repetitions)

    tg = schedule.add(TriggerCount(port="qe1:optical_readout", clock="qe1.ge0", duration=100e-9))
    schedule.add(
        MarkerPulse(duration=4e-9, port="qe1:switch"),
        rel_time=0,
        ref_op=tg,
        ref_pt="start",
    )
    for _ in range(3):
        schedule.add(MarkerPulse(duration=4e-9, port="qe1:switch"), rel_time=16e-9)

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["acquisitions"]["0"]["num_bins"]
        == repetitions
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
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
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
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


@pytest.mark.parametrize("bin_mode", [BinMode.APPEND, BinMode.AVERAGE])
def test_timetag_acq_compilation(mock_setup_basic_nv, assert_equal_q1asm, bin_mode):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(Timestamp(port="qe1:optical_readout", clock="qe1.ge0"))
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            bin_mode=bin_mode,
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.TIMESTAMP,
        ),
        rel_time=100e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "time_source"
        ]
        == TimeSource.FIRST
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "time_ref"
        ]
        == TimeRef.TIMESTAMP
    )

    if bin_mode == BinMode.APPEND:
        append_mode_init_str = "move 0,R0 # Initialize acquisition bin_idx for ch0"
        append_mode_update_str = "add R0,1,R0 # Increment bin_idx for ch0 by 1"
        bin_idx = "R0"
        fine_delay_init_str = "move 0,R10"
        fine_delay = "R10"
        loop_reg = 1
    else:
        append_mode_init_str = ""
        append_mode_update_str = ""
        bin_idx = "0"
        fine_delay_init_str = ""
        fine_delay = "0"
        loop_reg = 0
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
        f""" wait_sync 4
 upd_param 4
 {append_mode_init_str}
 wait 4 # latency correction of 4 + 0 ns
 move 1,R{loop_reg} # iterator for loop with label start
start:
 wait 4
 set_time_ref
 upd_param 4
 wait 96 # auto generated wait (96 ns)
 {fine_delay_init_str}
 acquire_timetags 0,{bin_idx},1,{fine_delay},4 # Enable timetag acquisition of acq_channel:0
 wait 92 # auto generated wait (92 ns)
 acquire_timetags 0,{bin_idx},0,{fine_delay},4 # Disable timetag acquisition of acq_channel:0
 {append_mode_update_str}
 loop R{loop_reg},@start
 stop
""",
    )


def test_trace_acq_qtm_compilation(mock_setup_basic_nv, assert_equal_q1asm):
    schedule = Schedule(name="Test")

    tg = schedule.add(
        Trace(
            duration=16e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            bin_mode=BinMode.FIRST,
        )
    )
    schedule.add(
        MarkerPulse(duration=4e-9, port="qe1:switch"),
        rel_time=0,
        ref_op=tg,
        ref_pt="start",
    )
    for _ in range(3):
        schedule.add(MarkerPulse(duration=4e-9, port="qe1:switch"), rel_time=16e-9)

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["acquisitions"]["0"]["num_bins"]
        == 1
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "scope_trace_type"
        ]
        == TimetagTraceType.SCOPE
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 wait 4
 set_scope_en 1
 acquire_timetags 0,0,1,0,4 # Enable timetag acquisition of acq_channel:0, bin_mode:average
 wait 15992 # auto generated wait (15992 ns)
 acquire_timetags 0,0,0,0,4 # Disable timetag acquisition of acq_channel:0, bin_mode:average
 set_scope_en 0
 loop R0,@start
 stop
""",
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
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
 wait 15932 # auto generated wait (15932 ns)
 loop R0,@start
 stop
""",
    )


def test_timetagtrace_acq_qtm_compilation(mock_setup_basic_nv, assert_equal_q1asm):
    schedule = Schedule(name="Test")

    tg = schedule.add(
        TimetagTrace(
            duration=16e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
        )
    )
    schedule.add(
        MarkerPulse(duration=4e-9, port="qe1:switch"),
        rel_time=0,
        ref_op=tg,
        ref_pt="start",
    )
    for _ in range(3):
        schedule.add(MarkerPulse(duration=4e-9, port="qe1:switch"), rel_time=16e-9)

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["acquisitions"]["0"]["num_bins"]
        == 1
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "scope_trace_type"
        ]
        == TimetagTraceType.TIMETAG
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 move 0,R0 # Initialize acquisition bin_idx for ch0
 wait 4 # latency correction of 4 + 0 ns
 move 1,R1 # iterator for loop with label start
start:
 wait 4
 set_scope_en 1
 move 0,R10
 acquire_timetags 0,R0,1,R10,4 # Enable timetag acquisition of acq_channel:0, store in bin:R0
 wait 15992 # auto generated wait (15992 ns)
 acquire_timetags 0,R0,0,R10,4 # Disable timetag acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 set_scope_en 0
 loop R1,@start
 stop
""",
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
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
 wait 15932 # auto generated wait (15932 ns)
 loop R0,@start
 stop
""",
    )


def test_timetag_different_source(mock_setup_basic_nv):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(Timestamp(port="qe1:optical_readout", clock="qe1.ge0"))
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.TIMESTAMP,
        ),
        rel_time=100e-9,
    )
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.SECOND,
            time_ref=TimeRef.TIMESTAMP,
        ),
        rel_time=100e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="time_source must be the same for all acquisitions on a port-clock combination.",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


def test_timetag_different_ref(mock_setup_basic_nv):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(Timestamp(port="qe1:optical_readout", clock="qe1.ge0"))
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.TIMESTAMP,
        ),
        rel_time=100e-9,
    )
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.END,
        ),
        rel_time=100e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="time_ref must be the same for all acquisitions on a port-clock combination.",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


def test_multiple_trace_acq(mock_setup_basic_nv, assert_equal_q1asm):
    schedule = Schedule(name="Test")

    schedule.add(
        Trace(
            duration=16e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            acq_index=0,
            bin_mode=BinMode.FIRST,
        )
    )
    schedule.add(
        Trace(
            duration=16e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            acq_index=1,
            bin_mode=BinMode.FIRST,
        ),
        rel_time=20e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["acquisitions"]["0"]["num_bins"]
        == 2
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "scope_trace_type"
        ]
        == TimetagTraceType.SCOPE
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 wait 4
 set_scope_en 1
 acquire_timetags 0,0,1,0,4 # Enable timetag acquisition of acq_channel:0, bin_mode:average
 wait 15992 # auto generated wait (15992 ns)
 acquire_timetags 0,0,0,0,4 # Disable timetag acquisition of acq_channel:0, bin_mode:average
 set_scope_en 0
 wait 20 # auto generated wait (20 ns)
 set_scope_en 1
 acquire_timetags 0,1,1,0,4 # Enable timetag acquisition of acq_channel:0, bin_mode:average
 wait 15992 # auto generated wait (15992 ns)
 acquire_timetags 0,1,0,0,4 # Disable timetag acquisition of acq_channel:0, bin_mode:average
 set_scope_en 0
 loop R0,@start
 stop
""",
    )


def test_multiple_timetagtrace_acq(mock_setup_basic_nv, assert_equal_q1asm):
    schedule = Schedule(name="Test")

    schedule.add(
        TimetagTrace(
            duration=16e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            acq_index=0,
        )
    )
    schedule.add(
        TimetagTrace(duration=16e-6, port="qe1:optical_readout", clock="qe1.ge0", acq_index=1),
        rel_time=20e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["acquisitions"]["0"]["num_bins"]
        == 2
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "scope_trace_type"
        ]
        == TimetagTraceType.TIMETAG
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 move 0,R0 # Initialize acquisition bin_idx for ch0
 wait 4 # latency correction of 4 + 0 ns
 move 1,R1 # iterator for loop with label start
start:
 wait 4
 set_scope_en 1
 move 0,R10
 acquire_timetags 0,R0,1,R10,4 # Enable timetag acquisition of acq_channel:0, store in bin:R0
 wait 15992 # auto generated wait (15992 ns)
 acquire_timetags 0,R0,0,R10,4 # Disable timetag acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 set_scope_en 0
 wait 20 # auto generated wait (20 ns)
 set_scope_en 1
 move 0,R10
 acquire_timetags 0,R0,1,R10,4 # Enable timetag acquisition of acq_channel:0, store in bin:R0
 wait 15992 # auto generated wait (15992 ns)
 acquire_timetags 0,R0,0,R10,4 # Disable timetag acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 set_scope_en 0
 loop R1,@start
 stop
""",
    )


def test_multiple_acq_channel_timetagtrace(mock_setup_basic_nv, assert_equal_q1asm):
    schedule = Schedule(name="Test")

    schedule.add(
        TimetagTrace(
            duration=16e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
        )
    )
    schedule.add(
        TimetagTrace(duration=16e-6, port="qe1:optical_readout", clock="qe1.ge0", acq_channel=1),
        rel_time=20e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Only one acquisition channel per port-clock can be specified, if the "
            "TimetagTrace acquisition protocol is used.\nAcquisition channels [0, 1] "
            "were found on port-clock qe1:optical_readout-qe1.ge0."
        ),
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


def test_timestamp_arg_but_no_operation(mock_setup_basic_nv):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.TIMESTAMP,
        ),
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.warns(
        UserWarning,
        match="A Timetag acquisition was scheduled with argument 'time_ref="
        "TimeRef.TIMESTAMP' on port 'qe1:optical_readout' and clock 'qe1.ge0', but no "
        "Timestamp operation was found with the same port and clock.",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


def test_timestamp_operation_but_no_arg(mock_setup_basic_nv):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(Timestamp(port="qe1:optical_readout", clock="qe1.ge0"))
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.START,
        ),
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.warns(
        UserWarning,
        match="A Timestamp operation was found on port 'qe1:optical_readout' and clock "
        "'qe1.ge0', but no Timetag acquisition was scheduled with argument 'time_ref="
        "TimeRef.TIMESTAMP'.",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


def test_marker_pulse_fine_delay(assert_equal_q1asm):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(
        MarkerPulse(
            duration=40e-9,
            fine_start_delay=391e-12,
            fine_end_delay=781e-12,
            port="qe1:switch",
        )
    )
    schedule.add(IdlePulse(duration=4e-9))

    quantum_device = QuantumDevice(name="quantum_device")

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq0"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 wait 4
 set_digital 1,1,50 # Set output high
 upd_param 4
 wait 36 # auto generated wait (36 ns)
 set_digital 0,1,100 # Set output low
 upd_param 4
 loop R0,@start
 stop
""",
    )


def test_trigger_count_fine_delay(mock_setup_basic_nv, assert_equal_q1asm):
    schedule = Schedule(name="Test")

    schedule.add(
        TriggerCount(
            port="qe1:optical_readout",
            clock="qe1.ge0",
            duration=100e-9,
            fine_start_delay=3125e-12,
            fine_end_delay=6250e-12,
        )
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 move 0,R0 # Initialize acquisition bin_idx for ch0
 wait 4 # latency correction of 4 + 0 ns
 move 1,R1 # iterator for loop with label start
start:
 wait 4
 move 400,R10
 move 800,R11
 acquire_timetags 0,R0,1,R10,4 # Enable TTL acquisition of acq_channel:0, store in bin:R0
 wait 92 # auto generated wait (92 ns)
 acquire_timetags 0,R0,0,R11,4 # Disable TTL acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 loop R1,@start
 stop
""",
    )


def test_timetag_fine_delay(mock_setup_basic_nv, assert_equal_q1asm):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(Timestamp(port="qe1:optical_readout", clock="qe1.ge0"))
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            bin_mode=BinMode.AVERAGE,
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.TIMESTAMP,
            fine_start_delay=1750e-12,
            fine_end_delay=0,
        ),
        rel_time=100e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"]["seq4"][
            "sequence"
        ]["program"],
        """ wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 wait 4
 set_time_ref
 upd_param 4
 wait 96 # auto generated wait (96 ns)
 acquire_timetags 0,0,1,224,4 # Enable timetag acquisition of acq_channel:0
 wait 92 # auto generated wait (92 ns)
 acquire_timetags 0,0,0,0,4 # Disable timetag acquisition of acq_channel:0
 loop R0,@start
 stop
""",
    )


def test_marker_pulse_fine_delay_error_within_op():
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(
        MarkerPulse(
            duration=6e-9,
            fine_start_delay=391e-12,
            fine_end_delay=781e-12,
            port="qe1:switch",
        )
    )
    schedule.add(IdlePulse(duration=4e-9))

    quantum_device = QuantumDevice(name="quantum_device")

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="there must be at least 7ns between the start and "
        "end of the operation including the fine delay, OR the time "
        "between the start and end must be an integer number of "
        "nanoseconds.",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


def test_marker_pulse_fine_delay_error_between_op():
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(
        MarkerPulse(
            duration=40e-9,
            fine_start_delay=0,
            fine_end_delay=80e-12,
            port="qe1:switch",
        )
    )
    schedule.add(
        MarkerPulse(
            duration=40e-9,
            fine_start_delay=40e-12,
            fine_end_delay=0,
            port="qe1:switch",
        ),
        rel_time=6e-9,
    )
    schedule.add(IdlePulse(duration=4e-9))

    quantum_device = QuantumDevice(name="quantum_device")

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="there must be at least 7ns between the end of the "
        "previous operation and the start of this one including the "
        "fine delay, OR the time between the end and start must be an "
        "integer number of nanoseconds.",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


def test_timetag_fine_delay_error_between_op(mock_setup_basic_nv):
    schedule = Schedule(name="Test", repetitions=1)

    schedule.add(Timestamp(port="qe1:optical_readout", clock="qe1.ge0"))
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            bin_mode=BinMode.AVERAGE,
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.TIMESTAMP,
            fine_start_delay=1750e-12,
            fine_end_delay=0,
        ),
        rel_time=100e-9,
    )
    schedule.add(
        Timetag(
            duration=100e-9,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            bin_mode=BinMode.AVERAGE,
            time_source=TimeSource.FIRST,
            time_ref=TimeRef.TIMESTAMP,
            fine_start_delay=50e-12,
            fine_end_delay=0,
        ),
        rel_time=4e-9,
    )

    quantum_device = mock_setup_basic_nv["quantum_device"]

    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="there must be at least 7ns between the end of the "
        "previous operation and the start of this one including the "
        "fine delay, OR the time between the end and start must be an "
        "integer number of nanoseconds.",
    ):
        _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())
