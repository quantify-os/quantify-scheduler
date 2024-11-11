# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the InstrumentCompiler subclasses."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler, corrections
from quantify_scheduler.backends.qblox import compiler_container, q1asm_instructions
from quantify_scheduler.backends.qblox.analog import AnalogSequencerCompiler
from quantify_scheduler.backends.qblox.helpers import (
    LoopBegin,
    _ControlFlowReturn,
    assign_pulse_and_acq_info_to_devices,
)
from quantify_scheduler.backends.qblox.instrument_compilers import ClusterCompiler
from quantify_scheduler.backends.qblox.operation_handling.factory_analog import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    UpdateParameterStrategy,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler
from quantify_scheduler.backends.qblox_backend import (
    QbloxHardwareCompilationConfig,
    _SequencerCompilationConfig,
)
from quantify_scheduler.backends.types.common import ModulationFrequencies
from quantify_scheduler.backends.types.qblox import (
    AnalogSequencerSettings,
    BoundedParameter,
    ComplexChannelDescription,
    OpInfo,
    SequencerOptions,
    SequencerSettings,
    StaticAnalogModuleProperties,
    StaticTimetagModuleProperties,
)
from quantify_scheduler.compilation import (
    _determine_absolute_timing,
)
from quantify_scheduler.operations.acquisition_library import Trace
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    LoopOperation,
)
from quantify_scheduler.operations.gate_library import Measure, X
from quantify_scheduler.operations.pulse_library import (
    DRAGPulse,
    ResetClockPhase,
    SetClockFrequency,
    ShiftClockPhase,
    SquarePulse,
    VoltageOffset,
)
from quantify_scheduler.resources import BasebandClockResource

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.operation_handling.base import (
        IOperationStrategy,
    )
    from quantify_scheduler.operations.operation import Operation


def _assert_update_parameters_op_list(
    op_list: list[Operation],
    expected_update_parameters: dict[int, float],
    hardware_cfg_cluster,
) -> None:
    schedule = Schedule("parameter update test")
    for op in op_list:
        schedule.add(op)
    _assert_update_parameters_schedule(
        schedule,
        expected_update_parameters,
        hardware_cfg_cluster,
    )


def _assert_update_parameters_schedule(
    schedule: Schedule,
    expected_update_parameters: dict[int, float],
    hardware_cfg_cluster,
) -> None:
    schedule = _determine_absolute_timing(schedule)
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        schedule, hardware_cfg_cluster
    )
    assign_pulse_and_acq_info_to_devices(schedule, container.clusters)
    container.prepare()

    cluster0 = container.instrument_compilers["cluster0"]
    assert isinstance(cluster0, ClusterCompiler)
    op_strategies = (
        cluster0.instrument_compilers["cluster0_module1"].sequencers["seq0"].op_strategies
    )

    assert len(expected_update_parameters) == len(
        [op for op in op_strategies if isinstance(op, UpdateParameterStrategy)]
    )

    for index, op_strategy in enumerate(op_strategies):
        if index in expected_update_parameters:
            assert isinstance(op_strategy, UpdateParameterStrategy)
            assert math.isclose(
                op_strategy.operation_info.timing,
                expected_update_parameters[index],
                abs_tol=0,
                rel_tol=1e-15,
            )
        else:
            assert not isinstance(op_strategy, UpdateParameterStrategy)


def square_pulse(duration: float, t0: float = 0) -> Operation:
    return SquarePulse(
        amp=0.5,
        port="q0:mw",
        duration=duration,
        clock="q0.01",
        t0=t0,
    )


def voltage_offset() -> Operation:
    return VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01")


def reset_clock_phase(t0: float = 0):
    return ResetClockPhase(clock="q0.01", t0=t0)


def shift_clock_phase(t0: float = 0):
    return ShiftClockPhase(phase_shift=0.5, clock="q0.01", t0=t0)


def set_clock_frequency(t0: float = 0):
    return SetClockFrequency(clock="q0.01", clock_freq_new=7.5e9, t0=t0)


@pytest.mark.parametrize(
    "op_list, expected_update_parameters",
    [
        (
            [
                square_pulse(duration=20e-9),
                voltage_offset(),
                square_pulse(duration=20e-9, t0=20e-9),
            ],
            {2: 20e-9},
        ),
        (
            [
                square_pulse(duration=20e-9),
                reset_clock_phase(),
                square_pulse(duration=20e-9, t0=20e-9),
            ],
            {2: 20e-9},
        ),
        (
            [
                square_pulse(duration=20e-9),
                shift_clock_phase(),
                square_pulse(duration=20e-9, t0=20e-9),
            ],
            {2: 20e-9},
        ),
        (
            [
                square_pulse(duration=20e-9),
                set_clock_frequency(),
                square_pulse(duration=20e-9, t0=20e-9),
            ],
            {2: 20e-9},
        ),
    ],
)
def test_param_update_after_param_op(
    op_list,
    expected_update_parameters,
    hardware_cfg_cluster,
):
    _assert_update_parameters_op_list(
        op_list,
        expected_update_parameters,
        QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster),
    )


@pytest.mark.parametrize(
    "op_list, expected_update_parameters",
    [
        (
            [
                square_pulse(duration=20e-9),
                voltage_offset(),
                square_pulse(duration=20e-9),
            ],
            {},
        ),
        (
            [
                square_pulse(duration=20e-9),
                voltage_offset(),
                square_pulse(duration=20e-9, t0=20e-9),
            ],
            {2: 20e-9},
        ),
        (
            [
                square_pulse(duration=20e-9),
                reset_clock_phase(),
                square_pulse(duration=20e-9),
            ],
            {},
        ),
        (
            [
                square_pulse(duration=20e-9),
                reset_clock_phase(),
                square_pulse(duration=20e-9, t0=20e-9),
            ],
            {2: 20e-9},
        ),
    ],
)
def test_param_update_after_param_op_except_if_simultaneous_play(
    op_list,
    expected_update_parameters,
    hardware_cfg_cluster,
):
    _assert_update_parameters_op_list(
        op_list,
        expected_update_parameters,
        QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster),
    )


@pytest.mark.parametrize(
    "op_list, expected_update_parameters",
    [
        (
            [
                square_pulse(duration=20e-9),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                square_pulse(duration=20e-9),
            ],
            {},
        ),
        (
            [
                square_pulse(duration=20e-9),
                voltage_offset(),
                reset_clock_phase(),
                voltage_offset(),
                square_pulse(duration=20e-9, t0=20e-9),
            ],
            {4: 20e-9},
        ),
        (
            [
                voltage_offset(),
                square_pulse(duration=20e-9),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                square_pulse(duration=20e-9),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                square_pulse(duration=20e-9),
            ],
            {},
        ),
        (
            [
                voltage_offset(),
                square_pulse(duration=20e-9),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                square_pulse(duration=20e-9, t0=20e-9),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                square_pulse(duration=20e-9, t0=20e-9),
                voltage_offset(),
                voltage_offset(),
                voltage_offset(),
                square_pulse(duration=20e-9),
            ],
            {5: 20e-9, 11: 60e-9},
        ),
    ],
)
def test_no_unnecessary_parameter_update(
    op_list,
    expected_update_parameters,
    hardware_cfg_cluster,
):
    _assert_update_parameters_op_list(
        op_list,
        expected_update_parameters,
        QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster),
    )


@pytest.mark.parametrize(
    "parameter_op",
    [
        voltage_offset(),
        reset_clock_phase(),
        shift_clock_phase(),
        set_clock_frequency(),
    ],
)
def test_error_parameter_end_of_schedule(
    parameter_op,
    compile_config_basic_transmon_qblox_hardware,
):
    schedule = Schedule("schedule")
    schedule.add(square_pulse(duration=20e-9))
    schedule.add(square_pulse(duration=20e-9))
    schedule.add(parameter_op)
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        RuntimeError,
        match="Parameter operation .* with start time 4e-08 "
        "cannot be scheduled at the very end of a Schedule. "
        "The Schedule can be extended by adding an IdlePulse "
        "operation with a duration of at least 4 ns, "
        "or the Parameter operation can be replaced by another operation.",
    ):
        schedule = compiler.compile(
            schedule=schedule,
            config=compile_config_basic_transmon_qblox_hardware,
        )


@pytest.mark.parametrize(
    "control_flow_op, control_flow_kwargs",
    [(LoopOperation, {"repetitions": 3}), (ConditionalOperation, {"qubit_name": "q0"})],
)
def test_no_remove_parameter_update_before_control_flow_begin(
    control_flow_op,
    control_flow_kwargs,
    compile_config_basic_transmon_qblox_hardware,
):
    """
    The compilation steps which removes the update parameter
    instructions should remove update parameters that happen at the same
    time, but it should not remove update parameters if a control flow
    begins between them: in case the control flow's body does not run,
    it should not remove the update parameter before the control flow.
    """
    schedule = Schedule("schedule")
    schedule.add(
        Measure(
            "q0",
            acq_protocol="ThresholdedAcquisition",
            feedback_trigger_label="q0",
        )
    )
    schedule.add(voltage_offset())

    subschedule = Schedule("inner")
    subschedule.add(voltage_offset())
    subschedule.add(square_pulse(duration=20e-9))

    schedule.add(
        control_flow_op(body=subschedule, **control_flow_kwargs),
    )

    schedule.add(square_pulse(duration=40e-9))

    schedule.add(square_pulse(duration=20e-9))
    compiler = SerialCompiler(name="compiler")

    # The update parameter operation cannot happen just before
    # the control flow begin, indicating that the update parameter
    # is preserved, not removed.
    with pytest.raises(
        RuntimeError,
        match='Parameter operation Pulse "UpdateParameters" '
        r"\(t0=1.1e-06, duration=0\) with start time 1.1e-06 "
        "cannot be scheduled exactly before the operation Pulse",
    ):
        schedule = compiler.compile(
            schedule=schedule,
            config=compile_config_basic_transmon_qblox_hardware,
        )


def test_no_remove_parameter_update_before_control_flow_end(
    compile_config_basic_transmon_qblox_hardware,
):
    """
    The compilation steps which removes the update parameter
    instructions should remove update parameters that happen at the same
    time, but it should not remove update parameters if a control flow
    ends between them: in case the control flow's body does a loop,
    it should not remove the update parameter before the control flow.
    """
    schedule = Schedule("schedule")

    subschedule = Schedule("inner")
    subschedule.add(square_pulse(duration=20e-9))
    subschedule.add(voltage_offset())

    schedule.add(LoopOperation(body=subschedule, repetitions=3))

    subschedule.add(voltage_offset())

    schedule.add(square_pulse(duration=40e-9))

    schedule.add(square_pulse(duration=20e-9))
    compiler = SerialCompiler(name="compiler")

    # The update parameter operation cannot happen just before
    # the control flow end, indicating that the update parameter
    # is preserved, not removed.
    with pytest.raises(
        RuntimeError,
        match="Parameter operation .* with start time 2e-08 "
        'cannot be scheduled exactly before the operation Pulse "ControlFlowReturn" '
        r"\(t0=2e-08, duration=0.0\) with the same start time. "
        "Insert an IdlePulse operation with a duration of at least 4 ns, "
        "or the Parameter operation can be replaced by another operation.",
    ):
        schedule = compiler.compile(
            schedule=schedule,
            config=compile_config_basic_transmon_qblox_hardware,
        )


@pytest.mark.parametrize(
    "parameter_op",
    [
        voltage_offset(),
        reset_clock_phase(),
        shift_clock_phase(),
        set_clock_frequency(),
    ],
)
def test_error_parameter_end_of_control_flow(
    parameter_op,
    compile_config_basic_transmon_qblox_hardware,
):
    schedule = Schedule("schedule")
    subschedule = Schedule("inner")
    subschedule.add(square_pulse(duration=20e-9))
    subschedule.add(parameter_op)
    schedule.add(LoopOperation(body=subschedule, repetitions=3))
    schedule.add(square_pulse(duration=20e-9))
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        RuntimeError,
        match="Parameter operation .* with start time 2e-08 "
        'cannot be scheduled exactly before the operation Pulse "ControlFlowReturn" '
        r"\(t0=2e-08, duration=0.0\) with the same start time. "
        "Insert an IdlePulse operation with a duration of at least 4 ns, "
        "or the Parameter operation can be replaced by another operation.",
    ):
        schedule = compiler.compile(
            schedule=schedule,
            config=compile_config_basic_transmon_qblox_hardware,
        )


DEFAULT_PORT = "q0:res"
DEFAULT_CLOCK = "q0.ro"


def pulse_with_waveform_op_info(
    timing: float, duration: float = 1e-7, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a non-idle pulse."""
    operation = SquarePulse(amp=1.0, duration=duration, port=port, clock=clock)
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def reset_clock_phase_op_info(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a virtual pulse."""
    operation = ResetClockPhase(clock=clock)
    operation.data["pulse_info"][0]["port"] = port
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def set_clock_frequency_op_info(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a virtual pulse."""
    operation = SetClockFrequency(clock=clock, clock_freq_new=1e9)
    operation.data["pulse_info"][0]["port"] = port
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def shift_clock_phase_op_info(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a virtual pulse."""
    operation = ShiftClockPhase(phase_shift=0.5, clock=clock)
    operation.data["pulse_info"][0]["port"] = port
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def offset_instruction_op_info(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as an offset instruction."""
    operation = VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port=port, clock=clock)
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def control_flow_return_op_info(
    timing: float, port: str = DEFAULT_PORT, clock: str = DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a control flow return operation."""
    operation = _ControlFlowReturn()
    operation["pulse_info"] = [
        {
            "wf_func": None,
            "clock": clock,
            "port": port,
            "duration": 0,
            "control_flow_end": True,
            **operation["control_flow_info"],
        }
    ]
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def loop_op_info(timing: float, repetitions: int = 1) -> OpInfo:
    """Create an OpInfo object that is recognized as a loop operation."""
    operation = LoopBegin(repetitions=repetitions)
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["control_flow_info"]
    )


# Total play time number does not matter here. The fixture needs it.
@pytest.mark.parametrize("total_play_time", [2e-7])
def test_too_many_instructions_warns(mock_sequencer: AnalogSequencerCompiler):
    max_num_instructions = 100
    max_operations_num = (max_num_instructions - 10) // 2
    mock_sequencer.parent.configure_mock(  # type: ignore # Member "configure_mock" is unknown
        max_number_of_instructions=max_num_instructions
    )
    mock_sequencer._default_marker = 0
    operations = [
        ioperation_strategy_from_op_info(
            offset_instruction_op_info(t * 8e-9), channel_name="real_output_0"
        )
        for t in range(0, max_operations_num + 1)
    ]
    with pytest.warns(
        RuntimeWarning,
        match="exceeds the maximum supported number of instructions in Q1ASM programs",
    ):
        mock_sequencer.generate_qasm_program(
            ordered_op_strategies=operations,
            total_sequence_time=max_operations_num * 8e-9,
            align_qasm_fields=False,
            acq_metadata=None,
            repetitions=1,
        )


_REAL_TIME_INSTRUCTIONS = {
    q1asm_instructions.UPDATE_PARAMETERS,
    q1asm_instructions.PLAY,
    q1asm_instructions.ACQUIRE,
    q1asm_instructions.ACQUIRE_TTL,
    q1asm_instructions.ACQUIRE_WEIGHED,
    q1asm_instructions.ACQUIRE_TTL,
    q1asm_instructions.FEEDBACK_TRIGGER_EN,
    q1asm_instructions.FEEDBACK_TRIGGERS_RST,
    q1asm_instructions.WAIT,
    q1asm_instructions.WAIT_SYNC,
    q1asm_instructions.WAIT_TRIGGER,
}


def _get_instruction_duration(instruction: str, arguments: str) -> int:
    if instruction not in _REAL_TIME_INSTRUCTIONS:
        return 0

    return int(arguments.split(",")[-1])


def test_write_repetition_loop_header_equal_time():
    sequencer_cfg = _SequencerCompilationConfig(
        sequencer_options=SequencerOptions(),
        hardware_description=ComplexChannelDescription(),
        portclock="port-clock",
        channel_name="channel_name_x",
        channel_name_measure=None,
        latency_correction=0,
        distortion_correction=None,
        lo_name=None,
        modulation_frequencies=ModulationFrequencies(),
        mixer_corrections=None,
    )
    analog_sequencer = AnalogSequencerCompiler(
        parent=Mock(),
        index=0,
        static_hw_properties=StaticAnalogModuleProperties(
            instrument_type="QRM",
            max_sequencers=6,
            max_awg_output_voltage=None,
            mixer_dc_offset_range=BoundedParameter(0, 0, ""),
        ),
        sequencer_cfg=sequencer_cfg,
    )
    analog_sequencer._default_marker = 0b1000
    timetag_sequencer = TimetagSequencerCompiler(
        parent=Mock(),
        index=0,
        static_hw_properties=StaticTimetagModuleProperties(
            instrument_type="QTM",
            max_sequencers=8,
        ),
        sequencer_cfg=sequencer_cfg,
    )

    durations = []
    for class_ in (analog_sequencer, timetag_sequencer):
        qasm_program = QASMProgram(
            static_hw_properties=Mock(),
            register_manager=Mock(),
            align_fields=False,
            acq_metadata=None,
        )
        class_._write_repetition_loop_header(qasm_program)
        durations.append(
            sum(
                map(
                    lambda instr_list: _get_instruction_duration(instr_list[1], instr_list[2]),
                    qasm_program.instructions,
                )
            )
        )

    assert all(dur == durations[0] for dur in durations)


def op_info_from_operation(operation: Operation, timing: float, data: dict) -> OpInfo:
    return OpInfo(
        name=operation.name,
        data=data,
        timing=timing,
    )


@pytest.fixture
def mock_sequencer(total_play_time) -> AnalogSequencerCompiler:
    mod = Mock()
    mod.configure_mock(total_play_time=total_play_time)
    sequencer_cfg = _SequencerCompilationConfig(
        sequencer_options=SequencerOptions(),
        hardware_description=ComplexChannelDescription(),
        portclock="q1:mw-q1.01",
        channel_name="channel_name_x",
        channel_name_measure=None,
        latency_correction=0,
        distortion_correction=None,
        lo_name=None,
        modulation_frequencies=ModulationFrequencies.model_validate(
            {"lo_freq": None, "interm_freq": 50e6}
        ),
        mixer_corrections=None,
    )
    return AnalogSequencerCompiler(
        parent=mod,
        index=0,
        static_hw_properties=StaticAnalogModuleProperties(
            instrument_type="QRM",
            max_sequencers=6,
            max_awg_output_voltage=None,
            mixer_dc_offset_range=BoundedParameter(0, 0, ""),
        ),
        sequencer_cfg=sequencer_cfg,
    )


def ioperation_strategy_from_op_info(op_info: OpInfo, channel_name: str) -> IOperationStrategy:
    return get_operation_strategy(op_info, channel_name)


def upd_param_op_info(
    timing: float, port: str = DEFAULT_PORT, clock: str = DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as upd_param operation."""
    return OpInfo(
        name="UpdateParameters",
        data={
            "t0": 0,
            "port": port,
            "clock": clock,
            "duration": 0,
            "instruction": q1asm_instructions.UPDATE_PARAMETERS,
        },
        timing=timing,
    )


@pytest.mark.parametrize("total_play_time", [2.08e-7])
def test_get_ordered_operations(mock_sequencer: AnalogSequencerCompiler):
    op_list = [
        reset_clock_phase_op_info(timing=0.0),
        set_clock_frequency_op_info(timing=0.0),
        upd_param_op_info(timing=0.0),
        loop_op_info(timing=4e-9 - 1e-12, repetitions=3),
        shift_clock_phase_op_info(timing=4e-09),
        pulse_with_waveform_op_info(timing=4e-09, duration=1e-07),
        control_flow_return_op_info(timing=1.04e-07),
        offset_instruction_op_info(timing=1.04e-07),
        upd_param_op_info(timing=1.04e-7),
        offset_instruction_op_info(timing=2.04e-07),
        upd_param_op_info(timing=2.04e-7),
    ]
    mock_sequencer.op_strategies = [
        ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list
    ]

    assert [op_strat.operation_info for op_strat in mock_sequencer._get_ordered_operations()] == [
        reset_clock_phase_op_info(timing=0.0),
        set_clock_frequency_op_info(timing=0.0),
        upd_param_op_info(timing=0.0),
        loop_op_info(timing=4e-9 - 1e-12, repetitions=3),
        shift_clock_phase_op_info(timing=4e-09),
        pulse_with_waveform_op_info(timing=4e-09, duration=1e-07),
        control_flow_return_op_info(timing=1.04e-07),
        offset_instruction_op_info(timing=1.04e-07),
        upd_param_op_info(timing=1.04e-7),
        offset_instruction_op_info(timing=2.04e-07),
        upd_param_op_info(timing=2.04e-7),
    ]
