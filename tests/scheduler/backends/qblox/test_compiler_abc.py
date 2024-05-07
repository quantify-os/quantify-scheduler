# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the InstrumentCompiler subclasses."""
from itertools import permutations, product
from typing import List
from unittest.mock import Mock

import pytest

from quantify_scheduler.backends.qblox import q1asm_instructions
from quantify_scheduler.backends.qblox.analog import AnalogSequencerCompiler
from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.operation_handling.factory import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    UpdateParameterStrategy,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler
from quantify_scheduler.backends.types.qblox import AnalogSequencerSettings, OpInfo
from quantify_scheduler.compilation import _ControlFlowReturn
from quantify_scheduler.operations.acquisition_library import Trace
from quantify_scheduler.operations.control_flow_library import Loop
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import (
    ResetClockPhase,
    SetClockFrequency,
    ShiftClockPhase,
    SquarePulse,
    VoltageOffset,
)

DEFAULT_PORT = "q0:res"
DEFAULT_CLOCK = "q0.ro"


@pytest.fixture
def mock_sequencer(total_play_time) -> AnalogSequencerCompiler:
    mod = Mock()
    mod.configure_mock(total_play_time=total_play_time)
    settings = AnalogSequencerSettings.initialize_from_config_dict(
        {
            "port": "q1:mw",
            "clock": "q1.01",
            "interm_freq": 50e6,
        },
        channel_name="complex_out_0",
        connected_input_indices=(),
        connected_output_indices=(0,),
    )
    return AnalogSequencerCompiler(
        parent=mod,
        index=0,
        portclock=(DEFAULT_PORT, DEFAULT_CLOCK),
        static_hw_properties=Mock(),
        settings=settings,
        latency_corrections={},
    )


def op_info_from_operation(operation: Operation, timing: float, data: dict) -> OpInfo:
    return OpInfo(
        name=operation.name,
        data=data,
        timing=timing,
    )


def pulse_with_waveform(
    timing: float, duration: float = 1e-7, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a non-idle pulse."""
    operation = SquarePulse(amp=1.0, duration=duration, port=port, clock=clock)
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def reset_clock_phase(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a virtual pulse."""
    operation = ResetClockPhase(clock=clock)
    operation.data["pulse_info"][0]["port"] = port
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def set_clock_frequency(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a virtual pulse."""
    operation = SetClockFrequency(clock=clock, clock_freq_new=1e9)
    operation.data["pulse_info"][0]["port"] = port
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def shift_clock_phase(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as a virtual pulse."""
    operation = ShiftClockPhase(phase_shift=0.5, clock=clock)
    operation.data["pulse_info"][0]["port"] = port
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def offset_instruction(
    timing: float, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as an offset instruction."""
    operation = VoltageOffset(
        offset_path_I=0.5, offset_path_Q=0.0, port=port, clock=clock
    )
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["pulse_info"][0]
    )


def acquisition(
    timing: float, duration: float = 1e-7, port: str = DEFAULT_PORT, clock=DEFAULT_CLOCK
) -> OpInfo:
    """Create an OpInfo object that is recognized as an acquisition."""
    operation = Trace(duration=duration, port=port, clock=clock)
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["acquisition_info"][0]
    )


def control_flow_return(timing: float) -> OpInfo:
    """Create an OpInfo object that is recognized as a control flow return operation."""
    operation = _ControlFlowReturn()
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["control_flow_info"]
    )


def loop(timing: float, repetitions: int = 1) -> OpInfo:
    """Create an OpInfo object that is recognized as a loop operation."""
    operation = Loop(repetitions=repetitions)
    return op_info_from_operation(
        operation=operation, timing=timing, data=operation.data["control_flow_info"]
    )


def upd_param(
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


def ioperation_strategy_from_op_info(
    op_info: OpInfo, channel_name: str
) -> IOperationStrategy:
    return get_operation_strategy(op_info, channel_name)


@pytest.mark.parametrize(
    "op_list, total_play_time",
    list(
        product(
            permutations(
                [
                    pulse_with_waveform(0.0),
                    offset_instruction(1e-7),
                    offset_instruction(2e-7),
                    pulse_with_waveform(2e-7),
                    reset_clock_phase(1e-7),
                ]
            ),
            [3e-7],
        ),
    ),
)
def test_param_update_after_param_op_except_if_simultaneous_play(
    op_list: List[OpInfo], mock_sequencer: AnalogSequencerCompiler
):
    """Test if upd_param is inserted after a VoltageOffset in the correct places."""
    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]

    for op in iop_list:
        mock_sequencer.op_strategies.append(op)
    mock_sequencer._insert_update_parameters()

    assert len(mock_sequencer.op_strategies) == 6
    upd_param_inserted = next(
        filter(
            lambda x: isinstance(x, UpdateParameterStrategy),
            mock_sequencer.op_strategies,
        )
    )
    assert upd_param_inserted.operation_info == OpInfo(
        name="UpdateParameters",
        data={
            "t0": 0,
            "duration": 0.0,
            "instruction": q1asm_instructions.UPDATE_PARAMETERS,
            "port": DEFAULT_PORT,
            "clock": DEFAULT_CLOCK,
        },
        timing=pytest.approx(1e-7),  # type: ignore
    )


@pytest.mark.parametrize(
    "op_list, total_play_time",
    list(
        product(
            permutations(
                [
                    pulse_with_waveform(0.0),
                    offset_instruction(1e-7),
                    acquisition(1e-7),
                    reset_clock_phase(1e-7),
                ]
            ),
            [2e-7],
        ),
    ),
)
def test_no_parameter_update(
    op_list: List[OpInfo], mock_sequencer: AnalogSequencerCompiler
):
    """Test if no upd_param is inserted where it is not necessary."""
    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]  # type: ignore

    for op in iop_list:
        mock_sequencer.op_strategies.append(op)

    mock_sequencer._insert_update_parameters()

    assert (
        len(
            [
                op_strategy
                for op_strategy in mock_sequencer.op_strategies
                if not op_strategy.operation_info.is_acquisition
            ]
        )
        == 3
    )
    assert (
        len(
            [
                op_strategy
                for op_strategy in mock_sequencer.op_strategies
                if op_strategy.operation_info.is_acquisition
            ]
        )
        == 1
    )
    for op in mock_sequencer.op_strategies:
        assert op.operation_info.name != "UpdateParameters"


@pytest.mark.parametrize("total_play_time", [2e-7])
def test_only_one_param_update(mock_sequencer: AnalogSequencerCompiler):
    """Test if no upd_param is inserted where it is not necessary."""
    op_list = [
        pulse_with_waveform(0.0),
        set_clock_frequency(1e-7),
        shift_clock_phase(1e-7),
        offset_instruction(1e-7),
        reset_clock_phase(1e-7),
        acquisition(2e-7),
    ]
    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]  # type: ignore

    for op in iop_list:
        mock_sequencer.op_strategies.append(op)

    mock_sequencer._insert_update_parameters()

    assert (
        len(
            [
                op_strategy
                for op_strategy in mock_sequencer.op_strategies
                if not op_strategy.operation_info.is_acquisition
            ]
        )
        == 6
    )
    assert (
        len(
            [
                op_strategy
                for op_strategy in mock_sequencer.op_strategies
                if op_strategy.operation_info.is_acquisition
            ]
        )
        == 1
    )
    upd_params = [
        op
        for op in mock_sequencer.op_strategies
        if op.operation_info.name == "UpdateParameters"
    ]
    assert len(upd_params) == 1


@pytest.mark.parametrize(
    "op_list, total_play_time",
    list(
        product(
            permutations(
                [
                    pulse_with_waveform(0.0),
                    offset_instruction(2e-7),
                    acquisition(1e-7),
                    reset_clock_phase(1e-7),
                ]
            ),
            [2e-7],
        ),
    ),
)
def test_error_parameter_update_end_of_schedule(
    op_list: List[OpInfo], mock_sequencer: AnalogSequencerCompiler
):
    """Test if no upd_param is inserted where it is not necessary."""
    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]  # type: ignore

    for op in iop_list:
        mock_sequencer.op_strategies.append(op)

    with pytest.raises(
        RuntimeError,
        match=f"start time {2e-7} cannot be scheduled at the very end of a Schedule",
    ):
        mock_sequencer._insert_update_parameters()


@pytest.mark.parametrize("total_play_time", [3e-7])
def test_error_parameter_update_at_control_flow_return(
    mock_sequencer: AnalogSequencerCompiler,
):
    """Test if no upd_param is inserted where it is not necessary."""
    op_list = [
        offset_instruction(1e-7),
        control_flow_return(1e-7),
    ]

    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]  # type: ignore

    for op in iop_list:
        mock_sequencer.op_strategies.append(op)

    with pytest.raises(
        RuntimeError,
        match=f"with start time {1e-7} cannot be scheduled at the same time as the end "
        "of a control-flow block",
    ):
        mock_sequencer._insert_update_parameters()


@pytest.mark.parametrize(
    "ordered_op_infos, op_index, expected",
    [
        (
            [
                offset_instruction(1 - 1e-12),
                offset_instruction(1),
                offset_instruction(1 + 1e-12),
                offset_instruction(1 + 1e-12),
            ],
            1,
            False,
        ),
        (
            [
                offset_instruction(1 - 1e-12),
                offset_instruction(1),
                offset_instruction(1 + 1e-12),
                acquisition(1 + 1e-12),
            ],
            1,
            True,
        ),
        (
            [
                acquisition(1 - 1e-12),
                offset_instruction(1 - 1e-12),
                offset_instruction(1),
                offset_instruction(1 + 1e-12),
            ],
            2,
            True,
        ),
        (
            [
                offset_instruction(1 - 1e-12),
                offset_instruction(1),
                offset_instruction(1 + 1e-12),
                acquisition(1 + 1e-9),
            ],
            1,
            False,
        ),
    ],
)
def test_any_other_updating_instruction_at_timing(
    ordered_op_infos: List[OpInfo], op_index: int, expected: bool
):
    pulse_list = [
        ioperation_strategy_from_op_info(op, "complex_out_0") for op in ordered_op_infos
    ]
    assert (
        AnalogSequencerCompiler._any_other_updating_instruction_at_timing_for_parameter_instruction(
            op_index=op_index, ordered_op_strategies=pulse_list
        )
        == expected
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
            offset_instruction(t * 8e-9), channel_name="real_output_0"
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
    analog_sequencer = AnalogSequencerCompiler(
        parent=Mock(),
        index=0,
        portclock=("foo", "bar"),
        static_hw_properties=Mock(),
        settings=Mock(),
        latency_corrections={},
    )
    analog_sequencer._default_marker = 0b1000
    timetag_sequencer = TimetagSequencerCompiler(
        parent=Mock(),
        index=0,
        portclock=("foo", "bar"),
        static_hw_properties=Mock(),
        settings=Mock(),
        latency_corrections={},
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
                    lambda instr_list: _get_instruction_duration(
                        instr_list[1], instr_list[2]
                    ),
                    qasm_program.instructions,
                )
            )
        )

    assert all(dur == durations[0] for dur in durations)


@pytest.mark.parametrize("total_play_time", [2.08e-7])
def test_get_ordered_operations(mock_sequencer: AnalogSequencerCompiler):
    op_list = [
        reset_clock_phase(timing=0.0),
        set_clock_frequency(timing=0.0),
        shift_clock_phase(timing=4e-09),
        loop(timing=4e-9 - 1e-12, repetitions=3),
        pulse_with_waveform(timing=4e-09, duration=1e-07),
        control_flow_return(timing=1.04e-07),
        offset_instruction(timing=1.04e-07),
        offset_instruction(timing=2.04e-07),
        upd_param(timing=0.0),
        upd_param(timing=2.04e-7),
        upd_param(timing=1.04e-7),
    ]
    mock_sequencer.op_strategies = [
        ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list
    ]

    assert [
        op_strat.operation_info for op_strat in mock_sequencer._get_ordered_operations()
    ] == [
        reset_clock_phase(timing=0.0),
        set_clock_frequency(timing=0.0),
        upd_param(timing=0.0),
        loop(timing=4e-9 - 1e-12, repetitions=3),
        shift_clock_phase(timing=4e-09),
        pulse_with_waveform(timing=4e-09, duration=1e-07),
        control_flow_return(timing=1.04e-07),
        offset_instruction(timing=1.04e-07),
        upd_param(timing=1.04e-7),
        offset_instruction(timing=2.04e-07),
        upd_param(timing=2.04e-7),
    ]
