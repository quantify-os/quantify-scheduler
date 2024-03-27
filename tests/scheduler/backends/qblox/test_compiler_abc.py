# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the InstrumentCompiler subclasses."""
from itertools import permutations, product
from typing import List
from unittest.mock import Mock

import pytest

from quantify_scheduler.backends.qblox import q1asm_instructions
from quantify_scheduler.backends.qblox.compiler_abc import Sequencer
from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.operation_handling.factory import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    UpdateParameterStrategy,
)
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.compilation import _ControlFlowReturn
from quantify_scheduler.operations.acquisition_library import Trace
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
def mock_sequencer(total_play_time) -> Sequencer:
    mod = Mock()
    mod.configure_mock(total_play_time=total_play_time)
    return Sequencer(
        parent=mod,
        index=0,
        portclock=(DEFAULT_PORT, DEFAULT_CLOCK),
        static_hw_properties=Mock(),
        channel_name="complex_out_0",
        sequencer_cfg={
            "port": "q1:mw",
            "clock": "q1.01",
            "interm_freq": 50e6,
        },
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
    op_list: List[OpInfo], mock_sequencer: Sequencer
):
    """Test if upd_param is inserted after a VoltageOffset in the correct places."""
    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]

    for op in iop_list:
        mock_sequencer.pulses.append(op)
    mock_sequencer._insert_update_parameters()

    assert len(mock_sequencer.pulses) == 6
    upd_param_inserted = next(
        filter(lambda x: isinstance(x, UpdateParameterStrategy), mock_sequencer.pulses)
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
def test_no_parameter_update(op_list: List[OpInfo], mock_sequencer: Sequencer):
    """Test if no upd_param is inserted where it is not necessary."""
    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]  # type: ignore

    for op in iop_list:
        if op.operation_info.is_acquisition:
            mock_sequencer.acquisitions.append(op)
        else:
            mock_sequencer.pulses.append(op)

    mock_sequencer._insert_update_parameters()

    assert len(mock_sequencer.pulses) == 3
    for op in mock_sequencer.pulses:
        assert op.operation_info.name != "UpdateParameters"


@pytest.mark.parametrize("total_play_time", [2e-7])
def test_only_one_param_update(mock_sequencer: Sequencer):
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
        if op.operation_info.is_acquisition:
            mock_sequencer.acquisitions.append(op)
        else:
            mock_sequencer.pulses.append(op)

    mock_sequencer._insert_update_parameters()

    assert len(mock_sequencer.pulses) == 6
    upd_params = [
        op
        for op in mock_sequencer.pulses
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
    op_list: List[OpInfo], mock_sequencer: Sequencer
):
    """Test if no upd_param is inserted where it is not necessary."""
    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]  # type: ignore

    for op in iop_list:
        if op.operation_info.is_acquisition:
            mock_sequencer.acquisitions.append(op)
        else:
            mock_sequencer.pulses.append(op)

    with pytest.raises(
        RuntimeError,
        match=f"start time {2e-7} cannot be scheduled at the very end of a Schedule",
    ):
        mock_sequencer._insert_update_parameters()


@pytest.mark.parametrize("total_play_time", [3e-7])
def test_error_parameter_update_at_control_flow_return(mock_sequencer: Sequencer):
    """Test if no upd_param is inserted where it is not necessary."""
    op_list = [
        offset_instruction(1e-7),
        control_flow_return(1e-7),
    ]

    iop_list = [ioperation_strategy_from_op_info(op, "complex_out_0") for op in op_list]  # type: ignore

    for op in iop_list:
        if op.operation_info.is_acquisition:
            mock_sequencer.acquisitions.append(op)
        else:
            mock_sequencer.pulses.append(op)

    with pytest.raises(
        RuntimeError,
        match=f"with start time {1e-7} cannot be scheduled at the same time as the end "
        "of a control-flow block",
    ):
        mock_sequencer._insert_update_parameters()


@pytest.mark.parametrize(
    "sorted_pulses_and_acqs, op_index, expected",
    [
        (
            [
                offset_instruction(1 - 1e-9),
                offset_instruction(1),
                offset_instruction(1 + 1e-9),
                offset_instruction(1 + 1.5e-9),
            ],
            1,
            False,
        ),
        (
            [
                offset_instruction(1 - 1e-9),
                offset_instruction(1),
                offset_instruction(1 + 1e-9),
                acquisition(1 + 1.5e-9),
            ],
            1,
            True,
        ),
        (
            [
                acquisition(1 - 1.5e-9),
                offset_instruction(1 - 1e-9),
                offset_instruction(1),
                offset_instruction(1 + 1e-9),
            ],
            2,
            True,
        ),
        (
            [
                offset_instruction(1 - 1e-9),
                offset_instruction(1),
                offset_instruction(1 + 1e-9),
                acquisition(1 + 2.5e-9),
            ],
            1,
            False,
        ),
    ],
)
def test_any_other_updating_instruction_at_timing(
    sorted_pulses_and_acqs: List[OpInfo], op_index: int, expected: bool
):
    pulse_list = [
        ioperation_strategy_from_op_info(op, "complex_out_0")
        for op in sorted_pulses_and_acqs
    ]
    assert (
        Sequencer._any_other_updating_instruction_at_timing_for_parameter_instruction(
            op_index=op_index, sorted_pulses_and_acqs=pulse_list
        )
        == expected
    )
