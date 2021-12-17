# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Tests for factory module."""
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument

import pytest

from typing import Type

from quantify_scheduler.enums import BinMode
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.backends.qblox.operation_handling import (
    base,
    factory,
    pulses,
    acquisitions,
)

TEST_OP_INFO_MAPPING = {
    "other": OpInfo(
        name="", data={"wf_func": "quantify_scheduler.waveforms.doesnotexist"}, timing=0
    ),
    "square": OpInfo(
        name="", data={"wf_func": "quantify_scheduler.waveforms.square"}, timing=0
    ),
    "staircase": OpInfo(
        name="", data={"wf_func": "quantify_scheduler.waveforms.staircase"}, timing=0
    ),
    "ssb": OpInfo(
        name="",
        data={
            "protocol": "ssb_integration_complex",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
        },
        timing=0,
    ),
    "weighted": OpInfo(
        name="",
        data={
            "protocol": "weighted_integrated_complex",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
        },
        timing=0,
    ),
    "trace": OpInfo(
        name="",
        data={
            "protocol": "trace",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
        },
        timing=0,
    ),
}


@pytest.mark.parametrize(
    "op_info, answer",
    [
        (TEST_OP_INFO_MAPPING["other"], pulses.GenericPulseStrategy),
        (TEST_OP_INFO_MAPPING["square"], pulses.StitchedSquarePulseStrategy),
        (TEST_OP_INFO_MAPPING["staircase"], pulses.StaircasePulseStrategy),
        (TEST_OP_INFO_MAPPING["ssb"], acquisitions.SquareAcquisitionStrategy),
        (TEST_OP_INFO_MAPPING["weighted"], acquisitions.WeightedAcquisitionStrategy),
        (TEST_OP_INFO_MAPPING["trace"], acquisitions.SquareAcquisitionStrategy),
    ],
)
def test_get_operation_strategy(
    op_info: OpInfo,
    answer: Type[base.IOperationStrategy],
):
    # arrange
    instruction_generated_pulses_enabled = True
    output_mode = "complex"

    # act
    obj = factory.get_operation_strategy(
        op_info, instruction_generated_pulses_enabled, output_mode
    )

    # assert
    assert isinstance(obj, answer)


@pytest.mark.parametrize(
    "op_info",
    [
        TEST_OP_INFO_MAPPING["other"],
        TEST_OP_INFO_MAPPING["square"],
        TEST_OP_INFO_MAPPING["staircase"],
    ],
)
def test_get_operation_strategy_no_instr_gen(
    op_info: OpInfo,
):
    # arrange
    instruction_generated_pulses_enabled = False
    output_mode = "complex"

    # act
    obj = factory.get_operation_strategy(
        op_info, instruction_generated_pulses_enabled, output_mode
    )

    # assert
    assert isinstance(obj, pulses.GenericPulseStrategy)


def test_invalid_protocol_exception():
    # arrange
    instruction_generated_pulses_enabled = True
    output_mode = "complex"
    op_info = OpInfo(
        name="",
        data={
            "duration": 12e-9,
            "protocol": "nonsense",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
        },
        timing=0,
    )

    # act
    with pytest.raises(ValueError) as exc:
        factory.get_operation_strategy(
            op_info, instruction_generated_pulses_enabled, output_mode
        )

    # assert
    assert (
        exc.value.args[0]
        == 'Unknown acquisition protocol "nonsense" encountered in Qblox backend when'
        " processing acquisition Acquisition  (t=0 to 1.2e-08)\ndata={'duration':"
        " 1.2e-08, 'protocol': 'nonsense', 'acq_channel': 0, 'acq_index': 0,"
        " 'bin_mode': <BinMode.AVERAGE: 'average'>}."
    )


def test_trace_append_exception():
    # arrange
    instruction_generated_pulses_enabled = True
    output_mode = "complex"
    op_info = OpInfo(
        name="",
        data={
            "duration": 12e-9,
            "protocol": "trace",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.APPEND,
        },
        timing=0,
    )

    # act
    with pytest.raises(ValueError) as exc:
        factory.get_operation_strategy(
            op_info, instruction_generated_pulses_enabled, output_mode
        )

    # assert
    assert (
        exc.value.args[0]
        == "Trace acquisition does not support APPEND bin mode.\n\nAcquisition  "
        "(t=0 to 1.2e-08)\ndata={'duration': 1.2e-08, 'protocol': 'trace', "
        "'acq_channel': 0, 'acq_index': 0, 'bin_mode': <BinMode.APPEND: "
        "'append'>} caused this exception to occur."
    )
