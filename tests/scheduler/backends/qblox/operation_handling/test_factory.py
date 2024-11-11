# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for factory module."""


from __future__ import annotations

import pytest

from quantify_scheduler.backends.qblox import q1asm_instructions
from quantify_scheduler.backends.qblox.operation_handling import (
    acquisitions,
    base,
    factory_analog,
    pulses,
    virtual,
)
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.enums import BinMode

TEST_OP_INFO_MAPPING = {
    "other": OpInfo(
        name="",
        data={
            "wf_func": "quantify_scheduler.waveforms.doesnotexist",
            "port": "some_port",
            "clock": "some_clock",
            "duration": 1e-7,
        },
        timing=0,
    ),
    "square": OpInfo(
        name="",
        data={
            "wf_func": "quantify_scheduler.waveforms.square",
            "port": "some_port",
            "clock": "some_clock",
            "duration": 1e-7,
        },
        timing=0,
    ),
    "staircase": OpInfo(
        name="",
        data={
            "wf_func": "quantify_scheduler.waveforms.staircase",
            "port": "some_port",
            "clock": "some_clock",
            "duration": 1e-7,
        },
        timing=0,
    ),
    "ssb": OpInfo(
        name="",
        data={
            "protocol": "SSBIntegrationComplex",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
            "port": "some_port",
            "clock": "some_clock",
            "duration": 1e-7,
        },
        timing=0,
    ),
    "weighted": OpInfo(
        name="",
        data={
            "protocol": "NumericalSeparatedWeightedIntegration",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
            "port": "some_port",
            "clock": "some_clock",
            "duration": 1e-7,
        },
        timing=0,
    ),
    "trace": OpInfo(
        name="",
        data={
            "protocol": "Trace",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
            "port": "some_port",
            "clock": "some_clock",
            "duration": 1e-7,
        },
        timing=0,
    ),
    "trigger_count": OpInfo(
        name="",
        data={
            "protocol": "TriggerCount",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
            "port": "some_port",
            "clock": "some_clock",
            "duration": 1e-7,
        },
        timing=0,
    ),
    "offset": OpInfo(
        name="",
        data={
            "wf_func": None,
            "offset_path_I": 0.5,
            "offset_path_Q": 0.5,
            "port": "some_port",
            "clock": "some_clock",
            "duration": 0,
        },
        timing=0,
    ),
    "upd_param": OpInfo(
        name="",
        data={
            "instruction": q1asm_instructions.UPDATE_PARAMETERS,
            "port": "some_port",
            "clock": "some_clock",
            "duration": 4e-9,
        },
        timing=0,
    ),
}


@pytest.mark.parametrize(
    "operation_info, answer",
    [
        (TEST_OP_INFO_MAPPING["other"], pulses.GenericPulseStrategy),
        (TEST_OP_INFO_MAPPING["square"], pulses.GenericPulseStrategy),
        (TEST_OP_INFO_MAPPING["staircase"], pulses.GenericPulseStrategy),
        (TEST_OP_INFO_MAPPING["ssb"], acquisitions.SquareAcquisitionStrategy),
        (TEST_OP_INFO_MAPPING["weighted"], acquisitions.WeightedAcquisitionStrategy),
        (TEST_OP_INFO_MAPPING["trace"], acquisitions.SquareAcquisitionStrategy),
        (
            TEST_OP_INFO_MAPPING["trigger_count"],
            acquisitions.TriggerCountAcquisitionStrategy,
        ),
        (TEST_OP_INFO_MAPPING["offset"], virtual.AwgOffsetStrategy),
        (TEST_OP_INFO_MAPPING["upd_param"], virtual.UpdateParameterStrategy),
    ],
)
def test_get_operation_strategy(
    operation_info: OpInfo,
    answer: type[base.IOperationStrategy],
):
    obj = factory_analog.get_operation_strategy(
        operation_info=operation_info,
        channel_name="complex_output_0",
    )

    # assert
    assert isinstance(obj, answer)


def test_invalid_protocol_exception():
    # arrange
    operation_info = OpInfo(
        name="",
        data={
            "duration": 12e-9,
            "protocol": "nonsense",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.AVERAGE,
            "port": "some_port",
            "clock": "some_clock",
        },
        timing=0,
    )

    # act
    with pytest.raises(ValueError) as exc:
        factory_analog.get_operation_strategy(
            operation_info=operation_info,
            channel_name="complex_output_0",
        )

    # assert
    assert (
        exc.value.args[0]
        == 'Unknown acquisition protocol "nonsense" encountered in Qblox backend when'
        " processing acquisition Acquisition  (t=0 to 1.2e-08)\ndata={'duration':"
        " 1.2e-08, 'protocol': 'nonsense', 'acq_channel': 0, 'acq_index': 0,"
        " 'bin_mode': <BinMode.AVERAGE: 'average'>, 'port': 'some_port', 'clock': "
        "'some_clock'}."
    )


def test_trace_append_exception():
    # arrange
    operation_info = OpInfo(
        name="",
        data={
            "duration": 12e-9,
            "protocol": "Trace",
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": BinMode.APPEND,
            "port": "some_port",
            "clock": "some_clock",
        },
        timing=0,
    )

    # act
    with pytest.raises(ValueError) as exc:
        factory_analog.get_operation_strategy(
            operation_info=operation_info,
            channel_name="complex_output_0",
        )

    # assert
    assert (
        exc.value.args[0] == "Trace acquisition does not support bin mode append.\n\nAcquisition  "
        "(t=0 to 1.2e-08)\ndata={'duration': 1.2e-08, 'protocol': 'Trace', "
        "'acq_channel': 0, 'acq_index': 0, 'bin_mode': <BinMode.APPEND: "
        "'append'>, 'port': 'some_port', 'clock': 'some_clock'} caused this exception "
        "to occur."
    )
