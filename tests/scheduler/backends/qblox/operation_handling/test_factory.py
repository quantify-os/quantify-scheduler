# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for factory module."""

from __future__ import annotations

import re

import pytest

from quantify_scheduler.backends.qblox import q1asm_instructions
from quantify_scheduler.backends.qblox.operation_handling import (
    acquisitions,
    base,
    bin_mode_compat,
    factory_analog,
    factory_timetag,
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
            "bin_mode": BinMode.DISTRIBUTION,
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


@pytest.mark.parametrize(
    "protocol, bin_mode",
    [
        ("Trace", BinMode.APPEND),
        ("Trace", BinMode.FIRST),
        ("Trace", BinMode.DISTRIBUTION),
        ("Trace", BinMode.SUM),
        ("SSBIntegrationComplex", BinMode.FIRST),
        ("SSBIntegrationComplex", BinMode.DISTRIBUTION),
        ("SSBIntegrationComplex", BinMode.SUM),
        ("NumericalSeparatedWeightedIntegration", BinMode.FIRST),
        ("NumericalSeparatedWeightedIntegration", BinMode.DISTRIBUTION),
        ("NumericalSeparatedWeightedIntegration", BinMode.SUM),
        ("ThresholdedAcquisition", BinMode.FIRST),
        ("ThresholdedAcquisition", BinMode.DISTRIBUTION),
        ("ThresholdedAcquisition", BinMode.SUM),
        ("TriggerCount", BinMode.AVERAGE),
        ("TriggerCount", BinMode.FIRST),
    ],
)
def test_incompatible_bin_mode_qrm_raises(protocol: str, bin_mode: BinMode):
    operation_info = OpInfo(
        name="",
        data={
            "duration": 12e-9,
            "protocol": protocol,
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": bin_mode,
            "port": "some_port",
            "clock": "some_clock",
        },
        timing=0,
    )

    with pytest.raises(
        bin_mode_compat.IncompatibleBinModeError,
        match=re.escape(
            f"{protocol} acquisition on the QRM does not support bin mode "
            f"{operation_info.data['bin_mode']}.\n\n{repr(operation_info)} caused "
            "this exception to occur."
        ),
    ):
        factory_analog.get_operation_strategy(
            operation_info=operation_info,
            channel_name="complex_output_0",
        )


@pytest.mark.parametrize(
    "protocol, bin_mode",
    [
        ("TriggerCount", BinMode.AVERAGE),
        ("TriggerCount", BinMode.DISTRIBUTION),
        ("TriggerCount", BinMode.FIRST),
        ("Timetag", BinMode.DISTRIBUTION),
        ("Timetag", BinMode.FIRST),
        ("Timetag", BinMode.SUM),
        ("Trace", BinMode.APPEND),
        ("Trace", BinMode.AVERAGE),
        ("Trace", BinMode.DISTRIBUTION),
        ("Trace", BinMode.SUM),
        ("TimetagTrace", BinMode.AVERAGE),
        ("TimetagTrace", BinMode.DISTRIBUTION),
        ("TimetagTrace", BinMode.FIRST),
        ("TimetagTrace", BinMode.SUM),
    ],
)
def test_incompatible_bin_mode_qtm_raises(protocol: str, bin_mode: BinMode):
    operation_info = OpInfo(
        name="",
        data={
            "duration": 12e-9,
            "protocol": protocol,
            "acq_channel": 0,
            "acq_index": 0,
            "bin_mode": bin_mode,
            "port": "some_port",
            "clock": "some_clock",
        },
        timing=0,
    )

    with pytest.raises(
        bin_mode_compat.IncompatibleBinModeError,
        match=re.escape(
            f"{protocol} acquisition on the QTM does not support bin mode "
            f"{operation_info.data['bin_mode']}.\n\n{repr(operation_info)} caused "
            "this exception to occur."
        ),
    ):
        factory_timetag.get_operation_strategy(
            operation_info=operation_info,
            channel_name="digital_input_0",
        )
