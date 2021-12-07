# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Tests for acquisitions module."""
from typing import Dict, Any

import pytest
import numpy as np

from quantify_scheduler import waveforms

from quantify_scheduler.helpers.waveforms import normalize_waveform_data
from quantify_scheduler.enums import BinMode

from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.register_manager import RegisterManager
from quantify_scheduler.backends.qblox.operation_handling import acquisitions


@pytest.fixture(name="empty_qasm_program")
def fixture_empty_qasm_program():
    static_hw_properties = types.StaticHardwareProperties(
        instrument_type="QRM",
        max_sequencers=constants.NUMBER_OF_SEQUENCERS_QRM,
        max_awg_output_voltage=0.5,
        marker_configuration=types.MarkerConfiguration(start=0b1111, end=0b0000),
        mixer_dc_offset_range=types.BoundedParameter(
            min_val=-0.5, max_val=0.5, units="V"
        ),
    )
    yield QASMProgram(static_hw_properties, RegisterManager())


class MockAcquisition(acquisitions.AcquisitionStrategyPartial):
    """Used for TestAcquisitionStrategyPartial."""

    def generate_data(self, wf_dict: Dict[str, Any]):
        pass

    def acquire_append(self, qasm_program: QASMProgram):
        pass

    def acquire_average(self, qasm_program: QASMProgram):
        pass


class TestAcquisitionStrategyPartial:
    """
    There is some logic in the AcquisitionStrategyPartial class that deserves
    testing.
    """

    def test_operation_info_property(self):
        # arrange
        data = {"bin_mode": BinMode.AVERAGE, "acq_channel": 0, "acq_index": 0}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_bin_mode(self, empty_qasm_program, bin_mode, mocker):
        # arrange
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)
        append_mock = mocker.patch.object(strategy, "acquire_append")
        average_mock = mocker.patch.object(strategy, "acquire_average")

        # act
        strategy.insert_qasm(empty_qasm_program)

        # assert
        if bin_mode == BinMode.AVERAGE:
            average_mock.assert_called_once()
            append_mock.assert_not_called()
        else:
            average_mock.assert_not_called()
            append_mock.assert_called_once()

    def test_invalid_bin_mode(self, empty_qasm_program):
        # arrange
        data = {"bin_mode": "nonsense", "acq_channel": 0, "acq_index": 0}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)

        # act
        with pytest.raises(RuntimeError) as exc:
            strategy.insert_qasm(empty_qasm_program)

        # assert
        assert (
            exc.value.args[0]
            == "Attempting to process an acquisition with unknown bin mode nonsense."
        )

    def test_start_acq_too_soon(self, empty_qasm_program):
        # arrange
        qasm = empty_qasm_program
        qasm.time_last_acquisition_triggered = 0
        data = {
            "bin_mode": "nonsense",
            "acq_channel": 0,
            "acq_index": 0,
            "duration": 1e-6,
        }
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)

        # act
        with pytest.raises(ValueError) as exc:
            strategy.insert_qasm(qasm)

        # assert
        assert (
            exc.value.args[0]
            == "Attempting to start an acquisition at t=0 ns, while the last "
            "acquisition was '\n 'started at t=0. Please ensure a minimum interval of "
            "1000 ns between '\n 'acquisitions.\n'\n '\n'\n 'Error caused by "
            "acquisition:\n'\n 'Acquisition  (t=0 to 1e-06)\n'\n \"data={"
            "'bin_mode': 'nonsense', 'acq_channel': 0, 'acq_index': 0, 'duration': "
            "\"\n '1e-06}."
        )


class TestSquareAcquisitionStrategy:
    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_constructor(self, bin_mode):
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )

    def test_generate_data(self):
        # arrange
        data = {"bin_mode": BinMode.AVERAGE, "acq_channel": 0, "acq_index": 0}
        strategy = acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        wf_dict = {}

        # act
        strategy.generate_data(wf_dict)

        # assert
        assert len(wf_dict) == 0

    def test_acquire_average(self, empty_qasm_program):
        # arrange
        qasm = empty_qasm_program
        data = {
            "bin_mode": BinMode.AVERAGE,
            "acq_channel": 0,
            "acq_index": 0,
            "duration": 1e-6,
        }
        strategy = acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.generate_data({})

        # act
        strategy.acquire_average(qasm)

        # assert
        assert qasm.instructions == [["", "acquire", "0,0,4", ""]]

    def test_acquire_append(self, empty_qasm_program):
        # arrange
        qasm = empty_qasm_program
        data = {
            "bin_mode": BinMode.APPEND,
            "acq_channel": 0,
            "acq_index": 1,
            "duration": 1e-6,
        }
        strategy = acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.operation_info.bin_idx_register = (
            qasm.register_manager.allocate_register()
        )
        strategy.generate_data({})

        # act
        strategy.acquire_append(qasm)

        # assert
        assert qasm.instructions == [
            ["", "", "", ""],
            ["", "acquire", "0,R0,4", ""],
            ["", "add", "R0,1,R0", "# Increment bin_idx for ch0"],
            ["", "", "", ""],
        ]
