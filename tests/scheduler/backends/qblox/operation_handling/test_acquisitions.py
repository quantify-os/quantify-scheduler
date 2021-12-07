# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Tests for acquisitions module."""

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


class TestSquareAcquisitionStrategy:
    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_constructor(self, bin_mode):
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
