from __future__ import annotations

from typing import Optional, Dict, Tuple, Any, Union

from abc import ABC, abstractmethod

import numpy as np

from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.waveforms import normalize_waveform_data

from quantify_scheduler.backends.qblox.operation_handling.base import (
    IOperationStrategy,
    get_indices_from_wf_dict,
)
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, constants, q1asm_instructions


class GenericPulseStrategy(IOperationStrategy):
    def __init__(self, operation_info: types.OpInfo):
        self._pulse_info: types.OpInfo = operation_info

        self.amplitude_path0: Optional[float] = None
        self.amplitude_path1: Optional[float] = None

    @property
    def operation_info(self) -> types.OpInfo:
        return self._pulse_info

    def generate_data(self, output_mode: str) -> np.ndarray:
        op_info = self.operation_info
        waveform_data = helpers.generate_waveform_data(
            op_info.data, sampling_rate=constants.SAMPLING_RATE
        )
        waveform_data, amp_real, amp_imag = normalize_waveform_data(waveform_data)

        if output_mode == "imag":
            self.amplitude_path0, self.amplitude_path1 = amp_imag, amp_real
        else:
            self.amplitude_path0, self.amplitude_path1 = amp_real, amp_imag

        return waveform_data

    def insert_qasm(self, qasm_program: QASMProgram, wf_dict: Dict[str, Any]):
        op_info = self.operation_info
        idx0, idx1 = get_indices_from_wf_dict(op_info.uuid, wf_dict=wf_dict)

        qasm_program.wait_till_start_operation(
            op_info
        )  # TODO move line outside of this class
        qasm_program.update_runtime_settings(op_info)
        qasm_program.emit(q1asm_instructions.PLAY, idx0, idx1, constants.GRID_TIME)
        qasm_program.elapsed_time += constants.GRID_TIME


class StitchedSquarePulseStrategy(IOperationStrategy):
    def __init__(self, operation_info: types.OpInfo):
        self._pulse_info: types.OpInfo = operation_info

        self.amplitude_path0: Optional[float] = None
        self.amplitude_path1: Optional[float] = None

    @property
    def operation_info(self) -> types.OpInfo:
        return self._pulse_info

    def generate_data(self, output_mode: str) -> np.ndarray:
        op_info = self.operation_info
        waveform_data = helpers.generate_waveform_data(
            op_info.data, sampling_rate=constants.SAMPLING_RATE
        )
        waveform_data, amp_real, amp_imag = normalize_waveform_data(waveform_data)

        if output_mode == "imag":
            self.amplitude_path0, self.amplitude_path1 = amp_imag, amp_real
        else:
            self.amplitude_path0, self.amplitude_path1 = amp_real, amp_imag

        return waveform_data

    def insert_qasm(self, qasm_program: QASMProgram, wf_dict: Dict[str, Any]):
        duration = self.operation_info.duration
        idx0, idx1 = get_indices_from_wf_dict(self.operation_info.uuid, wf_dict=wf_dict)

        repetitions = int(duration // constants.PULSE_STITCHING_DURATION)

        if repetitions > 0:
            with qasm_program.loop(
                label=f"stitch{len(qasm_program.instructions)}",
                repetitions=repetitions,
            ):
                qasm_program.emit(
                    q1asm_instructions.PLAY,
                    idx0,
                    idx1,
                    helpers.to_grid_time(constants.PULSE_STITCHING_DURATION),
                )
                qasm_program.elapsed_time += repetitions * helpers.to_grid_time(
                    constants.PULSE_STITCHING_DURATION
                )

        pulse_time_remaining = helpers.to_grid_time(
            duration % constants.PULSE_STITCHING_DURATION
        )
        if pulse_time_remaining > 0:
            qasm_program.emit(q1asm_instructions.PLAY, idx0, idx1, pulse_time_remaining)
            qasm_program.emit(
                q1asm_instructions.SET_AWG_GAIN,
                0,
                0,
                comment="set to 0 at end of pulse",
            )
        qasm_program.elapsed_time += pulse_time_remaining
