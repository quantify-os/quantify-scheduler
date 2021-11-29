from __future__ import annotations

from typing import Optional, Dict, Tuple, Any, Union

import numpy as np

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
        amplitude = op_info.data["amp"]

        array_with_ones = np.ones(
            int(constants.PULSE_STITCHING_DURATION * constants.SAMPLING_RATE)
        )
        if output_mode == "imag":
            waveform_data = 1j * array_with_ones
            self.amplitude_path0, self.amplitude_path1 = 0, amplitude
        else:
            waveform_data = array_with_ones
            self.amplitude_path0, self.amplitude_path1 = amplitude, 0

        return waveform_data

    def insert_qasm(self, qasm_program: QASMProgram, wf_dict: Dict[str, Any]):
        duration = self.operation_info.duration
        idx0, idx1 = get_indices_from_wf_dict(self.operation_info.uuid, wf_dict=wf_dict)

        repetitions = int(duration // constants.PULSE_STITCHING_DURATION)

        # TODO this has to be fixed to use the amp param instead. I want to get rid of
        #  the runtime settings
        qasm_program.update_runtime_settings(self.operation_info)
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


class StairCasePulseStrategy(IOperationStrategy):
    def __init__(self, operation_info: types.OpInfo):
        self._pulse_info: types.OpInfo = operation_info
        self.output_mode = ""

    @property
    def operation_info(self) -> types.OpInfo:
        return self._pulse_info

    def generate_data(self, output_mode: str) -> None:
        self.output_mode = output_mode
        return None

    def insert_qasm(self, qasm_program: QASMProgram, wf_dict: Dict[str, Any]):
        if self.output_mode == "":
            raise RuntimeError(
                f"Output_mode was never set for pulse {self.operation_info}"
            )

        pulse = self.operation_info

        with qasm_program.temp_register(2) as (offs_reg, offs_reg_zero):
            num_steps = pulse.data["num_steps"]
            start_amp = pulse.data["start_amp"]
            final_amp = pulse.data["final_amp"]
            step_duration = helpers.to_grid_time(pulse.duration / num_steps)

            amp_step = (final_amp - start_amp) / (num_steps - 1)
            amp_step_immediate = qasm_program._expand_from_normalised_range(
                amp_step
                / qasm_program.parent.static_hw_properties.max_awg_output_voltage,
                constants.IMMEDIATE_SZ_OFFSET,
                "offset_awg_path0",
                pulse,
            )
            start_amp_immediate = qasm_program._expand_from_normalised_range(
                start_amp
                / qasm_program.parent.static_hw_properties.max_awg_output_voltage,
                constants.IMMEDIATE_SZ_OFFSET,
                "offset_awg_path0",
                pulse,
            )
            if start_amp_immediate < 0:
                start_amp_immediate += constants.REGISTER_SIZE  # registers are unsigned

            qasm_program.emit(
                q1asm_instructions.SET_AWG_GAIN,
                constants.IMMEDIATE_SZ_GAIN // 2,
                constants.IMMEDIATE_SZ_GAIN // 2,
                comment="set gain to known value",
            )
            qasm_program.emit(
                q1asm_instructions.MOVE,
                start_amp_immediate,
                offs_reg,
                comment="keeps track of the offsets",
            )
            qasm_program.emit(
                q1asm_instructions.MOVE, 0, offs_reg_zero, comment="zero for Q channel"
            )
            qasm_program.emit(q1asm_instructions.NEW_LINE)
            with qasm_program.loop(
                f"ramp{len(qasm_program.instructions)}", repetitions=num_steps
            ):
                qasm_program.emit(
                    q1asm_instructions.SET_AWG_OFFSET, offs_reg, offs_reg_zero
                )
                qasm_program.emit(
                    q1asm_instructions.UPDATE_PARAMETERS,
                    constants.GRID_TIME,
                )
                qasm_program.elapsed_time += constants.GRID_TIME
                if amp_step_immediate >= 0:
                    qasm_program.emit(
                        q1asm_instructions.ADD,
                        offs_reg,
                        amp_step_immediate,
                        offs_reg,
                        comment=f"next incr offs by {amp_step_immediate}",
                    )
                else:
                    qasm_program.emit(
                        q1asm_instructions.SUB,
                        offs_reg,
                        -amp_step_immediate,
                        offs_reg,
                        comment=f"next incr offs by {amp_step_immediate}",
                    )
                qasm_program.auto_wait(step_duration - constants.GRID_TIME)
            qasm_program.elapsed_time += (
                step_duration * (num_steps - 1) if num_steps > 1 else 0
            )

            qasm_program.emit(q1asm_instructions.SET_AWG_OFFSET, 0, 0)
            qasm_program.emit(q1asm_instructions.NEW_LINE)
