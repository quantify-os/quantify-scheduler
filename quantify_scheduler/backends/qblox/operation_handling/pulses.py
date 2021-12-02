from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from quantify_scheduler.helpers.waveforms import normalize_waveform_data

from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy

from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, constants, q1asm_instructions


class PulseStrategyPartial(IOperationStrategy):
    def __init__(self, operation_info: types.OpInfo, output_mode: str):
        self._pulse_info: types.OpInfo = operation_info
        self.output_mode = output_mode

    @property
    def operation_info(self) -> types.OpInfo:
        return self._pulse_info


class GenericPulseStrategy(PulseStrategyPartial):
    def __init__(self, operation_info: types.OpInfo, output_mode: str):
        super().__init__(operation_info, output_mode)

        self.amplitude_path0: Optional[float] = None
        self.amplitude_path1: Optional[float] = None

        self.waveform_index0: Optional[int] = None
        self.waveform_index1: Optional[int] = None

        self.waveform_len: Optional[int] = None

    def generate_data(self, wf_dict: Dict[str, Any]):
        op_info = self.operation_info
        waveform_data = helpers.generate_waveform_data(
            op_info.data, sampling_rate=constants.SAMPLING_RATE
        )
        waveform_data, amp_real, amp_imag = normalize_waveform_data(waveform_data)
        self.waveform_len = len(waveform_data)

        _, _, idx_real = helpers.add_to_wf_dict_if_unique(wf_dict, waveform_data.real)
        _, _, idx_imag = helpers.add_to_wf_dict_if_unique(wf_dict, waveform_data.imag)

        if self.output_mode == "imag":
            self.waveform_index0, self.waveform_index1 = idx_real, idx_imag
            self.amplitude_path0, self.amplitude_path1 = amp_imag, amp_real
        else:
            self.waveform_index0, self.waveform_index1 = idx_real, idx_imag
            self.amplitude_path0, self.amplitude_path1 = amp_real, amp_imag

    def insert_qasm(self, qasm_program: QASMProgram):
        qasm_program.set_gain_from_amplitude(
            self.amplitude_path0, self.amplitude_path1, self.operation_info
        )
        qasm_program.emit(
            q1asm_instructions.PLAY,
            self.waveform_index0,
            self.waveform_index1,
            constants.GRID_TIME,
            comment=f"play {self.operation_info.name} ({self.waveform_len} ns)",
        )
        qasm_program.elapsed_time += constants.GRID_TIME


class StitchedSquarePulseStrategy(PulseStrategyPartial):
    def __init__(self, operation_info: types.OpInfo, output_mode: str):
        super().__init__(operation_info, output_mode)

        self.amplitude_path0: Optional[float] = None
        self.amplitude_path1: Optional[float] = None

        self.waveform_index0: Optional[int] = None
        self.waveform_index1: Optional[int] = None

    def generate_data(self, wf_dict: Dict[str, Any]):
        op_info = self.operation_info
        amplitude = op_info.data["amp"]

        array_with_ones = np.ones(
            int(constants.PULSE_STITCHING_DURATION * constants.SAMPLING_RATE)
        )
        _, _, idx_ones = helpers.add_to_wf_dict_if_unique(wf_dict, array_with_ones.real)
        _, _, idx_zeros = helpers.add_to_wf_dict_if_unique(
            wf_dict, array_with_ones.imag
        )
        if self.output_mode == "imag":
            self.waveform_index0, self.waveform_index1 = idx_zeros, idx_ones
            self.amplitude_path0, self.amplitude_path1 = 0, amplitude
        else:
            self.waveform_index0, self.waveform_index1 = idx_ones, idx_zeros
            self.amplitude_path0, self.amplitude_path1 = amplitude, 0

    def insert_qasm(self, qasm_program: QASMProgram):
        duration = self.operation_info.duration
        repetitions = int(duration // constants.PULSE_STITCHING_DURATION)

        qasm_program.set_gain_from_amplitude(
            self.amplitude_path0, self.amplitude_path1, self.operation_info
        )
        if repetitions > 0:
            with qasm_program.loop(
                label=f"stitch{len(qasm_program.instructions)}",
                repetitions=repetitions,
            ):
                qasm_program.emit(
                    q1asm_instructions.PLAY,
                    self.waveform_index0,
                    self.waveform_index1,
                    helpers.to_grid_time(constants.PULSE_STITCHING_DURATION),
                )
                qasm_program.elapsed_time += repetitions * helpers.to_grid_time(
                    constants.PULSE_STITCHING_DURATION
                )

        pulse_time_remaining = helpers.to_grid_time(
            duration % constants.PULSE_STITCHING_DURATION
        )
        if pulse_time_remaining > 0:
            qasm_program.emit(
                q1asm_instructions.PLAY,
                self.waveform_index0,
                self.waveform_index1,
                pulse_time_remaining,
            )
            qasm_program.emit(
                q1asm_instructions.SET_AWG_GAIN,
                0,
                0,
                comment="set to 0 at end of pulse",
            )
        qasm_program.elapsed_time += pulse_time_remaining


class StaircasePulseStrategy(PulseStrategyPartial):
    def __init__(self, operation_info: types.OpInfo, output_mode: str):
        super().__init__(operation_info, output_mode)

    def generate_data(self, wf_dict: Dict[str, Any]):
        return None

    def insert_qasm(self, qasm_program: QASMProgram):
        pulse = self.operation_info
        num_steps = pulse.data["num_steps"]
        start_amp = pulse.data["start_amp"]
        final_amp = pulse.data["final_amp"]
        step_duration_ns = helpers.to_grid_time(pulse.duration / num_steps)

        offset_param_label = (
            "offset_awg_path1" if self.output_mode == "imag" else "offset_awg_path0"
        )

        amp_step = (final_amp - start_amp) / (num_steps - 1)
        amp_step_immediate = qasm_program._expand_from_normalised_range(
            amp_step / qasm_program.parent.static_hw_properties.max_awg_output_voltage,
            constants.IMMEDIATE_SZ_OFFSET,
            offset_param_label,
            pulse,
        )
        start_amp_immediate = qasm_program._expand_from_normalised_range(
            start_amp / qasm_program.parent.static_hw_properties.max_awg_output_voltage,
            constants.IMMEDIATE_SZ_OFFSET,
            offset_param_label,
            pulse,
        )
        if start_amp_immediate < 0:
            start_amp_immediate += constants.REGISTER_SIZE  # registers are unsigned

        self._generate_staircase_loop(
            qasm_program,
            start_amp_immediate,
            amp_step_immediate,
            step_duration_ns,
            num_steps,
        )

    def _generate_staircase_loop(
        self,
        qasm_program: QASMProgram,
        start_amp_immediate: int,
        amp_step_immediate: int,
        step_duration_ns: int,
        num_steps: int,
    ):
        with qasm_program.temp_register(2) as (offs_reg, offs_reg_zero):
            qasm_program.emit(
                q1asm_instructions.SET_AWG_GAIN,
                constants.IMMEDIATE_SZ_GAIN // 2,
                constants.IMMEDIATE_SZ_GAIN // 2,
                comment="set gain to known value",
            )

            # Initialize registers
            qasm_program.emit(
                q1asm_instructions.MOVE,
                start_amp_immediate,
                offs_reg,
                comment="keeps track of the offsets",
            )
            qasm_program.emit(
                q1asm_instructions.MOVE,
                0,
                offs_reg_zero,
                comment="zero for unused output path",
            )

            qasm_program.emit(q1asm_instructions.NEW_LINE)
            with qasm_program.loop(
                f"ramp{len(qasm_program.instructions)}", repetitions=num_steps
            ):
                self._generate_step(
                    qasm_program,
                    offs_reg,
                    offs_reg_zero,
                    amp_step_immediate,
                )
                qasm_program.auto_wait(step_duration_ns - constants.GRID_TIME)

            qasm_program.elapsed_time += (
                step_duration_ns * (num_steps - 1) if num_steps > 1 else 0
            )

            qasm_program.emit(
                q1asm_instructions.SET_AWG_OFFSET,
                0,
                0,
                comment="return offset to 0 after staircase.",
            )
            qasm_program.emit(q1asm_instructions.NEW_LINE)

    def _generate_step(
        self,
        qasm_program: QASMProgram,
        offs_reg: str,
        offs_reg_zero: str,
        amp_step_immediate: int,
    ):
        if self.output_mode == "imag":
            qasm_program.emit(
                q1asm_instructions.SET_AWG_OFFSET, offs_reg_zero, offs_reg
            )
        else:
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
