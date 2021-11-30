from __future__ import annotations

from typing import Optional, Dict, Tuple, Any, Union

from abc import ABC, abstractmethod

import numpy as np

from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.waveforms import normalize_waveform_data

from quantify_scheduler.backends.qblox.operation_handling.base import (
    IOperationStrategy,
)
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, constants, q1asm_instructions


class AcquisitionStrategyPartial(IOperationStrategy):
    def __init__(self, operation_info: types.OpInfo):
        self._acq_info: types.OpInfo = operation_info
        self.bin_mode: BinMode = operation_info.data["bin_mode"]
        self.acq_channel = operation_info.data["acq_channel"]

    def insert_qasm(self, qasm_program: QASMProgram):
        if qasm_program.time_last_acquisition_triggered is not None:
            if (
                qasm_program.elapsed_time - qasm_program.time_last_acquisition_triggered
                < constants.MIN_TIME_BETWEEN_ACQUISITIONS
            ):
                raise ValueError(
                    f"Attempting to start an acquisition at t={qasm_program.elapsed_time} "
                    f"ns, while the last acquisition was started at "
                    f"t={qasm_program.time_last_acquisition_triggered}. Please ensure "
                    f"a minimum interval of "
                    f"{constants.MIN_TIME_BETWEEN_ACQUISITIONS} ns between "
                    f"acquisitions.\n\nError caused by acquisition:\n"
                    f"{repr(self.operation_info)}."
                )

        qasm_program.time_last_acquisition_triggered = qasm_program.elapsed_time

        if self.bin_mode == BinMode.AVERAGE:
            self.acquire_average(qasm_program)
        elif self.bin_mode == BinMode.APPEND:
            self.acquire_append(qasm_program)
        else:
            raise RuntimeError(
                f"Attempting to process an acquisition with unknown bin "
                f"mode {self.bin_mode}."
            )

    @abstractmethod
    def acquire_average(self, qasm_program: QASMProgram):
        pass

    @abstractmethod
    def acquire_append(self, qasm_program: QASMProgram):
        pass

    @property
    def operation_info(self) -> types.OpInfo:
        return self._acq_info


class SquareAcquisitionStrategy(AcquisitionStrategyPartial):
    def generate_data(self, wf_dict: Dict[str, Any]) -> None:
        return None

    def acquire_average(self, qasm_program: QASMProgram):
        bin_idx = self.operation_info.data["acq_index"]
        self._acquire_square(qasm_program, bin_idx)

    def acquire_append(self, qasm_program: QASMProgram):
        acquisition = self.operation_info
        acq_bin_idx_reg = acquisition.bin_idx_register

        qasm_program.emit(q1asm_instructions.NEW_LINE)

        self._acquire_square(qasm_program, acq_bin_idx_reg)

        qasm_program.emit(
            q1asm_instructions.ADD,
            acq_bin_idx_reg,
            1,
            acq_bin_idx_reg,
            comment=f"Increment bin_idx for ch{self.acq_channel}",
        )
        qasm_program.emit(q1asm_instructions.NEW_LINE)

    def _acquire_square(
        self, qasm_program: QASMProgram, bin_idx: Union[int, str]
    ) -> None:
        """
        Adds the instruction for performing acquisitions without weights playback.

        Parameters
        ----------
        qasm_program
            The qasm program to add the acquisition to.
        bin_idx
            The bin_idx to store the result in.
        """
        acquisition = self.operation_info

        qasm_program.verify_square_acquisition_duration(
            acquisition, acquisition.duration
        )

        qasm_program.emit(
            q1asm_instructions.ACQUIRE,
            self.acq_channel,
            bin_idx,
            constants.GRID_TIME,
        )
        qasm_program.elapsed_time += constants.GRID_TIME


class WeightedAcquisitionStrategy(AcquisitionStrategyPartial):
    def __init__(self, operation_info: types.OpInfo):
        super().__init__(operation_info)
        self.waveform_index0: Optional[int] = None
        self.waveform_index1: Optional[int] = None

    def generate_data(self, wf_dict: Dict[str, Any]):
        waveform_data = helpers.generate_waveform_data(
            self.operation_info.data, sampling_rate=constants.SAMPLING_RATE
        )
        _, _, idx_real = helpers.add_to_wf_dict_if_unique(wf_dict, waveform_data.real)
        _, _, idx_imag = helpers.add_to_wf_dict_if_unique(wf_dict, waveform_data.imag)
        self.waveform_index0, self.waveform_index1 = idx_real, idx_imag

    def acquire_average(self, qasm_program: QASMProgram):
        acquisition = self.operation_info

        bin_idx = self.operation_info.data["acq_index"]
        measurement_idx = acquisition.data["acq_channel"]

        qasm_program.emit(
            q1asm_instructions.ACQUIRE_WEIGHED,
            measurement_idx,
            bin_idx,
            self.waveform_index0,
            self.waveform_index1,
            constants.GRID_TIME,
            comment=f"Store acq in acq_channel:{measurement_idx}, bin_idx:{bin_idx}",
        )
        qasm_program.elapsed_time += constants.GRID_TIME

    def acquire_append(self, qasm_program: QASMProgram):
        acq_bin_idx_reg = self.operation_info.bin_idx_register

        with qasm_program.temp_register(2) as (acq_idx0_reg, acq_idx1_reg):
            qasm_program.emit(q1asm_instructions.NEW_LINE)
            qasm_program.emit(
                q1asm_instructions.MOVE,
                self.waveform_index0,
                acq_idx0_reg,
                comment=f"Store idx of acq I wave in {acq_idx0_reg}",
            )
            qasm_program.emit(
                q1asm_instructions.MOVE,
                self.waveform_index1,
                acq_idx1_reg,
                comment=f"Store idx of acq Q wave in {acq_idx1_reg}.",
            )

            qasm_program.emit(
                q1asm_instructions.ACQUIRE_WEIGHED,
                self.acq_channel,
                acq_bin_idx_reg,
                self.waveform_index0,
                self.waveform_index1,
                constants.GRID_TIME,
                comment=f"Store acq in acq_channel:{self.acq_channel}, "
                f"bin_idx:{acq_bin_idx_reg}",
            )
            qasm_program.emit(q1asm_instructions.NEW_LINE)
            qasm_program.elapsed_time += constants.GRID_TIME
