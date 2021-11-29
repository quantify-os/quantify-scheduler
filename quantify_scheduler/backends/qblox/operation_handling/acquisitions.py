from __future__ import annotations

from typing import Optional, Dict, Tuple, Any, Union

from abc import ABC, abstractmethod

import numpy as np

from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.waveforms import normalize_waveform_data

from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy, get_indices_from_wf_dict
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, constants, q1asm_instructions


class AcquisitionStrategyPartial(IOperationStrategy):
    def insert_qasm(self, qasm_program: QASMProgram, wf_dict: Dict[str, Any]):
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

        bin_mode = self.operation_info.data["bin_mode"]

        if bin_mode == BinMode.AVERAGE:
            self.acquire_average(qasm_program)
        elif bin_mode == BinMode.APPEND:
            self.acquire_append(qasm_program)
        else:
            raise RuntimeError(
                f"Attempting to process an acquisition with unknown bin "
                f"mode {bin_mode}."
            )

    @abstractmethod
    def acquire_average(self, qasm_program: QASMProgram):
        pass

    @abstractmethod
    def acquire_append(self, qasm_program: QASMProgram):
        pass


class SquareAcquisitionStrategy(AcquisitionStrategyPartial):
    def generate_data(self, output_mode: str) -> None:
        return None

    def acquire_average(self, qasm_program: QASMProgram):
        bin_idx = self.operation_info.data["acq_index"]
        self._acquire_square(qasm_program, bin_idx)

    def acquire_append(self, qasm_program: QASMProgram):
        acquisition = self.operation_info
        acq_bin_idx_reg = acquisition.bin_idx_register

        qasm_program.emit(q1asm_instructions.NEW_LINE)

        self._acquire_square(qasm_program, acq_bin_idx_reg)

        acq_channel = acquisition.data["acq_channel"]
        qasm_program.emit(
            q1asm_instructions.ADD,
            acq_bin_idx_reg,
            1,
            acq_bin_idx_reg,
            comment=f"Increment bin_idx for ch{acq_channel}",
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

        measurement_idx = acquisition.data["acq_channel"]
        qasm_program.emit(
            q1asm_instructions.ACQUIRE,
            measurement_idx,
            bin_idx,
            constants.GRID_TIME,
        )
        qasm_program.elapsed_time += constants.GRID_TIME
