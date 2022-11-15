# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Classes for handling acquisitions."""

from __future__ import annotations

from typing import Optional, Dict, Any, Union

from abc import abstractmethod

import numpy as np

from quantify_scheduler.enums import BinMode

from quantify_scheduler.backends.qblox.operation_handling.base import (
    IOperationStrategy,
)
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, constants, q1asm_instructions


class AcquisitionStrategyPartial(IOperationStrategy):
    """Contains the logic shared between all the acquisitions."""

    def __init__(self, operation_info: types.OpInfo):
        """Constructor of superclass."""
        self._acq_info: types.OpInfo = operation_info
        self.bin_mode: BinMode = operation_info.data["bin_mode"]
        self.acq_channel = operation_info.data["acq_channel"]
        self.bin_idx_register: Optional[str] = None
        """The register used to keep track of the bin index, only not None for append
        mode acquisitions."""

    def insert_qasm(self, qasm_program: QASMProgram):
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this acquisition. This function calls either acquire_average or acquire_append,
        depending on the bin mode.

        The acquire_average and acquire_append are to be implemented in the subclass.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        if qasm_program.time_last_acquisition_triggered is not None:
            if (
                qasm_program.elapsed_time - qasm_program.time_last_acquisition_triggered
                < constants.MIN_TIME_BETWEEN_ACQUISITIONS
            ):
                raise ValueError(
                    f"Attempting to start an acquisition at t="
                    f"{qasm_program.elapsed_time} ns, while the last acquisition was "
                    f"started at t={qasm_program.time_last_acquisition_triggered} ns. "
                    f"Please ensure a minimum interval of "
                    f"{constants.MIN_TIME_BETWEEN_ACQUISITIONS} ns between "
                    f"acquisitions.\n\nError caused by acquisition:\n"
                    f"{repr(self.operation_info)}."
                )

        qasm_program.time_last_acquisition_triggered = qasm_program.elapsed_time

        if self.bin_mode == BinMode.AVERAGE:
            if self.bin_idx_register is not None:
                raise ValueError(
                    "Attempting to add acquisition with average binmode. "
                    "bin_idx_register cannot be None."
                )
            self.acquire_average(qasm_program)
        elif self.bin_mode == BinMode.APPEND:
            if self.bin_idx_register is None:
                raise ValueError(
                    "Attempting to add acquisition with append binmode. "
                    "bin_idx_register cannot be None."
                )
            self.acquire_append(qasm_program)
        else:
            raise RuntimeError(
                f"Attempting to process an acquisition with unknown bin "
                f"mode {self.bin_mode}."
            )

    @abstractmethod
    def acquire_average(self, qasm_program: QASMProgram):
        """Adds the assembly to the program for a bin_mode==AVERAGE acquisition."""

    @abstractmethod
    def acquire_append(self, qasm_program: QASMProgram):
        """Adds the assembly to the program for a bin_mode==APPEND acquisition."""

    @property
    def operation_info(self) -> types.OpInfo:
        """Property for retrieving the operation info."""
        return self._acq_info


class SquareAcquisitionStrategy(AcquisitionStrategyPartial):
    """
    Performs a square acquisition (i.e. without acquisition weights).
    """

    def generate_data(self, wf_dict: Dict[str, Any]) -> None:
        """Returns None as no waveform is needed."""
        return None

    def acquire_average(self, qasm_program: QASMProgram):
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this acquisition, assuming averaging is used.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        bin_idx = self.operation_info.data["acq_index"]
        self._acquire_square(qasm_program, bin_idx)

    def acquire_append(self, qasm_program: QASMProgram):
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this acquisition, assuming append is used.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        acq_bin_idx_reg = self.bin_idx_register

        qasm_program.emit(q1asm_instructions.NEW_LINE)

        if self.bin_idx_register is None:
            raise ValueError(
                "Attempting to add acquisition with append binmode. "
                "bin_idx_register cannot be None."
            )
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
            The bin_idx to store the result in, can be either an int (for immediates) or
            a str (for registers).
        """
        qasm_program.verify_square_acquisition_duration(
            self.operation_info, self.operation_info.duration
        )

        qasm_program.emit(
            q1asm_instructions.ACQUIRE,
            self.acq_channel,
            bin_idx,
            constants.GRID_TIME,
        )
        qasm_program.elapsed_time += constants.GRID_TIME


class WeightedAcquisitionStrategy(AcquisitionStrategyPartial):
    """
    Performs a weighted acquisition.
    """

    def __init__(self, operation_info: types.OpInfo):
        """
        Constructor for this strategy.

        Parameters
        ----------
        operation_info
            The operation info that corresponds to this acquisition.
        """
        super().__init__(operation_info)
        self.waveform_index0: Optional[int] = None
        self.waveform_index1: Optional[int] = None

    def generate_data(self, wf_dict: Dict[str, Any]):
        """
        Generates the waveform data for both acquisition weights.

        Parameters
        ----------
        wf_dict
            The dictionary to add the waveform to. N.B. the dictionary is modified in
            function.
        """
        waveform_indices = []
        for idx, parameterized_waveform in enumerate(
            self.operation_info.data["waveforms"]
        ):
            if idx > 1:
                raise ValueError(
                    f"Too many waveforms ("
                    f"{len(self.operation_info.data['waveforms'])}) "
                    f"specified as acquisition weights. Qblox hardware "
                    f"only supports 2 real valued arrays as acquisition "
                    f"weights.\n\nException caused by "
                    f"{repr(self.operation_info)}."
                )
            waveform_data = helpers.generate_waveform_data(
                parameterized_waveform, sampling_rate=constants.SAMPLING_RATE
            )
            if not np.isrealobj(waveform_data):
                raise ValueError(
                    f"Complex weights not supported by hardware. Please use two 1d "
                    f"real-valued weights.\n\nException was triggered because of "
                    f"{repr(self.operation_info)}."
                )
            _, _, waveform_index = helpers.add_to_wf_dict_if_unique(
                wf_dict, waveform_data
            )
            waveform_indices.append(waveform_index)

        self.waveform_index0, self.waveform_index1 = waveform_indices

    def acquire_average(self, qasm_program: QASMProgram):
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this acquisition, assuming averaging is used.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        bin_idx = self.operation_info.data["acq_index"]

        qasm_program.emit(
            q1asm_instructions.ACQUIRE_WEIGHED,
            self.acq_channel,
            bin_idx,
            self.waveform_index0,
            self.waveform_index1,
            constants.GRID_TIME,
            comment=f"Store acq in acq_channel:{self.acq_channel}, bin_idx:{bin_idx}",
        )
        qasm_program.elapsed_time += constants.GRID_TIME

    def acquire_append(self, qasm_program: QASMProgram):
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this acquisition, assuming append is used. Registers will be used for the weight
        indexes and the bin index.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        acq_bin_idx_reg = self.bin_idx_register

        with qasm_program.temp_registers(2) as (acq_idx0_reg, acq_idx1_reg):
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
                acq_idx0_reg,
                acq_idx1_reg,
                constants.GRID_TIME,
                comment=f"Store acq in acq_channel:{self.acq_channel}, "
                f"bin_idx:{acq_bin_idx_reg}",
            )
            qasm_program.emit(q1asm_instructions.NEW_LINE)
            qasm_program.elapsed_time += constants.GRID_TIME
