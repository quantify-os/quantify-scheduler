# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Classes for handling operations that are neither pulses nor acquisitions."""

from typing import Dict, Any

import numpy as np

from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy

from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import constants, helpers, q1asm_instructions


class IdleStrategy(IOperationStrategy):
    """Defines the behavior for an operation that does not produce any output."""

    def __init__(self, operation_info: types.OpInfo):
        """
        Constructor for the IdleStrategy class.

        Parameters
        ----------
        operation_info:
            The operation info that corresponds to this operation.
        """
        self._op_info = operation_info

    @property
    def operation_info(self) -> types.OpInfo:
        """Property for retrieving the operation info."""
        return self._op_info

    def generate_data(self, wf_dict: Dict[str, Any]):
        """Returns None as no waveforms are generated in this strategy."""
        return None

    def insert_qasm(self, qasm_program: QASMProgram):
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this operation.

        Not an abstractmethod, since it is allowed to use the IdleStrategy directly
        (e.g. for IdlePulses), but can be overridden in subclass to add some assembly
        instructions despite not outputting any data.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """


class NcoPhaseShiftStrategy(IdleStrategy):
    """Strategy for operation that does not produce any output, but rather applies a
    phase shift to the NCO."""

    def insert_qasm(self, qasm_program: QASMProgram):
        """
        Inserts the instructions needed to shift the NCO phase by a specific amount.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        phase = self.operation_info.data.get("phase_shift")
        if phase != 0:
            phase_arg = helpers.get_nco_phase_arguments(phase)
            qasm_program.emit(
                q1asm_instructions.INCR_NCO_PHASE_OFFSET,
                phase_arg,
                comment=f"increment nco phase by {phase:.2f} deg",
            )


class NcoResetClockPhaseStrategy(IdleStrategy):
    """Strategy for operation that does not produce any output, but rather resets
    the phase of the NCO."""

    def insert_qasm(self, qasm_program: QASMProgram):
        """
        Inserts the instructions needed to reset the NCO phase.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        reset_clock_phase = self.operation_info.data.get("reset_clock_phase")
        if reset_clock_phase is None:
            raise KeyError(
                "NcoResetClockPhaseStrategy called, "
                "but reset_clock_phase not present in operation_info.data"
            )
        qasm_program.emit(q1asm_instructions.RESET_PHASE)


class NcoSetClockFrequencyStrategy(IdleStrategy):
    """
    Strategy for operation that does not produce any output, but rather sets
    the frequency of the NCO. Implemented as `set_freq` and an `upd_param` of 8 ns,
    leading to a total duration of 8 ns before the next command can be issued.
    """

    def __init__(self, operation_info: types.OpInfo):
        """
        Constructor for the NcoSetClockFrequencyStrategy class.

        Parameters
        ----------
        operation_info
            The operation info that corresponds to this operation.
        """
        super().__init__(operation_info)

    def insert_qasm(self, qasm_program: QASMProgram):
        """
        Inserts the instructions needed to set the NCO frequency.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        clock_freq_new = self.operation_info.data.get("clock_freq_new")
        clock_freq_old = self.operation_info.data.get("clock_freq_old")
        interm_freq_old = self.operation_info.data.get("interm_freq_old")

        if clock_freq_old is None or np.isnan(clock_freq_old):
            raise RuntimeError(
                f"Clock '{self.operation_info.data.get('clock')}' has an undefined "
                f"initial frequency ({clock_freq_old=}); "
                f"ensure this resource has been added to the schedule or to the device "
                f"config."
            )
        if interm_freq_old is None:
            raise RuntimeError(
                f"Clock '{self.operation_info.data.get('clock')}' has an undefined "
                f"associated intermodulation frequency ({interm_freq_old=}); make "
                f"sure an 'interm_freq' is supplied or that 'mix_lo' is set to true in "
                f"the hardware config."
            )
        iterm_freq_new = interm_freq_old + clock_freq_new - clock_freq_old

        frequency_args = helpers.get_nco_set_frequency_arguments(iterm_freq_new)
        qasm_program.emit(
            q1asm_instructions.SET_FREQUENCY,
            frequency_args,
            comment=f"set nco frequency to {iterm_freq_new:e} Hz",
        )
        qasm_program.emit(
            q1asm_instructions.UPDATE_PARAMETERS,
            constants.NCO_SET_FREQ_WAIT,
            comment=f"apply nco frequency change",
        )
        qasm_program.elapsed_time += constants.NCO_SET_FREQ_WAIT
