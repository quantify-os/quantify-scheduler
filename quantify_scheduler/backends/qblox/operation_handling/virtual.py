# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Classes for handling operations that are neither pulses nor acquisitions."""

from __future__ import annotations

from typing import Dict, Any

from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy

from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, q1asm_instructions


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
        Inserts the instructions needed to shift the nco phase by a specific amount.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
        phase = self.operation_info.data["phase"]
        phase_args = helpers.get_nco_phase_arguments(phase)
        qasm_program.emit(
            q1asm_instructions.INCR_NCO_PHASE_OFFSET,
            *phase_args,
            comment=f"increment nco phase by {phase:.2f} deg",
        )
