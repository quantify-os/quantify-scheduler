# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Classes for handling operations that are neither pulses nor acquisitions."""

from __future__ import annotations

from typing import Dict, Any

from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy

from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, q1asm_instructions


class Idle(IOperationStrategy):
    def __init__(self, operation_info: types.OpInfo):
        self._op_info = operation_info

    @property
    def operation_info(self) -> types.OpInfo:
        return self._op_info

    def generate_data(self, wf_dict: Dict[str, Any]):
        """Returns None as no waveforms are generated in this strategy."""
        return None

    def insert_qasm(self, qasm_program: QASMProgram):
        pass


class ClockPhaseShiftStrategy(Idle):
    def insert_qasm(self, qasm_program: QASMProgram):
        phase = self.operation_info.data["phase"]
        phase_args = helpers.get_nco_phase_arguments(phase)
        qasm_program.emit(
            q1asm_instructions.INCR_NCO_PHASE_OFFSET,
            *phase_args,
            comment=f"increment nco phase by {phase:.2f} deg",
        )
