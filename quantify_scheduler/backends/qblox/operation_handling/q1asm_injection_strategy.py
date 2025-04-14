# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Classes for handling operations that are neither pulses nor acquisitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.operations.inline_q1asm import Q1ASMOpInfo
    from quantify_scheduler.backends.types import qblox as types


class Q1ASMInjectionStrategy(IOperationStrategy):
    """Strategy for compiling an "inline Q1ASM" instruction block."""

    def __init__(self, operation_info: Q1ASMOpInfo) -> None:
        """
        Initialize the Q1ASMInjectionStrategy with the given operation information.

        Parameters
        ----------
        operation_info
            The operation information containing data required for this strategy.

        """
        self._op_info = operation_info

    @property
    def operation_info(self) -> types.OpInfo:
        """Property for retrieving the operation info."""
        return self._op_info

    def generate_data(self, wf_dict: dict[str, Any]) -> None:
        """
        Generates the waveform data and adds them to the wf_dict
        (if not already present).
        This is either the awg data, or the acquisition weights.

        Parameters
        ----------
        wf_dict
            The dictionary to add the waveform to.
            N.B. the dictionary is modified in function.

        """
        op_info = self.operation_info
        waveforms = op_info.data["waveforms"]
        for waveform in waveforms:
            if waveform in wf_dict:
                raise RuntimeError(
                    f"Duplicate waveform name {waveform} in list of defined waveforms."
                )
        wf_dict.update(waveforms)

    def insert_qasm(self, qasm_program: QASMProgram) -> None:
        """
        Add the inline Q1ASM program for the Q1 sequence processor.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.

        """
        safe_labels: bool = self.operation_info.data["safe_labels"]

        # split on newlines and only take non-empty lines
        program_lines = self.operation_info.data["program"].splitlines()

        register_mapping = {}
        label_prefix = f"inj{len(qasm_program.instructions)}_" if safe_labels else ""
        for line in program_lines:
            instruction, arguments, label, comment = QASMProgram.parse_program_line(line)
            if label is None and instruction == "" and arguments == [] and comment == "":
                # line was empty
                continue
            if label:
                label = f"{label_prefix}{label}"
            # map registers and labels
            for i, argument in enumerate(arguments):
                if argument[0] == "R":
                    register = argument
                    if register in register_mapping:
                        arguments[i] = register_mapping[register]
                    else:
                        arguments[i] = qasm_program.register_manager.allocate_register()
                        register_mapping[register] = arguments[i]
                elif argument[0] == "@":
                    arguments[i] = f"@{label_prefix}{argument[1:]}"

            qasm_program.emit(instruction, *arguments, label=label, comment="[inline] " + comment)

        # Free the registers
        for register in reversed(register_mapping.values()):
            qasm_program.register_manager.free_register(register)

        qasm_program.elapsed_time += round(self.operation_info.data["duration"] * 1e9)
