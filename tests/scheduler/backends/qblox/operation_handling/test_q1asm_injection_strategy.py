# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Unit tests for the Q1ASMInjectionStrategy class which handles inline Q1ASM compilation.

This test suite verifies:
- Safe label generation and handling
- Register allocation and mapping
- Register cleanup after operations
- Timing updates for Q1ASM operations
"""

from __future__ import annotations

import pytest

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.operation_handling.q1asm_injection_strategy import (
    Q1ASMInjectionStrategy,
)
from quantify_scheduler.backends.qblox.operations.inline_q1asm import (
    InlineQ1ASM,
    Q1ASMOpInfo,
)
from quantify_scheduler.backends.types.qblox import OpInfo


class TestInsertQasm:
    @staticmethod
    def _make_strategy(
        program: str = "",
        safe_labels: bool = True,
        duration: float = 0,
        operation_start_time: float = 0,
    ) -> Q1ASMInjectionStrategy:
        """Create a strategy object given the data for the Q1ASM operation"""
        return Q1ASMInjectionStrategy(
            Q1ASMOpInfo(
                InlineQ1ASM(
                    program=program,
                    duration=duration,
                    port="P",
                    clock="C",
                    safe_labels=safe_labels,
                ),
                operation_start_time=operation_start_time,
            )
        )

    @pytest.mark.parametrize("safe_labels", [True, False])
    def test_safe_labels(self, empty_qasm_program_qcm, safe_labels):
        program = "label2:"
        strategy = self._make_strategy(program, safe_labels)
        strategy.insert_qasm(empty_qasm_program_qcm)
        assert len(empty_qasm_program_qcm.instructions) == 1
        label = "inj0_label2" if safe_labels else "label2"
        expected_instructions = [f"{label}:", "", "", "# [inline] "]
        assert empty_qasm_program_qcm.instructions[0] == expected_instructions

    def test_register_mapping(self, empty_qasm_program_qcm):
        program = "instruction R93, R93, R134 # Comment\n copy R134"
        strategy = self._make_strategy(program, False)
        # Pre-allocate 10 registers to verify mapping works with non-zero starting point
        for _ in range(10):
            empty_qasm_program_qcm.register_manager.allocate_register()
        strategy.insert_qasm(empty_qasm_program_qcm)
        assert len(empty_qasm_program_qcm.instructions) == 2
        expected_instructions = [
            "",
            "instruction",
            "R10,R10,R11",
            "# [inline] Comment",
        ]
        assert empty_qasm_program_qcm.instructions[0] == expected_instructions
        expected_instructions2 = ["", "copy", "R11", "# [inline] "]
        assert empty_qasm_program_qcm.instructions[1] == expected_instructions2

    def test_registers_are_freed(self, empty_qasm_program_qcm):
        program = "instruction R93, R93, R134 # Comment\n copy R134"
        strategy = self._make_strategy(program, False)
        strategy.insert_qasm(empty_qasm_program_qcm)

        assert (
            len(empty_qasm_program_qcm.register_manager.available_registers)
            == constants.NUMBER_OF_REGISTERS
        )

    def test_elapsed_time_updated(self, empty_qasm_program_qcm):
        empty_qasm_program_qcm.elapsed_time = 1  # in seconds
        strategy = self._make_strategy("test", False, duration=3e-9, operation_start_time=5e-9)
        strategy.insert_qasm(empty_qasm_program_qcm)
        assert empty_qasm_program_qcm.elapsed_time == 4
        assert type(empty_qasm_program_qcm.elapsed_time) is int

    @pytest.mark.parametrize(
        "incorrect_program",
        [
            ":",  # Empty label
            "1,2,3",  # Arguments without an instruction
            "move 1,,4",  # Empty argument in middle
            "move 1,",  # Empty argument at end
            "move ,4",  # Empty argument at start
            "move ,",  # Empty argument at start and end
        ],
    )
    def test_incorrect_program(self, incorrect_program, empty_qasm_program_qcm):
        strategy = self._make_strategy(incorrect_program)
        with pytest.raises(ValueError):
            strategy.insert_qasm(empty_qasm_program_qcm)
