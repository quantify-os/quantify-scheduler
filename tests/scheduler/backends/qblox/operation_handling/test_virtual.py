# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for virtual strategy module."""
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=no-self-use
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

from typing import Tuple
import pytest

from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.instrument_compilers import QcmModule
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.register_manager import RegisterManager

from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.operation_handling import virtual


@pytest.fixture(name="empty_qasm_program_qcm")
def fixture_empty_qasm_program():
    yield QASMProgram(QcmModule.static_hw_properties, RegisterManager())


def _assert_none_data(strategy: IOperationStrategy):
    # pylint: disable=assignment-from-none
    # this is what we want to verify
    data = strategy.generate_data({})

    # assert
    assert data is None


class TestIdleStrategy:
    def test_constructor(self):
        virtual.IdleStrategy(types.OpInfo(name="", data={}, timing=0))

    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = virtual.IdleStrategy(op_info)

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    def test_generate_data(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = virtual.IdleStrategy(op_info)

        # act and assert
        _assert_none_data(strategy)

    def test_generate_qasm_program(self, empty_qasm_program_qcm: QASMProgram):
        # arrange
        qasm = empty_qasm_program_qcm
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = virtual.IdleStrategy(op_info)

        # act
        strategy.insert_qasm(qasm)

        # assert
        assert len(qasm.instructions) == 0


class TestNcoPhaseShiftStrategy:
    def test_constructor(self):
        virtual.NcoPhaseShiftStrategy(
            types.OpInfo(name="", data={"phase": 123.456}, timing=0)
        )

    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={"phase": 123.456}, timing=0)
        strategy = virtual.NcoPhaseShiftStrategy(op_info)

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    def test_generate_data(self):
        # arrange
        op_info = types.OpInfo(name="", data={"phase": 123.456}, timing=0)
        strategy = virtual.NcoPhaseShiftStrategy(op_info)

        # act and assert
        _assert_none_data(strategy)

    @pytest.mark.parametrize(
        "phase, answer",
        [
            (0.0, ("set_ph_delta", "0,0,0")),
            (360, ("set_ph_delta", "0,0,0")),
            (360.0, ("set_ph_delta", "0,0,0")),
            (359.99999999999999, ("set_ph_delta", "0,0,0")),
            (359.999, ("set_ph_delta", "399,399,3472")),
            (123.123, ("set_ph_delta", "136,321,2083")),
            (483.123, ("set_ph_delta", "136,321,2083")),
        ],
    )
    def test_generate_qasm_program(
        self, phase: float, answer: Tuple[str, str], empty_qasm_program_qcm: QASMProgram
    ):
        def extract_instruction_and_args(qasm_prog: QASMProgram) -> Tuple[str, str]:
            return qasm_prog.instructions[0][1], qasm_prog.instructions[0][2]

        # arrange
        qasm = empty_qasm_program_qcm
        op_info = types.OpInfo(name="", data={"phase": phase}, timing=0)
        strategy = virtual.NcoPhaseShiftStrategy(op_info)

        # act
        strategy.insert_qasm(qasm)

        # assert
        assert extract_instruction_and_args(qasm) == answer
