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

import numpy as np
import pytest

from quantify_scheduler.backends.types import qblox as types

from quantify_scheduler.backends.qblox import helpers
from quantify_scheduler.backends.qblox.instrument_compilers import QcmModule
from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.operation_handling import virtual
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.register_manager import RegisterManager


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
            types.OpInfo(name="", data={"phase_shift": 123.456}, timing=0)
        )

    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={"phase_shift": 123.456}, timing=0)
        strategy = virtual.NcoPhaseShiftStrategy(op_info)

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    def test_generate_data(self):
        # arrange
        op_info = types.OpInfo(name="", data={"phase_shift": 123.456}, timing=0)
        strategy = virtual.NcoPhaseShiftStrategy(op_info)

        # act and assert
        _assert_none_data(strategy)

    @pytest.mark.parametrize(
        "phase_shift, answer",
        [
            (0.0, ("set_ph_delta", "0")),
            (360, ("set_ph_delta", "0")),
            (360.0, ("set_ph_delta", "0")),
            (359.99999999999999, ("set_ph_delta", "0")),
            (359.999, ("set_ph_delta", "999997222")),
            (123.123, ("set_ph_delta", "342008333")),
            (483.123, ("set_ph_delta", "342008333")),
        ],
    )
    def test_generate_qasm_program(
        self,
        phase_shift: float,
        answer: Tuple[str, str],
        empty_qasm_program_qcm: QASMProgram,
    ):
        def extract_instruction_and_args(qasm_prog: QASMProgram) -> Tuple[str, str]:
            return qasm_prog.instructions[0][1], qasm_prog.instructions[0][2]

        # arrange
        qasm = empty_qasm_program_qcm
        op_info = types.OpInfo(name="", data={"phase_shift": phase_shift}, timing=0)
        strategy = virtual.NcoPhaseShiftStrategy(op_info)

        # act
        strategy.insert_qasm(qasm)

        # assert
        if phase_shift == 0.0:
            assert qasm.instructions == []
        else:
            assert extract_instruction_and_args(qasm) == answer


class TestNcoSetClockFrequencyStrategy:
    def test_constructor(self):
        op_info = types.OpInfo(name="", data={"clock_frequency": 1}, timing=0)
        virtual.NcoSetClockFrequencyStrategy(
            operation_info=op_info,
            frequencies=helpers.Frequencies(clock=2, IF=3),
        )

    def test_generate_data(self):
        # arrange
        op_info = types.OpInfo(name="", data={"clock_frequency": 1}, timing=0)
        strategy = virtual.NcoSetClockFrequencyStrategy(
            operation_info=op_info,
            frequencies=helpers.Frequencies(clock=2, IF=3),
        )

        # act and assert
        _assert_none_data(strategy)

    @pytest.mark.parametrize(
        "clock_freq_prev, clock_freq_new, interm_freq, expected_instruction",
        [
            (
                clock_freq_prev,
                clock_freq_new,
                interm_freq,
                (
                    "set_freq",
                    f"{round((interm_freq + clock_freq_new - clock_freq_prev)*4)}",
                    "upd_param",
                    "8",
                ),
            )
            for clock_freq_prev in np.append(
                np.geomspace(-1000e6, -1e-8, num=2), np.geomspace(1e-8, 1000e6, num=2)   # TODO: probably overkill, adding a lot of test cases
            )
            for clock_freq_new in [-2e9, 0, 600]
            for interm_freq in [-123, 50e6]
        ],
    )
    def test_generate_qasm_program(
        self,
        clock_freq_prev: float,
        clock_freq_new: float,
        interm_freq: float,
        expected_instruction: Tuple[str, str, str, str],
        empty_qasm_program_qcm: QASMProgram,
    ):
        def extract_instruction_and_args(
            qasm_prog: QASMProgram,
        ) -> Tuple[str, str, str, str]:
            return (
                qasm_prog.instructions[0][1],
                qasm_prog.instructions[0][2],
                qasm_prog.instructions[1][1],
                qasm_prog.instructions[1][2],
            )

        # arrange
        qasm = empty_qasm_program_qcm
        op_info = types.OpInfo(
            name="", data={"clock_frequency": clock_freq_new}, timing=0
        )

        strategy = virtual.NcoSetClockFrequencyStrategy(
            operation_info=op_info,
            frequencies=helpers.Frequencies(clock=clock_freq_prev, IF=interm_freq),
        )

        # act
        try:
            strategy.insert_qasm(qasm)
        except ValueError as error:
            interm_freq_new = interm_freq + clock_freq_new - clock_freq_prev
            limit = 500e6
            if interm_freq_new < -limit or interm_freq_new > limit:
                assert (
                    str(error) == f"Attempting to set NCO frequency. "
                    f"The frequency must be between and including "
                    f"-{limit} Hz and {limit} Hz. "
                    f"Got {interm_freq_new} Hz."
                )
                return
            raise

        # assert
        assert extract_instruction_and_args(qasm) == expected_instruction
