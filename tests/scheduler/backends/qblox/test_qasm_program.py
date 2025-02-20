# --------- Test QASMProgram class ---------
import pytest

from quantify_scheduler.backends.qblox import constants, q1asm_instructions
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.types import qblox as types


def test_emit(empty_qasm_program_qcm):
    qasm = empty_qasm_program_qcm
    qasm.emit(q1asm_instructions.PLAY, 0, 1, 120)
    qasm.emit(q1asm_instructions.STOP, comment="This is a comment that is added")

    assert len(qasm.instructions) == 2


def test_auto_wait(empty_qasm_program_qcm):
    qasm = empty_qasm_program_qcm
    qasm.auto_wait(120)
    assert len(qasm.instructions) == 1
    qasm.auto_wait(70000)
    assert len(qasm.instructions) == 3  # since it should split the waits
    assert qasm.elapsed_time == 70120
    qasm.auto_wait(700000)
    assert qasm.elapsed_time == 770120
    assert len(qasm.instructions) == 8  # now loops are used
    with pytest.raises(ValueError):
        qasm.auto_wait(-120)


@pytest.mark.parametrize(
    "val, expected_expanded_val",
    [
        (-1, -constants.IMMEDIATE_SZ_GAIN // 2),
        (-0.5, -constants.IMMEDIATE_SZ_GAIN // 4),
        (0.0, 0),
        (0.5, constants.IMMEDIATE_SZ_GAIN // 4),
        (1.0, constants.IMMEDIATE_SZ_GAIN // 2 - 1),
    ],
)
def test_expand_awg_gain_from_normalised_range(val, expected_expanded_val):
    minimal_pulse_data = {"duration": 20e-9}
    acq = types.OpInfo(name="test_acq", data=minimal_pulse_data, timing=4e-9)

    expanded_val = QASMProgram.expand_awg_from_normalised_range(
        val=val,
        immediate_size=constants.IMMEDIATE_SZ_GAIN,
        param="test_param",
        operation=acq,
    )
    assert expanded_val == expected_expanded_val


def test_loop(empty_qasm_program_qcm):
    num_rep = 10

    qasm = empty_qasm_program_qcm
    qasm.emit(q1asm_instructions.WAIT_SYNC, 4)
    with qasm.loop("this_loop", repetitions=num_rep):
        qasm.emit(q1asm_instructions.WAIT, 20)
    assert len(qasm.instructions) == 5
    assert qasm.instructions[1][1] == q1asm_instructions.MOVE
    num_rep_used, reg_used = qasm.instructions[1][2].split(",")
    assert int(num_rep_used) == num_rep


@pytest.mark.parametrize("amount", [1, 2, 3, 40])
def test_temp_register(amount, empty_qasm_program_qcm):
    qasm = empty_qasm_program_qcm
    with qasm.temp_registers(amount) as registers:
        for reg in registers:
            assert reg not in qasm.register_manager.available_registers
    for reg in registers:
        assert reg in qasm.register_manager.available_registers


def test_out_of_range_expand_awg_gain_from_normalised_range():
    minimal_pulse_data = {"duration": 20e-9}
    acq = types.OpInfo(name="test_acq", data=minimal_pulse_data, timing=4e-9)
    with pytest.raises(ValueError):
        QASMProgram.expand_awg_from_normalised_range(
            val=10,
            immediate_size=constants.IMMEDIATE_SZ_GAIN,
            param="test_param",
            operation=acq,
        )
