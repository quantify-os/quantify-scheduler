# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
from quantify.scheduler import Operation
from quantify.scheduler.gate_library import (
    Rxy,
    X,
    X90,
    Y,
    Y90,
    CNOT,
    CZ,
    Reset,
    Measure,
)


def test_rxy_is_valid():
    rxy_q5 = Rxy(theta=124, phi=23.9, qubit="q5")
    assert Operation.is_valid(rxy_q5)


def check_repr_equal(thing):
    """
    Asserts that evaulating the representation of a thing is identical to the thing
    itself.
    """
    # eval should be avoided for security reasons.
    # However, it is impossible to test this property using the safer ast.literal_eval
    assert eval(repr(thing)) == thing


def test_rxy_repr():
    check_repr_equal(Rxy(theta=124, phi=23.9, qubit="q5"))


def test_x90_repr():
    check_repr_equal(X90("q1"))


def test_x_repr():
    check_repr_equal(X("q0"))


def test_init_repr():
    check_repr_equal(Reset("q0", "q1"))

    check_repr_equal(Reset("q0"))


def test_y_repr():
    check_repr_equal(Y("q0"))


def test_y90_repr():
    check_repr_equal(Y90("q1"))


def test_cz_repr():
    check_repr_equal(CZ("q0", "q1"))


def test_cnot_repr():
    check_repr_equal(CNOT("q0", "q6"))


def test_measure_repr():
    check_repr_equal(Measure("q0", "q6"))
    check_repr_equal(Measure("q0"))

    check_repr_equal(Measure("q0", "q6", acq_channel=4))
    check_repr_equal(Measure("q0", "q6", acq_index=92))
