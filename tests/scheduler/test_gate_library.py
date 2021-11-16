# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
from typing import Any
from unittest import TestCase

import numpy as np
import pytest

from quantify_scheduler import Operation, Schedule
from quantify_scheduler.operations.gate_library import (
    CNOT,
    CZ,
    X90,
    Y90,
    Measure,
    Reset,
    Rxy,
    X,
    Y,
)


def test_schedule_add_timing_constraints() -> None:
    sched = Schedule("my exp")
    test_lab = "test label"
    x90_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"), label=test_lab)
    assert x90_label == test_lab

    with pytest.raises(ValueError):
        x90_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"), label=test_lab)

    uuid_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"))
    assert uuid_label != x90_label

    # not specifying a label should work
    sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op=None)

    # specifying existing label should work
    sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op=x90_label)

    # specifying non-existing label should raise an error
    with pytest.raises(ValueError):
        sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op="non-existing-operation")

    assert Schedule.is_valid(sched)


@pytest.mark.parametrize(
    "operation",
    [
        Reset("q0", "q1"),
        Rxy(theta=124, phi=23.9, qubit="q5"),
        X("q0"),
        X90("q1"),
        Y("q0"),
        Y90("q1"),
        CZ("q0", "q1"),
        CNOT("q0", "q6"),
        Measure("q0", "q9"),
    ],
)
def test_gate_is_valid(operation: Operation) -> None:
    assert Operation.is_valid(operation)


def test_rxy_is_valid() -> None:
    rxy_q5 = Rxy(theta=124, phi=23.9, qubit="q5")
    assert Operation.is_valid(rxy_q5)


def is__repr__equal(obj: Operation) -> None:
    """
    Asserts that evaulating the representation
    of a thing is identical to the thing
    itself.
    """
    # eval should be avoided for security reasons.
    # However, it is impossible to test this property using the safer ast.literal_eval
    assert eval(repr(obj)) == obj


def is__str__equal(obj: Any) -> None:
    """
    Asserts if the string representation
    equals the object type.
    """
    assert isinstance(eval(str(obj)), type(obj))


@pytest.mark.parametrize(
    "operation",
    [
        Rxy(theta=124, phi=23.9, qubit="q5"),
        X90("q1"),
        X("q0"),
        Y90("q1"),
        Y("q1"),
        Reset("q0"),
        Reset("q0", "q1"),
        CZ("q0", "q1"),
        CNOT("q0", "q6"),
        Measure("q0", "q6"),
        Measure("q0"),
        Measure("q0", "q6", acq_channel=4),
        Measure("q0", "q6", acq_index=92),
    ],
)
def test__repr__(operation: Operation) -> None:
    is__repr__equal(operation)


@pytest.mark.parametrize(
    "operation",
    [
        Rxy(theta=124, phi=23.9, qubit="q5"),
        X90("q1"),
        X("q0"),
        Y90("q1"),
        Y("q1"),
        Reset("q0"),
        Reset("q0", "q1"),
        CZ("q0", "q1"),
        CNOT("q0", "q6"),
        Measure("q0", "q6"),
        Measure("q0"),
        Measure("q0", "q6", acq_channel=4),
        Measure("q0", "q6", acq_index=92),
    ],
)
def test__str__(operation: Operation) -> None:
    is__str__equal(operation)


@pytest.mark.parametrize(
    "operation",
    [
        Rxy(theta=124, phi=23.9, qubit="q5"),
        X90("q1"),
        X("q0"),
        Y90("q1"),
        Y("q1"),
        Reset("q0"),
        Reset("q0", "q1"),
        CZ("q0", "q1"),
        CNOT("q0", "q6"),
        Measure("q0", "q6"),
        Measure("q0"),
        Measure("q0", "q6", acq_channel=4),
        Measure("q0", "q6", acq_index=92),
    ],
)
def test_deserialize(operation: Operation) -> None:
    # Arrange
    operation_repr: str = repr(operation)

    # Act
    obj = eval(operation_repr)

    # Assert
    if (
        "unitary" in operation.data["gate_info"]
        and not operation.data["gate_info"]["unitary"] is None
    ):
        assert isinstance(obj.data["gate_info"]["unitary"], (np.generic, np.ndarray))
        np.testing.assert_array_almost_equal(
            obj.data["gate_info"]["unitary"],
            operation.data["gate_info"]["unitary"],
            decimal=9,
        )

        # TestCase().assertDictEqual cannot compare numpy arrays for equality
        # therefore "unitary" is removed
        del obj.data["gate_info"]["unitary"]
        del operation.data["gate_info"]["unitary"]

    TestCase().assertDictEqual(obj.data, operation.data)


@pytest.mark.parametrize(
    "operation",
    [
        Rxy(theta=124, phi=23.9, qubit="q5"),
        X90("q1"),
        X("q0"),
        Y90("q1"),
        Y("q1"),
        Reset("q0"),
        Reset("q0", "q1"),
        CZ("q0", "q1"),
        CNOT("q0", "q6"),
        Measure("q0", "q6"),
        Measure("q0"),
        Measure("q0", "q6", acq_channel=4),
        Measure("q0", "q6", acq_index=92),
    ],
)
def test__repr__modify_not_equal(operation: Operation) -> None:
    # Arrange
    obj = eval(repr(operation))
    assert obj == operation

    # Act
    obj.data["pulse_info"].append({"clock": "q0.01"})

    # Assert
    assert obj != operation
