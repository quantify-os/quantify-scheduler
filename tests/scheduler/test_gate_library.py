# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
# pylint: disable=redefined-outer-name
import json
from typing import Any
from unittest import TestCase

import numpy as np
import pytest

from quantify_scheduler import Operation, Schedule, Schedulable
from quantify_scheduler.json_utils import ScheduleJSONEncoder, ScheduleJSONDecoder
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
from quantify_scheduler.operations.shared_native_library import SpectroscopyOperation
from quantify_scheduler.operations.nv_native_library import ChargeReset


def test_schedule_add_schedulables() -> None:
    sched = Schedule("my exp")
    test_lab = "test label"
    x90_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"), label=test_lab)["label"]
    assert x90_label == test_lab

    with pytest.raises(ValueError):
        x90_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"), label=test_lab)["label"]

    uuid_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"))["label"]
    assert uuid_label != x90_label

    # not specifying a label should work
    sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op=None)

    # specifying existing label should work
    sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op=x90_label)

    # specifying non-existing label should raise an error
    with pytest.raises(ValueError):
        sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op="non-existing-operation")

    # All schedulables should be valid
    for schedulable in sched.schedulables.values():
        assert Schedulable.is_valid(schedulable)

    assert Schedule.is_valid(sched)


def test_rxy_angle_modulo() -> None:
    """asserts that theta angles fall in the domain -180 to 180"""
    rxy_270 = Rxy(theta=270, phi=23.9, qubit="q5")
    rxy_m90 = Rxy(theta=-90, phi=23.9, qubit="q5")
    assert rxy_270 == rxy_m90

    assert rxy_270.data["gate_info"]["theta"] == -90.0

    rxy_360 = Rxy(theta=360, phi=23.9, qubit="q5")
    assert rxy_360.data["gate_info"]["theta"] == 0


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
        SpectroscopyOperation("q0"),
        ChargeReset("q0"),
    ],
)
def test_gate_is_valid(operation: Operation) -> None:
    assert Operation.is_valid(operation)


def test_rxy_is_valid() -> None:
    rxy_q5 = Rxy(theta=124, phi=23.9, qubit="q5")
    assert Operation.is_valid(rxy_q5)


def is__repr__equal(operation: Operation) -> None:
    """
    Asserts that evaluating the representation
    of a thing is identical to the thing
    itself.
    """
    # Arrange
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)
    assert obj == operation


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
        Measure("q0", "q6", acq_index=92),
        SpectroscopyOperation("q0"),
        ChargeReset("q0"),
    ],
)
def test__repr__(operation: Operation) -> None:
    is__repr__equal(operation)


def test_deprecated__repr__() -> None:
    """Tests deprecated ``acq_channel`` keyword. To be removed in
    quantify-scheduler >= 0.13.0."""
    with pytest.warns(FutureWarning):
        is__repr__equal(Measure("q0", "q6", acq_channel=4))


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
        Measure("q0", "q6", acq_index=92),
        SpectroscopyOperation("q0"),
        ChargeReset("q0"),
    ],
)
def test__str__(operation: Operation) -> None:
    is__str__equal(operation)


def test_deprecated__str__() -> None:
    """Tests deprecated ``acq_channel`` keyword. To be removed in
    quantify-scheduler >= 0.13.0."""
    with pytest.warns(FutureWarning):
        is__str__equal(Measure("q0", "q6", acq_channel=4))


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
        Measure("q0", "q6", acq_index=92),
        SpectroscopyOperation("q0"),
        ChargeReset("q0"),
    ],
)
def test_deserialize(operation: Operation) -> None:
    # Arrange
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)

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


def test_deprecated_deserialize() -> None:
    """Tests deprecated ``acq_channel`` keyword. To be removed in
    quantify-scheduler >= 0.13.0."""
    # Arrange
    with pytest.warns(FutureWarning):
        operation = Measure("q0", "q6", acq_channel=4)
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)

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
        Measure("q0", "q6", acq_index=92),
        SpectroscopyOperation("q0"),
        ChargeReset("q0"),
    ],
)
def test__repr__modify_not_equal(operation: Operation) -> None:
    # Arrange
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)
    assert obj == operation

    # Act
    obj.data["pulse_info"].append({"clock": "q0.01"})

    # Assert
    assert obj != operation


def test_deprecated__repr__modify_not_equal() -> None:
    """Tests deprecated ``acq_channel`` keyword. To be removed in
    quantify-scheduler >= 0.13.0."""
    # Arrange
    with pytest.warns(FutureWarning):
        operation = Measure("q0", "q6", acq_channel=4)
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)
    assert obj == operation

    # Act
    obj.data["pulse_info"].append({"clock": "q0.01"})

    # Assert
    assert obj != operation


def test_rotation_unitaries() -> None:
    # Set the tolerance in terms of machine precision, one machine epsilon by default
    # Could be increased to allow for less pretty computations with more round-off
    # error.

    atol = 1 * np.finfo(np.complex128).eps
    # Test Rxy for all angles:
    # The tests are written in form: target, desired
    np.testing.assert_allclose(
        Rxy(theta=0, phi=0, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) * np.array([[1, 0], [0, 1]]),
        atol=atol,
    )
    np.testing.assert_allclose(
        Rxy(theta=90, phi=0, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, -1j], [-1j, 1]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Rxy(theta=-90, phi=90, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, 1], [-1, 1]]),
        atol=atol,
    )

    # Test for the X180, X90, Y180 and Y90 gates which are derived from Rxy
    np.testing.assert_allclose(
        X(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) * np.array([[0, 1j], [1j, 0]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        X90(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, -1j], [-1j, 1]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Y(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) * np.array([[0, 1], [-1, 0]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Y90(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, -1], [1, 1]]),
        atol=atol,
    )
