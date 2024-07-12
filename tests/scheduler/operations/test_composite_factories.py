"""Tests for composite factory functions."""

from quantify_scheduler import Schedule
from quantify_scheduler.operations.composite_factories import (
    cnot_as_h_cz_h,
    hadamard_as_y90z,
)
from quantify_scheduler.operations.gate_library import (
    CZ,
    Y90,
    H,
    Z,
)


def test_hadamard_as_y90_z_gate():
    """Test hadamard gate as Y90*Z operation."""
    sched = hadamard_as_y90z("q0")

    expected_sched = Schedule("Expected schedule")
    expected_sched.add(Z("q0"))
    expected_sched.add(Y90("q0"))

    assert len(sched) == len(expected_sched)

    for schedulable, expected_schedulable in zip(
        sched.schedulables.values(),
        expected_sched.schedulables.values(),
    ):
        op = sched.operations[schedulable["operation_id"]]
        expected_op = expected_sched.operations[expected_schedulable["operation_id"]]
        assert op == expected_op


def test_cnot_gate():
    """Test cnot gate as H*CZ*H operation."""
    sched = cnot_as_h_cz_h("q0", "q2")

    expected_sched = Schedule("Expected schedule")
    expected_sched.add(H("q2"))
    expected_sched.add(CZ("q0", "q2"))
    expected_sched.add(H("q2"))

    assert len(sched) == len(expected_sched)

    for schedulable, expected_schedulable in zip(
        sched.schedulables.values(),
        expected_sched.schedulables.values(),
    ):
        op = sched.operations[schedulable["operation_id"]]
        expected_op = expected_sched.operations[expected_schedulable["operation_id"]]
        assert op == expected_op
