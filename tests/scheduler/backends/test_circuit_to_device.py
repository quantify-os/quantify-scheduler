# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

from copy import deepcopy

import numpy as np
import pytest

from quantify_scheduler import Operation, Schedule
from quantify_scheduler.compilation import (
    add_pulse_information_transmon,
    determine_absolute_timing,
    qcompile,
    validate_config,
)
from quantify_scheduler.backends.circuit_to_device import (
    compile_circuit_to_device,
    QubitKeyError,
    EdgeKeyError,
    OperationKeyError,
)


from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.gate_library import (
    CNOT,
    CZ,
    Measure,
    Reset,
    Rxy,
    X,
    Y,
    Y90,
)
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import BasebandClockResource, ClockResource, Resource

from quantify_scheduler.schemas.examples.circuit_to_device_example_cfgs import (
    example_transmon_cfg,
)


def test_compile_transmon_example_program():
    """
    Test if compilation using the new backend reproduces behavior of the old backend.
    """

    sched = Schedule("Test schedule")

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(Rxy(45, 0, qubit=q0))
    sched.add(Rxy(12, 0, qubit=q0))
    sched.add(Rxy(12, 0, qubit=q0))
    sched.add(X(qubit=q0))
    sched.add(Y(qubit=q0))
    sched.add(Y90(qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")

    # test that all these operations compile correctly.
    _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_Rxy_operations_compile():
    sched = Schedule("Test schedule")
    sched.add(Rxy(90, 0, qubit="q0"))
    sched.add(Rxy(180, 45, qubit="q0"))
    new_dev_sched = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_measurement_compile():
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", "q1"))  # acq_index should be 0 for both.
    sched.add(Measure("q0", acq_index=1))
    sched.add(Measure("q1", acq_index=2))  # acq_channel should be 1
    sched.add(Measure("q0", "q1", acq_index=2))
    new_dev_sched = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)

    operation_keys_list = list(new_dev_sched.operations.keys())

    M0_acq = new_dev_sched.operations[operation_keys_list[0]]["acquisition_info"]
    assert len(M0_acq) == 2  # both q0 and q1
    assert M0_acq[0]["acq_channel"] == 0
    assert M0_acq[1]["acq_channel"] == 1
    assert M0_acq[0]["acq_index"] == 0
    assert M0_acq[1]["acq_index"] == 0

    M1_acq = new_dev_sched.operations[operation_keys_list[1]]["acquisition_info"]
    assert len(M1_acq) == 1
    assert M1_acq[0]["acq_channel"] == 0
    assert M1_acq[0]["acq_index"] == 1

    M2_acq = new_dev_sched.operations[operation_keys_list[2]]["acquisition_info"]
    assert len(M2_acq) == 1
    assert M2_acq[0]["acq_channel"] == 1
    assert M2_acq[0]["acq_index"] == 2

    M3_acq = new_dev_sched.operations[operation_keys_list[3]]["acquisition_info"]
    assert len(M3_acq) == 2
    assert M3_acq[0]["acq_channel"] == 0
    assert M3_acq[1]["acq_channel"] == 1
    assert M3_acq[0]["acq_index"] == 2
    assert M3_acq[1]["acq_index"] == 2


def test_Reset_operations_compile():
    sched = Schedule("Test schedule")
    sched.add(Reset("q0"))
    sched.add(Reset("q0", "q1"))
    new_dev_sched = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_qubit_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(Rxy(90, 0, qubit="q20"))
    with pytest.raises(QubitKeyError):
        new_dev_sched = compile_circuit_to_device(
            sched, device_cfg=example_transmon_cfg
        )


def test_edge_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(CZ("q0", "q3"))
    with pytest.raises(EdgeKeyError):
        new_dev_sched = compile_circuit_to_device(
            sched, device_cfg=example_transmon_cfg
        )


def test_operation_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(CNOT("q0", "q1"))
    with pytest.raises(OperationKeyError):
        new_dev_sched = compile_circuit_to_device(
            sched, device_cfg=example_transmon_cfg
        )
