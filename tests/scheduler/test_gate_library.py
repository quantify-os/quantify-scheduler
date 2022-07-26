# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
from typing import Any
from unittest import TestCase

import numpy as np
import pytest

from quantify_scheduler import Operation, Schedule, Schedulable
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
    SpecPulseMicrowave,
)
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.compilation import device_compile


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
        Measure("q0", "q6", acq_channel=4),  # This operation should be invalid #262
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
        Measure("q0", "q6", acq_channel=4),  # This operation should be invalid #262
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


def get_nv_device_config():
    from quantify_scheduler.backends.circuit_to_device import OperationCompilationConfig

    spec_mw_cfg = OperationCompilationConfig(
        factory_func="quantify_scheduler.operations.pulse_factories.nv_spec_pulse_mw",
        factory_kwargs={"duration": 15e-6, "amplitude": 1, "clock": "qe0.clock_freqs.spec", "port": "mw"},
    )

    cfg_dict = {
        "backend": "quantify_scheduler.backends"
        ".circuit_to_device.compile_circuit_to_device",
        "elements": {
            f"qe0": {
                "spec_mw": spec_mw_cfg,
            }
        },
        "clocks": {
            f"qe0.clock_freqs.spec": 50e6,
        },
        "edges": {},
    }
    return cfg_dict


def test_pulse_compilation_spec_pulse_microwave():
    schedule = Schedule(name="Spec Pulse", repetitions=1)

    label1 = "MW pi pulse 1"
    label2 = "MW pi pulse 2"
    _ = schedule.add(SpecPulseMicrowave("qe0"), label=label1)
    _ = schedule.add(SpecPulseMicrowave("qe0"), label=label2)

    # SpecPulseMicrowave is added to the operations.
    # It has "gate_info", but no "pulse_info" yet.
    spec_pulse_str = str(SpecPulseMicrowave("qe0"))
    assert spec_pulse_str in schedule.operations
    assert "gate_info" in schedule.operations[spec_pulse_str]
    assert schedule.operations[spec_pulse_str]["pulse_info"] == []

    # Operation is added twice to schedulables and has no timing information yet.
    assert label1 in schedule.schedulables
    assert label2 in schedule.schedulables
    assert 'abs_time' not in schedule.schedulables[label1].data.keys() or schedule.schedulables[label1].data['abs_time'] is None
    assert 'abs_time' not in schedule.schedulables[label2].data.keys() or schedule.schedulables[label2].data['abs_time'] is None

    # We can plot the circuit diagram
    schedule.plot_circuit_diagram()

    # TODO: retrieve the device config from elsewhere?
    dev_cfg = get_nv_device_config()
    schedule_device = device_compile(schedule, dev_cfg)

    # The gate_info remains unchanged, but the pulse info has been added
    assert spec_pulse_str in schedule_device.operations
    assert "gate_info" in schedule_device.operations[spec_pulse_str]
    assert schedule_device.operations[spec_pulse_str]["gate_info"] == schedule.operations[spec_pulse_str]["gate_info"]
    assert not schedule_device.operations[spec_pulse_str]["pulse_info"] == []

    # Timing info has been added
    assert 'abs_time' in schedule_device.schedulables[label1].data.keys()
    assert 'abs_time' in schedule_device.schedulables[label2].data.keys()
    assert not schedule_device.schedulables[label1].data['abs_time'] is None
    assert not schedule_device.schedulables[label2].data['abs_time'] is None
