# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
import json

import numpy as np
import pandas as pd
import pytest
from quantify_core.data.handling import set_datadir

from quantify_scheduler import CompiledSchedule, Operation, Schedule
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
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
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import BasebandClockResource, ClockResource
from quantify_scheduler.schedules import timedomain_schedules
from quantify_scheduler.schemas.examples.utils import load_json_example_scheme


@pytest.fixture(scope="module", autouse=False)
def t1_schedule():
    schedule = Schedule("T1", 10)
    qubit = "q0"
    times = np.arange(0, 20e-6, 2e-6)
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X(qubit), label=f"pi {i}")
        schedule.add(
            Measure(qubit), ref_pt="start", rel_time=tau, label=f"Measurement {i}"
        )

    return schedule


def test_schedule_properties():
    # Act
    schedule = Schedule("Test", repetitions=1e3)

    # Assert
    assert schedule.name == "Test"
    assert schedule.repetitions == 1e3


def test_schedule_adding_double_resource():
    # clock associated with qubit
    sched = Schedule("Bell experiment")
    with pytest.raises(ValueError):
        sched.add_resource(BasebandClockResource(BasebandClockResource.IDENTITY))

    sched.add_resource(ClockResource("mystery", 6e9))
    with pytest.raises(ValueError):
        sched.add_resource(ClockResource("mystery", 6e9))


def test_schedule_bell():
    # Create an empty schedule
    sched = Schedule("Bell experiment")
    assert Schedule.is_valid(sched)

    assert len(sched.data["operation_dict"]) == 0
    assert len(sched.data["timing_constraints"]) == 0

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")

    # Define the operations, these will be added to the circuit
    init_all = Reset(q0, q1)  # instantiates
    x90_q0 = Rxy(theta=90, phi=0, qubit=q0)

    # we use a regular for loop as we have to unroll the changing theta variable here
    for theta in np.linspace(0, 360, 21):
        sched.add(init_all)
        sched.add(x90_q0)
        sched.add(operation=CNOT(qC=q0, qT=q1))
        sched.add(Rxy(theta=theta, phi=0, qubit=q0))
        sched.add(Measure(q0, q1), label="M {:.2f} deg".format(theta))

    assert len(sched.operations) == 24 - 1  # angle theta == 360 will evaluate to 0
    assert len(sched.timing_constraints) == 105

    assert Schedule.is_valid(sched)


def test_schedule_add_timing_constraints():
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


def test_gates_valid():
    init_all = Reset("q0", "q1")  # instantiates
    rxy_operation = Rxy(theta=124, phi=23.9, qubit="q5")
    x_operation = X("q0")
    x90_operation = X90("q1")
    y_operation = Y("q0")
    y90_operation = Y90("q1")

    cz_operation = CZ("q0", "q1")
    cnot_operation = CNOT("q0", "q6")

    measure_operation = Measure("q0", "q9")

    assert Operation.is_valid(init_all)
    assert Operation.is_valid(rxy_operation)
    assert Operation.is_valid(x_operation)
    assert Operation.is_valid(x90_operation)
    assert Operation.is_valid(y_operation)
    assert Operation.is_valid(y90_operation)
    assert Operation.is_valid(cz_operation)
    assert Operation.is_valid(cnot_operation)
    assert Operation.is_valid(measure_operation)


def test_operation_equality():
    xa_q0 = X("q0")
    xb_q0 = X("q0")
    assert xa_q0 == xb_q0
    # we now modify the contents of xa_q0.data
    # this does not change the repr but does change the content of the operation
    xa_q0.data["custom_key"] = 5
    assert xa_q0 != xb_q0


def test_type_properties():
    operation = Operation("blank op")
    assert not operation.valid_gate
    assert not operation.valid_pulse
    assert operation.name == "blank op"

    gate = X("q0")
    assert gate.valid_gate
    assert not gate.valid_pulse

    pulse = SquarePulse(1.0, 20e-9, "q0", clock="cl0.baseband")
    assert not pulse.valid_gate
    assert pulse.valid_pulse

    pulse.add_gate_info(X("q0"))
    assert pulse.valid_gate
    assert pulse.valid_pulse

    gate.add_pulse(SquarePulse(1.0, 20e-9, "q0", clock="cl0.baseband"))
    assert gate.valid_gate
    assert gate.valid_pulse


def test_operation_duration():
    # Arrange
    square_pulse_duration = 20e-9
    acquisition_duration = 300e-9

    # Act
    empty_measure = Measure("q0")
    empty_x_gate = X("q0")

    pulse = SquarePulse(1.0, square_pulse_duration, "q0", clock="cl0.baseband")

    x_gate = X("q0")
    x_gate.add_pulse(pulse)

    measure = Measure("q0")
    measure.add_acquisition(
        SSBIntegrationComplex(
            port="q0:res",
            clock="q0.ro",
            duration=acquisition_duration,
        )
    )

    # Assert
    assert empty_measure.duration == 0
    assert empty_x_gate.duration == 0
    assert pulse.duration == square_pulse_duration
    assert x_gate.duration == square_pulse_duration
    assert measure.duration == acquisition_duration


def test___repr__():
    operation = Operation("test", {"gate_info": {"clock": "q0.01"}})
    assert eval(repr(operation)) == operation


def test___str__():
    operation = Operation("test")
    assert eval(str(operation)) == operation


def test_schedule_to_json():
    # Arrange
    schedule = timedomain_schedules.t1_sched(np.zeros(1), "q0")

    # Act
    json_data = schedule.to_json()

    # Assert
    json.loads(json_data)


def test_schedule_from_json():
    # Arrange
    schedule = timedomain_schedules.t1_sched(np.zeros(1), "q0")

    # Act
    json_data = schedule.to_json()
    result = Schedule.from_json(json_data)

    # Assert
    assert schedule == result
    assert schedule.data == result.data


def test_t1_sched_valid(t1_schedule):
    """
    Tests that the test schedule is a valid Schedule and an invalid CompiledSchedule
    """
    test_schedule = t1_schedule
    assert Schedule.is_valid(test_schedule)

    assert not CompiledSchedule.is_valid(test_schedule)


def test_compiled_t1_sched_valid(t1_schedule):
    """
    Tests that the test schedule is a valid Schedule and a valid CompiledSchedule
    """
    test_schedule = CompiledSchedule(t1_schedule)

    assert Schedule.is_valid(test_schedule)
    assert CompiledSchedule.is_valid(test_schedule)


def test_t1_sched_circuit_diagram(t1_schedule):
    """
    Tests that the test schedule can be visualized
    """
    # will only test that a figure is created and runs without errors
    _ = t1_schedule.plot_circuit_diagram_mpl()


def test_t1_sched_pulse_diagram(t1_schedule, tmp_test_data_dir):
    """
    Tests that the test schedule can be visualized
    """

    set_datadir(tmp_test_data_dir)
    device_cfg = load_json_example_scheme("transmon_test_config.json")
    comp_sched = qcompile(t1_schedule, device_cfg=device_cfg)

    # will only test that a figure is created and runs without errors
    _ = comp_sched.plot_pulse_diagram_mpl()


def test_sched_timing_table(tmp_test_data_dir):

    schedule = Schedule(name="test_sched", repetitions=10)
    qubit = "q0"
    times = [0, 10e-6, 30e-6]
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X(qubit), label=f"pi {i}")
        schedule.add(
            Measure(qubit), ref_pt="start", rel_time=tau, label=f"Measurement {i}"
        )

    with pytest.raises(ValueError):
        _ = schedule.timing_table

    set_datadir(tmp_test_data_dir)
    device_cfg = load_json_example_scheme("transmon_test_config.json")
    comp_sched = qcompile(schedule, device_cfg=device_cfg)
    # will only test that a figure is created and runs without errors
    timing_table = comp_sched.timing_table

    assert set(timing_table.data.keys()) == {
        "abs_time",
        "clock",
        "duration",
        "is_acquisition",
        "port",
        "waveform_op_id",
        "operation",
        "wf_idx",
    }

    assert len(timing_table.data) == 12

    np.testing.assert_almost_equal(
        actual=np.array(timing_table.data["abs_time"]),
        desired=np.array(
            [
                0,
                200e-6,
                200e-6,
                200e-6 + 120e-9,  # acq delay
                200e-6 + 420e-9,
                400e-6 + 420e-9,
                410e-6 + 420e-9,
                410e-6 + 540e-9,
                410e-6 + 840e-9,
                610e-6 + 840e-9,
                640e-6 + 840e-9,
                640e-6 + 960e-9,
            ]
        ),
        decimal=10,
    )


def test_sched_hardware_timing_table(
    t1_schedule,
    load_example_transmon_config,
    load_example_zhinst_hardware_config,
):
    # assert that files properly compile
    compiled_schedule = qcompile(
        t1_schedule,  # pylint: disable=no-member
        load_example_transmon_config(),
        load_example_zhinst_hardware_config(),
    )
    hardware_timing_table = compiled_schedule.hardware_timing_table
    columns_of_hw_timing_table = hardware_timing_table.columns

    assert isinstance(hardware_timing_table, pd.io.formats.style.Styler)
    assert "clock_cycle_start" in columns_of_hw_timing_table
    assert "sample_start" in columns_of_hw_timing_table
    assert "hardware_channel" in columns_of_hw_timing_table
    assert "waveform_id" in columns_of_hw_timing_table


def test_sched_hardware_waveform_dict(
    t1_schedule,
    load_example_transmon_config,
    load_example_zhinst_hardware_config,
):
    # assert that files properly compile
    compiled_schedule = qcompile(
        t1_schedule,  # pylint: disable=no-member
        load_example_transmon_config(),
        load_example_zhinst_hardware_config(),
    )
    hardware_timing_table = compiled_schedule.hardware_timing_table
    hardware_waveform_dict = compiled_schedule.hardware_waveform_dict

    for waveform_id in hardware_timing_table.data.waveform_id:
        if "Reset" in waveform_id:
            # Ignore the reset operation because it will return None
            continue
        assert isinstance(hardware_waveform_dict.get(waveform_id), np.ndarray)
