# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
import copy
import json

import numpy as np
import pandas as pd
import pytest

from quantify_scheduler import enums, json_utils, Operation
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.json_utils import ScheduleJSONDecoder
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
from quantify_scheduler.schedules.schedule import (
    AcquisitionMetadata,
    CompiledSchedule,
    Schedule,
)
from quantify_scheduler.schedules.spectroscopy_schedules import heterodyne_spec_sched


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
    assert len(sched.data["schedulables"]) == 0

    # define the resources
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
        sched.add(Measure(q0, q1), label=f"M {theta:.2f} deg")

    assert len(sched.operations) == 24 - 1  # angle theta == 360 will evaluate to 0
    assert len(sched.schedulables) == 105

    assert Schedule.is_valid(sched)


def test_schedule_add_timing_constraints():
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
    operation = Operation("test")
    operation["gate_info"] = {"clock": "q0.01"}
    obj = ScheduleJSONDecoder().decode_dict(operation.__getstate__())
    assert obj == operation


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


def test_spec_schedule_from_json():
    # Arrange
    schedule = heterodyne_spec_sched(
        0.1, 0.1, 6e9, 1e-7, 1e-6, "q0:mw", "q0.01", 200e-6, 1024
    )

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
    _ = t1_schedule.plot_circuit_diagram()


def test_t1_sched_pulse_diagram(t1_schedule, device_compile_config_basic_transmon):
    """
    Tests that the test schedule can be visualized
    """
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=t1_schedule, config=device_compile_config_basic_transmon
    )
    # will only test that a figure is created and runs without errors
    _ = compiled_schedule.plot_pulse_diagram()


@pytest.mark.parametrize("reset_clock_phase", (True, False))
def test_sched_timing_table(
    mock_setup_basic_transmon_with_standard_params, reset_clock_phase
):
    schedule = Schedule(name="test_sched", repetitions=10)
    qubit = "q0"
    times = [0, 10e-6, 30e-6]
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X(qubit), label=f"pi {i}")
        schedule.add(
            Measure(qubit),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {i}",
        )

    with pytest.raises(ValueError):
        _ = schedule.timing_table

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    q0 = mock_setup_basic_transmon_with_standard_params["q0"]
    q0.measure.reset_clock_phase(reset_clock_phase)
    q0.measure.acq_delay(120e-9)
    q0.reset.duration(200e-6)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    timing_table_data = compiled_schedule.timing_table.data
    assert set(timing_table_data.keys()) == {
        "abs_time",
        "clock",
        "duration",
        "is_acquisition",
        "port",
        "waveform_op_id",
        "operation",
        "wf_idx",
    }
    assert len(timing_table_data) == 15 if reset_clock_phase else 12

    if reset_clock_phase:
        desired_timing = np.array(
            [
                0,
                200e-6,
                200e-6,
                200e-6,
                200e-6 + 120e-9,  # acq delay
                200e-6 + 1120e-9,
                400e-6 + 1120e-9,
                410e-6 + 1120e-9,
                410e-6 + 1120e-9,
                410e-6 + 1240e-9,
                410e-6 + 2240e-9,
                610e-6 + 2240e-9,
                640e-6 + 2240e-9,
                640e-6 + 2240e-9,
                640e-6 + 2360e-9,
            ]
        )
    else:
        desired_timing = np.array(
            [
                0,
                200e-6,
                200e-6,
                200e-6 + 120e-9,  # acq delay
                200e-6 + 1120e-9,
                400e-6 + 1120e-9,
                410e-6 + 1120e-9,
                410e-6 + 1240e-9,
                410e-6 + 2240e-9,
                610e-6 + 2240e-9,
                640e-6 + 2240e-9,
                640e-6 + 2360e-9,
            ]
        )
    np.testing.assert_almost_equal(
        actual=np.array(timing_table_data["abs_time"]),
        desired=desired_timing,
        decimal=10,
    )


def test_sched_hardware_timing_table(
    t1_schedule, compile_config_basic_transmon_zhinst_hardware
):
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=t1_schedule, config=compile_config_basic_transmon_zhinst_hardware
    )

    hardware_timing_table = compiled_schedule.hardware_timing_table
    columns_of_hw_timing_table = hardware_timing_table.columns

    assert isinstance(hardware_timing_table, pd.io.formats.style.Styler)
    assert "clock_cycle_start" in columns_of_hw_timing_table
    assert "sample_start" in columns_of_hw_timing_table
    assert "hardware_channel" in columns_of_hw_timing_table
    assert "waveform_id" in columns_of_hw_timing_table


def test_sched_hardware_waveform_dict(
    t1_schedule, compile_config_basic_transmon_zhinst_hardware
):
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=t1_schedule, config=compile_config_basic_transmon_zhinst_hardware
    )

    # Filter out operations that are not waveforms such as Reset and ClockPhaseReset,
    # that have port = None
    mask = compiled_schedule.hardware_timing_table.data.port.apply(
        lambda port: port is not None
    )
    hardware_timing_table = compiled_schedule.hardware_timing_table.data[mask]

    for waveform_id in hardware_timing_table.waveform_id:
        assert isinstance(
            compiled_schedule.hardware_waveform_dict.get(waveform_id), np.ndarray
        )


def test_acquisition_metadata():
    metadata = None
    for binmode in enums.BinMode:
        metadata = AcquisitionMetadata(
            acq_protocol="SSBIntegrationComplex",
            bin_mode=binmode,
            acq_return_type=complex,
            acq_indices={0: [0]},
            repetitions=1,
        )
        # test whether the copy function works correctly
        metadata_copy = copy.copy(metadata)
        assert metadata_copy == metadata
        assert isinstance(metadata_copy.bin_mode, enums.BinMode)
        assert isinstance(metadata_copy.acq_return_type, type)

    for return_type in complex, float, int, bool, str, np.ndarray:
        metadata = AcquisitionMetadata(
            acq_protocol="SSBIntegrationComplex",
            bin_mode=enums.BinMode.AVERAGE,
            acq_return_type=return_type,
            acq_indices={0: [0]},
            repetitions=1,
        )
        # test whether the copy function works correctly
        metadata_copy = copy.copy(metadata)
        assert metadata_copy == metadata
        assert isinstance(metadata_copy.bin_mode, enums.BinMode)
        assert isinstance(metadata_copy.acq_return_type, type)

    # Test that json serialization works correctly
    serialized = json.dumps(metadata, cls=json_utils.ScheduleJSONEncoder)
    # Test that json deserialization works correctly
    metadata_copy = json.loads(serialized, cls=json_utils.ScheduleJSONDecoder)
    assert metadata_copy == metadata
    assert isinstance(metadata_copy.bin_mode, enums.BinMode)
    assert isinstance(metadata_copy.acq_return_type, type)
