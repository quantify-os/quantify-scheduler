# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

import numpy as np

from quantify_scheduler import Schedule
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.schedules import trace_schedules


def test_trace_schedule() -> None:
    # Arrange
    init_duration: int = int(1e-5)
    integration_time = 1e-6
    pulse_duration = 500e-9
    pulse_delay = 0
    acquisition_delay = 2e-9
    clock_frequency = 7.04e9
    repetitions = 10

    # Act
    schedule = trace_schedules.trace_schedule(
        pulse_amp=1,
        pulse_duration=pulse_duration,
        pulse_delay=pulse_delay,
        frequency=clock_frequency,
        acquisition_delay=acquisition_delay,
        integration_time=integration_time,
        port="q0:res",
        clock="q0.ro",
        init_duration=init_duration,
        repetitions=repetitions,
    )

    # Assert
    assert isinstance(schedule, Schedule)
    assert schedule.name == "Raw trace acquisition"
    assert schedule.repetitions == repetitions
    assert schedule.resources["q0.ro"]["freq"] == clock_frequency
    assert len(schedule.schedulables) == 3
    # IdlePulse
    idle_pulse_op = schedule.operations[list(schedule.schedulables.values())[0]["operation_id"]]
    assert idle_pulse_op["pulse_info"][0]["duration"] == init_duration

    # SquarePulse
    square_pulse_op = schedule.operations[list(schedule.schedulables.values())[1]["operation_id"]]
    assert square_pulse_op["pulse_info"][0]["duration"] == pulse_duration
    assert (
        list(schedule.schedulables.values())[1]["timing_constraints"][0]["rel_time"] == pulse_delay
    )

    # Trace
    trace_acq_op = schedule.operations[list(schedule.schedulables.values())[2]["operation_id"]]
    assert trace_acq_op["acquisition_info"][0]["duration"] == integration_time
    assert (
        list(schedule.schedulables.values())[2]["timing_constraints"][0]["rel_time"]
        == acquisition_delay
    )


def test_two_tone_trace_schedule() -> None:
    # Arrange
    init_duration = 1e-5
    integration_time = 1e-6
    repetitions = 10

    # Act
    schedule = trace_schedules.two_tone_trace_schedule(
        qubit_pulse_amp=1,
        qubit_pulse_duration=16e-9,
        qubit_pulse_frequency=6.02e9,
        qubit_pulse_port="q0:mw",
        qubit_pulse_clock="q0.01",
        ro_pulse_amp=1,
        ro_pulse_duration=500e-9,
        ro_pulse_delay=2e-9,
        ro_pulse_port="q0:res",
        ro_pulse_clock="q0:ro",
        ro_pulse_frequency=6.02e9,
        ro_acquisition_delay=-20e-9,
        ro_integration_time=integration_time,
        init_duration=init_duration,
        repetitions=repetitions,
    )

    # Assert
    assert isinstance(schedule, Schedule)
    assert schedule.repetitions == repetitions
    assert schedule.name == "Two-tone Trace acquisition"
    assert schedule.resources["q0.01"]["freq"] == 6.02e9
    assert schedule.resources["q0:ro"]["freq"] == 6.02e9
    assert len(schedule.schedulables) == 4

    # IdlePulse
    schedulable = list(schedule.schedulables.values())[0]
    idle_pulse_op = schedule.operations[schedulable["operation_id"]]
    assert schedulable["label"] == "Reset"
    assert idle_pulse_op["pulse_info"][0]["duration"] == init_duration

    # Qubit pulse
    schedulable = list(schedule.schedulables.values())[1]
    square_pulse_op = schedule.operations[schedulable["operation_id"]]
    pulse_info = square_pulse_op["pulse_info"][0]
    assert schedulable["label"] == "qubit_pulse"
    assert pulse_info["port"] == "q0:mw"
    assert pulse_info["duration"] == 16e-9

    # Readout pulse
    schedulable = list(schedule.schedulables.values())[2]
    square_pulse_op = schedule.operations[schedulable["operation_id"]]
    pulse_info = square_pulse_op["pulse_info"][0]
    assert schedulable["label"] == "readout_pulse"
    assert schedulable["timing_constraints"][0]["rel_time"] == 2e-9
    assert pulse_info["duration"] == 500e-9
    assert pulse_info["port"] == "q0:res"

    # Trace Acquisition
    schedulable = list(schedule.schedulables.values())[3]
    trace_op = schedule.operations[schedulable["operation_id"]]
    acq_info = trace_op["acquisition_info"][0]
    assert schedulable["label"] == "acquisition"
    assert schedulable["timing_constraints"][0]["rel_time"] == -20e-9
    assert acq_info["duration"] == integration_time
    assert acq_info["port"] == "q0:res"


def test_long_trace_schedule() -> None:
    # Arrange
    pulse_amp = 0.5 + 0.25 * 1j
    pulse_delay = 0
    clock_frequency = 250e6
    acquisition_delay = 152e-9
    integration_time = 1e-6
    num_points = 100

    # Act
    schedule = trace_schedules.long_time_trace(
        pulse_amp=pulse_amp,
        pulse_delay=pulse_delay,
        frequency=clock_frequency,
        acquisition_delay=acquisition_delay,
        integration_time=integration_time,
        port="q0:res",
        clock="q0.ro",
        num_points=num_points,
    )

    # Assert
    assert isinstance(schedule, Schedule)
    assert schedule.name == "Long time trace acquisition"
    assert schedule.repetitions == 1
    assert schedule.resources["q0.ro"]["freq"] == clock_frequency
    assert len(schedule.schedulables) == 4

    # # VoltageOffset
    voltage_offset_op = schedule.operations[list(schedule.schedulables.values())[0]["operation_id"]]
    assert voltage_offset_op["name"] == "VoltageOffset"
    assert voltage_offset_op["pulse_info"][0]["offset_path_I"] == np.real(pulse_amp)
    assert voltage_offset_op["pulse_info"][0]["offset_path_Q"] == np.imag(pulse_amp)

    # # Control Flow loop
    control_flow_schedulable = list(schedule.schedulables.values())[1]
    inner_sched = schedule.operations[control_flow_schedulable["operation_id"]]
    assert isinstance(inner_sched, LoopOperation)
    assert inner_sched.data["control_flow_info"]["repetitions"] == num_points
    assert isinstance(inner_sched.body, Schedule)
    assert list(inner_sched.body.operations.values())[0]["name"] == "SSBIntegrationComplex"
    assert control_flow_schedulable["timing_constraints"][0]["rel_time"] == acquisition_delay

    # # VoltageOffset_off
    voltage_offset_op_off = schedule.operations[
        list(schedule.schedulables.values())[2]["operation_id"]
    ]
    assert voltage_offset_op_off["name"] == "VoltageOffset"
    assert voltage_offset_op_off["pulse_info"][0]["offset_path_I"] == 0
    assert voltage_offset_op_off["pulse_info"][0]["offset_path_Q"] == 0
