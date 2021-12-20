# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-function-docstring

from quantify_scheduler import Schedule
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
    assert len(schedule.timing_constraints) == 3
    # IdlePulse
    idle_pulse_op = schedule.operations[
        schedule.timing_constraints[0]["operation_repr"]
    ]
    assert idle_pulse_op["pulse_info"][0]["duration"] == init_duration

    # SquarePulse
    square_pulse_op = schedule.operations[
        schedule.timing_constraints[1]["operation_repr"]
    ]
    assert square_pulse_op["pulse_info"][0]["duration"] == pulse_duration
    assert schedule.timing_constraints[1]["rel_time"] == pulse_delay

    # Trace
    trace_acq_op = schedule.operations[schedule.timing_constraints[2]["operation_repr"]]
    assert trace_acq_op["acquisition_info"][0]["duration"] == integration_time
    assert schedule.timing_constraints[2]["rel_time"] == acquisition_delay


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
    assert len(schedule.timing_constraints) == 4

    # IdlePulse
    t_const = schedule.timing_constraints[0]
    idle_pulse_op = schedule.operations[t_const["operation_repr"]]
    assert t_const["label"] == "Reset"
    assert idle_pulse_op["pulse_info"][0]["duration"] == init_duration

    # Qubit pulse
    t_const = schedule.timing_constraints[1]
    square_pulse_op = schedule.operations[t_const["operation_repr"]]
    pulse_info = square_pulse_op["pulse_info"][0]
    assert t_const["label"] == "qubit_pulse"
    assert pulse_info["port"] == "q0:mw"
    assert pulse_info["duration"] == 16e-9

    # Readout pulse
    t_const = schedule.timing_constraints[2]
    square_pulse_op = schedule.operations[t_const["operation_repr"]]
    pulse_info = square_pulse_op["pulse_info"][0]
    assert t_const["label"] == "readout_pulse"
    assert t_const["rel_time"] == 2e-9
    assert pulse_info["duration"] == 500e-9
    assert pulse_info["port"] == "q0:res"

    # Trace Acquisition
    t_const = schedule.timing_constraints[3]
    trace_op = schedule.operations[t_const["operation_repr"]]
    acq_info = trace_op["acquisition_info"][0]
    assert t_const["label"] == "acquisition"
    assert t_const["rel_time"] == -20e-9
    assert acq_info["duration"] == integration_time
    assert acq_info["port"] == "q0:res"
