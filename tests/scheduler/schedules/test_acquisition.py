# -----------------------------------------------------------------------------
# Description:    Tests for Zurich Instruments backend.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------

from quantify.scheduler.types import Schedule
from quantify.scheduler.schedules.acquisition import (
    raw_trace_schedule,
    ssb_integration_complex_schedule,
)


def test_raw_trace_schedule():
    # Arrange
    init_duration = 1e-5
    integration_time = 1e-6
    clock_frequency = 7.04e9

    # Act
    schedule = raw_trace_schedule(
        port="q0:res",
        clock="q0.ro",
        integration_time=integration_time,
        spec_pulse_amp=1,
        frequency=clock_frequency,
        init_duration=init_duration,
    )

    # Assert
    assert isinstance(schedule, Schedule)
    assert schedule.name == "Raw trace acquisition"
    assert schedule.resources["q0.ro"]["freq"] == clock_frequency
    assert len(schedule.timing_constraints) == 3
    assert (
        schedule.operations[schedule.timing_constraints[0]["operation_hash"]][
            "pulse_info"
        ][0]["duration"]
        == init_duration
    )
    assert (
        schedule.operations[schedule.timing_constraints[1]["operation_hash"]][
            "acquisition_info"
        ][0]["duration"]
        == integration_time
    )
    assert schedule.timing_constraints[2]["rel_time"] == integration_time + 500e-9


def test_ssb_integration_complex_schedule():
    # Arrange
    init_duration = 1e-5
    integration_time = 1e-6
    clock_frequency = 7.04e9

    # Act
    schedule = ssb_integration_complex_schedule(
        port="q0:res",
        clock="q0.ro",
        integration_time=integration_time,
        spec_pulse_amp=1,
        frequency=clock_frequency,
        init_duration=init_duration,
    )

    # Assert
    assert isinstance(schedule, Schedule)
    assert schedule.name == "SSBIntegrationComplex acquisition"
    assert schedule.resources["q0.ro"]["freq"] == clock_frequency
    assert len(schedule.timing_constraints) == 5
    assert (
        schedule.operations[schedule.timing_constraints[0]["operation_hash"]][
            "pulse_info"
        ][0]["duration"]
        == init_duration
    )

    assert (
        schedule.operations[schedule.timing_constraints[1]["operation_hash"]][
            "acquisition_info"
        ][0]["duration"]
        == integration_time
    )
    assert (
        schedule.operations[schedule.timing_constraints[2]["operation_hash"]][
            "pulse_info"
        ][0]["duration"]
        == 100e-9
    )
    assert (
        schedule.operations[schedule.timing_constraints[3]["operation_hash"]][
            "acquisition_info"
        ][0]["duration"]
        == integration_time
    )
    assert (
        schedule.operations[schedule.timing_constraints[4]["operation_hash"]][
            "pulse_info"
        ][0]["duration"]
        == 100e-9
    )
    assert schedule.timing_constraints[2]["rel_time"] == 500e-9
    assert schedule.timing_constraints[4]["rel_time"] == 500e-9
