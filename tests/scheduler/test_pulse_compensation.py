import math

import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    LoopOperation,
)
from quantify_scheduler.operations.gate_library import (
    X,
)
from quantify_scheduler.operations.pulse_library import (
    RampPulse,
    SquarePulse,
    VoltageOffset,
)
from quantify_scheduler.pulse_compensation import determine_compensating_pulse
from quantify_scheduler.resources import BasebandClockResource


def test_determine_compensating_pulse():
    schedule = Schedule("Schedule")
    schedule.add(
        SquarePulse(
            amp=0.8, duration=1e-8, port="q0:mw", clock=BasebandClockResource.IDENTITY
        )
    )
    schedule.add(
        RampPulse(
            amp=0.5, duration=1e-8, port="q1:mw", clock=BasebandClockResource.IDENTITY
        )
    )
    schedule.add(
        LoopOperation(
            body=RampPulse(
                amp=0.3,
                duration=2e-8,
                port="q0:mw",
                clock=BasebandClockResource.IDENTITY,
            ),
            repetitions=3,
        )
    )

    compensating_max_amp = {
        ("q0:mw", BasebandClockResource.IDENTITY): 0.6,
        ("q1:mw", BasebandClockResource.IDENTITY): 0.7,
    }
    compensation_pulses_start_duration_amp = determine_compensating_pulse(
        schedule, compensating_max_amp, 4e-9, sampling_rate=1e9
    )

    assert compensation_pulses_start_duration_amp.keys() == {
        ("q1:mw", BasebandClockResource.IDENTITY),
        ("q0:mw", BasebandClockResource.IDENTITY),
    }

    assert (
        compensation_pulses_start_duration_amp[
            ("q0:mw", BasebandClockResource.IDENTITY)
        ].start
        == 8e-8
    )
    assert math.isclose(
        compensation_pulses_start_duration_amp[
            ("q0:mw", BasebandClockResource.IDENTITY)
        ].duration,
        2.8e-8,
    )
    assert math.isclose(
        compensation_pulses_start_duration_amp[
            ("q0:mw", BasebandClockResource.IDENTITY)
        ].amp,
        -0.5910714285714285,
    )

    assert (
        compensation_pulses_start_duration_amp[
            ("q1:mw", BasebandClockResource.IDENTITY)
        ].start
        == 2e-8
    )
    assert (
        compensation_pulses_start_duration_amp[
            ("q1:mw", BasebandClockResource.IDENTITY)
        ].duration
        == 4e-9
    )
    assert math.isclose(
        compensation_pulses_start_duration_amp[
            ("q1:mw", BasebandClockResource.IDENTITY)
        ].amp,
        -0.5625,
    )


@pytest.mark.parametrize(
    "operation, expected_error",
    [
        (
            SquarePulse(amp=0.8, duration=1e-8, port="q0:mw", clock="q0.01"),
            "Error calculating compensating pulse amplitude for "
            "'SquarePulse"
            "(amp=0.8,duration=1e-08,port='q0:mw',clock='q0.01',reference_magnitude=None,t0=0)'. "
            "Clock must be the baseband clock. ",
        ),
        (
            VoltageOffset(offset_path_I=1, offset_path_Q=1, port="q0:mw"),
            "Error calculating compensating pulse amplitude for "
            "'VoltageOffset"
            "(offset_path_I=1,offset_path_Q=1,port='q0:mw',clock='cl0.baseband',"
            "duration=0.0,t0=0,reference_magnitude=None)'. "
            "Voltage offset operation type is not allowed "
            "in a pulse compensating structure. ",
        ),
        (
            ConditionalOperation(body=X("q0"), qubit_name="q0"),
            "Error calculating compensating pulse amplitude for "
            "'ConditionalOperation(body=X(qubit='q0'),qubit_name='q0',t0=0.0)'. "
            "This control flow operation type is not allowed "
            "in a pulse compensating structure. ",
        ),
    ],
)
def test_determine_compensating_pulse_error(operation, expected_error):
    schedule = Schedule("Schedule")
    schedule.add(operation)

    compensating_max_amp = {
        ("q0:mw", BasebandClockResource.IDENTITY): 0.6,
    }

    with pytest.raises(ValueError) as exception:
        determine_compensating_pulse(
            schedule, compensating_max_amp, 4e-9, sampling_rate=1e9
        )

    assert exception.value.args[0] == expected_error
