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
from quantify_scheduler.operations.pulse_compensation_library import (
    PortClock,
    PulseCompensation,
)
from quantify_scheduler.operations.pulse_library import (
    RampPulse,
    SquarePulse,
    VoltageOffset,
)
from quantify_scheduler.pulse_compensation import (
    _determine_compensation_pulse,
    process_compensation_pulses,
)
from quantify_scheduler.resources import BasebandClockResource


def test_determine_compensation_pulse():
    schedule = Schedule("Schedule")
    schedule.add(
        SquarePulse(
            amp=0.8, duration=1e-8, port="q0:gt", clock=BasebandClockResource.IDENTITY
        )
    )
    schedule.add(
        RampPulse(
            amp=0.5, duration=1e-8, port="q1:gt", clock=BasebandClockResource.IDENTITY
        )
    )
    schedule.add(
        LoopOperation(
            body=RampPulse(
                amp=0.3,
                duration=2e-8,
                port="q0:gt",
                clock=BasebandClockResource.IDENTITY,
            ),
            repetitions=3,
        )
    )

    max_compensation_amp = {
        PortClock("q0:gt", BasebandClockResource.IDENTITY): 0.6,
        PortClock("q1:gt", BasebandClockResource.IDENTITY): 0.7,
    }
    compensation_pulses_start_duration_amp = _determine_compensation_pulse(
        schedule, max_compensation_amp, 4e-9, sampling_rate=1e9
    )

    assert compensation_pulses_start_duration_amp.keys() == {
        PortClock("q1:gt", BasebandClockResource.IDENTITY),
        PortClock("q0:gt", BasebandClockResource.IDENTITY),
    }

    assert (
        compensation_pulses_start_duration_amp[
            PortClock("q0:gt", BasebandClockResource.IDENTITY)
        ].start
        == 8e-8
    )
    assert math.isclose(
        compensation_pulses_start_duration_amp[
            PortClock("q0:gt", BasebandClockResource.IDENTITY)
        ].duration,
        2.8e-8,
    )
    assert math.isclose(
        compensation_pulses_start_duration_amp[
            PortClock("q0:gt", BasebandClockResource.IDENTITY)
        ].amp,
        -0.5910714285714285,
    )

    assert (
        compensation_pulses_start_duration_amp[
            PortClock("q1:gt", BasebandClockResource.IDENTITY)
        ].start
        == 2e-8
    )
    assert (
        compensation_pulses_start_duration_amp[
            PortClock("q1:gt", BasebandClockResource.IDENTITY)
        ].duration
        == 4e-9
    )
    assert math.isclose(
        compensation_pulses_start_duration_amp[
            PortClock("q1:gt", BasebandClockResource.IDENTITY)
        ].amp,
        -0.5625,
    )


@pytest.mark.parametrize(
    "operation, expected_error",
    [
        (
            SquarePulse(amp=0.8, duration=1e-8, port="q0:gt", clock="q0.01"),
            "Error calculating compensation pulse amplitude for "
            "'SquarePulse"
            "(amp=0.8,duration=1e-08,port='q0:gt',clock='q0.01',reference_magnitude=None,t0=0)'. "
            "Clock must be the baseband clock. ",
        ),
        (
            VoltageOffset(offset_path_I=1, offset_path_Q=1, port="q0:gt"),
            "Error calculating compensation pulse amplitude for "
            "'VoltageOffset"
            "(offset_path_I=1,offset_path_Q=1,port='q0:gt',clock='cl0.baseband',"
            "duration=0.0,t0=0,reference_magnitude=None)'. "
            "Voltage offset operation type is not allowed "
            "in a pulse compensation structure. ",
        ),
        (
            ConditionalOperation(body=X("q0"), qubit_name="q0"),
            "Error calculating compensation pulse amplitude for "
            "'ConditionalOperation(body=X(qubit='q0'),qubit_name='q0',t0=0.0)'. "
            "This control flow operation type is not allowed "
            "in a pulse compensation structure. ",
        ),
    ],
)
def test_determine_compensation_pulse_error(operation, expected_error):
    schedule = Schedule("Schedule")
    schedule.add(operation)

    max_compensation_amp = {
        PortClock("q0:gt", BasebandClockResource.IDENTITY): 0.6,
    }

    with pytest.raises(ValueError) as exception:
        _determine_compensation_pulse(
            schedule, max_compensation_amp, 4e-9, sampling_rate=1e9
        )

    assert exception.value.args[0] == expected_error


def test_insert_compensation_pulses(get_subschedule_operation):
    schedule = Schedule("Schedule")
    schedule.add(
        SquarePulse(
            amp=0.8, duration=1e-8, port="q0:gt", clock=BasebandClockResource.IDENTITY
        )
    )
    schedule.add(
        RampPulse(
            amp=0.5, duration=1e-8, port="q1:gt", clock=BasebandClockResource.IDENTITY
        )
    )
    schedule.add(
        LoopOperation(
            body=RampPulse(
                amp=0.3,
                duration=2e-8,
                port="q0:gt",
                clock=BasebandClockResource.IDENTITY,
            ),
            repetitions=3,
        )
    )

    max_compensation_amp = {
        PortClock("q0:gt", BasebandClockResource.IDENTITY): 0.6,
        PortClock("q1:gt", BasebandClockResource.IDENTITY): 0.7,
    }

    compensated_schedule = process_compensation_pulses(
        PulseCompensation(
            body=schedule,
            max_compensation_amp=max_compensation_amp,
            time_grid=4e-9,
            sampling_rate=1e9,
        )
    )

    assert isinstance(compensated_schedule, Schedule)

    subschedule_schedulable = list(compensated_schedule.schedulables.values())[0][
        "name"
    ]

    compensation_pulse_q0_schedulable = list(
        compensated_schedule.schedulables.values()
    )[1]
    compensation_pulse_q0 = compensated_schedule.operations[
        compensation_pulse_q0_schedulable["operation_id"]
    ]
    compensation_pulse_q1_schedulable = list(
        compensated_schedule.schedulables.values()
    )[2]
    compensation_pulse_q1 = compensated_schedule.operations[
        compensation_pulse_q1_schedulable["operation_id"]
    ]

    if compensation_pulse_q0["pulse_info"][0]["port"] == "q1:gt":
        compensation_pulse_q0_schedulable, compensation_pulse_q1_schedulable = (
            compensation_pulse_q1_schedulable,
            compensation_pulse_q0_schedulable,
        )
        compensation_pulse_q0, compensation_pulse_q1 = (
            compensation_pulse_q1,
            compensation_pulse_q0,
        )

    assert (
        compensation_pulse_q0_schedulable["timing_constraints"][0]["rel_time"] == 8e-8
    )
    assert (
        compensation_pulse_q0_schedulable["timing_constraints"][0]["ref_schedulable"]
        == subschedule_schedulable
    )
    assert (
        compensation_pulse_q0_schedulable["timing_constraints"][0]["ref_pt"] == "start"
    )
    assert (
        compensation_pulse_q0_schedulable["timing_constraints"][0]["ref_pt_new"]
        == "start"
    )
    assert len(compensation_pulse_q0["pulse_info"]) == 1
    assert (
        compensation_pulse_q0["pulse_info"][0]["wf_func"]
        == "quantify_scheduler.waveforms.square"
    )
    assert compensation_pulse_q0["pulse_info"][0]["reference_magnitude"] is None
    assert compensation_pulse_q0["pulse_info"][0]["t0"] == 0
    assert compensation_pulse_q0["pulse_info"][0]["port"] == "q0:gt"
    assert math.isclose(
        compensation_pulse_q0["pulse_info"][0]["amp"], -0.5910714285714285
    )
    assert math.isclose(compensation_pulse_q0["pulse_info"][0]["duration"], 2.8e-8)

    assert len(compensation_pulse_q1["pulse_info"]) == 1
    assert (
        compensation_pulse_q1["pulse_info"][0]["wf_func"]
        == "quantify_scheduler.waveforms.square"
    )
    assert compensation_pulse_q1["pulse_info"][0]["reference_magnitude"] is None
    assert compensation_pulse_q1["pulse_info"][0]["t0"] == 0
    assert compensation_pulse_q1["pulse_info"][0]["port"] == "q1:gt"
    assert math.isclose(compensation_pulse_q1["pulse_info"][0]["amp"], -0.5625)
    assert math.isclose(compensation_pulse_q1["pulse_info"][0]["duration"], 4e-9)
