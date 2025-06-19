import math
from typing import Union
from unittest.mock import MagicMock

import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends.graph_compilation import (
    SerialCompiler,
)
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    ControlFlowOperation,
    LoopOperation,
)
from quantify_scheduler.operations.gate_library import (
    X,
)
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_compensation_library import (
    PulseCompensation,
)
from quantify_scheduler.operations.pulse_library import (
    RampPulse,
    SquarePulse,
    VoltageOffset,
)
from quantify_scheduler.pulse_compensation import (
    _determine_compensation_pulse,
)
from quantify_scheduler.resources import BasebandClockResource


def test_determine_compensation_pulse():
    schedule = Schedule("Schedule")
    schedule.add(
        SquarePulse(amp=0.8, duration=1e-8, port="q0:gt", clock=BasebandClockResource.IDENTITY)
    )
    schedule.add(
        RampPulse(amp=0.5, duration=1e-8, port="q1:gt", clock=BasebandClockResource.IDENTITY)
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
        "q0:gt": 0.6,
        "q1:gt": 0.7,
    }
    compensation_pulses_start_duration_amp = _determine_compensation_pulse(
        schedule, max_compensation_amp, 4e-9, sampling_rate=1e9
    )

    assert compensation_pulses_start_duration_amp.keys() == {
        "q1:gt",
        "q0:gt",
    }

    assert compensation_pulses_start_duration_amp["q0:gt"].start == 8e-8
    assert math.isclose(
        compensation_pulses_start_duration_amp["q0:gt"].duration,
        2.8e-8,
    )
    assert math.isclose(
        compensation_pulses_start_duration_amp["q0:gt"].amp,
        -0.5910714285714285,
    )

    assert compensation_pulses_start_duration_amp["q1:gt"].start == 2e-8
    assert compensation_pulses_start_duration_amp["q1:gt"].duration == 4e-9
    assert math.isclose(
        compensation_pulses_start_duration_amp["q1:gt"].amp,
        -0.5625,
    )


class UnsupportedControlFlowOperation(ControlFlowOperation):
    def __init__(self, name):
        super().__init__(name)

    @property
    def body(self) -> Operation:
        return MagicMock()

    @body.setter
    def body(self, value: Union[Operation, Schedule]) -> None:
        pass

    def __str__(self) -> str:
        return "UnsupportedControlFlowOperation"


@pytest.mark.parametrize(
    "operation, expected_error",
    [
        (
            VoltageOffset(offset_path_I=1, offset_path_Q=1, port="q0:gt"),
            "Error calculating compensation pulse amplitude for "
            "'VoltageOffset"
            "(offset_path_I=1,offset_path_Q=1,port='q0:gt',clock='cl0.baseband',"
            "t0=0,reference_magnitude=None)'. "
            "Voltage offset operation type is not allowed "
            "in a pulse compensation structure. ",
        ),
        (
            UnsupportedControlFlowOperation(name="test"),
            "Error calculating compensation pulse amplitude for 'UnsupportedControlFlowOperation'. "
            "This control flow operation type is not allowed "
            "in a pulse compensation structure. ",
        ),
        (
            ConditionalOperation(
                body=SquarePulse(
                    amp=0.6,
                    duration=4e-9,
                    clock=BasebandClockResource.IDENTITY,
                    port="q0:gt",
                ),
                qubit_name="q0",
            ),
            "Error calculating compensation pulse amplitude for 'SquarePulse(amp=0.6,"
            "duration=4e-09,port='q0:gt',clock='cl0.baseband',reference_magnitude=None,t0=0)'. "
            "Operation is not allowed within a conditional operation. ",
        ),
    ],
)
def test_determine_compensation_pulse_error(operation, expected_error):
    schedule = Schedule("Schedule")
    schedule.add(operation)

    max_compensation_amp = {
        "q0:gt": 0.6,
    }
    if isinstance(operation, ConditionalOperation):
        schedule.add(
            operation=SquarePulse(
                amp=0.6,
                duration=4e-9,
                clock=BasebandClockResource.IDENTITY,
                port="q0:gt",
            ),
        )

    with pytest.raises(ValueError) as exception:
        _determine_compensation_pulse(schedule, max_compensation_amp, 4e-9, sampling_rate=1e9)

    assert exception.value.args[0] == expected_error


@pytest.mark.parametrize("is_circuit_level", [False, True])
def test_insert_compensation_pulses(  # noqa: PLR0915
    is_circuit_level,
    mock_setup_basic_transmon_with_standard_params,
    get_subschedule_operation,
):
    body = Schedule("schedule")
    body.add(
        SquarePulse(amp=0.8, duration=1e-8, port="q0:mw", clock=BasebandClockResource.IDENTITY)
    )
    body.add(RampPulse(amp=0.5, duration=1e-8, port="q1:mw", clock=BasebandClockResource.IDENTITY))
    body.add(
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
    body.add(X("q2"))

    schedule = Schedule("compensated_schedule")

    if is_circuit_level:
        schedule.add(PulseCompensation(body=body, qubits=["q0", "q1"]))

        q0 = mock_setup_basic_transmon_with_standard_params["q0"]
        q0.pulse_compensation.max_compensation_amp(0.6)
        q0.pulse_compensation.time_grid(4e-9)
        q0.pulse_compensation.sampling_rate(1e9)
        q1 = mock_setup_basic_transmon_with_standard_params["q1"]
        q1.pulse_compensation.max_compensation_amp(0.7)
        q1.pulse_compensation.time_grid(4e-9)
        q1.pulse_compensation.sampling_rate(1e9)
    else:
        schedule.add(
            PulseCompensation(
                body=body,
                max_compensation_amp={
                    "q0:mw": 0.6,
                    "q1:mw": 0.7,
                },
                time_grid=4e-9,
                sampling_rate=1e9,
            )
        )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule,
        config=mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )

    compensated_subschedule = get_subschedule_operation(compiled_schedule, [0])

    assert isinstance(compensated_subschedule, Schedule)

    subschedule_schedulable = list(compensated_subschedule.schedulables.values())[0]["name"]

    compensation_pulse_q0_schedulable = list(compensated_subschedule.schedulables.values())[1]
    compensation_pulse_q0 = compensated_subschedule.operations[
        compensation_pulse_q0_schedulable["operation_id"]
    ]
    compensation_pulse_q1_schedulable = list(compensated_subschedule.schedulables.values())[2]
    compensation_pulse_q1 = compensated_subschedule.operations[
        compensation_pulse_q1_schedulable["operation_id"]
    ]

    if compensation_pulse_q0["pulse_info"][0]["port"] == "q1:mw":
        compensation_pulse_q0_schedulable, compensation_pulse_q1_schedulable = (
            compensation_pulse_q1_schedulable,
            compensation_pulse_q0_schedulable,
        )
        compensation_pulse_q0, compensation_pulse_q1 = (
            compensation_pulse_q1,
            compensation_pulse_q0,
        )

    assert compensation_pulse_q0_schedulable["timing_constraints"][0]["rel_time"] == 8e-8
    assert (
        compensation_pulse_q0_schedulable["timing_constraints"][0]["ref_schedulable"]
        == subschedule_schedulable
    )
    assert compensation_pulse_q0_schedulable["timing_constraints"][0]["ref_pt"] == "start"
    assert compensation_pulse_q0_schedulable["timing_constraints"][0]["ref_pt_new"] == "start"
    assert len(compensation_pulse_q0["pulse_info"]) == 1
    assert (
        compensation_pulse_q0["pulse_info"][0]["wf_func"] == "quantify_scheduler.waveforms.square"
    )
    assert compensation_pulse_q0["pulse_info"][0]["reference_magnitude"] is None
    assert compensation_pulse_q0["pulse_info"][0]["t0"] == 0
    assert compensation_pulse_q0["pulse_info"][0]["port"] == "q0:mw"
    assert math.isclose(compensation_pulse_q0["pulse_info"][0]["amp"], -0.5910714285714285)
    assert math.isclose(compensation_pulse_q0["pulse_info"][0]["duration"], 2.8e-8)

    assert len(compensation_pulse_q1["pulse_info"]) == 1
    assert (
        compensation_pulse_q1["pulse_info"][0]["wf_func"] == "quantify_scheduler.waveforms.square"
    )
    assert compensation_pulse_q1["pulse_info"][0]["reference_magnitude"] is None
    assert compensation_pulse_q1["pulse_info"][0]["t0"] == 0
    assert compensation_pulse_q1["pulse_info"][0]["port"] == "q1:mw"
    assert math.isclose(compensation_pulse_q1["pulse_info"][0]["amp"], -0.5625)
    assert math.isclose(compensation_pulse_q1["pulse_info"][0]["duration"], 4e-9)

    # Check if the X operation has been device compiled which is inside the pulse compensation.
    compiled_x_op = get_subschedule_operation(compiled_schedule, [0, 0, 3])
    assert len(compiled_x_op["pulse_info"]) == 1
    assert compiled_x_op["pulse_info"][0]["wf_func"] == "quantify_scheduler.waveforms.drag"


def test_pulse_compensation_invalid_operation():
    with pytest.raises(ValueError) as exception:
        PulseCompensation(body=X("q0"), qubits=["q0"], time_grid=4e-9)

    assert exception.value.args[0] == (
        "PulseCompensation can only be defined on gate-level or device-level, "
        "but not both. If 'qubit' is defined, then 'max_compensation_amp', "
        "'time_grid' and 'sampling_rate' must be 'None'."
    )


def test_pulse_compensation_inconsistent_parameters(
    mock_setup_basic_transmon_with_standard_params,
):
    body = Schedule("schedule")
    body.add(
        SquarePulse(amp=0.8, duration=1e-8, port="q0:mw", clock=BasebandClockResource.IDENTITY)
    )
    body.add(RampPulse(amp=0.5, duration=1e-8, port="q1:mw", clock=BasebandClockResource.IDENTITY))

    schedule = Schedule("compensated_schedule")
    schedule.add(PulseCompensation(body=body, qubits=["q0", "q1"]))

    compiler = SerialCompiler(name="compiler")

    q0 = mock_setup_basic_transmon_with_standard_params["q0"]
    q1 = mock_setup_basic_transmon_with_standard_params["q1"]

    q0.pulse_compensation.max_compensation_amp(0.6)
    q0.pulse_compensation.time_grid(1e-9)
    q0.pulse_compensation.sampling_rate(1e9)
    q1.pulse_compensation.max_compensation_amp(0.7)
    q1.pulse_compensation.time_grid(4e-9)
    q1.pulse_compensation.sampling_rate(1e9)
    with pytest.raises(ValueError) as exception:
        compiler.compile(
            schedule,
            config=mock_setup_basic_transmon_with_standard_params[
                "quantum_device"
            ].generate_compilation_config(),
        )

    assert exception.value.args[0] == (
        "'time_grid' must be the same for every device element "
        "for pulse compensation. 'time_grid' for "
        "device element 'q1' is '4e-09', "
        "for others it is '1e-09'."
    )

    q0.pulse_compensation.time_grid(4e-9)
    q0.pulse_compensation.sampling_rate(2e9)
    q1.pulse_compensation.time_grid(4e-9)
    q1.pulse_compensation.sampling_rate(1e9)
    with pytest.raises(ValueError) as exception:
        compiler.compile(
            schedule,
            config=mock_setup_basic_transmon_with_standard_params[
                "quantum_device"
            ].generate_compilation_config(),
        )

    assert exception.value.args[0] == (
        "'sampling_rate' must be the same for every device element "
        "for pulse compensation. 'sampling_rate' for "
        "device element 'q1' is '1000000000.0', "
        "for others it is '2000000000.0'."
    )
