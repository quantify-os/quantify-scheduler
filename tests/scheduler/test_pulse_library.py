# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=eval-used
from unittest import TestCase

import json
import pytest

import numpy as np
from quantify_scheduler import Operation, Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.json_utils import ScheduleJSONDecoder, ScheduleJSONEncoder
from quantify_scheduler.operations.gate_library import X90, X
from quantify_scheduler.operations.pulse_library import (
    DRAGPulse,
    IdlePulse,
    NumericalPulse,
    RampPulse,
    ShiftClockPhase,
    SkewedHermitePulse,
    SoftSquarePulse,
    SquarePulse,
    create_dc_compensation_pulse,
    decompose_long_square_pulse,
)
from quantify_scheduler.resources import BasebandClockResource, ClockResource


# --------- Test classes and member methods ---------
def test_operation_duration_single_pulse() -> None:
    dgp = DRAGPulse(
        G_amp=0.8, D_amp=-0.3, phase=24.3, duration=20e-9, clock="cl:01", port="p.01"
    )
    assert dgp.duration == pytest.approx(20e-9)
    idle = IdlePulse(50e-9)
    assert idle.duration == pytest.approx(50e-9)


def test_operation_duration_single_pulse_delayed() -> None:
    dgp = DRAGPulse(
        G_amp=0.8,
        D_amp=-0.3,
        phase=24.3,
        duration=10e-9,
        clock="cl:01",
        port="p.01",
        t0=3.4e-9,
    )
    assert dgp.duration == pytest.approx(13.4e-9)


def test_operation_add_pulse() -> None:
    dgp1 = DRAGPulse(
        G_amp=0.8, D_amp=-0.3, phase=0, duration=20e-9, clock="cl:01", port="p.01", t0=0
    )
    assert len(dgp1["pulse_info"]) == 1
    dgp1.add_pulse(dgp1)
    assert len(dgp1["pulse_info"]) == 2

    x90 = X90("q1")
    assert len(x90["pulse_info"]) == 0
    dgp = DRAGPulse(
        G_amp=0.8, D_amp=-0.3, phase=0, duration=20e-9, clock="cl:01", port="p.01", t0=0
    )
    x90.add_pulse(dgp)
    assert len(x90["pulse_info"]) == 1


def test_operation_duration_composite_pulse() -> None:
    dgp1 = DRAGPulse(
        G_amp=0.8,
        D_amp=-0.3,
        phase=24.3,
        duration=10e-9,
        clock="cl:01",
        port="p.01",
        t0=0,
    )
    assert dgp1.duration == pytest.approx(10e-9)

    # Adding a shorter pulse is not expected to change the duration
    dgp2 = DRAGPulse(
        G_amp=0.8,
        D_amp=-0.3,
        phase=24.3,
        duration=7e-9,
        clock="cl:01",
        port="p.01",
        t0=2e-9,
    )
    dgp1.add_pulse(dgp2)
    assert dgp1.duration == pytest.approx(10e-9)

    # adding a longer pulse is expected to change the duration
    dgp3 = DRAGPulse(
        G_amp=0.8,
        D_amp=-0.3,
        phase=24.3,
        duration=12e-9,
        clock="cl:01",
        port="p.01",
        t0=3.4e-9,
    )
    dgp1.add_pulse(dgp3)
    assert dgp3.duration == pytest.approx(15.4e-9)
    assert dgp1.duration == pytest.approx(15.4e-9)


@pytest.mark.parametrize(
    "operation",
    [
        IdlePulse(duration=50e-9),
        ShiftClockPhase(clock="q0.01", phase_shift=180.0),
        SquarePulse(amp=0.5, duration=300e-9, port="p.01", clock="cl0.baseband"),
        SoftSquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0),
        RampPulse(1.0, 16e-9, "q0:mw"),
        NumericalPulse(
            np.linspace(0, 1, 1000), np.linspace(0, 20e-6, 1000), "q0:mw", "q0.01"
        ),
        DRAGPulse(
            G_amp=0.8,
            D_amp=-0.3,
            phase=24.3,
            duration=20e-9,
            clock="cl:01",
            port="p.01",
        ),
        SkewedHermitePulse(
            duration=2e-6,
            amplitude=0.05,
            skewness=-0.2,
            phase=90.0,
            port="qe0.mw",
            clock="qe0.spec",
        ),
    ],
)
def test_pulse_is_valid(operation: Operation) -> None:
    assert Operation.is_valid(operation)


def test_decompose_long_square_pulse() -> None:
    # Non matching durations ("extra" pulse needed to get the necessary duration)
    duration = 200e-9
    duration_sq_max = 16e-9
    amp = 1.0
    port = "LP"
    clock = "baseband"
    sums = [duration, duration + duration_sq_max - (duration % duration_sq_max)]
    for single_duration, sum_ in zip([False, True], sums):
        pulses = decompose_long_square_pulse(
            duration=duration,
            duration_max=duration_sq_max,
            single_duration=single_duration,
            amp=amp,
            port=port,
            clock=clock,
        )

        assert len(pulses) == int(duration // duration_sq_max) + bool(
            duration % duration_sq_max
        )
        assert sum(
            pulse["pulse_info"][0]["duration"] for pulse in pulses
        ) == pytest.approx(sum_)

    # Exactly matching durations
    duration = 200e-6
    duration_sq_max = 50e-6

    for single_duration in [False, True]:
        pulses = decompose_long_square_pulse(
            duration=duration,
            duration_max=duration_sq_max,
            single_duration=single_duration,
            amp=amp,
            port=port,
            clock=clock,
        )

        assert len(pulses) == int(duration // duration_sq_max) + bool(
            duration % duration_sq_max
        )
        assert sum(
            pulse["pulse_info"][0]["duration"] for pulse in pulses
        ) == pytest.approx(duration)


@pytest.mark.parametrize(
    "operation",
    [
        IdlePulse(16e-9),
        ShiftClockPhase(clock="q0.01", phase_shift=180.0),
        SquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0, 0),
        SoftSquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0),
        RampPulse(1.0, 16e-9, "q0:mw"),
        NumericalPulse(
            np.linspace(0, 1, 1000), np.linspace(0, 20e-6, 1000), "q0:mw", "q0.01"
        ),
        DRAGPulse(0.8, 0.83, 1.0, "q0:mw", 16e-9, "q0.01", 0),
        SkewedHermitePulse(
            duration=2e-6,
            amplitude=0.05,
            skewness=-0.2,
            phase=90.0,
            port="qe0.mw",
            clock="qe0.spec",
        ),
    ],
)
def test__repr__(operation: Operation) -> None:
    # Arrange
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)
    assert obj == operation


@pytest.mark.parametrize(
    "operation",
    [
        IdlePulse(16e-9),
        ShiftClockPhase(clock="q0.01", phase_shift=180.0),
        SquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0, 0),
        SoftSquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0),
        RampPulse(1.0, 16e-9, "q0:mw"),
        NumericalPulse(
            np.linspace(0, 1, 1000), np.linspace(0, 20e-6, 1000), "q0:mw", "q0.01"
        ),
        DRAGPulse(0.8, 0.83, 1.0, "q0:mw", 16e-9, "q0.01", 0),
        SkewedHermitePulse(
            duration=2e-6,
            amplitude=0.05,
            skewness=-0.2,
            phase=90.0,
            port="qe0.mw",
            clock="qe0.spec",
        ),
    ],
)
def test__str__(operation: Operation) -> None:
    assert isinstance(eval(str(operation)), type(operation))


@pytest.mark.parametrize(
    "operation",
    [
        IdlePulse(16e-9),
        SquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0, 0),
        ShiftClockPhase(clock="q0.01", phase_shift=180.0),
        SoftSquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0),
        RampPulse(1.0, 16e-9, "q0:mw"),
        NumericalPulse(
            np.linspace(0, 1, 1000), np.linspace(0, 20e-6, 1000), "q0:mw", "q0.01"
        ),
        DRAGPulse(0.8, 0.83, 1.0, "q0:mw", 16e-9, "q0.01", 0),
        SkewedHermitePulse(
            duration=2e-6,
            amplitude=0.05,
            skewness=-0.2,
            phase=90.0,
            port="qe0.mw",
            clock="qe0.spec",
        ),
    ],
)
def test_deserialize(operation: Operation) -> None:
    # Arrange
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)

    # Assert
    TestCase().assertDictEqual(obj.data, operation.data)


@pytest.mark.parametrize(
    "operation",
    [
        IdlePulse(16e-9),
        ShiftClockPhase(clock="q0.01", phase_shift=180.0),
        SquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0, 0),
        SoftSquarePulse(1.0, 16e-9, "q0:mw", "q0.01", 0),
        RampPulse(1.0, 16e-9, "q0:mw"),
        NumericalPulse(
            np.linspace(0, 1, 1000), np.linspace(0, 20e-6, 1000), "q0:mw", "q0.01"
        ),
        DRAGPulse(0.8, 0.83, 1.0, "q0:mw", 16e-9, "q0.01", 0),
        SkewedHermitePulse(
            duration=2e-6,
            amplitude=0.05,
            skewness=-0.2,
            phase=90.0,
            port="qe0.mw",
            clock="qe0.spec",
        ),
    ],
)
def test__repr__modify_not_equal(operation: Operation) -> None:
    # Arrange
    # Arrange
    operation_state: str = json.dumps(operation, cls=ScheduleJSONEncoder)

    # Act
    obj = json.loads(operation_state, cls=ScheduleJSONDecoder)
    assert obj == operation

    # Act
    obj.data["pulse_info"][0]["foo"] = "bar"

    # Assert
    assert obj != operation


def test_dccompensation_pulse_amp() -> None:
    pulse0 = SquarePulse(
        amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY
    )
    pulse1 = RampPulse(
        amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY
    )

    pulse2 = create_dc_compensation_pulse(
        duration=1e-8,
        pulses=[pulse0, pulse1],
        sampling_rate=int(1e16),
        port="LP",
    )
    TestCase().assertAlmostEqual(pulse2.data["pulse_info"][0]["amp"], -1.5)


def test_dccompensation_pulse_modulated() -> None:
    clock = ClockResource("clock", 1.0)
    pulse0 = SquarePulse(amp=1, duration=1e-8, port="LP", clock=clock)
    pulse = create_dc_compensation_pulse(
        amp=1, pulses=[pulse0], port="LP", sampling_rate=int(1e9)
    )
    assert pulse.data["pulse_info"][0]["duration"] == 0.0


def test_dccompensation_pulse_duration() -> None:
    pulse0 = SquarePulse(
        amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY
    )
    pulse1 = RampPulse(
        amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY
    )

    pulse2 = create_dc_compensation_pulse(
        amp=1,
        pulses=[pulse0, pulse1],
        sampling_rate=int(1e9),
        port="LP",
    )
    TestCase().assertAlmostEqual(pulse2.data["pulse_info"][0]["duration"], 1.5e-8)
    TestCase().assertAlmostEqual(pulse2.data["pulse_info"][0]["amp"], -1.0)


def test_dccompensation_pulse_both_params() -> None:
    with pytest.raises(ValueError):
        pulse0 = SquarePulse(
            amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY
        )
        create_dc_compensation_pulse(
            amp=1,
            duration=1e-8,
            pulses=[pulse0],
            sampling_rate=int(1e9),
            port="LP",
        )


# --------- Test pulse compilation ---------
def test_dragpulse_motzoi(mock_setup_basic_transmon_with_standard_params):
    mock_setup_basic_transmon_with_standard_params["q0"].rxy.amp180(0.2)
    mock_setup_basic_transmon_with_standard_params["q0"].rxy.motzoi(0.02)

    sched = Schedule("Test DRAG Pulse")
    sched.add(X("q0"))

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )

    D_amp = (
        list(compiled_sched.operations.values())[0].data["pulse_info"][0].get("D_amp")
    )
    assert D_amp == mock_setup_basic_transmon_with_standard_params["q0"].rxy.motzoi(), (
        "The amplification of the derivative DRAG pulse is not equal to the motzoi "
        "parameter"
    )
