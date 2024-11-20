import json
from unittest import TestCase

import numpy as np
import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.qblox.operations.pulse_factories import (
    long_square_pulse,
)
from quantify_scheduler.json_utils import SchedulerJSONDecoder, SchedulerJSONEncoder
from quantify_scheduler.operations.gate_library import X90, X
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import (
    ChirpPulse,
    DRAGPulse,
    GaussPulse,
    IdlePulse,
    NumericalPulse,
    RampPulse,
    ResetClockPhase,
    SetClockFrequency,
    ShiftClockPhase,
    SkewedHermitePulse,
    SoftSquarePulse,
    SquarePulse,
    StaircasePulse,
    SuddenNetZeroPulse,
    Timestamp,
    VoltageOffset,
    create_dc_compensation_pulse,
    decompose_long_square_pulse,
)
from quantify_scheduler.resources import BasebandClockResource, ClockResource
from quantify_scheduler.schedules.schedule import Schedule


# --------- Test classes and member methods ---------
def test_operation_duration_single_pulse() -> None:
    dgp = DRAGPulse(G_amp=0.8, D_amp=-0.3, phase=24.3, duration=20e-9, clock="cl:01", port="p.01")
    assert dgp.duration == pytest.approx(20e-9)
    idle = IdlePulse(50e-9)
    assert idle.duration == pytest.approx(50e-9)

    gauss = GaussPulse(G_amp=0.2, phase=27, duration=200e-9, clock="cl:01", port="p.01")
    assert gauss.duration == pytest.approx(200e-9)


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

    gp = GaussPulse(
        G_amp=0.5,
        phase=21.3,
        duration=17e-9,
        clock="cl:01",
        port="p.01",
        t0=8.4e-9,
    )
    assert gp.duration == pytest.approx(25.4e-9)


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

    gauss = GaussPulse(G_amp=0.2, phase=27, duration=200e-9, clock="cl:01", port="p.01")
    x90.add_pulse(gauss)
    assert len(x90["pulse_info"]) == 2


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

        assert len(pulses) == int(duration // duration_sq_max) + bool(duration % duration_sq_max)
        assert sum(pulse["pulse_info"][0]["duration"] for pulse in pulses) == pytest.approx(sum_)

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

        assert len(pulses) == int(duration // duration_sq_max) + bool(duration % duration_sq_max)
        assert sum(pulse["pulse_info"][0]["duration"] for pulse in pulses) == pytest.approx(
            duration
        )


@pytest.mark.parametrize(
    "operation",
    [
        [
            ShiftClockPhase(clock=clock, phase_shift=180.0),
            ResetClockPhase(clock=clock),
            SetClockFrequency(clock=clock, clock_freq_new=1e6),
            IdlePulse(duration=duration),
            RampPulse(amp=1.0, duration=duration, port=port),
            StaircasePulse(start_amp=0, final_amp=1, num_steps=5, duration=duration, port=port),
            SquarePulse(amp=1.0, duration=duration, port=port, clock=clock),
            SuddenNetZeroPulse(
                amp_A=0.4,
                amp_B=0.2,
                net_zero_A_scale=0.95,
                t_pulse=20e-9,
                t_phi=2e-9,
                t_integral_correction=10e-9,
                port=port,
            ),
            SoftSquarePulse(amp=1.0, duration=duration, port=port, clock=clock),
            ChirpPulse(
                amp=1.0,
                start_freq=1e6,
                end_freq=2e6,
                duration=duration,
                port=port,
                clock=clock,
            ),
            DRAGPulse(
                G_amp=0.8,
                D_amp=0.83,
                phase=1.0,
                duration=duration,
                port=port,
                clock=clock,
            ),
            GaussPulse(
                G_amp=0.2,
                phase=0.1,
                duration=duration,
                port=port,
                clock=clock,
            ),
            NumericalPulse(
                samples=np.linspace(0, 1, 1000),
                t_samples=np.linspace(0, 20e-6, 1000),
                port=port,
                clock=clock,
            ),
            SkewedHermitePulse(
                amplitude=0.05,
                skewness=-0.2,
                phase=90.0,
                duration=duration,
                port="qe0.mw",
                clock="qe0.spec",
            ),
            VoltageOffset(
                offset_path_I=0.5,
                offset_path_Q=0.1,
                port=port,
                clock=clock,
            ),
            Timestamp(
                port=port,
                clock=clock,
            ),
        ]
        for clock in ["q0.01"]
        for port in ["q0:mw"]
        for duration in [16e-9]
    ][0],
)
class TestPulseLevelOperation:
    def test__repr__(self, operation: Operation) -> None:
        # Arrange
        operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

        # Act
        obj = json.loads(operation_state, cls=SchedulerJSONDecoder)
        assert obj == operation

    def test__str__(self, operation: Operation) -> None:
        assert isinstance(eval(str(operation)), type(operation))

    def test_is_valid(self, operation: Operation) -> None:
        assert Operation.is_valid(operation)

    def test_duration(self, operation: Operation) -> None:
        pulse_info = operation.data["pulse_info"][0]
        if operation.__class__ in [
            SetClockFrequency,
            ShiftClockPhase,
            ResetClockPhase,
            VoltageOffset,
            Timestamp,
        ]:
            assert operation.duration == 0, operation
        elif operation.__class__ is SuddenNetZeroPulse:
            assert (
                operation.duration
                == pulse_info["t_pulse"] + pulse_info["t_phi"] + pulse_info["t_integral_correction"]
            )
        elif operation.__class__ is NumericalPulse:
            assert operation.duration == pulse_info["t_samples"][-1] - pulse_info["t_samples"][0]
        else:
            assert operation.duration == 16e-9, operation

    def test_deserialize(self, operation: Operation) -> None:
        # Arrange
        operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

        # Act
        obj = json.loads(operation_state, cls=SchedulerJSONDecoder)

        # Assert
        TestCase().assertDictEqual(obj.data, operation.data)

    def test__repr__modify_not_equal(self, operation: Operation) -> None:
        # Arrange
        operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

        # Act
        obj = json.loads(operation_state, cls=SchedulerJSONDecoder)
        assert obj == operation

        # Act
        obj.data["pulse_info"][0]["foo"] = "bar"

        # Assert
        assert obj != operation


def test_complex_square_pulse(mock_setup_basic_transmon_with_standard_params):
    pulse0 = SquarePulse(
        amp=1 + 1j,
        duration=1e-8,
        port="q0:fl",
        clock=BasebandClockResource.IDENTITY,
    )

    assert pulse0["pulse_info"][0]["amp"] == 1 + 1j

    pulse1 = long_square_pulse(
        amp=1 + 1j,
        duration=1.2e-8,
        port="q0:fl",
        clock=BasebandClockResource.IDENTITY,
    )

    assert pulse1["pulse_info"][0]["offset_path_I"] == 1
    assert pulse1["pulse_info"][0]["offset_path_Q"] == 1
    assert pulse1["pulse_info"][1]["offset_path_I"] == 0
    assert pulse1["pulse_info"][1]["offset_path_Q"] == 0
    assert pulse1["pulse_info"][2]["amp"] == 1 + 1j


def test_dc_compensation_pulse_amp() -> None:
    pulse0 = SquarePulse(amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY)
    pulse1 = RampPulse(amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY)

    pulse2 = create_dc_compensation_pulse(
        duration=1e-8,
        pulses=[pulse0, pulse1],
        sampling_rate=int(1e16),
        port="LP",
    )
    TestCase().assertAlmostEqual(pulse2.data["pulse_info"][0]["amp"], -1.5)


def test_dc_compensation_pulse_modulated() -> None:
    clock = ClockResource("clock", 1.0)
    pulse0 = SquarePulse(amp=1, duration=1e-8, port="LP", clock=clock.name)
    pulse = create_dc_compensation_pulse(amp=1, pulses=[pulse0], port="LP", sampling_rate=int(1e9))
    assert pulse.data["pulse_info"][0]["duration"] == 0.0


def test_dc_compensation_pulse_duration() -> None:
    pulse0 = SquarePulse(amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY)
    pulse1 = RampPulse(amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY)

    pulse2 = create_dc_compensation_pulse(
        amp=1,
        pulses=[pulse0, pulse1],
        sampling_rate=int(1e9),
        port="LP",
    )
    TestCase().assertAlmostEqual(pulse2.data["pulse_info"][0]["duration"], 1.5e-8)
    TestCase().assertAlmostEqual(pulse2.data["pulse_info"][0]["amp"], -1.0)


def test_dc_compensation_pulse_both_params() -> None:
    with pytest.raises(ValueError):
        pulse0 = SquarePulse(amp=1, duration=1e-8, port="LP", clock=BasebandClockResource.IDENTITY)
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

    D_amp = list(compiled_sched.operations.values())[0].data["pulse_info"][0].get("D_amp")
    assert D_amp == mock_setup_basic_transmon_with_standard_params["q0"].rxy.motzoi(), (
        "The amplification of the derivative DRAG pulse is not equal to the motzoi " "parameter"
    )
