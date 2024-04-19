# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

from __future__ import annotations

import inspect
from typing import List
from unittest.case import TestCase

import numpy as np
import pytest
from pytest_mock.plugin import MockerFixture

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.helpers.schedule import get_pulse_uuid
from quantify_scheduler.helpers.waveforms import (
    apply_mixer_skewness_corrections,
    area_pulse,
    area_pulses,
    exec_custom_waveform_function,
    exec_waveform_function,
    get_waveform,
    get_waveform_by_pulseid,
    get_waveform_size,
    modulate_waveform,
    resize_waveform,
    shift_waveform,
)
from quantify_scheduler.operations.gate_library import X90
from quantify_scheduler.operations.pulse_library import (
    RampPulse,
    SquarePulse,
    StaircasePulse,
)


@pytest.mark.parametrize(
    "size,granularity,expected",
    [
        (0, 16, 16),
        (10, 16, 16),
        (16, 16, 16),
        (30, 16, 32),
        (33, 16, 48),
    ],
)
def test_resize_waveform(size: int, granularity: int, expected: int) -> None:
    # Arrange
    waveform = np.arange(0, size, 1)

    # Act
    waveform = resize_waveform(waveform, granularity)

    # Assert
    assert len(waveform) == expected


@pytest.mark.parametrize(
    "wf_func,sampling_rate",
    [
        ("quantify_scheduler.waveforms.square", 2.4e9),
        ("quantify_scheduler.waveforms.ramp", 2.4e9),
        ("quantify_scheduler.waveforms.soft_square", 2.4e9),
        ("quantify_scheduler.waveforms.drag", 2.4e9),
    ],
)
def test_get_waveform(
    mocker: MockerFixture, wf_func: str, sampling_rate: float
) -> None:
    # Arrange
    mock = mocker.patch(
        "quantify_scheduler.helpers.waveforms.exec_waveform_function", return_value=[]
    )
    pulse_info_mock = {"duration": 1.6e-08, "wf_func": wf_func}

    # Act
    get_waveform(pulse_info_mock, sampling_rate)

    # Assert
    args, _ = mock.call_args
    assert args[0] == wf_func
    assert isinstance(args[1], np.ndarray)
    assert args[2] == pulse_info_mock


def test_get_waveform_by_pulseid(
    schedule_with_pulse_info: Schedule,
) -> None:
    # Arrange
    operation_id = list(schedule_with_pulse_info.schedulables.values())[0][
        "operation_id"
    ]
    pulse_info_0 = schedule_with_pulse_info.operations[operation_id]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)
    expected_keys: List[int] = [pulse_id]

    # Act
    waveform_dict = get_waveform_by_pulseid(schedule_with_pulse_info)

    # Assert
    assert len(waveform_dict) == 1
    assert list(waveform_dict.keys()) == expected_keys
    assert callable(waveform_dict[pulse_id])


def test_get_waveform_by_pulseid_are_unique(
    device_compile_config_basic_transmon,
) -> None:
    # Arrange
    schedule = Schedule("my-schedule")
    schedule.add(X90("q0"))
    schedule.add(X90("q0"))

    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule, config=device_compile_config_basic_transmon
    )

    operation_id = list(schedule.schedulables.values())[0]["operation_id"]
    pulse_info_0 = schedule.operations[operation_id]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)
    expected_keys: List[int] = [pulse_id]

    # Act
    waveform_dict = get_waveform_by_pulseid(schedule)

    # Assert
    assert len(waveform_dict) == 1
    assert list(waveform_dict.keys()) == expected_keys
    assert callable(waveform_dict[pulse_id])


def test_get_waveform_by_pulseid_empty(empty_schedule: Schedule) -> None:
    # Arrange
    # Act
    waveform_dict = get_waveform_by_pulseid(empty_schedule)

    # Assert
    assert len(waveform_dict) == 0


@pytest.mark.parametrize(
    "wf_func",
    [
        ("quantify_scheduler.waveforms.square"),
        ("quantify_scheduler.waveforms.ramp"),
        ("quantify_scheduler.waveforms.soft_square"),
        ("quantify_scheduler.waveforms.drag"),
    ],
)
def test_exec_waveform_function(wf_func: str, mocker: MockerFixture) -> None:
    # Arrange
    pulse_duration = 1e-08
    t: np.ndarray = np.arange(0, 0 + pulse_duration, 1 / 1e9)
    pulse_info_stub = {
        "amp": 0.5,
        "offset": 0,
        "duration": pulse_duration,
        "G_amp": 0.7,
        "D_amp": -0.2,
        "nr_sigma": 3,
        "sigma": None,
        "phase": 90,
    }
    wavefn_stub = mocker.patch(wf_func, return_value=[])

    # Act
    waveform = exec_waveform_function(wf_func, t, pulse_info=pulse_info_stub)

    # Assert
    wavefn_stub.assert_called()
    assert waveform == []


@pytest.mark.parametrize(
    "wf_func",
    [
        ("foo.bar.square"),
        ("bar.foo.sawtooth"),
        ("module.function"),
    ],
)
def test_exec_waveform_function_with_custom(
    wf_func: str, mocker: MockerFixture
) -> None:
    # Arrange
    pulse_duration = 1e-08
    t: np.ndarray = np.arange(0, 0 + pulse_duration, 1 / 1e9)
    pulse_info_stub = {
        "amp": 0.5,
        "duration": pulse_duration,
        "G_amp": 0.7,
        "D_amp": -0.2,
        "nr_sigma": 3,
        "phase": 90,
    }
    wavefn_stub = mocker.patch(
        "quantify_scheduler.helpers.waveforms.exec_custom_waveform_function",
        return_value=[],
    )

    # Act
    waveform = exec_waveform_function(wf_func, t, pulse_info=pulse_info_stub)

    # Assert
    wavefn_stub.assert_called()
    assert waveform == []


def test_exec_custom_waveform_function(mocker: MockerFixture) -> None:
    # Arrange
    t = np.arange(0, 10, 1)
    pulse_info_mock = {"duration": 1.4e-9, "t0": 0}

    def custom_function(t: int, duration: float, t0: float) -> None:
        pass

    mock = mocker.Mock()
    mock.__signature__ = inspect.signature(custom_function)
    mocker.patch(
        # We need to patch function that is available in the module, which is not
        #   "quantify_scheduler.helpers.importers.import_python_object_from_string",
        # but ...
        "quantify_scheduler.helpers.waveforms.import_python_object_from_string",
        return_value=mock,
    )

    # Act
    exec_custom_waveform_function("mock_custom_function", t, pulse_info_mock)

    # Assert
    mock.assert_called_with(t=t, duration=1.4e-9, t0=0)


def test_shift_waveform_misaligned() -> None:
    # Arrange
    clock_rate: int = 2400000000
    t = np.arange(0, 16e-9, 1 / clock_rate)
    waveform = np.ones(len(t))
    start_in_seconds = 16e-9  # 16ns
    resolution = 8

    start_in_clocks = start_in_seconds * clock_rate
    n_samples = int(start_in_clocks % resolution)

    # 16e-9 / (8 / 2.4e9) = 4.8
    # waveform must start at: floor(4.8)

    expected = np.concatenate([np.zeros(n_samples), waveform])

    # Act
    clock, shifted_waveform = shift_waveform(
        waveform, start_in_seconds, clock_rate, resolution
    )

    # Assert
    assert clock == 4
    assert n_samples == 6
    assert len(shifted_waveform) == 45
    np.testing.assert_array_equal(shifted_waveform, expected)


def test_shift_waveform_aligned() -> None:
    # Arrange
    clock_rate: int = 2400000000
    t = np.arange(0, 16e-9, 1 / clock_rate)
    waveform = np.ones(len(t))
    start_in_seconds = 3.3333e-9
    resolution = 8

    # Act
    clock, shifted_waveform = shift_waveform(
        waveform, start_in_seconds, clock_rate, resolution
    )

    # Assert
    assert clock == 1
    np.testing.assert_array_equal(shifted_waveform, waveform)


@pytest.mark.parametrize(
    "size,granularity,expected",
    [
        (0, 16, 16),
        (10, 16, 16),
        (16, 16, 16),
        (30, 16, 32),
        (33, 16, 48),
    ],
)
def test_get_waveform_size(size: int, granularity: int, expected: int) -> None:
    # Act
    new_size = get_waveform_size(np.ones(size), granularity)

    # Assert
    assert expected == new_size


def test_apply_mixer_skewness_corrections() -> None:
    # Arrange
    frequency = 10e6
    t = np.linspace(0, 1e-6, 1000)

    amplitude_ratio: float = 2.1234
    phase_shift: float = 90

    real = np.cos(2 * np.pi * frequency * t)
    imag = np.sin(2 * np.pi * frequency * t)
    waveform = real + 1.0j * imag

    # Act
    waveform = apply_mixer_skewness_corrections(waveform, amplitude_ratio, phase_shift)
    amp_ratio_after = np.max(np.abs(waveform.real)) / np.max(np.abs(waveform.imag))

    # Assert
    assert isinstance(waveform, np.ndarray)
    assert amp_ratio_after == pytest.approx(amplitude_ratio, 1e-4)
    normalized_real = waveform.real / np.max(np.abs(waveform.real))
    normalized_imag = waveform.imag / np.max(np.abs(waveform.imag))
    assert np.allclose(normalized_real, normalized_imag)


@pytest.mark.parametrize(
    "frequency,t0,points",
    [(10e6, 50e-9, 1000), (-10e6, 0.0, 1000), (0.0, -10e-9, 2), (10e6, -10e-9, 1)],
)
def test_modulate_waveform(frequency, t0, points) -> None:
    # Arrange
    t = np.linspace(0, 1e-6, points)
    envelope = np.ones(len(t))

    expected_real = np.cos(2 * np.pi * frequency * (t + t0))
    expected_imag = np.sin(2 * np.pi * frequency * (t + t0))

    # Act
    waveform = modulate_waveform(t, envelope, frequency, t0)

    # Assert
    assert np.allclose(waveform.real, expected_real)
    assert np.allclose(waveform.imag, expected_imag)


@pytest.mark.parametrize(
    "frequency,t0,points",
    [(10e6, 50e-9, 1000), (-10e6, 0.0, 1000), (0.0, -10e-9, 2), (10e6, -10e-9, 1)],
)
def test_demodulate_waveform(frequency, t0, points) -> None:
    # Arrange
    t = np.linspace(0, 1e-6, points)
    envelopes = [np.ones(len(t)), np.sin(2 * np.pi * t * 1e6)]

    for envelope in envelopes:
        # Act
        mod_waveform = modulate_waveform(t, envelope, frequency, t0)
        demod_waveform = modulate_waveform(t, mod_waveform, -frequency, t0)

        # Assert
        assert np.allclose(demod_waveform, envelope)


def test_area_pulse() -> None:
    pulse = {
        "wf_func": "quantify_scheduler.waveforms.square",
        "amp": 1,
        "offset": 0,
        "duration": 1e-08,
        "phase": 0,
        "t0": 0,
        "clock": "cl0.baseband",
        "port": "LP",
    }
    result = area_pulse(pulse, int(1e9))
    TestCase().assertAlmostEqual(result, 1e-8)


def test_area_pulses() -> None:
    test_list = [
        {
            "wf_func": "quantify_scheduler.waveforms.square",
            "amp": 1,
            "offset": 0,
            "duration": 1e-08,
            "phase": 0,
            "t0": 0,
            "clock": "cl0.baseband",
            "port": "LP",
        },
        {
            "wf_func": "quantify_scheduler.waveforms.ramp",
            "amp": 1,
            "offset": 0,
            "duration": 1e-08,
            "t0": 0,
            "clock": "cl0.baseband",
            "port": "LP",
        },
    ]

    result = area_pulses(test_list, int(1e9))
    TestCase().assertAlmostEqual(result, 1.5e-8)


def test_area_pulses_half_sampling() -> None:
    operation = SquarePulse(amp=1, duration=10.5e-9, port="P")
    area = area_pulses(operation.data["pulse_info"], sampling_rate=1e9)
    print(area)
    TestCase().assertAlmostEqual(area, 10.5e-9)


def test_area_pulses_long_pulse() -> None:
    operation = SquarePulse(amp=1, duration=1e6, port="P")
    area = area_pulses(operation.data["pulse_info"], sampling_rate=1e10)
    TestCase().assertAlmostEqual(area, 1e6)


def test_area_pulses_ramp_pulse_regression() -> None:
    operation = RampPulse(amp=0, offset=1, duration=10.5e-9, port="P")
    area = area_pulses(operation.data["pulse_info"], sampling_rate=1e9)
    TestCase().assertAlmostEqual(area, 10.5e-9)

    operation = RampPulse(amp=1, offset=0, duration=10e-9, port="P")
    area = area_pulses(operation.data["pulse_info"], sampling_rate=1e9)
    TestCase().assertAlmostEqual(area, 5e-9)


def test_area_pulses_staircase_pulse() -> None:
    operation = StaircasePulse(
        start_amp=0, final_amp=1, num_steps=5, duration=10e-9, port="P"
    )
    area = area_pulses(operation.data["pulse_info"], sampling_rate=1e9)
    TestCase().assertAlmostEqual(area, 5e-9)
