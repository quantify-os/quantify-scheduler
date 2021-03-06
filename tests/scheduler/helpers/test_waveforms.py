# -----------------------------------------------------------------------------
# Description:    Tests schedule helper functions.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
# pylint: disable=missing-function-docstring
from __future__ import annotations

import inspect
from typing import List

import numpy as np
import pytest

from quantify_scheduler.gate_library import X90
from quantify_scheduler.helpers.schedule import get_pulse_uuid
from quantify_scheduler.helpers.waveforms import (
    exec_custom_waveform_function,
    exec_waveform_function,
    get_waveform,
    get_waveform_by_pulseid,
    get_waveform_size,
    resize_waveform,
    shift_waveform,
    apply_mixer_skewness_corrections,
    modulate_waveform,
)
from quantify_scheduler.types import Schedule


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
def test_resize_waveform(size: int, granularity: int, expected: int):
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
def test_get_waveform(mocker, wf_func: str, sampling_rate: float):
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
):
    # Arrange
    operation_repr = schedule_with_pulse_info.timing_constraints[0]["operation_repr"]
    pulse_info_0 = schedule_with_pulse_info.operations[operation_repr]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)
    expected_keys: List[int] = [pulse_id]

    # Act
    waveform_dict = get_waveform_by_pulseid(schedule_with_pulse_info)

    # Assert
    assert len(waveform_dict) == 1
    assert list(waveform_dict.keys()) == expected_keys
    assert callable(waveform_dict[pulse_id])


def test_get_waveform_by_pulseid_are_unique(
    create_schedule_with_pulse_info,
):
    # Arrange
    schedule = Schedule("my-schedule")
    schedule.add(X90("q0"))
    schedule.add(X90("q0"))
    create_schedule_with_pulse_info(schedule)

    operation_repr = schedule.timing_constraints[0]["operation_repr"]
    pulse_info_0 = schedule.operations[operation_repr]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)
    expected_keys: List[int] = [pulse_id]

    # Act
    waveform_dict = get_waveform_by_pulseid(schedule)

    # Assert
    assert len(waveform_dict) == 1
    assert list(waveform_dict.keys()) == expected_keys
    assert callable(waveform_dict[pulse_id])


def test_get_waveform_by_pulseid_empty(empty_schedule: Schedule):
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
def test_exec_waveform_function(wf_func: str, mocker):
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
def test_exec_waveform_function_with_custom(wf_func: str, mocker):
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


def test_exec_custom_waveform_function(mocker):
    # Arrange
    t = np.arange(0, 10, 1)
    pulse_info_mock = {"duration": 1.4e-9, "t0": 0}

    def custom_function(
        t: int, duration: float, t0: float
    ):  # pylint: disable=unused-argument
        pass

    mock = mocker.Mock()
    mock.__signature__ = inspect.signature(custom_function)
    mocker.patch(
        "quantify_core.utilities.general.import_func_from_string",
        return_value=mock,
    )

    # Act
    exec_custom_waveform_function("mock_custom_function", t, pulse_info_mock)

    # Assert
    mock.assert_called_with(t=t, duration=1.4e-9, t0=0)


def test_shift_waveform_misaligned():
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


def test_shift_waveform_aligned():
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
def test_get_waveform_size(size: int, granularity: int, expected: int):
    # Act
    new_size = get_waveform_size(np.ones(size), granularity)

    # Assert
    assert expected == new_size


def test_apply_mixer_skewness_corrections():
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
    assert isinstance(waveform, (np.ndarray, np.generic))
    assert pytest.approx(amp_ratio_after, amplitude_ratio)
    normalized_real = waveform.real / np.max(np.abs(waveform.real))
    normalized_imag = waveform.imag / np.max(np.abs(waveform.imag))
    assert np.allclose(normalized_real, normalized_imag)


def test_modulate_waveform():
    # Arrange
    frequency = 10e6
    t0 = 50e-9
    t = np.linspace(0, 1e-6, 1000)
    envelope = np.ones(len(t))

    expected_real = np.cos(2 * np.pi * frequency * (t + t0))
    expected_imag = np.sin(2 * np.pi * frequency * (t + t0))

    # Act
    waveform = modulate_waveform(t, envelope, frequency, t0)

    # Assert
    assert np.allclose(waveform.real, expected_real)
    assert np.allclose(waveform.imag, expected_imag)
