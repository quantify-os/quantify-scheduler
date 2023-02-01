# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

import numpy as np
import numpy.testing as npt
import pytest

from quantify_scheduler.waveforms import (
    drag,
    staircase,
    interpolated_complex_waveform,
    modulate_wave,
    rotate_wave,
    square,
    staircase,
    sudden_net_zero,
    skewed_hermite,
)


def test_square_wave() -> None:
    amped_sq = square(np.arange(50), 2.44)
    npt.assert_array_equal(amped_sq, np.linspace(2.44, 2.44, 50))

    amped_sq_iq = square(np.arange(20), 6.88)
    npt.assert_array_equal(amped_sq_iq.real, np.linspace(6.88, 6.88, 20))
    npt.assert_array_equal(amped_sq_iq.imag, np.linspace(0, 0, 20))


def test_staircase() -> None:
    t = np.linspace(0, 1e-6, 20)
    sig = staircase(t, -1, 2, 4)
    answer = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
    npt.assert_array_equal(sig, answer)


def test_drag_ns() -> None:
    duration = 20e-9
    nr_sigma = 3
    G_amp = 0.5
    D_amp = 1

    times = np.arange(0, duration, 1e-9)  # sampling rate set to 1 GSPs
    mu = times[0] + duration / 2
    sigma = duration / (2 * nr_sigma)
    gauss_env = G_amp * np.exp(-(0.5 * ((times - mu) ** 2) / sigma**2))
    deriv_gauss_env = D_amp * -1 * (times - mu) / (sigma**1) * gauss_env
    exp_waveform = gauss_env + 1j * deriv_gauss_env

    # quantify
    waveform = drag(
        times,
        G_amp=G_amp,
        D_amp=D_amp,
        duration=duration,
        nr_sigma=nr_sigma,
        subtract_offset="none",
    )

    np.testing.assert_array_almost_equal(waveform, exp_waveform, decimal=3)
    assert np.max(waveform) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        drag(times, 0.5, D_amp, duration, subtract_offset="bad!")

    waveform = drag(
        times,
        G_amp=G_amp,
        D_amp=D_amp,
        duration=duration,
        nr_sigma=nr_sigma,
        subtract_offset="average",
    )
    exp_waveform.real -= np.mean([exp_waveform.real[0], exp_waveform.real[-1]])
    exp_waveform.imag -= np.mean([exp_waveform.imag[0], exp_waveform.imag[-1]])
    np.testing.assert_array_almost_equal(waveform, exp_waveform, decimal=3)


def test_sudden_net_zero() -> None:
    times = np.arange(0, 40e-9, 1e-9)
    amp_A = 0.4
    amp_B = 0.2
    net_zero_A_scale = 0.95

    waveform = sudden_net_zero(
        times,
        amp_A=amp_A,
        amp_B=amp_B,
        net_zero_A_scale=net_zero_A_scale,
        t_pulse=20e-9,
        t_phi=2e-9,
        t_integral_correction=10e-9,
    )

    assert np.sum(waveform) == pytest.approx(0, abs=1e-12)
    assert np.max(waveform) == amp_A
    assert np.min(waveform) == -1 * amp_A * net_zero_A_scale


@pytest.mark.parametrize(
    "test_wf, test_time, answer",
    [
        (
            square(np.linspace(0, 50e-6, 1000), 2.44),
            np.linspace(0, 50e-6, 1000),
            np.array([2.44] * 2000),
        ),
        (
            square(np.linspace(0, 50e-6, 1000), 2.44)
            + 1.0j * square(np.linspace(0, 50e-6, 1000), 1),
            np.linspace(0, 50e-6, 1000),
            np.array([2.44 + 1.0j] * 2000),
        ),
        (
            square(np.linspace(0, 50e-6, 1000), -2.1j),
            np.linspace(0, 50e-6, 1000),
            np.array([-2.1j] * 2000),
        ),
    ],
)
def test_interpolated_complex_waveform(test_wf, test_time, answer):
    t_answer = np.linspace(0, 50e-6, 2000)
    result = interpolated_complex_waveform(
        t=t_answer, samples=test_wf, t_samples=test_time
    )
    npt.assert_array_equal(answer, result)


def test_rotate_wave() -> None:
    I = np.ones(10)  # noqa # Q component is zero
    Q = np.zeros(10)  # noqa # not used as input, only used for testing

    rot_wf = rotate_wave(I, 0)

    npt.assert_array_almost_equal(I, rot_wf.real)
    npt.assert_array_almost_equal(I.imag, rot_wf.imag)

    rot_wf = rotate_wave(I, 90)

    npt.assert_array_almost_equal(I, rot_wf.imag)
    npt.assert_array_almost_equal(Q, -rot_wf.real)

    rot_wf = rotate_wave(I, 180)

    npt.assert_array_almost_equal(I, -rot_wf.real)
    npt.assert_array_almost_equal(Q, -rot_wf.imag)

    rot_wf = rotate_wave(I, 360)

    npt.assert_array_almost_equal(I, rot_wf.real)
    npt.assert_array_almost_equal(Q, rot_wf.imag)


def test_modulate() -> None:
    fs = 100
    f = 4
    t = np.arange(fs)
    I = np.sin(2 * np.pi * f * (t / fs))  # noqa
    Q = np.sin(2 * np.pi * f * (t / fs) + (np.pi / 2))  # noqa
    wf = I + 1j * Q

    mod_wf = modulate_wave(np.linspace(0, 1, fs), wf, 2)
    npt.assert_array_almost_equal(
        mod_wf.real, np.sin(2 * np.pi * (f + 2) * (t / fs)), decimal=1
    )

    mod_wf = modulate_wave(np.linspace(0, 1, fs), wf, -2)
    npt.assert_array_almost_equal(
        mod_wf.imag, np.sin(2 * np.pi * (f - 2) * (t / fs) + (np.pi / 2)), decimal=1
    )


@pytest.fixture
def hermite_kwargs():
    """Define 'random' keyword arguments for skewed hermite pulses"""
    t = np.linspace(0, 1e-5, 20)
    kwargs = {
        "t": t,
        "duration": 1.03e-5,
        "amplitude": 0.983,
        "phase": 0.3,
        "skewness": -0.7,
        "pi2_pulse": True,
        "center": None,
        "duration_over_char_time": 6.0,
    }
    return kwargs


def test_hermite_real(hermite_kwargs):
    """Hermite pulse is real if skewness and phase are 0."""
    hermite_kwargs["skewness"] = 0.0
    hermite_kwargs["phase"] = 0.0
    assert (skewed_hermite(**hermite_kwargs).imag == 0).all()


def test_hermite_amp_linear_scaling(hermite_kwargs):
    """Hermite pulse scales linearly with the amplitude."""
    del hermite_kwargs["amplitude"]
    approx = np.frompyfunc(pytest.approx, 1, 1)
    assert (
        2 * skewed_hermite(amplitude=0.032, **hermite_kwargs)
        == approx(skewed_hermite(amplitude=0.064, **hermite_kwargs))
    ).all()


def test_hermite_duration_scaling(hermite_kwargs):
    """When time and duration are scaled by a factor, the result stays unchanged."""
    dur = hermite_kwargs["duration"]
    del hermite_kwargs["duration"]
    t = hermite_kwargs["t"]
    del hermite_kwargs["t"]

    # Note that we also have to scale the skewness. This is unintuitive, but can be
    # understood from eqs. (A.12) and (A.36) in H.K.C.Beukers MSc Thesis (2019), where
    # the skewness factor b has the unit time.
    skewness = hermite_kwargs["skewness"]
    del hermite_kwargs["skewness"]

    scaling = 2.7

    assert skewed_hermite(
        t=t, duration=dur, skewness=skewness, **hermite_kwargs
    ) == pytest.approx(
        skewed_hermite(
            t=scaling * t,
            duration=scaling * dur,
            skewness=scaling * skewness,
            **hermite_kwargs,
        )
    )
