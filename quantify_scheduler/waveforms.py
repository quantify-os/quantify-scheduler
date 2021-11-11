# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Contains function to generate most basic waveforms.

These functions are intended to be used to generate waveforms defined in the
:mod:`~.pulse_library`.
Examples of waveforms that are too advanced are flux pulses that require knowledge of
the flux sensitivity and interaction strengths and qubit frequencies.
"""
from typing import List, Union

import numpy as np
from scipy import signal


def square(t: Union[np.ndarray, List[float]], amp: Union[float, complex]) -> np.ndarray:
    return amp * np.ones(len(t))


def square_imaginary(
    t: Union[np.ndarray, List[float]], amp: Union[float, complex]
) -> np.ndarray:
    return square(t, 1j * amp)


def ramp(t, amp, offset=0) -> np.ndarray:
    return np.linspace(offset, amp + offset, len(t), endpoint=False)


def staircase(
    t: Union[np.ndarray, List[float]],
    start_amp: Union[float, complex],
    final_amp: Union[float, complex],
    num_steps: int,
) -> np.ndarray:
    """
    Ramps from zero to a finite value in discrete steps.

    Parameters
    ----------
    t
        Times at which to evaluate the function.
    start_amp
        Starting amplitude.
    final_amp
        Final amplitude to reach on the last step.
    num_steps
        Number of steps to reach final value.

    Returns
    -------
    :
        The real valued waveform.
    """
    amp_step = (final_amp - start_amp) / (num_steps - 1)
    t_arr_plateau_len = int(len(t) // num_steps)

    waveform = np.array([])
    for i in range(num_steps):
        t_current_plateau = t[i * t_arr_plateau_len : (i + 1) * t_arr_plateau_len]
        waveform = np.append(
            waveform,
            square(
                t_current_plateau,
                i * amp_step,
            )
            + start_amp,
        )
    t_rem = t[num_steps * t_arr_plateau_len :]
    waveform = np.append(waveform, square(t_rem, final_amp))
    return waveform


def soft_square(t, amp):
    """A softened square pulse.

    Parameters
    ----------
    t

    amp

    """
    data = square(t, amp)
    if len(t) > 1:
        window = signal.windows.hann(int(len(t) / 2))
        data = signal.convolve(data, window, mode="same") / sum(window)
    return data


def chirp(t: np.ndarray, amp: float, start_freq: float, end_freq: float) -> np.ndarray:
    r"""
    Produces a linear chirp signal. The frequency is determined according to the
    relation:

    .. math:

        f(t) = ct + f_0,
        c = \frac{f_1 - f_0}{T}

    The waveform is produced simply by multiplying with a complex exponential.

    Parameters
    ----------
    t
        Times at which to evaluate the function.
    amp
        Amplitude of the envelope.
    start_freq
        Start frequency of the Chirp.
    end_freq
        End frequency of the Chirp.

    Returns
    -------
    :
        The complex waveform.
    """
    chirp_rate = (end_freq - start_freq) / (t[-1] - t[0])
    return amp * np.exp(1.0j * 2 * np.pi * (chirp_rate * t / 2 + start_freq) * t)


# pylint: disable=too-many-arguments
def drag(
    t: np.ndarray,
    G_amp: float,
    D_amp: float,
    duration: float,
    nr_sigma: int = 3,
    phase: float = 0,
    subtract_offset: str = "average",
) -> np.ndarray:
    r"""
    Generates a DRAG pulse consisting of a Gaussian :math:`G` as the I- and a
    Derivative :math:`D` as the Q-component (:cite:t:`motzoi_simple_2009` and
    :cite:t:`gambetta_analytic_2011`).

    All inputs are in s and Hz.
    phases are in degree.

    :math:`G(t) = G_{amp} e^{-(t-\mu)^2/(2\sigma^2)}`.

    :math:`D(t) = -D_{amp} \frac{(t-\mu)}{\sigma} G(t)`.

    .. note:

        One would expect a factor :math:`1/\sigma^2` in the prefactor of
        :math:`1/\sigma^2`, we absorb this in the scaling factor :math:`D_{amp}` to
        ensure the derivative component is scale invariant with the duration of
        the pulse.


    Parameters
    ----------
    t
        Times at which to evaluate the function.
    G_amp
        Amplitude of the Gaussian envelope.
    D_amp
        Amplitude of the derivative component, the DRAG-pulse parameter.
    duration
        Duration of the pulse in seconds.
    nr_sigma
        After how many sigma the Gaussian is cut off.
    phase
        Phase of the pulse in degrees.
    subtract_offset
        Instruction on how to subtract the offset in order to avoid jumps in the
        waveform due to the cut-off.

        - 'average': subtract the average of the first and last point.
        - 'first': subtract the value of the waveform at the first sample.
        - 'last': subtract the value of the waveform at the last sample.
        - 'none', None: don't subtract any offset.

    Returns
    -------
    :
        complex waveform

    """
    mu = t[0] + duration / 2

    sigma = duration / (2 * nr_sigma)

    gauss_env = G_amp * np.exp(-(0.5 * ((t - mu) ** 2) / sigma ** 2))
    deriv_gauss_env = -D_amp * (t - mu) / (sigma ** 1) * gauss_env

    # Subtract offsets
    if subtract_offset.lower() == "none" or subtract_offset is None:
        # Do not subtract offset
        pass
    elif subtract_offset.lower() == "average":
        gauss_env -= (gauss_env[0] + gauss_env[-1]) / 2.0
        deriv_gauss_env -= (deriv_gauss_env[0] + deriv_gauss_env[-1]) / 2.0
    elif subtract_offset.lower() == "first":
        gauss_env -= gauss_env[0]
        deriv_gauss_env -= deriv_gauss_env[0]
    elif subtract_offset.lower() == "last":
        gauss_env -= gauss_env[-1]
        deriv_gauss_env -= deriv_gauss_env[-1]
    else:
        raise ValueError(
            'Unknown value "{}" for keyword argument subtract_offset".'.format(
                subtract_offset
            )
        )

    # generate pulses
    drag_wave = gauss_env + 1j * deriv_gauss_env

    # Apply phase rotation
    rot_drag_wave = rotate_wave(drag_wave, phase=phase)

    return rot_drag_wave


# ----------------------------------
# Utility functions
# ----------------------------------


def rotate_wave(wave: np.ndarray, phase: float) -> np.ndarray:
    """
    Rotate a wave in the complex plane.

    Parameters
    ----------
    wave
        Complex waveform, real component corresponds to I, imaginary component to Q.
    phase
        Rotation angle in degrees.

    Returns
    -------
    :
        Rotated complex waveform.
    """
    angle = np.deg2rad(phase)

    rot = (np.cos(angle) + 1.0j * np.sin(angle)) * wave
    return rot


def modulate_wave(t: np.ndarray, wave: np.ndarray, freq_mod: float) -> np.ndarray:
    """
    Apply single sideband (SSB) modulation to a waveform.

    The frequency convention we adhere to is:

        freq_base + freq_mod = freq_signal

    Parameters
    ----------
    t :
        Times at which to determine the modulation.
    wave :
        Complex waveform, real component corresponds to I, imaginary component to Q.
    freq_mod :
        Modulation frequency in Hz.


    Returns
    -------
    :
        modulated waveform.


    .. note::

        Pulse modulation is generally not included when specifying waveform envelopes
        as there are many hardware backends include this capability.
    """
    cos_mod = np.cos(2 * np.pi * freq_mod * t)
    sin_mod = np.sin(2 * np.pi * freq_mod * t)
    mod_I = cos_mod * wave.real + sin_mod * wave.imag
    mod_Q = -sin_mod * wave.real + cos_mod * wave.imag

    return mod_I + 1j * mod_Q
