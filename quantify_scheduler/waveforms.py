# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Contains function to generate most basic waveforms.

These functions are intended to be used to generate waveforms defined in the
:mod:`~.pulse_library`.
Examples of waveforms that are too advanced are flux pulses that require knowledge of
the flux sensitivity and interaction strengths and qubit frequencies.
"""
from typing import List, Optional, Union

import numpy as np
from scipy import signal, interpolate


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
        :math:`D(t)`, we absorb this in the scaling factor :math:`D_{amp}` to
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

    gauss_env = G_amp * np.exp(-(0.5 * ((t - mu) ** 2) / sigma**2))
    deriv_gauss_env = -D_amp * (t - mu) / (sigma**1) * gauss_env

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


def sudden_net_zero(
    t: np.ndarray,
    amp_A: float,
    amp_B: float,
    net_zero_A_scale: float,
    t_pulse: float,
    t_phi: float,
    t_integral_correction: float,
):
    """
    Generates the sudden net zero waveform from :cite:t:`negirneac_high_fidelity_2021`.

    Parameters
    ----------
    t
        Times at which to evaluate the function.
    amp_A
        amplitude of the main square pulse
    amp_B
        scaling correction for the final sample of the first square and first sample
        of the second square pulse.
    net_zero_A_scale
        amplitude scaling correction factor of the negative arm of the net-zero pulse.
    t_pulse
        the total duration of the two half square pulses
    t_phi
        the idling duration between the two half pulses
    t_integral_correction
        the duration in which any non-zero pulse amplitude needs to be corrected.
    """

    # this transform is because all step functions are defined with respect to the
    # start of the waveform.
    t = t - min(t)

    def _square(t, start: float, stop: float, start_amp=1, stop_amp=0):
        """square pulses with a start and stop using a heaviside function."""
        return np.heaviside(
            np.around(t - start, decimals=12), start_amp
        ) - np.heaviside(np.around(t - stop, decimals=12), 1 - stop_amp)

    # the waveform itself
    first_arm = amp_A * _square(t, start=0, stop=t_pulse / 2, stop_amp=amp_B)
    second_arm = (
        -1
        * amp_A
        * net_zero_A_scale
        * _square(
            t,
            start=t_pulse / 2 + t_phi,
            stop=t_pulse + t_phi,
            start_amp=amp_B,
            stop_amp=0,
        )
    )
    waveform_amps = first_arm + second_arm

    # adding a correction to ensure the integral evaluates to 0
    sampling_rate = t[1] - t[0]
    num_corr_samples = t_integral_correction / sampling_rate
    corr_amp = -np.sum(waveform_amps) / num_corr_samples

    corr_waveform_amps = waveform_amps + corr_amp * _square(
        t,
        start=t_pulse + t_phi,
        stop=t_pulse + t_phi + t_integral_correction,
        start_amp=0,
        stop_amp=1,
    )

    return corr_waveform_amps


def interpolated_complex_waveform(
    t: np.ndarray,
    samples: np.ndarray,
    t_samples: np.ndarray,
    interpolation: str = "linear",
    **kwargs,
) -> np.ndarray:
    """
    Wrapper function around `scipy.interpolate.interp1d`, which takes the array of
    (complex) samples, interpolates the real and imaginary parts separately and returns
    the interpolated values at the specified times.

    Parameters
    ----------
    t
        Times at which to evaluated the to be returned waveform.
    samples
        An array of (possibly complex) values specifying the shape of the waveform.
    t_samples
        An array of values specifying the corresponding times at which the `samples`
        are evaluated.
    kwargs
        Optional keyword arguments to pass to `scipy.interpolate.interp1d`.

    Returns
    -------
    :
        An array containing the interpolated values.
    """
    if isinstance(samples, list):
        samples = np.array(samples)
    real_interpolator = interpolate.interp1d(
        t_samples, samples.real, kind=interpolation, **kwargs
    )
    imag_interpolator = interpolate.interp1d(
        t_samples, samples.imag, kind=interpolation, **kwargs
    )
    return real_interpolator(t) + 1.0j * imag_interpolator(t)


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


def skewed_hermite(
    t: np.ndarray,
    duration: float,
    amplitude: float,
    skewness: float,
    phase: float,
    pi2_pulse: bool = False,
    center: Optional[float] = None,
    duration_over_char_time: float = 6.0,
) -> np.ndarray:
    """Generates a skewed hermite pulse for single qubit rotations in NV centers.

    A Hermite pulse is a Gaussian multiplied by a second degree Hermite polynomial.
    See :cite:t:`Beukers_MSc_2019`, Appendix A.2.

    The skew parameter is a first order amplitude correction to the hermite pulse. It
    increases the fidelity of the performed gates.
    See :cite:t:`Beukers_MSc_2019`, section 4.2. To get a "standard" hermite
    pulse, use ``skewness=0``.

    The hermite factors are taken from equation 44 and 45 of
    :cite:t:`Warren_NMR_pulse_shapes_1984`.

    Parameters
    ----------
    t
        Times at which to evaluate the function.
    duration
        Duration of the pulse in seconds.
    amplitude
        Amplitude of the pulse.
    skewness
        Skewness in the frequency space
    phase
        Phase of the pulse in degrees.
    pi2_pulse
        if True, the pulse will be pi/2 otherwise pi pulse
    center
        Optional: time after which the pulse center occurs. If ``None``, it is
        automatically set to duration/2.
    duration_over_char_time
        Ratio of the pulse duration and the characteristic time of the hermite
        polynomial. Increasing this number will compress the pulse. By default, 6.

    Returns
    -------
    :
        complex skewed waveform

    """
    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name

    # Hermite factors are taken from paper cited in docstring.
    PI_HERMITE_FACTOR = 0.956
    PI2_HERMITE_FACTOR = 0.667

    # Determine parameters based on switches:
    #   - characteristic time of hermite polynomial
    #   - hermite factor
    #   - center position of pulse
    t_hermite = duration / duration_over_char_time
    hermite_factor = PI2_HERMITE_FACTOR if pi2_pulse else PI_HERMITE_FACTOR
    if center is None:
        center = duration / 2.0

    # normalize time array for easier evaluation
    center_total = center + t[0]
    normalized_time = (t - center_total) / t_hermite

    # Hermite pulse with zero skewness
    h_t = (1 - hermite_factor * normalized_time**2) * np.exp(-(normalized_time**2))

    # I and Q components
    I = amplitude * h_t
    Q = (
        amplitude
        * (skewness / np.pi)
        * (normalized_time / t_hermite)
        * (hermite_factor + 1 - hermite_factor * normalized_time**2)
        * np.exp(-(normalized_time**2))
    )
    hermite = I + 1j * Q

    # Rotate pulse to get correct phase
    rotated_hermite = rotate_wave(hermite, phase)

    return rotated_hermite


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
