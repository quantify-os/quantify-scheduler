# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Contains function to generate most basic waveforms.

These functions are intended to be used to generate waveforms defined in the
:mod:`~quantify_scheduler.operations.pulse_library`.
Examples of waveforms that are too advanced are flux pulses that require knowledge of
the flux sensitivity and interaction strengths and qubit frequencies.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import interpolate, signal

if TYPE_CHECKING:
    from numpy.typing import NDArray


def square(t: np.ndarray | list[float], amp: float | complex) -> np.ndarray:
    """Generate a square pulse."""
    return amp * np.ones(len(t))


def square_imaginary(t: np.ndarray | list[float], amp: float | complex) -> np.ndarray:
    """Generate a square pulse with imaginary amplitude."""
    return square(t, 1j * amp)


def ramp(t: np.ndarray, amp: float, offset: float = 0, duration: float | None = None) -> np.ndarray:
    """Generate a ramp pulse."""
    if duration is None:
        warnings.warn(
            "Defining the duration of a ramp pulse based on the given timings "
            "is deprecated. Please provide the desired pulse duration to the "
            "`duration` keyword. E.g. `ramp(..., duration=100e-9)`. "
            "This will be enforced in quantify-scheduler>=0.24",
            FutureWarning,
        )
        return np.linspace(offset, amp + offset, len(t), endpoint=False)
    return amp / duration * t + offset


def staircase(
    t: np.ndarray[float, np.dtype],
    start_amp: float | complex,
    final_amp: float | complex,
    num_steps: int,
    duration: float | None = None,
) -> np.ndarray[float, np.dtype] | np.ndarray[complex, np.dtype]:
    """
    Ramps from zero to a finite value in discrete steps.

    Parameters
    ----------
    t
        Times at which to evaluate the function.
        Times < 0 will output `start_amp`.
        Times >= `duration` will output `final_amp`.
    start_amp
        Starting amplitude.
    final_amp
        Final amplitude to reach on the last step.
    num_steps
        Number of steps to reach final value.
    duration
        Duration of the pulse in seconds.

    Returns
    -------
    :
        The real valued waveform.

    """
    if duration is None:
        warnings.warn(
            "Defining the plateau length of a staircase pulse based "
            "on the given timings is deprecated. "
            "Please provide the desired pulse duration to the "
            "`duration` keyword. E.g. `staircase(..., duration=100e-9)`. "
            "This will be enforced in quantify-scheduler>=0.24",
            FutureWarning,
        )
        duration = float(t[-1] - t[0])

    amp_step = (final_amp - start_amp) / (num_steps - 1)
    waveform = np.floor(t * (num_steps / duration)) * amp_step + start_amp
    # Replace out of range values with start_amp or final_amp.
    # In most cases that would only happen when `duration in t`
    return np.minimum(np.maximum(waveform, start_amp), final_amp)


def soft_square(t: NDArray | list[float], amp: float | complex) -> NDArray:
    """
    A softened square pulse.

    Parameters
    ----------
    t :
        Times at which to evaluate the function.

    amp :
        Amplitude of the pulse.

    """
    data = square(t, amp)
    if len(t) > 1:
        window = signal.windows.hann(int(len(t) / 2))
        data = signal.convolve(data, window, mode="same") / sum(window)
    return data


def chirp(
    t: np.ndarray,
    amp: float,
    start_freq: float,
    end_freq: float,
    duration: float | None = None,
) -> np.ndarray:
    r"""
    Produces a linear chirp signal.

    The frequency is determined according to the
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
    duration
        Duration of the pulse in seconds.

    Returns
    -------
    :
        The complex waveform.

    """
    if duration is None:
        warnings.warn(
            "Defining the duration of a chirp pulse based on the given timings is deprecated."
            "Please provide the duration."
            "Previously the duration was calculated using the first and last point in t. "
            "Not only should the shape of t, or its values, not be assumed, "
            "in most default cases the last point of t was "
            "`duration - 1/sampling_rate instead` instead of `duration`.  "
            "Please provide the desired pulse duration to the "
            "`duration` keyword. E.g. `chirp(..., duration=100e-9)`. "
            "This will be enforced in quantify-scheduler>=0.24",
            FutureWarning,
        )
        # This was technically incorrect (hence the deprecation)
        # as in most calls the actual duration was this + 1/sampling_rate
        duration = float(t[-1] - t[0])
    chirp_rate = (end_freq - start_freq) / duration
    return amp * np.exp(1.0j * 2 * np.pi * (chirp_rate * t / 2 + start_freq) * t)


def drag(
    t: np.ndarray,
    G_amp: float,
    D_amp: float,
    duration: float,
    nr_sigma: float,
    sigma: float | int | None = None,
    phase: float = 0,
    subtract_offset: Literal["average", "first", "last", "none"] = "average",
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
    sigma
        Width of the Gaussian envelope. If None, it is calculated with nr_sigma, which is set to 4.
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

    if sigma is not None and nr_sigma is not None:
        raise ValueError("Both sigma and nr_sigma are specified. Please specify only one.")

    if sigma is None:
        sigma = duration / (2 * nr_sigma)

    gauss_env = G_amp * np.exp(-(0.5 * ((t - mu) ** 2) / sigma**2))
    deriv_gauss_env = -D_amp * (t - mu) / sigma * gauss_env

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
            f'Unknown value "{subtract_offset}" for keyword argument subtract_offset".'
        )

    # generate pulses
    drag_wave = gauss_env + 1j * deriv_gauss_env

    # Apply phase rotation
    rot_drag_wave = rotate_wave(drag_wave, phase=phase)

    return rot_drag_wave


def sudden_net_zero(
    t: np.ndarray,
    amp_A: float,  # noqa N803
    amp_B: float,  # noqa N803
    net_zero_A_scale: float,  # noqa N803: upper case in variable
    t_pulse: float,
    t_phi: float,
    t_integral_correction: float,
) -> NDArray:
    """
    Generates the sudden net zero waveform from :cite:t:`negirneac_high_fidelity_2021`.

    The waveform consists of a square pulse with a duration of half
    ``t_pulse`` and an amplitude of ``amp_A``, followed by an idling period (0
    V) with duration ``t_phi``, followed again by a square pulse with amplitude
    ``-amp_A * net_zero_A_scale`` and a duration of half ``t_pulse``, followed
    by a integral correction period with duration ``t_integral_correction``.

    The last sample of the first pulse has amplitude ``amp_A * amp_B``. The
    first sample of the second pulse has amplitude ``-amp_A * net_zero_A_scale *
    amp_B``.

    The amplitude of the integral correction period is such that ``sum(waveform)
    == 0``.

    If the total duration of the pulse parts is less than the duration set by
    the ``t`` array, the remaining samples will be set to 0 V.

    The various pulse part durations are rounded **down** (floor) to the sample
    rate of the ``t`` array. Since ``t_pulse`` is the total duration of the two
    square pulses, half this duration is rounded to the sample rate. For
    example:

    .. jupyter-execute::

        import numpy as np
        from quantify_scheduler.waveforms import sudden_net_zero

        t = np.linspace(0, 9e-9, 10)  # 1 GSa/s
        amp_A = 1.0
        amp_B = 0.5
        net_zero_A_scale = 0.8
        t_pulse = 5.0e-9  # will be rounded to 2 pulses of 2 ns
        t_phi = 2.6e-9  # rounded to 2 ns
        t_integral_correction = 4.4e-9  # rounded to 4 ns

        sudden_net_zero(
            t, amp_A, amp_B, net_zero_A_scale, t_pulse, t_phi, t_integral_correction
        )

    Parameters
    ----------
    t
        A uniformly sampled array of times at which to evaluate the function.
    amp_A
        Amplitude of the main square pulse
    amp_B
        Scaling correction for the final sample of the first square and first sample
        of the second square pulse.
    net_zero_A_scale
        Amplitude scaling correction factor of the negative arm of the net-zero pulse.
    t_pulse
        The total duration of the two half square pulses. The duration of each
        half is rounded to the sample rate of the ``t`` array.
    t_phi
        The idling duration between the two half pulses. The duration is rounded
        to the sample rate of the ``t`` array.
    t_integral_correction
        The duration in which any non-zero pulse amplitude needs to be
        corrected. The duration is rounded to the sample rate of the ``t`` array.

    """
    sampling_rate = t[1] - t[0]
    single_arm_samples = int(t_pulse / 2 / sampling_rate)
    mid_samples = int(t_phi / sampling_rate)
    num_corr_samples = int(t_integral_correction / sampling_rate)

    if 2 * single_arm_samples + mid_samples + num_corr_samples > len(t):
        raise ValueError(
            "Specified pulse part durations add up to longer than the given time array."
        )

    waveform = np.zeros(len(t))
    waveform[:single_arm_samples] = amp_A
    waveform[single_arm_samples - 1] = amp_A * amp_B
    waveform[single_arm_samples + mid_samples : 2 * single_arm_samples + mid_samples] = (
        -amp_A * net_zero_A_scale
    )
    waveform[single_arm_samples + mid_samples] = -amp_A * net_zero_A_scale * amp_B
    integral_value = -sum(waveform) / num_corr_samples
    waveform[
        2 * single_arm_samples
        + mid_samples : 2 * single_arm_samples
        + mid_samples
        + num_corr_samples
    ] = integral_value
    return waveform


def interpolated_complex_waveform(
    t: np.ndarray,
    samples: np.ndarray,
    t_samples: np.ndarray,
    interpolation: str = "linear",
    **kwargs,
) -> np.ndarray:
    """
    Wrapper function around :class:`scipy.interpolate.interp1d`, which takes the
    array of (complex) samples, interpolates the real and imaginary parts
    separately and returns the interpolated values at the specified times.

    Parameters
    ----------
    t
        Times at which to evaluated the to be returned waveform.
    samples
        An array of (possibly complex) values specifying the shape of the waveform.
    t_samples
        An array of values specifying the corresponding times at which the ``samples``
        are evaluated.
    interpolation:
        The interpolation method to use, by default "linear".
    kwargs
        Optional keyword arguments to pass to ``scipy.interpolate.interp1d``.

    Returns
    -------
    :
        An array containing the interpolated values.

    """
    if ("bounds_error" in kwargs) or ("fill_value" in kwargs):
        raise ValueError(
            "The 'bounds_error' and `fill_value` arguments are fixed to 'False' and "
            "'\"extrapolate\", respectively, and cannot be changed'."
        )

    samples = np.array(samples)

    # Allow extrapolation only when t starts less than one t_sample before the start
    # of t_samples, and when t ends less than one t_sample after the end of t_samples.
    # This is necessary to fill in the last sample(s) when upsampling.
    delta_t_samples = t_samples[1] - t_samples[0]
    if t[0] < t_samples[0] - delta_t_samples or t[-1] > t_samples[-1] + delta_t_samples:
        raise ValueError(
            "Interpolation out of bounds: 't' should start at or after the first 't_sample'"
            " and end at or before the last 't_sample'"
        )

    real_interpolator = interpolate.interp1d(
        t_samples,
        samples.real,
        kind=interpolation,
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
        **kwargs,
    )

    if np.all(np.isreal(samples)):
        # If samples is purely real, early return with purely real result, since the
        # calling code might not expect complex values
        return real_interpolator(t)

    imag_interpolator = interpolate.interp1d(
        t_samples,
        samples.imag,
        kind=interpolation,
        bounds_error=False,
        fill_value="extrapolate",  # type: ignore
        **kwargs,
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
    center: float | None = None,
    duration_over_char_time: float = 6.0,
) -> np.ndarray:
    """
    Generates a skewed hermite pulse for single qubit rotations in NV centers.

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
    # Hermite factors are taken from paper cited in docstring.
    pi_hermite_factor = 0.956
    pi2_hermite_factor = 0.667

    # Determine parameters based on switches:
    #   - characteristic time of hermite polynomial
    #   - hermite factor
    #   - center position of pulse
    t_hermite = duration / duration_over_char_time
    hermite_factor = pi2_hermite_factor if pi2_pulse else pi_hermite_factor
    if center is None:
        center = duration / 2.0

    # normalize time array for easier evaluation
    center_total = center + t[0]
    normalized_time = (t - center_total) / t_hermite

    # Hermite pulse with zero skewness
    h_t = (1 - hermite_factor * normalized_time**2) * np.exp(-(normalized_time**2))

    # I and Q components
    I = amplitude * h_t  # noqa: E741 ambiguous rules about ambiguousness
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
