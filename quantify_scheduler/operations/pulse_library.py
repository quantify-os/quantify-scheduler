# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard pulse-level operations for use with the quantify_scheduler."""
# pylint: disable= too-many-arguments, too-many-ancestors
from __future__ import annotations

from typing import List, Optional, Dict, Any, Union
import warnings

import numpy as np
from numpy.typing import NDArray
from qcodes import validators

from quantify_scheduler import Operation
from quantify_scheduler.backends.qblox import constants as qblox_constants
from quantify_scheduler.helpers.waveforms import area_pulses
from quantify_scheduler.resources import BasebandClockResource


class ShiftClockPhase(Operation):
    """An operation that shifts the phase of a clock by a specified amount."""

    def __init__(
        self, phase_shift: float, clock: str, t0: float = 0, data: Optional[dict] = None
    ):
        """
        Create a new instance of ShiftClockPhase.

        Parameters
        ----------
        phase_shift
            The phase shift in degrees.
        clock
            The clock of which to shift the phase.
        t0
            Time in seconds when to execute the command relative
            to the start time of the Operation in the Schedule.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            super().__init__(name="ShiftClockPhase")
            self.data.update(
                {
                    "name": "ShiftClockPhase",
                    "pulse_info": [
                        {
                            "wf_func": None,
                            "t0": t0,
                            "phase_shift": phase_shift,
                            "clock": clock,
                            "port": None,
                            "duration": 0,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class ResetClockPhase(Operation):
    """An operation that resets the phase of a clock."""

    def __init__(self, clock: str, t0: float = 0, data: Optional[dict] = None):
        """
        Create a new instance of ResetClockPhase.

        Parameters
        ----------
        clock
            The clock of which to reset the phase.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            super().__init__(name="ResetClockPhase")
            self.data.update(
                {
                    "name": "ResetClockPhase",
                    "pulse_info": [
                        {
                            "wf_func": None,
                            "clock": clock,
                            "t0": t0,
                            "duration": 0,
                            "port": None,
                            "reset_clock_phase": True,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class SetClockFrequency(Operation):
    """
    Operation that sets updates the frequency of a clock. This is a low-level operation
    and therefore depends on the backend.

    Currently only implemented for Qblox backend, refer to
    :class:`~quantify_scheduler.backends.qblox.operation_handling.virtual.NcoSetClockFrequencyStrategy`
    for more details.
    """

    def __init__(
        self,
        clock: str,
        clock_freq_new: float,
        t0: float = 0,
        duration: float = qblox_constants.NCO_SET_FREQ_WAIT * 1e-9,
    ):
        """

        Parameters
        ----------
        clock
            The clock for which a new frequency is to be set.
        clock_freq_new
            The new frequency in Hz.
        t0
            Time in seconds when to execute the command relative to the start time of
            the Operation in the Schedule.
        duration
            The duration of the operation in seconds.
        """
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "t0": t0,
                "clock": clock,
                "clock_freq_new": clock_freq_new,
                "clock_freq_old": None,
                "interm_freq_old": None,
                "port": None,
                "duration": duration,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class IdlePulse(Operation):
    """
    The IdlePulse Operation is a placeholder for a specified duration of time.
    """

    def __init__(self, duration: float, data: Optional[dict] = None):
        """
        Create a new instance of IdlePulse.

        The IdlePulse Operation is a placeholder for a specified duration
        of time.

        Parameters
        ----------
        duration
            The duration of idle time in seconds.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            super().__init__(name="Idle")
            self.data.update(
                {
                    "name": "Idle",
                    "pulse_info": [
                        {
                            "wf_func": None,
                            "t0": 0,
                            "duration": duration,
                            "clock": BasebandClockResource.IDENTITY,
                            "port": None,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class RampPulse(Operation):
    """
    The RampPulse Operation is a real-valued pulse that ramps from the specified offset
    to the specified amplitude + offset during the duration of the pulse.
    """

    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        offset: float = 0,
        clock: str = BasebandClockResource.IDENTITY,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        r"""
        Create a new instance of RampPulse.

        The RampPulse Operation is a real-valued pulse that ramps from zero
        to the specified amplitude during the duration of the pulse.

        The pulse is given as a function of time :math:`t` and the parameters offset and
        amplitude by

        .. math::

            P(t) = \mathrm{offset} + t \times \mathrm{amp}.

        Parameters
        ----------
        amp
            Amplitude of the ramp envelope function.
        duration
            The pulse duration in seconds.
        offset
            Starting point of the ramp pulse
        port
            Port of the pulse.
        clock
            Clock used to modulate the pulse, by default a
            BasebandClock is used.
        t0
            Time in seconds when to start the pulses relative
            to the start time
            of the Operation in the Schedule.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            super().__init__(name="RampPulse")
            self.data.update(
                {
                    "name": "RampPulse",
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.ramp",
                            "amp": amp,
                            "duration": duration,
                            "offset": offset,
                            "t0": t0,
                            "clock": clock,
                            "port": port,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class StaircasePulse(Operation):  # pylint: disable=too-many-ancestors
    """
    A real valued staircase pulse, which reaches it's final amplitude in discrete
    steps. In between it will maintain a plateau.
    """

    def __init__(
        self,
        start_amp: float,
        final_amp: float,
        num_steps: int,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Constructor for a staircase.

        Parameters
        ----------
        start_amp
            Starting amplitude of the staircase envelope function.
        final_amp
            Final amplitude of the staircase envelope function.
        num_steps
            The number of plateaus.
        duration
            Duration of the pulse in seconds.
        port
            Port of the pulse.
        clock
            Clock used to modulate the pulse.
        t0
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            super().__init__(name="StaircasePulse")

            self.data.update(
                {
                    "name": "StaircasePulse",
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.staircase",
                            "start_amp": start_amp,
                            "final_amp": final_amp,
                            "num_steps": num_steps,
                            "duration": duration,
                            "t0": t0,
                            "clock": clock,
                            "port": port,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class SquarePulse(Operation):
    """
    The SquarePulse Operation is a real-valued pulse with the specified
    amplitude during the pulse.
    """

    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        phase: float = 0,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Create a new instance of SquarePulse.

        The SquarePulse Operation is a real-valued pulse with the specified
        amplitude during the pulse.

        Parameters
        ----------
        amp
            Amplitude of the envelope.
        duration
            The pulse duration in seconds.
        port
            Port of the pulse, must be capable of playing a complex waveform.
        clock
            Clock used to modulate the pulse.
        phase
            Phase of the pulse in degrees.
        t0
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to # pylint: disable=fixme
            # the waveform separate from the clock.
            raise NotImplementedError

        if data is None:
            super().__init__(name="ModSquarePulse")
            self.data.update(
                {
                    "name": "ModSquarePulse",
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.square",
                            "amp": amp,
                            "duration": duration,
                            "phase": phase,
                            "t0": t0,
                            "clock": clock,
                            "port": port,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class SuddenNetZeroPulse(Operation):
    """The sudden net-zero (SNZ) pulse from :cite:t:`negirneac_high_fidelity_2021`."""

    def __init__(
        self,
        amp_A: float,
        amp_B: float,
        net_zero_A_scale: float,
        t_pulse: float,
        t_phi: float,
        t_integral_correction: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        The sudden net-zero (SNZ) pulse from :cite:t:`negirneac_high_fidelity_2021`.

        The SuddenNetZeroPulse is a real-valued pulse that can be used to implement a
        conditional phase gate in transmon qubits.

        Parameters
        ----------
        amp_A
            amplitude of the main square pulse
        amp_B
            scaling correction for the final sample of the first square and first sample
            of the second square pulse.
        net_zero_A_scale
            amplitude scaling correction factor of the negative arm of the net-zero
            pulse.
        t_pulse
            the total duration of the two half square pulses
        t_phi
            the idling duration between the two half pulses
        t_integral_correction
            the duration in which any non-zero pulse amplitude needs to be corrected.
        port
            Port of the pulse, must be capable of playing a complex waveform.
        clock
            Clock used to modulate the pulse.
        t0
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        duration = t_pulse + t_phi + t_integral_correction

        if data is None:
            super().__init__(name="SuddenNetZeroPulse")
            self.data.update(
                {
                    "name": "SuddenNetZeroPulse",
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.sudden_net_zero",
                            "amp_A": amp_A,
                            "amp_B": amp_B,
                            "net_zero_A_scale": net_zero_A_scale,
                            "t_pulse": t_pulse,
                            "t_phi": t_phi,
                            "t_integral_correction": t_integral_correction,
                            "duration": duration,
                            "phase": 0,
                            "t0": t0,
                            "clock": clock,
                            "port": port,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


def decompose_long_square_pulse(
    duration: float, duration_max: float, single_duration: bool = False, **kwargs
) -> list:
    """
    Generates a list of square pulses equivalent to a  (very) long square pulse.

    Intended to be used for waveform-memory-limited devices. Effectively, only two
    square pulses, at most, will be needed: a main one of duration `duration_max` and
    a second one for potential mismatch between N `duration_max` and overall `duration`.

    Parameters
    ----------
    duration
        Duration of the long pulse in seconds.
    duration_max
        Maximum duration of square pulses to be generated in seconds.
    single_duration
        If `True`, only square pulses of duration `duration_max` will be generated.
        If `False`, a square pulse of `duration` < `duration_max` might be generated if
        necessary.
    **kwargs
        Other keyword arguments to be passed to the :class:`~SquarePulse`.

    Returns
    -------
    :
        A list of :class`SquarePulse` s equivalent to the desired long pulse.
    """
    # Sanity checks
    validator_dur = validators.Numbers(min_value=0.0, max_value=7 * 24 * 3600.0)
    validator_dur.validate(duration)

    validator_dur_max = validators.Numbers(min_value=0.0, max_value=duration)
    validator_dur_max.validate(duration_max)

    duration_last_pulse = duration % duration_max
    num_pulses = int(duration // duration_max)

    pulses = [SquarePulse(duration=duration_max, **kwargs) for _ in range(num_pulses)]

    if duration_last_pulse != 0.0:
        duration_last_pulse = duration_max if single_duration else duration_last_pulse
        pulses.append(SquarePulse(duration=duration_last_pulse, **kwargs))

    return pulses


class SoftSquarePulse(Operation):
    """
    The SoftSquarePulse Operation is a real valued square pulse convolved with
    a Hann window for smoothing.
    """

    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Create a new instance of SoftSquarePulse.

        The SoftSquarePulse Operation is a real valued square pulse convolved with
        a Hann window for smoothing.

        Parameters
        ----------
        amp
            Amplitude of the envelope.
        duration
            The pulse duration in seconds.
        port
            Port of the pulse, must be capable of playing a complex waveform.
        clock
            Clock used to modulate the pulse.
        t0
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            super().__init__(name="SoftSquarePulse")
            self.data.update(
                {
                    "name": "SoftSquarePulse",
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.soft_square",
                            "amp": amp,
                            "duration": duration,
                            "t0": t0,
                            "clock": clock,
                            "port": port,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class ChirpPulse(Operation):  # pylint: disable=too-many-ancestors
    """
    A linear chirp signal. A sinusoidal signal that ramps up in frequency.
    """

    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str,
        start_freq: float,
        end_freq: float,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Constructor for a chirp pulse.

        Parameters
        ----------
        amp
            Amplitude of the envelope.
        duration
            Duration of the pulse.
        port
            The port of the pulse.
        clock
            Clock used to modulate the pulse.
        start_freq
            Start frequency of the Chirp. Note that this is the frequency at which the
            waveform is calculated, this may differ from the clock frequency.
        end_freq
            End frequency of the Chirp.
        t0
            Shift of the start time with respect to the start of the operation.
        """
        if data is None:
            super().__init__(name="ChirpPulse")
            self.data.update(
                {
                    "name": "ChirpPulse",
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.chirp",
                            "amp": amp,
                            "duration": duration,
                            "start_freq": start_freq,
                            "end_freq": end_freq,
                            "t0": t0,
                            "clock": clock,
                            "port": port,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class DRAGPulse(Operation):
    # pylint: disable=line-too-long, too-many-ancestors
    r"""
    DRAG pulse intended for single qubit gates in transmon based systems.

    A DRAG pulse is a gaussian pulse with a derivative component added to the
    out-of-phase channel to reduce unwanted excitations of the
    :math:`|1\rangle - |2\rangle` transition (:cite:t:`motzoi_simple_2009` and
    :cite:t:`gambetta_analytic_2011`).

    The waveform is generated using :func:`.waveforms.drag` .
    """

    def __init__(
        self,
        G_amp: float,
        D_amp: float,
        phase: float,
        clock: str,
        duration: float,
        port: str,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Create a new instance of DRAGPulse.

        Parameters
        ----------
        G_amp
            Amplitude of the Gaussian envelope.
        D_amp
            Amplitude of the derivative component, the DRAG-pulse parameter.
        duration
            The pulse duration in seconds.
        phase
            Phase of the pulse in degrees.
        clock
            Clock used to modulate the pulse.
        port
            Port of the pulse, must be capable of carrying a complex waveform.
        t0
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """

        if data is None:
            super().__init__(name="DRAG")
            self.data.update(
                {
                    "name": "DRAG",
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.drag",
                            "G_amp": G_amp,
                            "D_amp": D_amp,
                            "duration": duration,
                            "phase": phase,
                            "nr_sigma": 4,
                            "clock": clock,
                            "port": port,
                            "t0": t0,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


def create_dc_compensation_pulse(
    pulses: List[Operation],
    sampling_rate: float,
    port: str,
    t0: float = 0,
    amp: Optional[float] = None,
    duration: Optional[float] = None,
    data: Optional[Dict[str, Any]] = None,
) -> SquarePulse:
    """
    Calculates a SquarePulse to counteract charging effects based on a list of pulses.

    The compensation is calculated by summing the area of all pulses on the specified
    port.
    This gives a first order approximation for the pulse required to compensate the
    charging. All modulated pulses ignored in the calculation.

    Parameters
    ----------
    pulses
        List of pulses to compensate
    sampling_rate
        Resolution to calculate the enclosure of the
        pulses to calculate the area to compensate.
    amp
        Desired amplitude of the DCCompensationPulse.
        Leave to None to calculate the value for compensation,
        in this case you must assign a value to duration.
        The sign of the amplitude is ignored and adjusted
        automatically to perform the compensation.
    duration
        Desired pulse duration in seconds.
        Leave to None to calculate the value for compensation,
        in this case you must assign a value to amp.
        The sign of the value of amp given in the previous step
        is adjusted to perform the compensation.
    port
        Port to perform the compensation. Any pulse that does not
        belong to the specified port is ignored.
    clock
        Clock used to modulate the pulse.
    phase
        Phase of the pulse in degrees.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.
    data
        The operation's dictionary, by default None\n
        Note: if the data parameter is not None all other parameters are
        overwritten using the contents of data.\n
        Deprecated: support for the data argument will be dropped in
        quantify-scheduler >= 0.13.0. Please consider updating the data
        dictionary after initialization.

    Returns
    -------

    :
        Returns a SquarePulse object that compensates all pulses passed as argument.
    """
    # Make sure that the list contains at least one element
    if len(pulses) == 0:
        raise ValueError(
            "Attempting to create a DC compensation SquarePulse with no pulses. "
            "At least one pulse is necessary."
        )

    pulse_info_list: List[Dict[str, Any]] = _extract_pulses(pulses, port)

    # Calculate the area given by the list of pulses
    area: float = area_pulses(pulse_info_list, sampling_rate)

    # Calculate the compensation amplitude and duration based on area
    c_duration: float
    c_amp: float
    if amp is None and duration is not None:
        if not duration > 0:
            raise ValueError(
                f"Attempting to create a DC compensation SquarePulse specified by {duration=}. "
                f"Duration must be a positive number."
            )
        c_duration = duration
        c_amp = -area / c_duration
    elif amp is not None and duration is None:
        if area > 0:
            c_amp = -abs(amp)
        else:
            c_amp = abs(amp)
        c_duration = abs(area / c_amp)
    else:
        raise ValueError(
            "The `DCCompensationPulse` allows either amp or duration to "
            + "be specified, not both. Both amp and duration were passed."
        )

    if data is not None:
        warnings.warn(
            "Support for the data argument will be dropped in"
            "quantify-scheduler >= 0.13.0.\n"
            "Please consider updating the data "
            "dictionary after initialization.",
            FutureWarning,
        )

    return SquarePulse(
        amp=c_amp,
        duration=c_duration,
        port=port,
        clock=BasebandClockResource.IDENTITY,
        phase=0,
        t0=t0,
        data=data,
    )


def _extract_pulses(pulses: List[Operation], port: str) -> List[Dict[str, Any]]:
    # Collect all pulses for the given port
    pulse_info_list: List[Dict[str, Any]] = []

    for pulse in pulses:
        for pulse_info in pulse["pulse_info"]:
            if (
                pulse_info["port"] == port
                and pulse_info["clock"] == BasebandClockResource.IDENTITY
            ):
                pulse_info_list.append(pulse_info)

    return pulse_info_list


class WindowOperation(Operation):
    """
    The WindowOperation is an operation for visualization purposes.

    The `WindowOperation` has a starting time and duration.
    """

    def __init__(
        self,
        window_name: str,
        duration: float,
        t0: float = 0.0,
        data: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a new instance of WindowOperation.

        """
        if data is None:
            super().__init__(name="WindowOperation")
            self.data.update(
                {
                    "name": "WindowOperation",
                    "pulse_info": [
                        {
                            "wf_func": None,
                            "window_name": window_name,
                            "duration": duration,
                            "t0": t0,
                            "port": None,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    @property
    def window_name(self) -> str:
        """Return the window name of this operation"""
        return self.data["pulse_info"][0]["window_name"]

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class NumericalPulse(Operation):
    """
    Defines a pulse where the shape is determined by specifying an array of (complex)
    points. If points are required between the specified samples (such as could be
    required by the sampling rate of the hardware), meaning :math:`t[n] < t' < t[n+1]`,
    `scipy.interpolate.interp1d` will be used to interpolate between the two points and
    determine the value.
    """

    def __init__(
        self,
        samples: Union[np.ndarray, list],
        t_samples: Union[np.ndarray, list],
        port: str,
        clock: str,
        t0: float = 0,
        interpolation: str = "linear",
        data: Optional[dict] = None,
    ):
        """
        Creates an instance of the `NumericalPulse`.

        Parameters
        ----------
        samples
            An array of (possibly complex) values specifying the shape of the pulse.
        t_samples
            An array of values specifying the corresponding times at which the
            `samples` are evaluated.
        port
            The port that the pulse should be played on.
        clock
            Clock used to (de)modulate the pulse.
        t0
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        interpolation
            Specifies the type of interpolation used. This is passed as the "kind"
            argument to `scipy.interpolate.interp1d`.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """

        def make_list_from_array(
            val: Union[NDArray[float], List[float]]
        ) -> List[float]:
            """Needed since numpy arrays break the (de)serialization code (#146)."""
            if isinstance(val, np.ndarray):
                new_val: List[float] = val.tolist()
                return new_val
            return val

        duration = t_samples[-1] - t_samples[0]
        samples, t_samples = map(make_list_from_array, [samples, t_samples])
        if data is None:
            super().__init__(name="NumericalPulse")
            self.data.update(
                {
                    "name": "NumericalPulse",
                    "pulse_info": [
                        {  # pylint: disable=line-too-long
                            "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
                            "samples": samples,
                            "t_samples": t_samples,
                            "duration": duration,
                            "interpolation": interpolation,
                            "clock": clock,
                            "port": port,
                            "t0": t0,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        """Provides a string representation of the Pulse."""
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class SkewedHermitePulse(Operation):
    # pylint: disable=line-too-long, too-many-ancestors
    """
    Hermite pulse intended for single qubit gates in diamond based systems.

    The waveform is generated using :func:`~quantify_scheduler.waveforms.skewed_hermite`.
    """

    def __init__(
        self,
        duration: float,
        amplitude: float,
        skewness: float,
        phase: float,
        port: str,
        clock: str,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Create a new instance of SkewedHermitePulse.

        Parameters
        ----------
        duration
            The pulse duration in seconds.
        amplitude
            Amplitude of the hermite pulse.
        skewness
            Skewness in the frequency space.
        phase
            Phase of the pulse in degrees.
        clock
            Clock used to modulate the pulse.
        port
            Port of the pulse, must be capable of carrying a complex waveform.
        t0
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule. By default 0.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """

        if data is None:
            super().__init__(name="hermite")
            self.data.update(
                {
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.skewed_hermite",
                            "duration": duration,
                            "amplitude": amplitude,
                            "skewness": skewness,
                            "phase": phase,
                            "clock": clock,
                            "port": port,
                            "t0": t0,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                DeprecationWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)
