# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Standard pulses for use with the quantify_scheduler."""
# pylint: disable= too-many-arguments, too-many-ancestors
from __future__ import annotations

from typing import Any, Dict, List, Optional

from qcodes import validators

from quantify_scheduler import Operation
from quantify_scheduler.helpers.waveforms import area_pulses
from quantify_scheduler.resources import BasebandClockResource


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
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """

        if data is None:
            data = {
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
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        if data is None:
            data = {
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
        """

        data = {
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
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to # pylint: disable=fixme
            # the waveform separate from the clock.
            raise NotImplementedError

        if data is None:
            data = {
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
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        if data is None:
            data = {
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
        data = {
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
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """

        if data is None:
            data = {
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
        The operation's dictionary, by default None
        Note: if the data parameter is not None all other parameters are
        overwritten using the contents of data.

    Returns
    -------

    :
        Returns a SquarePulse object that compensates all pulses passed as argument.
    """
    # Make sure that the list contains at least one element
    assert len(pulses) > 0

    pulse_info_list: List[Dict[str, Any]] = _extract_pulses(pulses, port)

    # Calculate the area given by the list of pulses
    area: float = area_pulses(pulse_info_list, sampling_rate)

    # Calculate the compensation amplitude and duration based on area
    c_duration: float
    c_amp: float
    if amp is None and duration is not None:
        assert duration > 0
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
            data = {
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
        super().__init__(name=data["name"], data=data)

    @property
    def window_name(self) -> str:
        """Return the window name of this operation"""
        return self.data["pulse_info"][0]["window_name"]

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)
