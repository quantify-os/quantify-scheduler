# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Standard pulses for use with the quantify.scheduler."""
from __future__ import annotations
from qcodes import validators
from quantify.scheduler.types import Operation
from quantify.scheduler.resources import BasebandClockResource


class IdlePulse(Operation):
    def __init__(self, duration):
        """
        An idle pulse performing no actions for a certain duration.

        Parameters
        ------------
        duration : float
            Duration of the idling in seconds.
        """
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


class RampPulse(Operation):
    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        t0: float = 0,
    ):
        """
        A real valued ramp pulse.

        Parameters
        ------------
        amp : float
            Final amplitude of the ramp envelope function.
        duration : float
            Duration of the pulse in seconds.
        port : str
            Port of the pulse.
        clock : str
            Clock used to modulate the pulse, by default a BasebandClock is used.
        t0 : float
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        """

        data = {
            "name": "RampPulse",
            "pulse_info": [
                {
                    "wf_func": "quantify.scheduler.waveforms.ramp",
                    "amp": amp,
                    "duration": duration,
                    "t0": t0,
                    "clock": clock,
                    "port": port,
                }
            ],
        }
        super().__init__(name=data["name"], data=data)


class SquarePulse(Operation):
    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str,
        phase: float = 0,
        t0: float = 0,
    ):
        """
        A real valued square pulse.

        Parameters
        ------------
        amp : float
            Amplitude of the envelope.
        duration : float
            Duration of the pulse in seconds.
        port : str
            Port of the pulse, must be capable of playing a complex waveform.
        clock : str
            Clock used to modulate the pulse.
        phase : float
            Phase of the pulse in degrees.
        t0 : float
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError

        data = {
            "name": "ModSquarePulse",
            "pulse_info": [
                {
                    "wf_func": "quantify.scheduler.waveforms.square",
                    "amp": amp,
                    "duration": duration,
                    "t0": t0,
                    "clock": clock,
                    "port": port,
                }
            ],
        }
        super().__init__(name=data["name"], data=data)


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
    def __init__(
        self, amp: float, duration: float, port: str, clock: str, t0: float = 0
    ):
        """
        A real valued square pulse convolved with a hann window in order to smoothen it.

        Parameters
        ------------
        amp : float
            Amplitude of the envelope.
        duration : float
            Duration of the pulse in seconds.
        port : str
            Port of the pulse, must be capable of playing a complex waveform.
        clock : str
            Clock used to modulate the pulse.
        t0 : float
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        """
        data = {
            "name": "SoftSquarePulse",
            "pulse_info": [
                {
                    "wf_func": "quantify.scheduler.waveforms.soft_square",
                    "amp": amp,
                    "duration": duration,
                    "t0": t0,
                    "clock": clock,
                    "port": port,
                }
            ],
        }
        super().__init__(name=data["name"], data=data)


class DRAGPulse(Operation):
    """
    DRAG pulse intended for single qubit gates in transmon based systems.

    A DRAG pulse is a gaussian pulse with a derivative component added to the out-of-phase channel to
    reduce unwanted excitations of the :math:`|1\\rangle - |2\\rangle` transition.


    The waveform is generated using :func:`.waveforms.drag` .

    References:
        1. |citation1|_

        .. _citation1: https://link.aps.org/doi/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).*

        2. |citation2|_

        .. _citation2: https://link.aps.org/doi/10.1103/PhysRevLett.103.110501

        .. |citation2| replace:: *F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
           Phys. Rev. Lett. 103, 110501 (2009).*
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
    ):
        """
        Parameters
        ------------
        G_amp : float
            Amplitude of the Gaussian envelope.
        D_amp : float
            Amplitude of the derivative component, the DRAG-pulse parameter.
        duration : float
            Duration of the pulse in seconds.
        phase : float
            Phase of the pulse in degrees.
        clock : str
            Clock used to modulate the pulse.
        port : str
            Port of the pulse, must be capable of carrying a complex waveform.
        t0 : float
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.
        """

        data = {
            "name": "DRAG",
            "pulse_info": [
                {
                    "wf_func": "quantify.scheduler.waveforms.drag",
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
