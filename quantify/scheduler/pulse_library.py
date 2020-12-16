# -----------------------------------------------------------------------------
# Description:    Library standard pulses for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
from .types import Operation
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
        data = {'name': 'Idle', 'pulse_info': [{
            'wf_func': None,
            't0': 0,
            'duration': duration,
            'clock': BasebandClockResource.IDENTITY,
            'port': None}]}
        super().__init__(name=data['name'], data=data)


class RampPulse(Operation):

    def __init__(self, amp: float, duration: float, port: str, clock: str, t0: float = 0):
        """
        A single-channel square pulse.

        Parameters
        ------------
        amp : float
            Amplitude of the Gaussian envelope.
        duration : float
            Duration of the pulse in seconds.
        port : str
            Port of the pulse.
        """

        data = {'name': 'RampPulse', 'pulse_info': [{
            'wf_func': 'quantify.scheduler.waveforms.ramp',
            'amp': amp, 'duration': duration,
            't0': t0,
            'clock': clock,
            'port': port}]}
        super().__init__(name=data['name'], data=data)


class SquarePulse(Operation):

    def __init__(self, amp: float, duration: float, port: str, clock: str, phase: float = 0, t0: float = 0):
        """
        A two-channel square pulse.

        Parameters
        ------------
        amp : float
            Amplitude of the envelope.
        duration : float
            Duration of the pulse in seconds.
        port : str
            Port of the pulse, must be capable of playing a complex waveform.
        phase : float
            Phase of the pulse in degrees.
        clock : str
            Clock used to modulate the pulse.
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError

        data = {'name': 'ModSquarePulse', 'pulse_info': [{
            'wf_func': 'quantify.scheduler.waveforms.square',
            'amp': amp, 'duration': duration,
            't0': t0,
            'clock': clock,
            'port': port}]}
        super().__init__(name=data['name'], data=data)


class SoftSquarePulse(Operation):
    """
    Place holder pulse for mocking the CZ pulse until proper implementation. Replicates parameters.
    """

    def __init__(self, amp: float, duration: float, port: str, clock: str, t0: float = 0):
        data = {'name': 'SoftSquarePulse', 'pulse_info': [{
            'wf_func': 'quantify.scheduler.waveforms.soft_square',
            'amp': amp, 'duration': duration,
            't0': t0,
            'clock': clock,
            'port': port}]}
        super().__init__(name=data['name'], data=data)


class DRAGPulse(Operation):
    """
    DRAG pulse inteded for single qubit gates in transmon based systems.

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

    def __init__(self, G_amp: float, D_amp: float, phase: float, clock: str, duration: float, port: str, t0: float = 0):
        """
        Parameters
        ------------
        G_amp : float
            Amplitude of the Gaussian envelope.
        D_amp : float
            Amplitude of the derivative component, the DRAG-pulse parameter.
        duration : float
            Duration of the pulse in seconds.
        nr_sigma : int
            After how many sigma the Gaussian is cut off.
        phase : float
            Phase of the pulse in degrees.
        clock : str
            Clock used to modulate the pulse.
        port : str
            Port of the pulse, must be capable of carrying a complex waveform.
        """

        data = {'name': "DRAG", 'pulse_info': [{
            'wf_func': 'quantify.scheduler.waveforms.drag',
            'G_amp': G_amp, 'D_amp': D_amp, 'duration': duration,
            'phase': phase, 'nr_sigma': 4, 'clock': clock,
            'port': port,  't0': t0}]}

        super().__init__(name=data['name'], data=data)
