# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard pulse-level operations for use with the quantify_scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from qcodes import InstrumentChannel, validators

from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.resources import BasebandClockResource, DigitalClockResource


@dataclass
class ReferenceMagnitude:
    """Dataclass defining a reference level for pulse amplitudes in units of 'V', 'dBm', or 'A'."""

    value: float
    unit: Literal["V", "dBm", "A"]

    @classmethod
    def from_parameter(cls, parameter: InstrumentChannel) -> ReferenceMagnitude | None:
        """Initialize from ReferenceMagnitude QCoDeS InstrumentChannel values."""
        value, unit = parameter.get_val_unit()
        if np.isnan(value):
            return None
        if unit not in (allowed_units := ["V", "dBm", "A", "W"]):
            raise ValueError(f"Invalid unit: {unit}. Allowed units: {allowed_units}")

        return cls(value, unit)

    def __hash__(self) -> int:
        return hash((self.value, self.unit))


class ShiftClockPhase(Operation):
    """
    Operation that shifts the phase of a clock by a specified amount.

    This is a low-level operation and therefore depends on the backend.

    Currently only implemented for Qblox backend, refer to
    :class:`~quantify_scheduler.backends.qblox.operation_handling.virtual.NcoPhaseShiftStrategy`
    for more details.

    Parameters
    ----------
    phase_shift
        The phase shift in degrees.
    clock
        The clock of which to shift the phase.
    t0
        Time in seconds when to execute the command relative
        to the start time of the Operation in the Schedule.
    duration
        (deprecated) The duration of the operation in seconds.

    """

    def __init__(
        self,
        phase_shift: float,
        clock: str,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "t0": t0,
                "phase_shift": phase_shift,
                "clock": clock,
                "port": None,
                "duration": 0,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class ResetClockPhase(Operation):
    """
    An operation that resets the phase of a clock.

    Parameters
    ----------
    clock
        The clock of which to reset the phase.

    """

    def __init__(self, clock: str, t0: float = 0) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "clock": clock,
                "t0": t0,
                "duration": 0,
                "port": None,
                "reset_clock_phase": True,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class SetClockFrequency(Operation):
    """
    Operation that sets updates the frequency of a clock.

    This is a low-level operation and therefore depends on the backend.

    Currently only implemented for Qblox backend, refer to
    :class:`~quantify_scheduler.backends.qblox.operation_handling.virtual.NcoSetClockFrequencyStrategy`
    for more details.

    Parameters
    ----------
    clock
        The clock for which a new frequency is to be set.
    clock_freq_new
        The new frequency in Hz.
        If None, it will reset to the clock frequency set by the configuration or resource.
    t0
        Time in seconds when to execute the command relative to the start time of
        the Operation in the Schedule.
    duration
        (deprecated) The duration of the operation in seconds.

    """

    def __init__(
        self,
        clock: str,
        clock_freq_new: float | None,
        t0: float = 0,
    ) -> None:
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
                "duration": 0,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class VoltageOffset(Operation):
    """
    Operation that represents setting a constant offset to the output voltage.

    Please refer to :ref:`sec-qblox-offsets-long-voltage-offsets` in the reference guide
    for more details.

    Parameters
    ----------
    offset_path_I : float
        Offset of path I.
    offset_path_Q : float
        Offset of path Q.
    port : str
        Port of the voltage offset.
    clock : str, optional
        Clock used to modulate the voltage offset.
        By default the baseband clock.
    duration : float, optional
        (deprecated) The time to hold the offset for (in seconds).
    t0 : float, optional
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.
    reference_magnitude :
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.

    """

    def __init__(
        self,
        offset_path_I: float,
        offset_path_Q: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        t0: float = 0,
        reference_magnitude: ReferenceMagnitude | None = None,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "t0": t0,
                "offset_path_I": offset_path_I,
                "offset_path_Q": offset_path_Q,
                "clock": clock,
                "port": port,
                "duration": 0.0,
                "reference_magnitude": reference_magnitude,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class IdlePulse(Operation):
    """
    The IdlePulse Operation is a placeholder for a specified duration of time.

    Parameters
    ----------
    duration
        The duration of idle time in seconds.

    """

    def __init__(self, duration: float) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "t0": 0,
                "duration": duration,
                "clock": BasebandClockResource.IDENTITY,
                "port": None,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class RampPulse(Operation):
    r"""
    RampPulse Operation is a pulse that ramps from zero to a set amplitude over its duration.

    The pulse is given as a function of time :math:`t` and the parameters offset and
    amplitude by

    .. math::

        P(t) = \mathrm{offset} + t \times \mathrm{amp}.

    Parameters
    ----------
    amp
        Unitless amplitude of the ramp envelope function.
    duration
        The pulse duration in seconds.
    offset
        Starting point of the ramp pulse
    port
        Port of the pulse.
    clock
        Clock used to modulate the pulse.
        By default the baseband clock.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative
        to the start time
        of the Operation in the Schedule.

    """

    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        offset: float = 0,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.ramp",
                "amp": amp,
                "reference_magnitude": reference_magnitude,
                "duration": duration,
                "offset": offset,
                "t0": t0,
                "clock": clock,
                "port": port,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class StaircasePulse(Operation):
    """
    A real valued staircase pulse, which reaches it's final amplitude in discrete steps.

    In between it will maintain a plateau.

    Parameters
    ----------
    start_amp
        Starting unitless amplitude of the staircase envelope function.
    final_amp
        Final unitless amplitude of the staircase envelope function.
    num_steps
        The number of plateaus.
    duration
        Duration of the pulse in seconds.
    port
        Port of the pulse.
    clock
        Clock used to modulate the pulse.
        By default the baseband clock.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    """

    def __init__(
        self,
        start_amp: float,
        final_amp: float,
        num_steps: int,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.staircase",
                "start_amp": start_amp,
                "final_amp": final_amp,
                "reference_magnitude": reference_magnitude,
                "num_steps": num_steps,
                "duration": duration,
                "t0": t0,
                "clock": clock,
                "port": port,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class MarkerPulse(Operation):
    """
    Digital pulse that is HIGH for the specified duration.

    Marker pulse is played on marker output. Currently only implemented for Qblox
    backend.

    Parameters
    ----------
    duration
        Duration of the HIGH signal.
    port
        Name of the associated port.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.
    clock
        Name of the associated clock. By default
        :class:`~quantify_scheduler.resources.DigitalClockResource`. This only needs to
        be specified if a custom clock name is used for a digital channel (for example,
        when a port-clock combination of a device element is used with a digital
        channel).
    fine_start_delay
        Delays the start of the pulse by the given amount in seconds. Does not
        delay the start time of the operation in the schedule. If the hardware
        supports it, this parameter can be used to shift the pulse by a small
        amount of time, independent of the hardware instruction timing grid.
        Currently only implemented for Qblox QTM modules, which allow only
        positive values for this parameter. By default 0.
    fine_end_delay
        Delays the end of the pulse by the given amount in seconds. Does not
        delay the end time of the operation in the schedule. If the hardware
        supports it, this parameter can be used to shift the pulse by a small
        amount of time, independent of the hardware instruction timing grid.
        Currently only implemented for Qblox QTM modules, which allow only
        positive values for this parameter. By default 0.

    """

    def __init__(
        self,
        duration: float,
        port: str,
        t0: float = 0,
        clock: str = DigitalClockResource.IDENTITY,
        fine_start_delay: float = 0,
        fine_end_delay: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "marker_pulse": True,  # This distinguishes MarkerPulse from other operations
                "t0": t0,
                "clock": clock,
                "port": port,
                "duration": duration,
                "fine_start_delay": fine_start_delay,
                "fine_end_delay": fine_end_delay,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class SquarePulse(Operation):
    """
    A real-valued pulse with the specified amplitude during the pulse.

    Parameters
    ----------
    amp
        Unitless complex valued amplitude of the envelope.
    duration
        The pulse duration in seconds.
    port
        Port of the pulse, must be capable of playing a complex waveform.
    clock
        Clock used to modulate the pulse.
        By default the baseband clock.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    """

    def __init__(
        self,
        amp: complex,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": amp,
                "reference_magnitude": reference_magnitude,
                "duration": duration,
                "t0": t0,
                "clock": clock,
                "port": port,
            }
        ]
        self._update()

    def __str__(self) -> str:
        # Some pulse infos do not have amps in them, we need to select the correct pulse info.
        pulse_info = [d for d in self.data["pulse_info"] if "amp" in d][0]
        return self._get_signature(pulse_info)


class SuddenNetZeroPulse(Operation):
    """
    A pulse that can be used to implement a conditional phase gate in transmon qubits.

    The sudden net-zero (SNZ) pulse is defined in
    :cite:t:`negirneac_high_fidelity_2021`.

    Parameters
    ----------
    amp_A
        Unitless amplitude of the main square pulse.
    amp_B
        Unitless scaling correction for the final sample of the first square and first
        sample of the second square pulse.
    net_zero_A_scale
        Amplitude scaling correction factor of the negative arm of the net-zero
        pulse.
    t_pulse
        The total duration of the two half square pulses
    t_phi
        The idling duration between the two half pulses
    t_integral_correction
        The duration in which any non-zero pulse amplitude needs to be corrected.
    port
        Port of the pulse, must be capable of playing a complex waveform.
    clock
        Clock used to modulate the pulse.
        By default the baseband clock.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    """

    def __init__(
        self,
        amp_A: float,  # noqa N803: upper case in variable
        amp_B: float,  # noqa N803: upper case in variable
        net_zero_A_scale: float,  # noqa N803: upper case in variable
        t_pulse: float,
        t_phi: float,
        t_integral_correction: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
    ) -> None:
        duration = t_pulse + t_phi + t_integral_correction

        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.sudden_net_zero",
                "amp_A": amp_A,
                "amp_B": amp_B,
                "reference_magnitude": reference_magnitude,
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
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


def decompose_long_square_pulse(
    duration: float, duration_max: float, single_duration: bool = False, **kwargs
) -> list:
    """
    Generates a list of square pulses equivalent to a  (very) long square pulse.

    Intended to be used for waveform-memory-limited devices. Effectively, only two
    square pulses, at most, will be needed: a main one of duration ``duration_max`` and
    a second one for potential mismatch between N ``duration_max`` and overall `duration`.

    Parameters
    ----------
    duration
        Duration of the long pulse in seconds.
    duration_max
        Maximum duration of square pulses to be generated in seconds.
    single_duration
        If ``True``, only square pulses of duration ``duration_max`` will be generated.
        If ``False``, a square pulse of ``duration`` < ``duration_max`` might be generated if
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
    A real valued square pulse convolved with a Hann window for smoothing.

    Parameters
    ----------
    amp
        Unitless amplitude of the envelope.
    duration
        The pulse duration in seconds.
    port
        Port of the pulse, must be capable of playing a complex waveform.
    clock
        Clock used to modulate the pulse.
        By default the baseband clock.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    """

    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.soft_square",
                "amp": amp,
                "reference_magnitude": reference_magnitude,
                "duration": duration,
                "t0": t0,
                "clock": clock,
                "port": port,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class ChirpPulse(Operation):
    """
    A linear chirp signal. A sinusoidal signal that ramps up in frequency.

    Parameters
    ----------
    amp
        Unitless amplitude of the envelope.
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
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Shift of the start time with respect to the start of the operation.

    """

    def __init__(
        self,
        amp: float,
        duration: float,
        port: str,
        clock: str,
        start_freq: float,
        end_freq: float,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.chirp",
                "amp": amp,
                "reference_magnitude": reference_magnitude,
                "duration": duration,
                "start_freq": start_freq,
                "end_freq": end_freq,
                "t0": t0,
                "clock": clock,
                "port": port,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class DRAGPulse(Operation):
    r"""
    A Gaussian pulse with a derivative component added to the out-of-phase channel.
    It uses the specified amplitude and sigma.
    If sigma is not specified it is set to 1/4 of the duration.

    The DRAG pulse is intended for single qubit gates in transmon based systems.
    It can be calibrated to reduce unwanted excitations of the
    :math:`|1\rangle - |2\rangle` transition (:cite:t:`motzoi_simple_2009` and
    :cite:t:`gambetta_analytic_2011`).

    The waveform is generated using :func:`.waveforms.drag` .

    Parameters
    ----------
    G_amp
        Unitless amplitude of the Gaussian envelope.
    D_amp
        Unitless amplitude of the derivative component, the DRAG-pulse parameter.
    duration
        The pulse duration in seconds.
    phase
        Phase of the pulse in degrees.
    clock
        Clock used to modulate the pulse.
    port
        Port of the pulse, must be capable of carrying a complex waveform.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    sigma
        Width of the Gaussian envelope in seconds. If not provided, the sigma
        is set to 1/4 of the duration.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    """

    def __init__(
        self,
        G_amp: float,
        D_amp: float,
        phase: float,
        duration: float,
        port: str,
        clock: str,
        reference_magnitude: ReferenceMagnitude | None = None,
        sigma: float = None,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.drag",
                "G_amp": G_amp,
                "D_amp": D_amp,
                "reference_magnitude": reference_magnitude,
                "duration": duration,
                "phase": phase,
                "nr_sigma": 4 if sigma is None else None,
                "sigma": sigma,
                "clock": clock,
                "port": port,
                "t0": t0,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class GaussPulse(Operation):
    r"""
    The GaussPulse Operation is a real-valued pulse with the specified
    amplitude and sigma.
    If sigma is not specified it is set to 1/4 of the duration.

    The waveform is generated using :func:`.waveforms.drag` whith a D_amp set to zero,
    corresponding to a Gaussian pulse.

    Parameters
    ----------
    G_amp
        Unitless amplitude of the Gaussian envelope.
    duration
        The pulse duration in seconds.
    phase
        Phase of the pulse in degrees.
    clock
        Clock used to modulate the pulse.
        By default the baseband clock.
    port
        Port of the pulse, must be capable of carrying a complex waveform.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    sigma
        Width of the Gaussian envelope in seconds. If not provided, the sigma
        is set to 1/4 of the duration.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    """

    def __init__(
        self,
        G_amp: float,
        phase: float,
        duration: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        sigma: float = None,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.drag",
                "G_amp": G_amp,
                "D_amp": 0,
                "reference_magnitude": reference_magnitude,
                "duration": duration,
                "phase": phase,
                "nr_sigma": 4 if sigma is None else None,
                "sigma": sigma,
                "clock": clock,
                "port": port,
                "t0": t0,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


def create_dc_compensation_pulse(
    pulses: list[Operation],
    sampling_rate: float,
    port: str,
    t0: float = 0,
    amp: float | None = None,
    reference_magnitude: ReferenceMagnitude | None = None,  # noqa: ARG001
    duration: float | None = None,
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
        Desired unitless amplitude of the DC compensation SquarePulse.
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
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    phase
        Phase of the pulse in degrees.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    Returns
    -------
    :
        Returns a SquarePulse object that compensates all pulses passed as argument.

    """
    # Prevent circular import.
    from quantify_scheduler.helpers.waveforms import area_pulses

    def _extract_pulses(pulses: list[Operation], port: str) -> list[dict[str, Any]]:
        # Collect all pulses for the given port
        pulse_info_list: list[dict[str, Any]] = []

        for pulse in pulses:
            for pulse_info in pulse["pulse_info"]:
                if (
                    pulse_info["port"] == port
                    and pulse_info["clock"] == BasebandClockResource.IDENTITY
                ):
                    pulse_info_list.append(pulse_info)

        return pulse_info_list

    # Make sure that the list contains at least one element
    if len(pulses) == 0:
        raise ValueError(
            "Attempting to create a DC compensation SquarePulse with no pulses. "
            "At least one pulse is necessary."
        )

    pulse_info_list: list[dict[str, Any]] = _extract_pulses(pulses, port)

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
        c_amp = -abs(amp) if area > 0 else abs(amp)
        c_duration = abs(area / c_amp)
    else:
        raise ValueError(
            "The DC compensation SquarePulse allows either amp or duration to "
            + "be specified, not both. Both amp and duration were passed."
        )

    return SquarePulse(
        amp=c_amp,
        duration=c_duration,
        port=port,
        clock=BasebandClockResource.IDENTITY,
        t0=t0,
    )


class WindowOperation(Operation):
    """
    The WindowOperation is an operation for visualization purposes.

    The :class:`~WindowOperation` has a starting time and duration.
    """

    def __init__(
        self,
        window_name: str,
        duration: float,
        t0: float = 0.0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "window_name": window_name,
                "duration": duration,
                "t0": t0,
                "port": None,
            }
        ]
        self._update()

    @property
    def window_name(self) -> str:
        """Return the window name of this operation."""
        return self.data["pulse_info"][0]["window_name"]

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class NumericalPulse(Operation):
    """
    A pulse where the shape is determined by specifying an array of (complex) points.

    If points are required between the specified samples (such as could be
    required by the sampling rate of the hardware), meaning :math:`t[n] < t' < t[n+1]`,
    `scipy.interpolate.interp1d` will be used to interpolate between the two points and
    determine the value.

    Parameters
    ----------
    samples
        An array of (possibly complex) values specifying the shape of the pulse.
    t_samples
        An array of values specifying the corresponding times at which the
        ``samples`` are evaluated.
    port
        The port that the pulse should be played on.
    clock
        Clock used to (de)modulate the pulse.
        By default the baseband clock.
    reference_magnitude
        Scaling value and unit for the unitless samples. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.
    interpolation
        Specifies the type of interpolation used. This is passed as the "kind"
        argument to `scipy.interpolate.interp1d`.

    """

    def __init__(
        self,
        samples: np.ndarray | list,
        t_samples: np.ndarray | list,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
        interpolation: str = "linear",
    ) -> None:
        def make_list_from_array(val: np.ndarray[float] | list[float]) -> list[float]:
            """Needed since numpy arrays break the (de)serialization code (#146)."""
            if isinstance(val, np.ndarray):
                new_val: list[float] = val.tolist()
                return new_val
            return val

        duration = t_samples[-1] - t_samples[0]
        samples, t_samples = map(make_list_from_array, [samples, t_samples])

        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
                "samples": samples,
                "t_samples": t_samples,
                "reference_magnitude": reference_magnitude,
                "duration": duration,
                "interpolation": interpolation,
                "clock": clock,
                "port": port,
                "t0": t0,
            }
        ]
        self._update()

    def __str__(self) -> str:
        """Provides a string representation of the Pulse."""
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class SkewedHermitePulse(Operation):
    """
    Hermite pulse intended for single qubit gates in diamond based systems.

    The waveform is generated using :func:`~quantify_scheduler.waveforms.skewed_hermite`.

    Parameters
    ----------
    duration
        The pulse duration in seconds.
    amplitude
        Unitless amplitude of the hermite pulse.
    skewness
        Skewness in the frequency space.
    phase
        Phase of the pulse in degrees.
    clock
        Clock used to modulate the pulse.
    port
        Port of the pulse, must be capable of carrying a complex waveform.
    reference_magnitude
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule. By default 0.

    """

    def __init__(
        self,
        duration: float,
        amplitude: float,
        skewness: float,
        phase: float,
        port: str,
        clock: str,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": "quantify_scheduler.waveforms.skewed_hermite",
                "duration": duration,
                "amplitude": amplitude,
                "reference_magnitude": reference_magnitude,
                "skewness": skewness,
                "phase": phase,
                "clock": clock,
                "port": port,
                "t0": t0,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)


class Timestamp(Operation):
    """
    Operation that marks a time reference for timetags.

    Specifically, all timetags in
    :class:`~quantify_scheduler.operations.acquisition_library.Timetag` and
    :class:`~quantify_scheduler.operations.acquisition_library.TimetagTrace` are
    measured relative to the timing of this operation, if they have a matching port and
    clock, and if ``time_ref=TimeRef.TIMESTAMP`` is given as an argument.

    Parameters
    ----------
    port
        The same port that the timetag acquisition is defined on.
    clock
        The same clock that the timetag acquisition is defined on.
    t0
        Time offset (in seconds) of this Operation, relative to the start time in the
        Schedule. By default 0.

    """

    def __init__(
        self,
        port: str,
        t0: float = 0,
        clock: str = DigitalClockResource.IDENTITY,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "t0": t0,
                "duration": 0,
                "clock": clock,
                "port": port,
                "timestamp": True,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)
