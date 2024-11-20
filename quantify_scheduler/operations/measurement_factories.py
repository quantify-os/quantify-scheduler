# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
A module containing factory functions for measurements on the quantum-device layer.

These factories are used to take a parametrized representation of on a operation
and use that to create an instance of the operation itself.
"""
from __future__ import annotations

import math
import warnings
from typing import Hashable, Literal

import numpy as np

from quantify_scheduler.enums import BinMode, TimeRef, TimeSource
from quantify_scheduler.operations.acquisition_library import (
    NumericalSeparatedWeightedIntegration,
    NumericalWeightedIntegration,
    SSBIntegrationComplex,
    ThresholdedAcquisition,
    Timetag,
    TimetagTrace,
    Trace,
    TriggerCount,
)
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    ReferenceMagnitude,
    ResetClockPhase,
    SetClockFrequency,
    SquarePulse,
    VoltageOffset,
)
from quantify_scheduler.schedules.schedule import Schedule


def _dispersive_measurement(  # noqa: PLR0915
    pulse_amp: float,
    pulse_duration: float,
    port: str,
    gate_pulse_amp: float | None,
    gate_port: str | None,
    clock: str,
    acq_duration: float,
    acq_delay: float,
    acq_channel: Hashable,
    acq_channel_override: Hashable | None,
    acq_index: int,
    acq_protocol: str | None,
    pulse_type: Literal["SquarePulse"],
    bin_mode: BinMode | None,
    acq_protocol_default: str,
    reset_clock_phase: bool,
    reference_magnitude: ReferenceMagnitude | None,
    acq_weights_a: list[complex] | np.ndarray | None,
    acq_weights_b: list[complex] | np.ndarray | None,
    acq_weights_sampling_rate: float | None,
    feedback_trigger_label: str | None,
    acq_rotation: float | None,
    acq_threshold: float | None,
    num_points: float | None,
    freq: float | None,
) -> Schedule:
    """
    Generator function for a standard dispersive measurement.

    A dispersive measurement (typically) consists of a pulse being applied to the device
    followed by an acquisition protocol to interpret the signal coming back from the
    device.

    Parameters
    ----------
    pulse_amp
        The amplitude of the pulse.
    pulse_duration
        The duration of the pulse.
    port
        The port for the pulse.
    gate_pulse_amp
        Optional amplitude for the gate pulse.
    gate_port
        Optional port for the gate pulse.
    clock
        The clock for the pulse.
    acq_duration
        The duration of the acquisition.
    acq_delay
        The delay before the acquisition starts.
    acq_channel
        The acquisition channel.
    acq_channel_override
        An optional override for the acquisition channel.
    acq_index
        The index of the acquisition.
    acq_protocol
        The acquisition protocol to use.
    pulse_type
        The type of pulse to use. Default is "SquarePulse".
    bin_mode
        The binning mode for the acquisition. Default is BinMode.AVERAGE.
    acq_protocol_default
        The default acquisition protocol to use. Default is "SSBIntegrationComplex".
    reset_clock_phase
        Whether to reset the clock phase. Default is True.
    reference_magnitude
        An optional reference magnitude.
    acq_weights_a
        Optional acquisition weights A.
    acq_weights_b
        Optional acquisition weights B.
    acq_weights_sampling_rate
        The sampling rate for the acquisition weights.
    feedback_trigger_label
        Optional feedback trigger label.
    acq_rotation
        Optional acquisition rotation.
    acq_threshold
        Optional acquisition threshold.
    num_points
        Optional number of points for the acquisition.
    freq
        Optional frequency to override clock for this operation.

    Returns
    -------
    :
        The resulting schedule for the dispersive measurement.

    """
    if bin_mode is None:
        bin_mode = BinMode.AVERAGE

    subschedule = Schedule("dispersive_measurement")

    if acq_protocol != "LongTimeTrace":

        if reset_clock_phase:
            subschedule.add(ResetClockPhase(clock=clock))

        if pulse_type == "SquarePulse":
            subschedule.add(
                SquarePulse(
                    amp=pulse_amp,
                    duration=pulse_duration,
                    port=port,
                    clock=clock,
                    reference_magnitude=reference_magnitude,
                ),
                ref_pt="start",
            )
        else:
            raise NotImplementedError(
                f'Invalid pulse_type "{pulse_type}" specified as argument to '
                "dispersive_measurement. Currently dispersive_measurement only"
                ' allows "SquarePulse". Please correct your device config.'
            )

        if gate_pulse_amp is not None or gate_port is not None:
            subschedule.add(
                SquarePulse(
                    amp=gate_pulse_amp,
                    duration=pulse_duration,
                    port=gate_port,
                    clock="cl0.baseband",
                    reference_magnitude=reference_magnitude,
                ),
                ref_pt="start",
            )

    if acq_protocol is None:
        acq_protocol = acq_protocol_default

    if acq_channel_override is not None:
        acq_channel = acq_channel_override

    if acq_protocol == "SSBIntegrationComplex":
        subschedule.add(
            SSBIntegrationComplex(
                port=port,
                clock=clock,
                duration=acq_duration,
                acq_channel=acq_channel,
                acq_index=acq_index,
                bin_mode=bin_mode,
                t0=acq_delay,
            ),
            ref_pt="start",
        )
    elif acq_protocol in (
        "NumericalSeparatedWeightedIntegration",
        "NumericalWeightedIntegration",
    ):
        if acq_weights_a is None or acq_weights_b is None or acq_weights_sampling_rate is None:
            raise TypeError(
                f"Keyword arguments 'acq_weights_a', 'acq_weights_b' and "
                f"'acq_weights_sampling_rate' must not be None when {acq_protocol=} is "
                f"selected. These arguments can be specified in the device "
                f"configuration."
            )
        dur_from_weights = len(acq_weights_a) / acq_weights_sampling_rate
        if not math.isclose(acq_duration, dur_from_weights):
            warnings.warn(
                f"The specified weights and sampling rate lead to a weighted "
                f"integration duration of {dur_from_weights:0.1e} s, which is "
                f"different from the specified default acquisition duration of "
                f"{acq_duration:0.1e} s. The default acquisition duration will be "
                f"ignored for weighted acquisition.",
                UserWarning,
            )
        if acq_protocol == "NumericalSeparatedWeightedIntegration":
            subschedule.add(
                NumericalSeparatedWeightedIntegration(
                    port=port,
                    clock=clock,
                    weights_a=acq_weights_a,
                    weights_b=acq_weights_b,
                    weights_sampling_rate=acq_weights_sampling_rate,
                    acq_channel=acq_channel,
                    acq_index=acq_index,
                    bin_mode=bin_mode,
                    t0=acq_delay,
                ),
                ref_pt="start",
            )
        elif acq_protocol == "NumericalWeightedIntegration":
            subschedule.add(
                NumericalWeightedIntegration(
                    port=port,
                    clock=clock,
                    weights_a=acq_weights_a,
                    weights_b=acq_weights_b,
                    weights_sampling_rate=acq_weights_sampling_rate,
                    acq_channel=acq_channel,
                    acq_index=acq_index,
                    bin_mode=bin_mode,
                    t0=acq_delay,
                ),
                ref_pt="start",
            )
    elif acq_protocol == "ThresholdedAcquisition":
        subschedule.add(
            ThresholdedAcquisition(
                port=port,
                clock=clock,
                duration=acq_duration,
                acq_channel=acq_channel,
                acq_index=acq_index,
                bin_mode=bin_mode,
                t0=acq_delay,
                feedback_trigger_label=feedback_trigger_label,
                acq_rotation=acq_rotation,
                acq_threshold=acq_threshold,
            ),
            ref_pt="start",
        )
    elif acq_protocol == "Trace":
        subschedule.add(
            Trace(
                port=port,
                clock=clock,
                duration=acq_duration,
                acq_channel=acq_channel,
                acq_index=acq_index,
                t0=acq_delay,
            ),
            ref_pt="start",
        )
    elif acq_protocol == "LongTimeTrace":
        if bin_mode != BinMode.APPEND:
            raise ValueError(
                f"For measurement protocol '{acq_protocol}' "
                f"bin_mode set to '{bin_mode}', "
                f"but only 'BinMode.APPEND' is supported."
            )
        if not isinstance(num_points, int):
            raise ValueError(
                f"For measurement protocol '{acq_protocol}', "
                f"num_points is set to '{num_points}',"
                f"but only integer values are supported."
            )

        pulse_op = VoltageOffset(
            offset_path_I=np.real(pulse_amp),
            offset_path_Q=np.imag(pulse_amp),
            port=port,
            clock=clock,
        )

        subschedule.add(pulse_op)

        if gate_pulse_amp is not None or gate_port is not None:
            gate_pulse_op = SquarePulse(
                amp=np.real(gate_pulse_amp),
                duration=num_points * acq_duration,
                port=gate_port,
                clock="cl0.baseband",
            )
            subschedule.add(gate_pulse_op)

        loop_subschedule = Schedule("loop_SSB")

        acquisition = SSBIntegrationComplex(
            port=port,
            clock=clock,
            duration=acq_duration,
            acq_channel=0,
            acq_index=acq_index,
            bin_mode=bin_mode,
            t0=0,
        )

        loop_subschedule.add(ResetClockPhase(clock=clock))
        loop_subschedule.add(acquisition, ref_pt="start")

        subschedule.add(
            LoopOperation(
                body=loop_subschedule,
                repetitions=num_points,
            ),
            rel_time=acq_delay,
            ref_pt="start",
        )

        pulse_op_off = VoltageOffset(
            offset_path_I=0,
            offset_path_Q=0,
            port=port,
            clock=clock,
        )
        subschedule.add(pulse_op_off)

        subschedule.add(IdlePulse(duration=4e-9))

    else:
        raise ValueError(f'Acquisition protocol "{acq_protocol}" is not supported.')

    if freq is not None:
        subschedule_with_freq = Schedule("dispersive_measurement_with_freq")
        subschedule_with_freq.add(SetClockFrequency(clock=clock, clock_freq_new=freq))
        subschedule_with_freq.add(subschedule)
        subschedule_with_freq.add(SetClockFrequency(clock=clock, clock_freq_new=None))
        return subschedule_with_freq
    else:
        return subschedule


def dispersive_measurement_transmon(
    pulse_amp: float,
    pulse_duration: float,
    port: str,
    clock: str,
    acq_duration: float,
    acq_delay: float,
    acq_channel: Hashable,
    acq_channel_override: Hashable | None,
    acq_index: int,
    acq_protocol: str | None,
    pulse_type: Literal["SquarePulse"] = "SquarePulse",
    bin_mode: BinMode | None = BinMode.AVERAGE,
    acq_protocol_default: str = "SSBIntegrationComplex",
    reset_clock_phase: bool = True,
    reference_magnitude: ReferenceMagnitude | None = None,
    acq_weights_a: list[complex] | np.ndarray | None = None,
    acq_weights_b: list[complex] | np.ndarray | None = None,
    acq_weights_sampling_rate: float | None = None,
    feedback_trigger_label: str | None = None,
    acq_rotation: float | None = None,
    acq_threshold: float | None = None,
    num_points: float | None = None,
    freq: float | None = None,
) -> Schedule:
    """
    Creates a dispersive measurement schedule for a transmon qubit.

    Parameters
    ----------
    pulse_amp
        The amplitude of the pulse.
    pulse_duration
        The duration of the pulse.
    port
        The port for the pulse.
    clock
        The clock for the pulse.
    acq_duration
        The duration of the acquisition.
    acq_delay
        The delay before the acquisition starts.
    acq_channel
        The acquisition channel.
    acq_channel_override
        An optional override for the acquisition channel.
    acq_index
        The index of the acquisition.
    acq_protocol
        The acquisition protocol to use.
    pulse_type
        The type of pulse to use. Default is "SquarePulse".
    bin_mode
        The binning mode for the acquisition. Default is BinMode.AVERAGE.
    acq_protocol_default
        The default acquisition protocol to use. Default is "SSBIntegrationComplex".
    reset_clock_phase
        Whether to reset the clock phase. Default is True.
    reference_magnitude
        An optional reference magnitude.
    acq_weights_a
        Optional acquisition weights A.
    acq_weights_b
        Optional acquisition weights B.
    acq_weights_sampling_rate
        The sampling rate for the acquisition weights.
    feedback_trigger_label
        Optional feedback trigger label.
    acq_rotation
        Optional acquisition rotation.
    acq_threshold
        Optional acquisition threshold.
    num_points
        Optional number of points for the acquisition.
    freq
        Optional frequency to override clock for this operation.

    Returns
    -------
    :
        The resulting schedule for the dispersive measurement.

    """
    return _dispersive_measurement(
        pulse_amp=pulse_amp,
        pulse_duration=pulse_duration,
        port=port,
        gate_pulse_amp=None,
        gate_port=None,
        clock=clock,
        acq_duration=acq_duration,
        acq_delay=acq_delay,
        acq_channel=acq_channel,
        acq_channel_override=acq_channel_override,
        acq_index=acq_index,
        acq_protocol=acq_protocol,
        pulse_type=pulse_type,
        bin_mode=bin_mode,
        acq_protocol_default=acq_protocol_default,
        reset_clock_phase=reset_clock_phase,
        reference_magnitude=reference_magnitude,
        acq_weights_a=acq_weights_a,
        acq_weights_b=acq_weights_b,
        acq_weights_sampling_rate=acq_weights_sampling_rate,
        feedback_trigger_label=feedback_trigger_label,
        acq_rotation=acq_rotation,
        acq_threshold=acq_threshold,
        num_points=num_points,
        freq=freq,
    )


def dispersive_measurement_spin(
    pulse_amp: float,
    pulse_duration: float,
    port: str,
    gate_pulse_amp: float | None,
    gate_port: str | None,
    clock: str,
    acq_duration: float,
    acq_delay: float,
    acq_channel: Hashable,
    acq_channel_override: Hashable | None,
    acq_index: int,
    acq_protocol: str | None,
    pulse_type: Literal["SquarePulse"] = "SquarePulse",
    bin_mode: BinMode | None = BinMode.AVERAGE,
    acq_protocol_default: str = "SSBIntegrationComplex",
    reset_clock_phase: bool = True,
    reference_magnitude: ReferenceMagnitude | None = None,
    acq_weights_a: list[complex] | np.ndarray | None = None,
    acq_weights_b: list[complex] | np.ndarray | None = None,
    acq_weights_sampling_rate: float | None = None,
    feedback_trigger_label: str | None = None,
    acq_rotation: float | None = None,
    acq_threshold: float | None = None,
    num_points: float | None = None,
    freq: float | None = None,
) -> Schedule:
    """
    Creates a dispersive measurement schedule for a spin qubit.

    Parameters
    ----------
    pulse_amp
        The amplitude of the pulse.
    pulse_duration
        The duration of the pulse.
    port
        The port for the pulse.
    clock
        The clock for the pulse.
    acq_duration
        The duration of the acquisition.
    acq_delay
        The delay before the acquisition starts.
    acq_channel
        The acquisition channel.
    acq_channel_override
        An optional override for the acquisition channel.
    acq_index
        The index of the acquisition.
    acq_protocol
        The acquisition protocol to use.
    pulse_type
        The type of pulse to use. Default is "SquarePulse".
    bin_mode
        The binning mode for the acquisition. Default is BinMode.AVERAGE.
    acq_protocol_default
        The default acquisition protocol to use. Default is "SSBIntegrationComplex".
    reset_clock_phase
        Whether to reset the clock phase. Default is True.
    reference_magnitude
        An optional reference magnitude.
    acq_weights_a
        Optional acquisition weights A.
    acq_weights_b
        Optional acquisition weights B.
    acq_weights_sampling_rate
        The sampling rate for the acquisition weights.
    feedback_trigger_label
        Optional feedback trigger label.
    acq_rotation
        Optional acquisition rotation.
    acq_threshold
        Optional acquisition threshold.
    num_points
        Optional number of points for the acquisition.
    gate_pulse_amp
        Optional amplitude for the gate pulse.
    gate_port
        Optional port for the gate pulse.
    freq
        Optional frequency to override clock for this operation.

    Returns
    -------
    :
        The resulting schedule for the dispersive measurement.

    """
    return _dispersive_measurement(
        pulse_amp=pulse_amp,
        pulse_duration=pulse_duration,
        port=port,
        gate_pulse_amp=gate_pulse_amp,
        gate_port=gate_port,
        clock=clock,
        acq_duration=acq_duration,
        acq_delay=acq_delay,
        acq_channel=acq_channel,
        acq_channel_override=acq_channel_override,
        acq_index=acq_index,
        acq_protocol=acq_protocol,
        pulse_type=pulse_type,
        bin_mode=bin_mode,
        acq_protocol_default=acq_protocol_default,
        reset_clock_phase=reset_clock_phase,
        reference_magnitude=reference_magnitude,
        acq_weights_a=acq_weights_a,
        acq_weights_b=acq_weights_b,
        acq_weights_sampling_rate=acq_weights_sampling_rate,
        feedback_trigger_label=feedback_trigger_label,
        acq_rotation=acq_rotation,
        acq_threshold=acq_threshold,
        num_points=num_points,
        freq=freq,
    )


def optical_measurement(
    pulse_amplitudes: list[float],
    pulse_durations: list[float],
    pulse_ports: list[str],
    pulse_clocks: list[str],
    acq_duration: float,
    acq_delay: float,
    acq_port: str,
    acq_clock: str,
    acq_channel: Hashable,
    acq_channel_override: Hashable | None,
    acq_index: int,
    bin_mode: BinMode | None,
    acq_protocol: Literal["Trace", "TriggerCount", "Timetag", "TimetagTrace"] | None,
    acq_protocol_default: Literal["Trace", "TriggerCount"],
    pulse_type: Literal["SquarePulse"],
    acq_time_source: TimeSource | None = None,
    acq_time_ref: TimeRef | None = None,
) -> Operation:
    """
    Generator function for an optical measurement with multiple excitation pulses.

    An optical measurement generates a square pulse in the optical range and uses
    either the Trace acquisition to return the output of a photon detector as a
    function of time or the TriggerCount acquisition to return the number of photons
    that are collected.

    All pulses can have different amplitudes, durations, ports and clocks. All pulses
    start simultaneously. The acquisition can have an ``acq_delay`` with respect to the
    pulses. A negative ``acq_delay`` causes the acquisition to be scheduled at time 0
    and the pulses at the positive time ``-acq_delay``.

    Parameters
    ----------
    pulse_amplitudes
        list of amplitudes of the corresponding pulses
    pulse_durations
        list of durations of the corresponding pulses
    pulse_ports
        Port names, where the corresponding pulses are applied
    pulse_clocks
        Clock names of the corresponding pulses
    acq_duration
        Duration of the acquisition
    acq_delay
        Delay between the start of the readout pulse and the start of the acquisition:
        acq_delay = t0_pulse - t0_acquisition.
    acq_port
        Port name of the acquisition
    acq_clock
        Clock name of the acquisition
    acq_channel
        Default acquisition channel of the device element
    acq_channel_override
        Acquisition channel of the operation
    acq_index
        Acquisition index as defined in the Schedule
    bin_mode
        Describes what is done when data is written to a register that already
        contains a value. Options are "append" which appends the result to the
        list. "average" which stores the count value of the new result and the
        old register value is not currently implemented. ``None`` internally
        resolves to ``BinMode.APPEND``.
    acq_protocol
        Acquisition protocol. "Trace" returns a time trace of the collected signal.
        "TriggerCount" returns the number of times the trigger threshold is surpassed.
    acq_protocol_default
        Acquisition protocol if ``acq_protocol`` is None
    pulse_type
        Shape of the pulse to be generated
    acq_time_source
        Selects the timetag data source for this acquisition type.
    acq_time_ref
        Selects the time reference that the timetag is recorded in relation to.

    Returns
    -------
    :
        Operation with the generated pulses and acquisition

    Raises
    ------
    ValueError
        If first four function arguments do not have the same length.
    NotImplementedError
        If an unknown ``pulse_type`` or ``acq_protocol`` are used.

    """
    # ensures default argument is used if not specified at gate level.
    # ideally, this input would not be accepted, but this is a workaround for #267
    if bin_mode is None:
        bin_mode = BinMode.APPEND

    # All lists should be of equal length so this should be ensured
    if not len(pulse_amplitudes) == len(pulse_durations) == len(pulse_ports) == len(pulse_clocks):
        raise ValueError(
            "For multiple optical excitations, lists must have same length:\n"
            + f"{len(pulse_amplitudes)=},\n"
            + f"{len(pulse_durations)=},\n"
            + f"{len(pulse_ports)=},\n"
            + f"{len(pulse_clocks)=}"
        )

    # If acq_delay >= 0, the pulse starts at 0 and the acquisition at acq_delay
    # If acq_delay < 0, the acquisition starts at 0 and the pulse at -acq_delay (which is positive)
    t0_pulse = max(0, -acq_delay)
    t0_acquisition = max(0, acq_delay)

    # This operation will contain all pulses and the acquisition
    device_op = Operation("OpticalMeasurement")

    if pulse_type == "SquarePulse":
        settings = zip(pulse_amplitudes, pulse_durations, pulse_ports, pulse_clocks)
        for amp, dur, port, clock in settings:
            device_op.add_pulse(
                SquarePulse(
                    amp=amp,
                    duration=dur,
                    port=port,
                    clock=clock,
                    t0=t0_pulse,
                )
            )
    else:
        raise NotImplementedError(
            f"Invalid pulse_type '{pulse_type}' specified as argument to "
            f"optical_measurement. Currently, only 'SquarePulse' is accepted. "
            f"Please correct your device config."
        )

    if acq_protocol is None:
        acq_protocol = acq_protocol_default

    if acq_channel_override is not None:
        acq_channel = acq_channel_override

    if acq_protocol == "TriggerCount":
        device_op.add_acquisition(
            TriggerCount(
                port=acq_port,
                clock=acq_clock,
                duration=acq_duration,
                t0=t0_acquisition,
                acq_channel=acq_channel,
                acq_index=acq_index,
                bin_mode=bin_mode,
            )
        )
    elif acq_protocol == "Trace":
        device_op.add_acquisition(
            Trace(
                port=acq_port,
                clock=acq_clock,
                duration=acq_duration,
                t0=t0_acquisition,
                acq_channel=acq_channel,
                acq_index=acq_index,
                bin_mode=bin_mode,
            )
        )
    elif acq_protocol == "Timetag":
        # Add time_source and time_ref to the dict only if they are not None, so that
        # they do not override operation defaults (these variables are passed as an
        # unpacked **dict below).
        timetag_args = {}
        if acq_time_source is not None:
            timetag_args["time_source"] = acq_time_source
        if acq_time_ref is not None:
            timetag_args["time_ref"] = acq_time_ref
        device_op.add_acquisition(
            Timetag(
                port=acq_port,
                clock=acq_clock,
                duration=acq_duration,
                t0=t0_acquisition,
                acq_channel=acq_channel,
                acq_index=acq_index,
                bin_mode=bin_mode,
                **timetag_args,
            )
        )
    elif acq_protocol == "TimetagTrace":
        timetag_args = {}
        if acq_time_ref is not None:
            timetag_args["time_ref"] = acq_time_ref
        device_op.add_acquisition(
            TimetagTrace(
                port=acq_port,
                clock=acq_clock,
                duration=acq_duration,
                t0=t0_acquisition,
                acq_channel=acq_channel,
                acq_index=acq_index,
                bin_mode=bin_mode,
                **timetag_args,
            )
        )
    else:
        raise NotImplementedError(
            f"Acquisition protocol '{acq_protocol}' is not supported. "
            f"Currently, only 'TriggerCount' and 'Trace' are accepted."
        )

    return device_op
