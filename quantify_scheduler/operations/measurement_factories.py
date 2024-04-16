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
from typing import List, Literal, Optional, Hashable

import numpy as np

from quantify_scheduler import Operation
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.acquisition_library import (
    NumericalSeparatedWeightedIntegration,
    NumericalWeightedIntegration,
    NumericalWeightedIntegrationComplex,
    SSBIntegrationComplex,
    ThresholdedAcquisition,
    Trace,
    TriggerCount,
)
from quantify_scheduler.operations.pulse_library import (
    SquarePulse,
    ReferenceMagnitude,
    ResetClockPhase,
)


def dispersive_measurement(
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
    reference_magnitude: Optional[ReferenceMagnitude] = None,
    acq_weights_a: List[complex] | np.ndarray | None = None,
    acq_weights_b: List[complex] | np.ndarray | None = None,
    acq_weights_sampling_rate: float | None = None,
    feedback_trigger_label: Optional[str] = None,
    acq_rotation: float | None = None,
    acq_threshold: float | None = None,
) -> Operation:
    """
    Generator function for a standard dispersive measurement.

    A dispersive measurement (typically) exists of a pulse being applied to the device
    followed by an acquisition protocol to interpret the signal coming back from the
    device.
    """
    # ensures default argument is used if not specified at gate level.
    # ideally, this input would not be accepted, but this is a workaround for #267
    if bin_mode is None:
        bin_mode = BinMode.AVERAGE

    # Note that the funny structure here comes from the fact that the measurement
    # is a composite operation. We need to either introduce some level of nesting
    # in the structure of arguments (to re-use our custom structure), or just keep
    # this as a simple piece of code and accept that different protocols (e.g.,
    # using different measurement pulses) would require a different generator function.

    if pulse_type == "SquarePulse":
        pulse_op = SquarePulse(
            amp=pulse_amp,
            duration=pulse_duration,
            port=port,
            clock=clock,
            reference_magnitude=reference_magnitude,
        )
    else:
        # here we need to add support for SoftSquarePulse
        raise NotImplementedError(
            f'Invalid pulse_type "{pulse_type}" specified as argument to '
            + "dispersive_measurement. Currently dispersive_measurement only"
            + ' allows "SquarePulse". Please correct your device config.'
        )

    if reset_clock_phase:
        device_op = ResetClockPhase(clock=clock)
        device_op.add_pulse(pulse_op)
    else:
        device_op = pulse_op

    if acq_protocol is None:
        acq_protocol = acq_protocol_default

    if acq_channel_override is not None:
        acq_channel = acq_channel_override

    if acq_protocol == "SSBIntegrationComplex":
        # readout pulse
        device_op.add_acquisition(
            SSBIntegrationComplex(
                port=port,
                clock=clock,
                duration=acq_duration,
                acq_channel=acq_channel,
                acq_index=acq_index,
                bin_mode=bin_mode,
                t0=acq_delay,
            )
        )
    elif acq_protocol in (
        "NumericalSeparatedWeightedIntegration",
        "NumericalWeightedIntegration",
        "NumericalWeightedIntegrationComplex",
    ):
        if (
            acq_weights_a is None
            or acq_weights_b is None
            or acq_weights_sampling_rate is None
        ):
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
            device_op.add_acquisition(
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
                )
            )
        elif acq_protocol == "NumericalWeightedIntegration":
            device_op.add_acquisition(
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
                )
            )
        elif acq_protocol == "NumericalWeightedIntegrationComplex":
            device_op.add_acquisition(
                NumericalWeightedIntegrationComplex(
                    port=port,
                    clock=clock,
                    weights_a=acq_weights_a,
                    weights_b=acq_weights_b,
                    weights_sampling_rate=acq_weights_sampling_rate,
                    acq_channel=acq_channel,
                    acq_index=acq_index,
                    bin_mode=bin_mode,
                    t0=acq_delay,
                )
            )
    elif acq_protocol == "ThresholdedAcquisition":
        device_op.add_acquisition(
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
            )
        )
    elif acq_protocol == "Trace":
        device_op.add_acquisition(
            Trace(
                port=port,
                clock=clock,
                duration=acq_duration,
                acq_channel=acq_channel,
                acq_index=acq_index,
                t0=acq_delay,
            )
        )
    else:
        raise ValueError(f'Acquisition protocol "{acq_protocol}" is not supported.')

    return device_op


def optical_measurement(
    pulse_amplitudes: List[float],
    pulse_durations: List[float],
    pulse_ports: List[str],
    pulse_clocks: List[str],
    acq_duration: float,
    acq_delay: float,
    acq_port: str,
    acq_clock: str,
    acq_channel: Hashable,
    acq_channel_override: Hashable | None,
    acq_index: int,
    bin_mode: BinMode | None,
    acq_protocol: Literal["Trace", "TriggerCount"] | None,
    acq_protocol_default: Literal["Trace", "TriggerCount"],
    pulse_type: Literal["SquarePulse"],
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
    if (
        not len(pulse_amplitudes)
        == len(pulse_durations)
        == len(pulse_ports)
        == len(pulse_clocks)
    ):
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
            )
        )
    else:
        raise NotImplementedError(
            f"Acquisition protocol '{acq_protocol}' is not supported. "
            f"Currently, only 'TriggerCount' and 'Trace' are accepted."
        )

    return device_op
