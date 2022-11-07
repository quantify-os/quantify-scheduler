# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
A module containing factory functions for measurements on the quantum-device layer.

These factories are used to take a parametrized representation of on a operation
and use that to create an instance of the operation itself.
"""
from typing import Literal, Union

from quantify_scheduler import Operation
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.acquisition_library import (
    SSBIntegrationComplex,
    Trace,
    TriggerCount,
)
from quantify_scheduler.operations.pulse_library import SquarePulse, ResetClockPhase


# pylint: disable=too-many-arguments
def dispersive_measurement(
    pulse_amp: float,
    pulse_duration: float,
    port: str,
    clock: str,
    acq_duration: float,
    acq_delay: float,
    acq_channel: int,
    acq_index: int,
    acq_protocol: Literal["SSBIntegrationComplex", "Trace", None],
    pulse_type: Literal["SquarePulse"] = "SquarePulse",
    bin_mode: Union[BinMode, None] = BinMode.AVERAGE,
    acq_protocol_default: Literal[
        "SSBIntegrationComplex", "Trace"
    ] = "SSBIntegrationComplex",
    reset_clock_phase: bool = True,
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

    if acq_protocol == "SSBIntegrationComplex":
        # readout pulse
        device_op.add_acquisition(
            SSBIntegrationComplex(
                duration=acq_duration,
                t0=acq_delay,
                acq_channel=acq_channel,
                acq_index=acq_index,
                port=port,
                clock=clock,
                bin_mode=bin_mode,
            )
        )
    elif acq_protocol == "Trace":
        device_op.add_acquisition(
            Trace(
                clock=clock,
                duration=acq_duration,
                t0=acq_delay,
                acq_channel=acq_channel,
                acq_index=acq_index,
                port=port,
            )
        )
    else:
        raise ValueError(f'Acquisition protocol "{acq_protocol}" is not supported.')

    return device_op


def optical_measurement(
    pulse_amplitude: float,
    pulse_duration: float,
    pulse_port: str,
    pulse_clock: str,
    acq_duration: float,
    acq_delay: float,
    acq_port: str,
    acq_clock: str,
    acq_channel: int,
    acq_index: int,
    bin_mode: Union[BinMode, None],
    acq_protocol: Literal["Trace", "TriggerCount"],
    acq_protocol_default: Literal["Trace", "TriggerCount"],
    pulse_type: Literal["SquarePulse"],
) -> Operation:
    # pylint: disable=too-many-locals
    """Generator function for a standard optical measurement.

    An optical measurement generates a square pulse in the optical range and uses the
    Trace acquisition to return the output of a photon detector as a function of time.
    Alternatively, the TriggerCount counts the number of photons that are collected.


    Parameters
    ----------
    pulse_amplitude
        Amplitude of the generated pulse
    pulse_duration
        Duration of the generated pulse
    pulse_port
        Port name, where the pulse is applied
    pulse_clock
        Clock name of the generated pulse
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
        Acquisition channel of the device element
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
    acq_protocol_default, optional
        Acquisition protocol if ``acq_protocol`` is None, by default "TriggerCount"
    pulse_type, optional
        Shape of the pulse to be generated, by default "SquarePulse"

    Returns
    -------
        Operation with the generated pulse and acquisition

    Raises
    ------
    NotImplementedError
        If an unknown ``pulse_type`` or ``acq_protocol`` are used.
    """
    # ensures default argument is used if not specified at gate level.
    # ideally, this input would not be accepted, but this is a workaround for #267
    if bin_mode is None:
        bin_mode = BinMode.APPEND

    # If acq_delay >= 0, the pulse starts at 0 and the acquisition at acq_delay
    # If acq_delay < 0, the pulse starts at -acq_delay and the acquisition at 0
    t0_pulse = max(0, -acq_delay)
    t0_acquisition = max(0, acq_delay)

    device_op = Operation("OpticalMeasurement")
    if pulse_type == "SquarePulse":
        device_op.add_pulse(
            SquarePulse(
                amp=pulse_amplitude,
                duration=pulse_duration,
                port=pulse_port,
                clock=pulse_clock,
                t0=t0_pulse,
            )
        )
    else:
        raise NotImplementedError(
            f'Invalid pulse_type "{pulse_type}" specified as argument to '
            'optical_measurement. Currently, only "SquarePulse" is accepted. '
            "Please correct your device config."
        )

    if acq_protocol is None:
        acq_protocol = acq_protocol_default

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
            f'Acquisition protocol "{acq_protocol}" is not supported. '
            'Currently, only "TriggerCount" and "Trace" are accepted.'
        )

    return device_op
