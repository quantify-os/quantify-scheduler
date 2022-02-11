# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
A module containing factory functions for measurements on the quantum-device layer.

These factories are used to take a parametrized representation of on a operation
and use that to create an instance of the operation itself.
"""
from typing import Union

from quantify_scheduler import Operation
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.acquisition_library import (
    SSBIntegrationComplex,
    Trace,
)
from quantify_scheduler.operations.pulse_library import SquarePulse

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
    acq_protocol: str,
    pulse_type: str = "SquarePulse",
    bin_mode: Union[BinMode, None] = BinMode.AVERAGE,
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
        device_op = SquarePulse(
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

    if acq_protocol == "Trace":
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
    return device_op
