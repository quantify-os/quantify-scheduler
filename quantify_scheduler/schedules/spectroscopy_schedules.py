# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Module containing schedules for common spectroscopy experiments.
"""
from __future__ import annotations

from typing import Optional

from quantify_scheduler import Schedule
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.pulse_library import IdlePulse, SquarePulse
from quantify_scheduler.resources import ClockResource


# pylint: disable=too-many-arguments
def heterodyne_spec_sched(
    pulse_amp: float,
    pulse_duration: float,
    frequency: float,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    init_duration: float = 10e-6,
    repetitions: int = 1,
    port_out: Optional[str] = None,
) -> Schedule:
    """
    Generate a schedule for performing heterodyne spectroscopy.

    Parameters
    ----------
    pulse_amp
        amplitude of the spectroscopy pulse in Volt.
    pulse_duration
        duration of the spectroscopy pulse in seconds.
    frequency
        frequency of the spectroscopy pulse and of the data acquisition in Hertz.
    acquisition_delay
        start of the data acquisition with respect to the start of the spectroscopy
        pulse in seconds.
    integration_time
        integration time of the data acquisition in seconds.
    port
        Location on the device where the acquisition is performed.
    clock
        reference clock used to track the spectroscopy frequency.
    init_duration :
        The relaxation time or dead time.
    repetitions
        The amount of times the Schedule will be repeated.
    port_out:
        Output port on the device where the pulse should be applied. If `None`, then use the same as `port`.
    """
    sched = Schedule("Heterodyne spectroscopy", repetitions)
    sched.add_resource(ClockResource(name=clock, freq=frequency))

    sched.add(IdlePulse(duration=init_duration), label="buffer")

    if port_out is None:
        port_out = port

    pulse = sched.add(
        SquarePulse(
            duration=pulse_duration,
            amp=pulse_amp,
            port=port_out,
            clock=clock,
        ),
        label="spec_pulse",
    )

    sched.add(
        SSBIntegrationComplex(
            duration=integration_time,
            port=port,
            clock=clock,
            acq_index=0,
            acq_channel=0,
        ),
        ref_op=pulse,
        ref_pt="start",
        rel_time=acquisition_delay,
        label="acquisition",
    )

    return sched


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def two_tone_spec_sched(
    spec_pulse_amp: float,
    spec_pulse_duration: float,
    spec_pulse_frequency: float,
    spec_pulse_port: str,
    spec_pulse_clock: str,
    ro_pulse_amp: float,
    ro_pulse_duration: float,
    ro_pulse_delay: float,
    ro_pulse_port: str,
    ro_pulse_clock: str,
    ro_pulse_frequency: float,
    ro_acquisition_delay: float,
    ro_integration_time: float,
    init_duration: float = 10e-6,
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a schedule for performing two-tone spectroscopy.

    Parameters
    ----------
    spec_pulse_amp
        amplitude of the spectroscopy pulse in Volt.
    spec_pulse_duration
        duration of the spectroscopy pulse in seconds.
    spec_pulse_frequency
        frequency of the spectroscopy pulse in Hertz.
    spec_pulse_port
        location on the device where the spectroscopy pulse should be applied.
    spec_pulse_clock
        reference clock used to track the spectroscopy frequency.
    ro_pulse_amp
        amplitude of the readout (spectroscopy) pulse in Volt.
    ro_pulse_duration
        duration of the readout (spectroscopy) pulse in seconds.
    ro_pulse_delay
        time between the end of the spectroscopy pulse and the start of the readout
        (spectroscopy) pulse.
    ro_pulse_port
        location on the device where the readout (spectroscopy) pulse should be applied.
    ro_pulse_clock
        reference clock used to track the readout (spectroscopy) frequency.
    ro_pulse_frequency
        frequency of the spectroscopy pulse and of the data acquisition in Hertz.
    ro_acquisition_delay
        start of the data acquisition with respect to the start of the spectroscopy
        pulse in seconds.
    ro_integration_time
        integration time of the data acquisition in seconds.
    init_duration :
        The relaxation time or dead time.
    repetitions
        The amount of times the Schedule will be repeated.
    """
    sched = Schedule("Pulsed spectroscopy", repetitions)
    sched.add_resource(ClockResource(name=spec_pulse_clock, freq=spec_pulse_frequency))
    sched.add_resource(ClockResource(name=ro_pulse_clock, freq=ro_pulse_frequency))

    sched.add(
        IdlePulse(duration=init_duration),
        label="buffer",
    )

    sched.add(
        SquarePulse(
            duration=spec_pulse_duration,
            amp=spec_pulse_amp,
            port=spec_pulse_port,
            clock=spec_pulse_clock,
        ),
        label="spec_pulse",
    )

    ro_pulse = sched.add(
        SquarePulse(
            duration=ro_pulse_duration,
            amp=ro_pulse_amp,
            port=ro_pulse_port,
            clock=ro_pulse_clock,
        ),
        label="readout_pulse",
        rel_time=ro_pulse_delay,
    )

    sched.add(
        SSBIntegrationComplex(
            duration=ro_integration_time,
            port=ro_pulse_port,
            clock=ro_pulse_clock,
            acq_index=0,
            acq_channel=0,
        ),
        ref_op=ro_pulse,
        ref_pt="start",
        rel_time=ro_acquisition_delay,
        label="acquisition",
    )

    return sched
