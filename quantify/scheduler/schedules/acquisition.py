# -----------------------------------------------------------------------------
# Description:    Enums for quantify-scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from quantify.scheduler.types import Schedule
from quantify.scheduler.pulse_library import SquarePulse, IdlePulse
from quantify.scheduler.acquisition_library import Trace, SSBIntegrationComplex
from quantify.scheduler.resources import ClockResource


def raw_trace_schedule(
    port: str,
    clock: str,
    integration_time: float,
    spec_pulse_amp: float,
    frequency: float,
    init_duration: int = 1e-6,
):
    """
    Generate a schedule to perform raw trace acquisition.

    Parameters
    ----------
    port
        location on the chip where the spec pulse should be applied and the
        signal interpreted.
    clock
        name of the location in frequency space where to apply the pulse
        and interpret the signal.
    integration_time
        time in seconds to integrate
    spec_pulse_amp
        amplitude of the spectroscopy pulse in Volt.
    frequency
        frequency to which to set the clock.
    """
    sched = Schedule("Raw trace acquisition")
    sched.add_resource(ClockResource(name=clock, freq=frequency))

    sched.add(
        IdlePulse(duration=init_duration),
    )

    # start = 0ns           duration = integration_time
    acq = sched.add(
        Trace(
            duration=integration_time,
            port=port,
            acq_index=0,
            acq_channel=0,
        ),
    )

    # start = 0ns + 500ns   duration = 100ns
    sched.add(
        SquarePulse(
            duration=100e-9,
            amp=spec_pulse_amp,
            port=port,
            clock=clock,
        ),
        ref_op=acq,
        ref_pt="start",
        rel_time=integration_time + 500e-9,
    )

    return sched


def ssb_integration_complex_schedule(
    port: str,
    clock: str,
    integration_time: float,
    spec_pulse_amp: float,
    frequency: float,
    init_duration: int = 1e-6,
):
    """
    Generate a schedule to perform Single Sideband integration acquisition
    using complex numbers.

    Parameters
    ----------
    port
        location on the chip where the spec pulse should be applied and the
        signal interpreted.
    clock
        name of the location in frequency space where to apply the pulse
        and interpret the signal.
    integration_time
        time in seconds to integrate
    spec_pulse_amp
        amplitude of the spectroscopy pulse in Volt.
    frequency
        frequency to which to set the clock.
    """
    sched = Schedule("SSBIntegrationComplex acquisition")
    sched.add_resource(ClockResource(name=clock, freq=frequency))

    sched.add(
        IdlePulse(duration=init_duration),
    )

    # start = 0ns           duration = integration_time
    acq0 = sched.add(
        SSBIntegrationComplex(
            duration=integration_time,
            port=port,
            clock=clock,
            acq_index=0,
            acq_channel=0,
        ),
    )

    # start = 0ns + 500ns   duration = 100ns
    sched.add(
        SquarePulse(
            duration=100e-9,
            amp=spec_pulse_amp,
            port=port,
            clock=clock,
        ),
        ref_op=acq0,
        ref_pt="start",
        rel_time=500e-9,
    )

    # start = 0ns           duration = integration_time
    acq1 = sched.add(
        SSBIntegrationComplex(
            duration=integration_time,
            port=port,
            clock=clock,
            acq_index=0,
            acq_channel=0,
        ),
    )

    # start = 0ns + 500ns   duration = 100ns
    sched.add(
        SquarePulse(
            duration=100e-9,
            amp=-spec_pulse_amp,
            port=port,
            clock=clock,
        ),
        ref_op=acq1,
        ref_pt="start",
        rel_time=500e-9,
    )

    return sched
