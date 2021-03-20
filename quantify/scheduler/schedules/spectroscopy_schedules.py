from quantify.scheduler.types import Schedule
from quantify.scheduler.pulse_library import SquarePulse, IdlePulse
from quantify.scheduler.acquisition_library import SSBIntegrationComplex
from quantify.scheduler.resources import ClockResource

def heterodyne_spec_sched(
    pulse_amp: float,
    pulse_duration: float,
    port: str,
    clock: str,
    frequency: float,
    acquisition_delay: float,
    integration_time: float,
    buffer_time: float = 18e-6, # min based on QRM repetition rate.
):
    """
    Generate a schedule for performing heterodyne spectroscopy.

    Parameters
    ----------
    port
        location on the chip where the pulse should be applied and the
        signal interpreted.
    clock
        name of the location in frequency space where to apply the pulse and interpret the signal.
    integration_time
        time in seconds to integrate
    pulse_amp
        amplitude of the spectroscopy pulse in Volt.
    frequency
        frequency to which to set the clock.
    """
    sched = Schedule("Heterodyne spectroscopy")
    sched.add_resource(ClockResource(name=clock, freq=frequency))

    sched.add(
        IdlePulse(duration=buffer_time),
        label="buffer",
    )  # QRM can only start acquisition every 17 microseconds

    pulse = sched.add(
        SquarePulse(
            duration=pulse_duration,
            amp=pulse_amp,
            port=port,
            clock=clock,
        ),
        label="Spec_pulse",
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
        label='Acquisition'
    )

    return sched


def pulsed_spec_sched(
    spec_pulse_amp: float,
    spec_pulse_duration: float,
    spec_pulse_port: str,
    spec_pulse_clock: str,
    spec_pulse_frequency: float,
    ro_pulse_amp: float,
    ro_pulse_duration: float,
    ro_pulse_delay: float,
    ro_pulse_port: str,
    ro_pulse_clock: str,
    ro_pulse_frequency: float,
    ro_acquisition_delay: float,
    ro_integration_time: float,
    buffer_time: float = 18e-6, # min based on QRM repetition rate.

):
    """
    Generate a schedule for performing heterodyne spectroscopy.

    Parameters
    ----------
    port
        location on the chip where the spec pulse should be applied and the
        signal interpreted.
    clock
        name of the location in frequency space where to apply the pulse and interpret the signal.
    integration_time
        time in seconds to integrate
    spec_pulse_amp
        amplitude of the spectroscopy pulse in Volt.
    frequency
        frequency to which to set the clock.
    """
    sched = Schedule("Pulsed spectroscopy")
    sched.add_resource(ClockResource(name=spec_pulse_clock, freq=spec_pulse_frequency))
    sched.add_resource(ClockResource(name=ro_pulse_clock, freq=ro_pulse_frequency))

    # wait time between different repetitions of the schedule.
    sched.add(
        IdlePulse(duration=buffer_time),
    )

    sched.add(
        SquarePulse(
            duration=spec_pulse_duration,
            amp=spec_pulse_amp,
            port=spec_pulse_port,
            clock=spec_pulse_clock,
        ),
        label="spec_pulse",
        ref_pt="end",
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
