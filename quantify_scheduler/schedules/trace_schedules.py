# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
from quantify_scheduler import Schedule
from quantify_scheduler.operations.acquisition_library import Trace
from quantify_scheduler.operations.pulse_library import IdlePulse, SquarePulse
from quantify_scheduler.resources import ClockResource


# pylint: disable=too-many-arguments
def trace_schedule(
    pulse_amp: float,
    pulse_duration: float,
    pulse_delay: float,
    frequency: float,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    init_duration: int = 200e-6,
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a schedule to perform raw trace acquisition.

    Parameters
    ----------
    pulse_amp :
        The amplitude of the pulse in Volt.
    pulse_duration
        The duration of the pulse in seconds.
    pulse_delay :
        The pulse delay in seconds.
    frequency :
        The frequency of the pulse
        and of the data acquisition in Hertz.
    acquisition_delay
        The start of the data acquisition with respect to
        the start of the pulse in seconds.
    integration_time :
        The time in seconds to integrate.
    port :
        The location on the device where the
        pulse should be applied.
    clock :
        The reference clock used to track the
        pulse frequency.
    init_duration :
        The relaxation time or dead time.
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        The Raw Trace acquisition Schedule.
    """
    schedule = Schedule("Raw trace acquisition", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    schedule.add(IdlePulse(duration=init_duration), label="Dead time")

    pulse = schedule.add(
        SquarePulse(
            duration=pulse_duration,
            amp=pulse_amp,
            port=port,
            clock=clock,
        ),
        label="trace_pulse",
        rel_time=pulse_delay,
    )

    schedule.add(
        Trace(
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

    return schedule


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def two_tone_trace_schedule(
    qubit_pulse_amp: float,
    qubit_pulse_duration: float,
    qubit_pulse_frequency: float,
    qubit_pulse_port: str,
    qubit_pulse_clock: str,
    ro_pulse_amp: float,
    ro_pulse_duration: float,
    ro_pulse_delay: float,
    ro_pulse_port: str,
    ro_pulse_clock: str,
    ro_pulse_frequency: float,
    ro_acquisition_delay: float,
    ro_integration_time: float,
    init_duration: float = 200e-6,
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a schedule for performing a two-tone raw trace acquisition.

    Parameters
    ----------
    qubit_pulse_amp
        The amplitude of the pulse in Volt.
    qubit_pulse_duration
        The duration of the pulse in seconds.
    qubit_pulse_frequency
        The pulse frequency in Hertz.
    qubit_pulse_port
        The location on the device where the
        qubit pulse should be applied.
    spec_pulse_clock
        The reference clock used to track the
        pulse frequency.
    ro_pulse_amp
        The amplitude of the readout pulse in Volt.
    ro_pulse_duration
        The duration of the readout pulse in seconds.
    ro_pulse_delay
        The time between the end of the pulse and the start
        of the readout pulse.
    ro_pulse_port
        The location on the device where the
        readout pulse should be applied.
    ro_pulse_clock
        The reference clock used to track the
        readout pulse frequency.
    ro_pulse_frequency
        The readout pulse frequency in Hertz.
    ro_acquisition_delay
        The start of the data acquisition with respect to
        the start of the pulse in seconds.
    ro_integration_time
        The integration time of the data acquisition in seconds.
    init_duration :
        The relaxation time or dead time.
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        The Two-tone Trace acquisition Schedule.
    """
    schedule = Schedule("Two-tone Trace acquisition", repetitions)
    schedule.add_resource(
        ClockResource(name=qubit_pulse_clock, freq=qubit_pulse_frequency)
    )
    schedule.add_resource(ClockResource(name=ro_pulse_clock, freq=ro_pulse_frequency))

    schedule.add(
        IdlePulse(duration=init_duration),
        label="Reset",
    )

    schedule.add(
        SquarePulse(
            duration=qubit_pulse_duration,
            amp=qubit_pulse_amp,
            port=qubit_pulse_port,
            clock=qubit_pulse_clock,
        ),
        label="qubit_pulse",
    )

    ro_pulse = schedule.add(
        SquarePulse(
            duration=ro_pulse_duration,
            amp=ro_pulse_amp,
            port=ro_pulse_port,
            clock=ro_pulse_clock,
        ),
        label="readout_pulse",
        rel_time=ro_pulse_delay,
    )

    schedule.add(
        Trace(
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

    return schedule
