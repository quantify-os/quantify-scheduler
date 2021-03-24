"""
Module containing schedules for common timedomain experiments such as a Rabi and T1 measurement.
"""
from quantify.scheduler.types import Schedule
from quantify.scheduler.pulse_library import SquarePulse, IdlePulse
from quantify.scheduler.acquisition_library import SSBIntegrationComplex
from quantify.scheduler.resources import ClockResource



def rabi_sched(
    pulse_amp: float,
    pulse_duration: float,
    frequency: float,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    buffer_time: float = 18e-6,  # min based on QRM repetition rate.
) -> Schedule:
    pass



def rabi_pulse_sched(
    reset_duration: float,
    mw_G_amp: float,
    mw_D_amp: float,
    mw_frequency: float,
    mw_clock: str,
    mw_port: str,
    mw_pulse_duration: float,
    ro_pulse_amp: float,
    ro_pulse_duration: float,
    ro_pulse_delay: float,
    ro_pulse_port: str,
    ro_pulse_clock: str,
    ro_pulse_frequency: float,
    ro_integration_time: float,
):
    """
    Generete a schedule for performing a Rabi experiment.

    Parameters
    ----------
    reset_duration
        time it takes for the qubit to initialize.



    """
    sched = Schedule("Rabi schedule (pulse)")
    sched.add_resource(ClockResource(name=mw_clock, freq=mw_frequency))
    sched.add_resource(ClockResource(name=ro_pulse_clock, freq=ro_pulse_frequency))

    # minimum sequence duration
    sched.add(
        IdlePulse(duration=reset_duration),
    )
    # QRM can only start acquisition every 17 microseconds this should be included in the backend

    sched.add(
        DRAGPulse(
            duration=mw_pulse_duration,
            G_amp=mw_G_amp,
            D_amp=mw_D_amp,
            port=mw_port,
            clock=mw_clock,
            phase=0,
        ),
        label="Rabi_pulse",
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
        rel_time=1e-6 + ro_pulse_duration / 2,
        label="acquisition",
    )

    return sched
