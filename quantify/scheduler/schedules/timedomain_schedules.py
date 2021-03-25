"""
Module containing schedules for common timedomain experiments such as a Rabi and T1 measurement.
"""
import numpy as np
from quantify.scheduler.types import Schedule
from quantify.scheduler.pulse_library import SquarePulse, IdlePulse, DRAGPulse
from quantify.scheduler.acquisition_library import SSBIntegrationComplex
from quantify.scheduler.resources import ClockResource


def rabi_sched(
    pulse_amp: float,
    pulse_duration: float,
    qubit: str,
    frequency: float,
    port: str,
    clock: str,
) -> Schedule:
    """
    Generete a schedule for performing a Rabi experiment.

        Parameters
        ----------
        reset_duration
            time it takes for the qubit to initialize.
    """
    schedule = Schedule("Rabi schedule")
    sched.add_resource(ClockResource(name=mw_clock, freq=mw_frequency))

    sched.add(Reset(qubit))
    sched.add(
        DRAGPulse(
            duration=mw_pulse_duration,
            G_amp=mw_G_amp,
            D_amp=0,
            port=mw_port,
            clock=mw_clock,
            phase=0,
        ),
        label="Rabi_pulse",
    )
    schedule.add(Measure(qubit))


def t1_sched(
    times: np.ndarray,
    qubit: str,
) -> Schedule:
    """
    Generete a schedule for performing a T1 experiment to measure the qubit relaxation time.

    Parameters
    ----------
    qubit
        the name of the qubit e.g., "q0" to perform the T1 experiment on.
    """
    schedule = Schedule("T1 schedule")
    sched.add_resource(ClockResource(name=mw_clock, freq=mw_frequency))

    for tau in times:
        sched.add(Reset(qubit))
        sched.add(X180(qubit))
        schedule.add(Measure(qubit), ref_pt="start", rel_time=tau)


def ramsey_sched(
    times: np.ndarray,
    qubit: str,
) -> Schedule:
    """
    Generete a schedule for performing a Ramsey experiment to measure the dephasing time T2*.

    Parameters
    ----------
    qubit
        the name of the qubit e.g., "q0" to perform the T1 experiment on.
    """
    schedule = Schedule("Ramsey schedule")
    sched.add_resource(ClockResource(name=mw_clock, freq=mw_frequency))

    for tau in times:
        sched.add(Reset(qubit))
        sched.add(X90(qubit))
        # to be added artificial detuning
        sched.add(Rxy(qubit, theta=90, phi=0), ref_pt="start", rel_time=tau)
        schedule.add(Measure(qubit))


def echo_sched(
    times: np.ndarray,
    qubit: str,
) -> Schedule:
    """
    Generete a schedule for performing an Echo experiment to measure the qubit echo-dephasing time T2.

    Parameters
    ----------
    qubit
        the name of the qubit e.g., "q0" to perform the T1 experiment on.
    """
    schedule = Schedule("Echo schedule")
    sched.add_resource(ClockResource(name=mw_clock, freq=mw_frequency))

    for tau in times:
        sched.add(Reset(qubit))
        sched.add(X90(qubit))
        sched.add(X180(qubit), ref_pt="start", rel_time=tau / 2)
        sched.add(X90(qubit), ref_pt="start", rel_time=tau / 2)
        schedule.add(Measure(qubit))


def rabi_pulse_sched(
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
    ro_acquisition_delay: float,
    ro_integration_time: float,
    reset_duration: float,
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
        label='qubit reset'
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
        rel_time=ro_acquisition_delay,
        label="acquisition",
    )

    return sched
