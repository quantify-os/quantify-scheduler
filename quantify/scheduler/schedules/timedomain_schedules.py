"""
Module containing schedules for common timedomain experiments such as a Rabi and T1 measurement.
"""
import numpy as np
from quantify.scheduler.types import Schedule
from quantify.scheduler.pulse_library import SquarePulse, IdlePulse, DRAGPulse
from quantify.scheduler.gate_library import Rxy, X, X90, Reset, Measure
from quantify.scheduler.acquisition_library import SSBIntegrationComplex
from quantify.scheduler.resources import ClockResource

# pylint: disable=too-many-arguments
def rabi_sched(
    pulse_amplitude: float,
    pulse_duration: float,
    frequency: float,
    qubit: str,
    port: str = None,
    clock: str = None,
) -> Schedule:
    """
    Generete a schedule for performing a Rabi using a Gaussian pulse.

    Parameters
    ----------
    pulse_amplitude
        amplitude of the Rabi pulse in V.
    pulse_duration
        duration of the gaussian shaped Rabi pulse. Corresponds to 4 sigma.
    frequency
        frequency of the qubit 01 transition.
    qubit
        the qubit on which to perform a Rabi experiment.
    port
        location on the chip where the Rabi pulse should be applied.
        if set to None, will use the naming convention "<qubit>:mw" to infer the port.
    clock
        name of the location in frequency space where to apply the Rabi pulse.
        if set to None, will use the naming convention "<qubit>.01" to infer the clock.

    """
    schedule = Schedule("Rabi schedule")
    if port is None:
        port = f"{qubit}:mw"
    if clock is None:
        clock = f"{qubit}:01"

    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    schedule.add(Reset(qubit))
    schedule.add(
        DRAGPulse(
            duration=pulse_duration,
            G_amp=pulse_amplitude,
            D_amp=0,
            port=port,
            clock=clock,
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

    A T1 experiment consists of
        pi -- Idle(tau) - Measure

    Parameters
    ----------
    times
        an array of wait times tau between the pi-pulse and the measurement.
    qubit
        the name of the qubit e.g., "q0" to perform the T1 experiment on.
    """
    schedule = Schedule("T1 schedule")
    for tau in times:
        schedule.add(Reset(qubit))
        schedule.add(X(qubit))
        schedule.add(Measure(qubit), ref_pt="start", rel_time=tau)
    return schedule


def ramsey_sched(
    times: np.ndarray,
    qubit: str,
) -> Schedule:
    """
    Generete a schedule for performing a Ramsey experiment to measure the dephasing time :math:`T_2^{\\star}`.

    A Ramsey experiment consists of
        pi/2 -- Idle(tau) -- pi/2 - Measure

    Parameters
    ----------
    times
        an array of wait times tau between the pi/2 pulses.
    qubit
        the name of the qubit e.g., "q0" to perform the Ramsey experiment on.
    """
    schedule = Schedule("Ramsey schedule")

    for tau in times:
        schedule.add(Reset(qubit))
        schedule.add(X90(qubit))
        # to be added artificial detuning
        schedule.add(Rxy(theta=90, phi=0, qubit=qubit), ref_pt="start", rel_time=tau)
        schedule.add(Measure(qubit))
    return schedule


def echo_sched(
    times: np.ndarray,
    qubit: str,
) -> Schedule:
    """
    Generete a schedule for performing an Echo experiment to measure the qubit echo-dephasing time :math:`T_2^{E}`.

    An Echo experiment consists of
        pi/2 -- Idle(tau/2) -- pi -- Idle(tau/2) -- pi/2 - Measure


    Parameters
    ----------
    qubit
        the name of the qubit e.g., "q0" to perform the Echo experiment on.
    times
        an array of wait times between the
    """
    schedule = Schedule("Echo schedule")
    for tau in times:
        schedule.add(Reset(qubit))
        schedule.add(X90(qubit))
        schedule.add(X(qubit), ref_pt="start", rel_time=tau / 2)
        schedule.add(X90(qubit), ref_pt="start", rel_time=tau / 2)
        schedule.add(Measure(qubit))
    return schedule


def allxy_sched(qubit: str) -> Schedule:
    """
    Generete a schedule for performing an AllXY experiment.

    An AllXY experiment consists of
        Rxy[0] -- Rxy[1] -- Measure

    for a specific set of combinations of x90, x180, y90, y180 and idle rotations.


    Parameters
    ----------
    qubit
        the name of the qubit e.g., "q0" to perform the experiment on.
    """

    # all combinations of Idle, X90, Y90, X180 and Y180 gates that are part of the AllXY experiment
    allXY_combinations = [
        [(0, 0), (0, 0)],
        [(180, 0), (180, 0)],
        [(180, 0), (180, 0)],
        [(180, 0), (180, 90)],
        [(180, 90), (180, 0)],
        [(90, 0), (0, 0)],
        [(90, 90), (0, 0)],
        [(90, 0), (90, 90)],
        [(90, 90), (90, 0)],
        [(90, 0), (180, 90)],
        [(90, 90), (180, 0)],
        [(180, 0), (90, 90)],
        [(180, 90), (90, 0)],
        [(90, 0), (180, 0)],
        [(180, 0), (90, 0)],
        [(90, 90), (180, 90)],
        [(180, 90), (90, 90)],
        [(180, 0), (0, 0)],
        [(180, 90), (0, 0)],
        [(90, 0), (90, 0)],
        [(90, 90), (90, 90)],
    ]
    schedule = Schedule("AllXY schedule")
    for p0, p1 in allXY_combinations:
        schedule.add(Reset(qubit))
        schedule.add(Rxy(qubit=qubit, theta=p0[0], phi=p0[1]))
        schedule.add(Rxy(qubit=qubit, theta=p1[0], phi=p1[1]))
        schedule.add(Measure(qubit))
    return schedule


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
    schedule = Schedule("Rabi schedule (pulse)")
    schedule.add_resource(ClockResource(name=mw_clock, freq=mw_frequency))
    schedule.add_resource(ClockResource(name=ro_pulse_clock, freq=ro_pulse_frequency))

    # minimum sequence duration
    schedule.add(IdlePulse(duration=reset_duration), label="qubit reset")
    # QRM can only start acquisition every 17 microseconds this should be included in the backend

    schedule.add(
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

    return schedule
