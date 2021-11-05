# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Module containing schedules for common time domain experiments such as a Rabi and
T1 measurement.
"""
from typing import List, Union

import numpy as np
from typing_extensions import Literal

from quantify_scheduler import Schedule
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.gate_library import X90, Measure, Reset, Rxy, X
from quantify_scheduler.operations.pulse_library import (
    DRAGPulse,
    IdlePulse,
    SquarePulse,
)
from quantify_scheduler.resources import ClockResource


# pylint: disable=too-many-arguments
def rabi_sched(
    pulse_amp: Union[np.ndarray, float],
    pulse_duration: Union[np.ndarray, float],
    frequency: float,
    qubit: str,
    port: str = None,
    clock: str = None,
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a schedule for performing a Rabi using a Gaussian pulse.

    Schedule sequence
        .. centered:: Reset -- DRAG -- Measure

    Parameters
    ----------
    pulse_amp
        amplitude of the Rabi pulse in V.
    pulse_duration
        duration of the Gaussian shaped Rabi pulse. Corresponds to 4 sigma.
    frequency
        frequency of the qubit 01 transition.
    qubit
        the qubit on which to perform a Rabi experiment.
    port
        location on the chip where the Rabi pulse should be applied.
        if set to :code:`None`, will use the naming convention :code:`"<qubit>:mw"` to
        infer the port.
    clock
        name of the location in frequency space where to apply the Rabi pulse.
        if set to :code:`None`, will use the naming convention :code:`"<qubit>.01"` to
        infer the clock.
    repetitions
        The amount of times the Schedule will be repeated.
    """

    # ensure pulse_amplitude and pulse_duration are iterable.
    amps = np.asarray(pulse_amp)
    amps = amps.reshape(amps.shape or (1,))
    durations = np.asarray(pulse_duration)
    durations = durations.reshape(durations.shape or (1,))

    # either the shapes of the amp and duration must match or one of
    # them must be a constant floating point value.
    if len(amps) == 1:
        amps = np.ones(np.shape(durations)) * amps
    elif len(durations) == 1:
        durations = np.ones(np.shape(amps)) * durations
    elif len(durations) != len(amps):
        raise ValueError(
            f"Shapes of pulse_amplitude ({pulse_amp.shape}) and "
            f"pulse_duration ({pulse_duration.shape}) are incompatible."
        )

    if port is None:
        port = f"{qubit}:mw"
    if clock is None:
        clock = f"{qubit}.01"

    schedule = Schedule("Rabi", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    for i, (amp, duration) in enumerate(zip(amps, durations)):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(
            DRAGPulse(
                duration=duration,
                G_amp=amp,
                D_amp=0,
                port=port,
                clock=clock,
                phase=0,
            ),
            label=f"Rabi_pulse {i}",
        )
        # N.B. acq_channel is not specified
        schedule.add(Measure(qubit, acq_index=i), label=f"Measurement {i}")

    return schedule


def t1_sched(
    times: Union[np.ndarray, float],
    qubit: str,
    repetitions: int = 1,
) -> Schedule:
    # pylint: disable=line-too-long
    """
    Generate a schedule for performing a :math:`T_1` experiment to measure the qubit
    relaxation time.

    Schedule sequence
        .. centered:: Reset -- pi -- Idle(tau) -- Measure

    See section III.B.2. of :cite:t:`krantz_quantum_2019` for an explanation of the Bloch-Redfield
    model of decoherence and the :math:`T_1` experiment.

    Parameters
    ----------
    times
        an array of wait times tau between the pi-pulse and the measurement.
    qubit
        the name of the qubit e.g., :code:`"q0"` to perform the T1 experiment on.
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        An experiment schedule.

    """
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("T1", repetitions)
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X(qubit), label=f"pi {i}")
        schedule.add(
            Measure(qubit, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {i}",
        )
    return schedule


def ramsey_sched(
    times: Union[np.ndarray, float],
    qubit: str,
    artificial_detuning: float = 0,
    repetitions: int = 1,
) -> Schedule:
    # pylint: disable=line-too-long
    r"""
    Generate a schedule for performing a Ramsey experiment to measure the
    dephasing time :math:`T_2^{\star}`.

    Schedule sequence
        .. centered:: Reset -- pi/2 -- Idle(tau) -- pi/2 -- Measure

    See section III.B.2. of :cite:t:`krantz_quantum_2019` for an explanation of the Bloch-Redfield
    model of decoherence and the Ramsey experiment.

    Parameters
    ----------
    times
        an array of wait times tau between the pi/2 pulses.
    artificial_detuning
        frequency in Hz of the software emulated, or `artificial` qubit detuning, which is
        implemented by changing the phase of the second pi/2 (recovery) pulse. The
        artificial detuning changes the observed frequency of the Ramsey oscillation,
        which can be useful to distinguish a slow oscillation due to a small physical
        detuning from the decay of the dephasing noise.
    qubit
        the name of the qubit e.g., :code:`"q0"` to perform the Ramsey experiment on.
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        An experiment schedule.

    """
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("Ramsey", repetitions)

    if isinstance(times, float):
        times = [times]

    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X90(qubit))

        # the phase of the second pi/2 phase progresses to propagate
        recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
        schedule.add(
            Rxy(theta=90, phi=recovery_phase, qubit=qubit), ref_pt="start", rel_time=tau
        )
        schedule.add(Measure(qubit, acq_index=i), label=f"Measurement {i}")
    return schedule


def echo_sched(
    times: Union[np.ndarray, float],
    qubit: str,
    repetitions: int = 1,
) -> Schedule:
    # pylint: disable=line-too-long
    """
    Generate a schedule for performing an Echo experiment to measure the qubit
    echo-dephasing time :math:`T_2^{E}`.

    Schedule sequence
        .. centered:: Reset -- pi/2 -- Idle(tau/2) -- pi -- Idle(tau/2) -- pi/2 -- Measure

    See section III.B.2. of :cite:t:`krantz_quantum_2019` for an explanation of the Bloch-Redfield
    model of decoherence and the echo experiment.

    Parameters
    ----------
    qubit
        the name of the qubit e.g., "q0" to perform the echo experiment on.
    times
        an array of wait times between the
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        An experiment schedule.


    """  # pylint: disable=line-too-long

    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("Echo", repetitions)
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X90(qubit))
        schedule.add(X(qubit), ref_pt="start", rel_time=tau / 2)
        schedule.add(X90(qubit), ref_pt="start", rel_time=tau / 2)
        schedule.add(Measure(qubit, acq_index=i), label=f"Measurement {i}")
    return schedule


def allxy_sched(
    qubit: str,
    element_select_idx: Union[np.ndarray, int] = np.arange(21),
    repetitions: int = 1,
) -> Schedule:
    # pylint: disable=line-too-long
    """
    Generate a schedule for performing an AllXY experiment.

    Schedule sequence
        .. centered:: Reset -- Rxy[0] -- Rxy[1] -- Measure

    for a specific set of combinations of x90, x180, y90, y180 and idle rotations.

    See section 2.3.2 of :cite:t:`reed_entanglement_2013` for an explanation of
    the AllXY experiment and it's applications in diagnosing errors in single-qubit
    control pulses.

    Parameters
    ----------
    qubit
        the name of the qubit e.g., :code:`"q0"` to perform the experiment on.
    element_select_idx
        the index of the particular element of the AllXY experiment to exectute.
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        An experiment schedule.

    """

    element_idxs = np.asarray(element_select_idx)
    element_idxs = element_idxs.reshape(element_idxs.shape or (1,))

    # all combinations of Idle, X90, Y90, X180 and Y180 gates that are part of
    # the AllXY experiment
    allxy_combinations = [
        [(0, 0), (0, 0)],
        [(180, 0), (180, 0)],
        [(180, 90), (180, 90)],
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
    schedule = Schedule("AllXY", repetitions)

    for i, elt_idx in enumerate(element_idxs):
        # check index valid
        if elt_idx > len(allxy_combinations) or elt_idx < 0:
            raise ValueError(
                f"Invalid index selected: {elt_idx}. "
                "Index must be in range 0 to 21 inclusive."
            )

        ((th0, phi0), (th1, phi1)) = allxy_combinations[elt_idx]

        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(Rxy(qubit=qubit, theta=th0, phi=phi0))
        schedule.add(Rxy(qubit=qubit, theta=th1, phi=phi1))
        schedule.add(Measure(qubit, acq_index=i), label=f"Measurement {i}")
    return schedule


def readout_calibration_sched(
    qubit: str,
    prepared_states: List[int],
    repetitions: int = 1,
) -> Schedule:
    """
    A schedule for readout calibration. Prepares a state and immediately performs
    a measurement.

    Parameters
    ----------
    qubit
        the name of the qubit e.g., :code:`"q0"` to perform the experiment on.
    prepared_states
        the states to prepare the qubit in before measuring as in integer corresponding
        to the ground (0), first-excited (1) or second-excited (2) state.
    repetitions
        The number of shots to acquire, sets the number of times the schedule will
        be repeated.

    Returns
    -------
    :
        An experiment schedule.

    Raises
    ------
    ValueError
        If the prepared state is not either 0, 1, or 2.
    NotImplementedError
        If the prepared state is 2.
    """

    schedule = Schedule(f"Readout calibration {qubit}, {prepared_states}", repetitions)

    for i, prep_state in enumerate(prepared_states):

        schedule.add(Reset(qubit), label=f"Reset {i}")
        if prep_state == 0:
            pass
        elif prep_state == 1:
            schedule.add(Rxy(qubit=qubit, theta=180, phi=0))
        elif prep_state == 2:
            raise NotImplementedError(
                "Preparing the qubit in the second excited (2) "
                "state is not supported yet."
            )
        else:
            raise ValueError(f"Prepared state ({prep_state}) must be either 0, 1 or 2.")
        schedule.add(
            Measure(qubit, acq_index=i, bin_mode=BinMode.APPEND),
            label=f"Measurement {i}",
        )

    return schedule


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
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
    init_duration: float,
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a schedule for performing a Rabi experiment using a
    :func:`quantify_scheduler.waveforms.drag` pulse.

    .. note::

        This function allows specifying a Rabi experiment directly using the pulse-level
        abstraction. For most applications we recommend using :func:`rabi_sched`
        instead.

    Parameters
    ----------
    mw_G_amp
        amplitude of the gaussian component of a DRAG pulse.
    mw_D_amp
        amplitude of the derivative-of-gaussian component of a DRAG pulse.
    mw_frequency
        frequency of the DRAG pulse.
    mw_clock
        reference clock used to track the qubit 01 transition.
    mw_port
        location on the device where the pulse should be applied.
    mw_pulse_duration
        duration of the DRAG pulse. Corresponds to 4 sigma.
    ro_pulse_amp
        amplitude of the readout pulse in Volt.
    ro_pulse_duration
        duration of the readout pulse in seconds.
    ro_pulse_delay
        time between the end of the spectroscopy pulse and the start of the readout
        pulse.
    ro_pulse_port
        location on the device where the readout pulse should be applied.
    ro_pulse_clock
        reference clock used to track the readout frequency.
    ro_pulse_frequency
        frequency of the spectroscopy pulse and of the data acquisition in Hertz.
    ro_acquisition_delay
        start of the data acquisition with respect to the start of the readout pulse
        in seconds.
    ro_integration_time
        integration time of the data acquisition in seconds.
    init_duration :
        The relaxation time or dead time.
    repetitions
        The amount of times the Schedule will be repeated.
    """
    schedule = Schedule("Rabi schedule (pulse)", repetitions)
    schedule.add_resource(ClockResource(name=mw_clock, freq=mw_frequency))
    schedule.add_resource(ClockResource(name=ro_pulse_clock, freq=ro_pulse_frequency))

    schedule.add(IdlePulse(duration=init_duration), label="qubit reset")
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
