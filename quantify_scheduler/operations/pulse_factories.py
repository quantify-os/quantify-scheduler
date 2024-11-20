# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
A module containing factory functions for pulses on the quantum-device layer.

These factories are used to take a parametrized representation of on a operation
and use that to create an instance of the operation itself.
"""
from __future__ import annotations

from quantify_scheduler.operations import pulse_library
from quantify_scheduler.schedules import Schedule


def rxy_drag_pulse(
    amp180: float,
    motzoi: float,
    theta: float,
    phi: float,
    port: str,
    duration: float,
    clock: str,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
) -> pulse_library.DRAGPulse:
    """
    Generate a :class:`~.operations.pulse_library.DRAGPulse` that achieves the right
    rotation angle ``theta`` based on a calibrated pi-pulse amplitude and motzoi
    parameter based on linear interpolation of the pulse amplitudes.

    Parameters
    ----------
    amp180
        Unitless amplitude of excitation pulse to get the maximum 180 degree theta.
    motzoi
        Unitless amplitude of the derivative component, the DRAG-pulse parameter.
    theta
        Angle in degrees to rotate around an equatorial axis on the Bloch sphere.
    phi
        Phase of the pulse in degrees.
    port
        Name of the port where the pulse is played.
    duration
        Duration of the pulse in seconds.
    clock
        Name of the clock used to modulate the pulse.
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`,
        Optional scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.

    Returns
    -------
    :
        DRAGPulse operation.

    """
    # G_amp is the gaussian amplitude introduced in
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501
    # 180 refers to the normalization, theta is in degrees, and
    # amp180 is the amplitude necessary to get the
    # maximum 180 degree theta (experimentally)
    G_amp = amp180 * theta / 180
    D_amp = motzoi

    return pulse_library.DRAGPulse(
        G_amp=G_amp,
        D_amp=D_amp,
        phase=phi,
        port=port,
        duration=duration,
        clock=clock,
        reference_magnitude=reference_magnitude,
    )


def rxy_gauss_pulse(
    amp180: float,
    theta: float,
    phi: float,
    port: str,
    duration: float,
    clock: str,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
) -> pulse_library.GaussPulse:
    """
    Generate a Gaussian drive with :class:`~.operations.pulse_library.GaussPulse` that achieves
    the right rotation angle ``theta`` based on a calibrated pi-pulse amplitude.

    Parameters
    ----------
    amp180
        Unitless amplitude of excitation pulse to get the maximum 180 degree theta.
    theta
        Angle in degrees to rotate around an equatorial axis on the Bloch sphere.
    phi
        Phase of the pulse in degrees.
    port
        Name of the port where the pulse is played.
    duration
        Duration of the pulse in seconds.
    clock
        Name of the clock used to modulate the pulse.
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`,
        Optional scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.

    Returns
    -------
    :
        GaussPulse operation.

    """
    # theta is in degrees, and
    # amp180 is the amplitude necessary to get the
    # maximum 180 degree theta (experimentally)
    G_amp = amp180 * theta / 180

    return pulse_library.GaussPulse(
        G_amp=G_amp,
        phase=phi,
        port=port,
        duration=duration,
        clock=clock,
        reference_magnitude=reference_magnitude,
    )


def phase_shift(
    theta: float,
    clock: str,
) -> pulse_library.ShiftClockPhase:
    """
    Generate a :class:`~.operations.pulse_library.ShiftClockPhase` that shifts the phase of the
    ``clock`` by an angle `theta`.

    Parameters
    ----------
    theta
        Angle to shift the clock by, in degrees.
    clock
        Name of the clock to shift.

    Returns
    -------
    :
        ShiftClockPhase operation.

    """
    return pulse_library.ShiftClockPhase(
        phase_shift=theta,
        clock=clock,
    )


def composite_square_pulse(
    square_amp: float,
    square_duration: float,
    square_port: str,
    square_clock: str,
    virt_z_parent_qubit_phase: float,
    virt_z_parent_qubit_clock: str,
    virt_z_child_qubit_phase: float,
    virt_z_child_qubit_clock: str,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
    t0: float = 0,
) -> pulse_library.SquarePulse:
    """
    An example composite pulse to implement a CZ gate.

    It applies the
    square pulse and then corrects for the phase shifts on both the qubits.

    Parameters
    ----------
    square_amp
        Amplitude of the square envelope.
    square_duration
        The square pulse duration in seconds.
    square_port
        Port of the pulse, must be capable of playing a complex waveform.
    square_clock
        Clock used to modulate the pulse.
    virt_z_parent_qubit_phase
        The phase shift in degrees applied to the parent qubit.
    virt_z_parent_qubit_clock
        The clock of which to shift the phase applied to the parent qubit.
    virt_z_child_qubit_phase
        The phase shift in degrees applied to the child qubit.
    virt_z_child_qubit_clock
        The clock of which to shift the phase applied to the child qubit.
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`,
        Optional scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.

    Returns
    -------
    :
        SquarePulse operation.

    """
    # Start the flux pulse
    composite_pulse = pulse_library.SquarePulse(
        amp=square_amp,
        reference_magnitude=reference_magnitude,
        duration=square_duration,
        port=square_port,
        clock=square_clock,
        t0=t0,
    )

    # And at the same time apply clock phase corrections
    composite_pulse.add_pulse(
        pulse_library.ShiftClockPhase(
            phase_shift=virt_z_parent_qubit_phase,
            clock=virt_z_parent_qubit_clock,
            t0=t0,
        )
    )
    composite_pulse.add_pulse(
        pulse_library.ShiftClockPhase(
            phase_shift=virt_z_child_qubit_phase,
            clock=virt_z_child_qubit_clock,
            t0=t0,
        )
    )

    return composite_pulse


def rxy_hermite_pulse(
    amp180: float,
    skewness: float,
    theta: float,
    phi: float,
    port: str,
    duration: float,
    clock: str,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
) -> pulse_library.SkewedHermitePulse:
    """
    Generate a Gaussian drive with :class:`~.operations.pulse_library.GaussPulse` that achieves
    the right rotation angle ``theta`` based on a calibrated pi-pulse amplitude.

    Parameters
    ----------
    amp180
        Unitless amplitude of excitation pulse to get the maximum 180 degree theta.
    skewness
        First-order amplitude correction to the Hermite pulse. Skewness of 0 returns a standard
        hermite pulse.
    theta
        Angle in degrees to rotate around an equatorial axis on the Bloch sphere.
    phi
        Phase of the pulse in degrees.
    port
        Name of the port where the pulse is played.
    duration
        Duration of the pulse in seconds.
    clock
        Name of the clock used to modulate the pulse.
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`,
        hardware config if not provided.

    Returns
    -------
    :
        GaussPulse operation.

    """
    # theta is in degrees, and
    # amp180 is the amplitude necessary to get the
    # maximum 180 degree theta (experimentally)
    amp_theta = amp180 * theta / 180

    return pulse_library.SkewedHermitePulse(
        amplitude=amp_theta,
        skewness=skewness,
        phase=phi,
        port=port,
        duration=duration,
        clock=clock,
        reference_magnitude=reference_magnitude,
    )


def nv_spec_pulse_mw(
    duration: float,
    amplitude: float,
    clock: str,
    port: str,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
) -> pulse_library.SkewedHermitePulse:
    """
    Generate hermite pulse for spectroscopy experiment.

    This is a simplified version of the SkewedHermitePulse. It is not skewed. It also
    sets the phase to 0. This means that no rotation about the z-axis is applied on the
    qubit.

    Parameters
    ----------
    duration
        Pulse duration in seconds
    amplitude
        Amplitude of the hermite pulse
    skewness
        Skewness of hermite pulse
    clock
        Name of clock for frequency modulation of hermite pulse
    port
        Name of port where hermite pulse is applied
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`,
        Optional scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.

    Returns
    -------
    :
        Hermite pulse operation

    """
    return pulse_library.SkewedHermitePulse(
        duration=duration,
        amplitude=amplitude,
        reference_magnitude=reference_magnitude,
        skewness=0,
        phase=0,
        clock=clock,
        port=port,
    )


def spin_init_pulse(
    square_duration: float,
    ramp_diff: float,
    parent_port: str,
    parent_clock: str,
    parent_square_amp: float,
    parent_ramp_amp: float,
    parent_ramp_rate: float,
    child_port: str,
    child_clock: str,
    child_square_amp: float,
    child_ramp_amp: float,
    child_ramp_rate: float,
) -> Schedule:
    """Device compilation of the spin init operation."""
    spin_init_schedule = Schedule("spin_init")

    spin_init_schedule.add(
        pulse_library.SquarePulse(
            amp=parent_square_amp,
            duration=square_duration,
            port=parent_port,
            clock=parent_clock,
        )
    )

    spin_init_schedule.add(
        pulse_library.SquarePulse(
            amp=child_square_amp,
            duration=square_duration,
            port=child_port,
            clock=child_clock,
        ),
        ref_pt="start",
    )

    parent_ramp_rel_time = abs(min(ramp_diff, 0))

    spin_init_schedule.add(
        pulse_library.RampPulse(
            amp=parent_ramp_amp,
            duration=parent_ramp_amp / parent_ramp_rate,
            port=parent_port,
            clock=parent_clock,
        ),
        ref_pt="end",
        rel_time=parent_ramp_rel_time,
    )

    spin_init_schedule.add(
        pulse_library.RampPulse(
            amp=child_ramp_amp,
            duration=child_ramp_amp / child_ramp_rate,
            port=child_port,
            clock=child_clock,
        ),
        ref_pt="start",
        rel_time=ramp_diff,
    )

    return spin_init_schedule
