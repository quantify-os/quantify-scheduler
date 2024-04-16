# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
A module containing factory functions for pulses on the quantum-device layer.

These factories are used to take a parametrized representation of on a operation
and use that to create an instance of the operation itself.
"""
from __future__ import annotations

from quantify_core.utilities import deprecated

from quantify_scheduler.backends.qblox.operations import (
    pulse_factories as qblox_pulse_factories,
)
from quantify_scheduler.backends.qblox.operations.stitched_pulse import StitchedPulse
from quantify_scheduler.operations import pulse_library


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
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`, optional
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
    Generate a Gaussian drive with :class:`~.operations.pulse_library.GaussPulse` that achieves the right
    rotation angle ``theta`` based on a calibrated pi-pulse amplitude.

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
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`, optional
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
    Generate a :class:`~.operations.pulse_library.ShiftClockPhase` that shifts the phase of the ``clock`` by an angle `theta`.

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
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`, optional
        Scaling value and unit for the unitless amplitude. Uses settings in
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
    reference_magnitude : :class:`~quantify_scheduler.operations.pulse_library.ReferenceMagnitude`, optional
        Scaling value and unit for the unitless amplitude. Uses settings in
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


@deprecated("0.20.0", qblox_pulse_factories.long_square_pulse)
def long_square_pulse() -> StitchedPulse:
    """Deprecated long_square_pulse."""


@deprecated("0.20.0", qblox_pulse_factories.staircase_pulse)
def staircase_pulse() -> StitchedPulse:
    """Deprecated staircase_pulse."""


@deprecated("0.20.0", qblox_pulse_factories.long_ramp_pulse)
def long_ramp_pulse() -> StitchedPulse:
    """Deprecated long_ramp_pulse."""
