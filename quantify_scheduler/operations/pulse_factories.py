# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
A module containing factory functions for pulses on the quantum-device layer.

These factories are used to take a parametrized representation of on a operation
and use that to create an instance of the operation itself.
"""
from quantify_scheduler.operations import pulse_library


def rxy_drag_pulse(
    amp180, motzoi, theta, phi, port, duration, clock
) -> pulse_library.DRAGPulse:
    """
    Generate a :class:`~.operations.pulse_library.DRAGPulse` that achieves the right
    rotation angle `theta` based on a calibrated pi-pulse amplitude and motzoi
    parameter based on linear interpolation of the pulse amplitudes.
    """
    # G_amp is the gaussian amplitude introduced in
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501
    # 180 refers to the normalization, theta is in degrees, and
    # mw_amp180 is the amplitude necessary to get the
    # maximum 180 degree theta (experimentally)
    G_amp = amp180 * theta / 180
    D_amp = motzoi

    return pulse_library.DRAGPulse(
        G_amp=G_amp, D_amp=D_amp, phase=phi, port=port, duration=duration, clock=clock
    )


def composite_square_pulse(  # pylint: disable=too-many-arguments
    square_amp: float,
    square_duration: float,
    square_port: str,
    square_clock: str,
    virt_z_parent_qubit_phase: float,
    virt_z_parent_qubit_clock: str,
    virt_z_child_qubit_phase: float,
    virt_z_child_qubit_clock: str,
    t0: float = 0,
) -> pulse_library.SquarePulse:
    """
    This is an example composite pulse to implement a CZ gate. It applies the
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
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.
    """

    # Start the flux pulse
    composite_pulse = pulse_library.SquarePulse(
        amp=square_amp,
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
) -> pulse_library.SkewedHermitePulse:
    """Generate hermite pulse for spectroscopy experiment.

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

    Returns
    -------
    :
        Hermite pulse operation
    """
    return pulse_library.SkewedHermitePulse(
        duration=duration,
        amplitude=amplitude,
        skewness=0,
        phase=0,
        clock=clock,
        port=port,
    )
