# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing factory functions for pulses on the quantum-device layer.

These factories take a parametrized representation of an operation and create an
instance of the operation itself. The created operations make use of Qblox-specific
hardware features.
"""
from __future__ import annotations

import math

import numpy as np

from quantify_scheduler.backends.qblox import constants, helpers
from quantify_scheduler.backends.qblox.operations.stitched_pulse import (
    StitchedPulse,
    StitchedPulseBuilder,
)
from quantify_scheduler.operations import pulse_library
from quantify_scheduler.resources import BasebandClockResource


def long_square_pulse(
    amp: float,
    duration: float,
    port: str,
    clock: str = BasebandClockResource.IDENTITY,
    t0: float = 0,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
) -> StitchedPulse:
    """
    Create a long square pulse using DC voltage offsets.

    .. warning::

        This function creates a
        :class:`~quantify_scheduler.backends.qblox.operations.stitched_pulse.StitchedPulse`
        object, containing a combination of voltage offsets and waveforms. Overlapping
        StitchedPulses on the same port and clock may lead to unexpected results.

    Parameters
    ----------
    amp : float
        Amplitude of the envelope.
    duration : float
        The pulse duration in seconds.
    port : str
        Port of the pulse, must be capable of playing a complex waveform.
    clock : str, optional
        Clock used to modulate the pulse. By default the baseband clock.
    t0 : float, optional
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule. By default 0.
    min_operation_time_ns : int, optional
        Min operation time in ns. The duration of the long_square_pulse must be a multiple
        of this. By default equal to the min operation time time of Qblox modules.
    reference_magnitude : optional
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.

    Returns
    -------
    StitchedPulse
        A StitchedPulse object containing an offset instruction with the specified
        amplitude.

    Raises
    ------
    ValueError
        When the duration of the pulse is not a multiple of ``grid_time_ns``.
    """
    if duration * 1e9 < constants.MIN_TIME_BETWEEN_OPERATIONS:
        raise ValueError(
            f"The duration of a long_square_pulse must be at least "
            f"{constants.MIN_TIME_BETWEEN_OPERATIONS} ns."
            f" Duration of offending operation: {duration}."
            f" Start time: {t0}"
        )

    pulse = (
        StitchedPulseBuilder(
            name=long_square_pulse.__name__, port=port, clock=clock, t0=t0
        )
        .add_voltage_offset(
            path_I=np.real(amp),
            path_Q=np.imag(amp),
            reference_magnitude=reference_magnitude,
        )
        # The last bit, with duration 'grid time' ns, is replaced by a normal pulse. The
        # offset is set back to 0 before this pulse, because the Qblox backend might
        # otherwise lengthen the full operation by adding an 'UpdateParameters'
        # instruction at the end.
        .add_voltage_offset(
            path_I=0.0,
            path_Q=0.0,
            rel_time=duration - constants.MIN_TIME_BETWEEN_OPERATIONS * 1e-9,
        )
        .add_pulse(
            pulse_library.SquarePulse(
                amp=amp,
                duration=constants.MIN_TIME_BETWEEN_OPERATIONS * 1e-9,
                port=port,
                clock=clock,
                reference_magnitude=reference_magnitude,
            )
        )
        .build()
    )
    return pulse


def staircase_pulse(
    start_amp: float,
    final_amp: float,
    num_steps: int,
    duration: float,
    port: str,
    clock: str = BasebandClockResource.IDENTITY,
    t0: float = 0,
    min_operation_time_ns: int = constants.MIN_TIME_BETWEEN_OPERATIONS,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
) -> StitchedPulse:
    """
    Create a staircase-shaped pulse using DC voltage offsets.

    This function generates a real valued staircase pulse, which reaches its final
    amplitude in discrete steps. In between it will maintain a plateau.

    .. warning::

        This function creates a
        :class:`~quantify_scheduler.backends.qblox.operations.stitched_pulse.StitchedPulse`
        object, containing a combination of voltage offsets and waveforms. Overlapping
        StitchedPulses on the same port and clock may lead to unexpected results.

    Parameters
    ----------
    start_amp : float
        Starting amplitude of the staircase envelope function.
    final_amp : float
        Final amplitude of the staircase envelope function.
    num_steps : int
        The number of plateaus.
    duration : float
        Duration of the pulse in seconds.
    port : str
        Port of the pulse.
    clock : str, optional
        Clock used to modulate the pulse. By default the baseband clock.
    t0 : float, optional
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule. By default 0.
    min_operation_time_ns : int, optional
        Min operation time in ns. The duration of the long_square_pulse must be a multiple
        of this. By default equal to the min operation time time of Qblox modules.
    reference_magnitude : optional
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.

    Returns
    -------
    StitchedPulse
        A StitchedPulse object containing incrementing or decrementing offset
        instructions.

    Raises
    ------
    ValueError
        When the duration of a step is not a multiple of ``grid_time_ns``.
    """
    builder = StitchedPulseBuilder(
        name=staircase_pulse.__name__, port=port, clock=clock, t0=t0
    )

    try:
        step_duration = (
            helpers.to_grid_time(duration / num_steps, min_operation_time_ns) * 1e-9
        )
    except ValueError as err:
        raise ValueError(
            f"The duration of each step of the staircase must be a multiple of"
            f" {min_operation_time_ns} ns."
        ) from err

    amps = np.linspace(start_amp, final_amp, num_steps)

    # The final step is a special case, see below.
    for amp in amps[:-1]:
        builder.add_voltage_offset(
            path_I=amp,
            path_Q=0.0,
            duration=step_duration,
            min_duration=min_operation_time_ns * 1e-9,
            reference_magnitude=reference_magnitude,
        )

    # The final step is an offset with the last part (of duration 'grid time' ns)
    # replaced by a pulse. The offset is set back to 0 before the pulse, because the
    # Qblox backend might otherwise lengthen the full operation by adding an
    # 'UpdateParameters' instruction at the end.
    builder.add_voltage_offset(
        path_I=amps[-1],
        path_Q=0.0,
        duration=step_duration - min_operation_time_ns * 1e-9,
        min_duration=min_operation_time_ns * 1e-9,
        reference_magnitude=reference_magnitude,
    )
    builder.add_voltage_offset(path_I=0.0, path_Q=0.0)
    builder.add_pulse(
        pulse_library.SquarePulse(
            amp=amps[-1],
            duration=min_operation_time_ns * 1e-9,
            port=port,
            clock=clock,
            reference_magnitude=reference_magnitude,
        )
    )

    pulse = builder.build()

    return pulse


def long_ramp_pulse(
    amp: float,
    duration: float,
    port: str,
    offset: float = 0,
    clock: str = BasebandClockResource.IDENTITY,
    t0: float = 0,
    part_duration_ns: int = constants.STITCHED_PULSE_PART_DURATION_NS,
    reference_magnitude: pulse_library.ReferenceMagnitude | None = None,
) -> StitchedPulse:
    """
    Creates a long ramp pulse by stitching together shorter ramps.

    This function creates a long ramp pulse by stitching together ramp pulses of the
    specified duration ``part_duration_ns``, with DC voltage offset instructions placed
    in between.

    .. warning::

        This function creates a
        :class:`~quantify_scheduler.backends.qblox.operations.stitched_pulse.StitchedPulse`
        object, containing a combination of voltage offsets and waveforms. Overlapping
        StitchedPulses on the same port and clock may lead to unexpected results.

    Parameters
    ----------
    amp : float
        Amplitude of the ramp envelope function.
    duration : float
        The pulse duration in seconds.
    port : str
        Port of the pulse.
    offset : float, optional
        Starting point of the ramp pulse. By default 0.
    clock : str, optional
        Clock used to modulate the pulse, by default a BasebandClock is used.
    t0 : float, optional
        Time in seconds when to start the pulses relative to the start time of the
        Operation in the Schedule. By default 0.
    part_duration_ns : int, optional
        Duration of each partial ramp in nanoseconds, by default
        :class:`~quantify_scheduler.backends.qblox.constants.STITCHED_PULSE_PART_DURATION_NS`.
    reference_magnitude : optional
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.

    Returns
    -------
    StitchedPulse
        A ``StitchedPulse`` composed of shorter ramp pulses with varying DC offsets,
        forming one long ramp pulse.
    """
    dur_ns = helpers.to_grid_time(duration)
    num_whole_parts = (dur_ns - 1) // part_duration_ns
    amp_part = part_duration_ns / dur_ns * amp
    dur_left = (dur_ns - num_whole_parts * part_duration_ns) * 1e-9
    amp_left = amp - num_whole_parts * amp_part

    builder = StitchedPulseBuilder(
        long_ramp_pulse.__name__, port=port, clock=clock, t0=t0
    )

    last_sample_voltage = offset
    for _ in range(num_whole_parts):
        # Add an offset for each ramp part, except for the first one if the overall ramp
        # offset is 0.
        if not (last_sample_voltage == offset and math.isclose(offset, 0.0)):
            builder.add_voltage_offset(
                path_I=last_sample_voltage,
                path_Q=0.0,
                reference_magnitude=reference_magnitude,
            )
        builder.add_pulse(
            pulse_library.RampPulse(
                amp=amp_part,
                duration=part_duration_ns * 1e-9,
                port=port,
                reference_magnitude=reference_magnitude,
            )
        )
        last_sample_voltage += amp_part

    # For the final part, the voltage offset is set to 0, because the Qblox
    # backend might otherwise lengthen the full operation by adding an
    # 'UpdateParameters' instruction at the end.

    # Insert a 0 offset if offsets were inserted above and the last offset is not 0.
    if not math.isclose(last_sample_voltage, offset) and not math.isclose(
        last_sample_voltage - amp_part, 0.0
    ):
        builder.add_voltage_offset(path_I=0.0, path_Q=0.0)
    builder.add_pulse(
        pulse_library.RampPulse(
            amp=amp_left,
            offset=last_sample_voltage,
            duration=dur_left,
            port=port,
            reference_magnitude=reference_magnitude,
        )
    )

    pulse = builder.build()

    return pulse
