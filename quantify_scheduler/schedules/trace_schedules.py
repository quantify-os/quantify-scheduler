# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains various examples of trace schedules."""

import numpy as np

from quantify_scheduler.operations.control_flow_library import Loop
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.enums import BinMode
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.operations.acquisition_library import (
    SSBIntegrationComplex,
    Trace,
)

from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    SquarePulse,
    VoltageOffset,
)
from quantify_scheduler.resources import ClockResource


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
    pulse_delay
        The pulse delay in seconds.
    frequency
        The frequency of the pulse and of the data acquisition in Hertz.
    acquisition_delay
        The start of the data acquisition with respect to the start of the pulse in
        seconds.
    integration_time
        The time in seconds to integrate.
    port
        The location on the device where the
        pulse should be applied.
    clock
        The reference clock used to track the pulse frequency.
    init_duration
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


def trace_schedule_circuit_layer(
    qubit_name: str,
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a simple schedule at circuit layer to perform raw trace acquisition.

    Parameters
    ----------
    qubit_name
        Name of a device element.
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        The Raw Trace acquisition Schedule.
    """
    schedule = Schedule("Raw trace acquisition", repetitions)
    schedule.add(Measure(qubit_name, acq_protocol="Trace"))
    return schedule


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
    qubit_pulse_clock
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


def long_time_trace(
    pulse_amp: complex,
    frequency: float,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    num_points: int,
    pulse_delay: float = 0,
    acq_index: int = 0,
    repetitions: int = 1,
) -> Schedule:
    """
    The function generates a :class:`~Schedule` for a time trace experiment
    where single side band integration (i.e. ``SSBIntegrationComplex``) is
    applied repeatedly. Compared to the Trace schedule which returns one point
    each ns for 16.4 μs (qblox instrument), the long time trace allows the
    user to customize the integration time and the number of data points in the trace (a minimum of
    300 ns (qblox instrument) for the integration time is needed in
    between two points). The resulting :class:`~Schedule` can be used for
    quantum experiments involving dynamic processes (charge transition) and
    time-dependent measurements (Elzerman Readout).

    Parameters
    ----------
    pulse_amp
        The amplitude of the pulse in Volt.
    frequency
        The frequency of the pulse and of the data acquisition in Hertz.
    acquisition_delay
        The start of the data acquisition with respect to the start of the pulse in
        seconds.
    integration_time
        Integration time per point in the trace. A minimum of 300ns is required du
        to the spacing of acquisition protocols in QbloxInstrument.
    port
        The location on the device where the
        pulse should be applied.
    clock
        The reference clock used to track the pulse frequency.
    num_points
        Number of points the output long_trace contains.
        The total time of the long_trace is then nb_pts*integration_time
    pulse_delay
        The pulse delay in seconds.
    acq_index
        The data register in which the acquisition is stored, by default 0.
        Describes the "when" information of the measurement, used to label or
        tag individual measurements in a large circuit. Typically corresponds
        to the setpoints of a schedule (e.g., tau in a T1 experiment).
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        The custom long Trace acquisition Schedule.
    """
    schedule = Schedule("Long time trace acquisition", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    pulse_op = VoltageOffset(
        offset_path_I=np.real(pulse_amp),
        offset_path_Q=np.imag(pulse_amp),
        port=port,
        clock=clock,
    )

    schedule.add(pulse_op)

    op = SSBIntegrationComplex(
        port=port,
        clock=clock,
        duration=integration_time,
        acq_channel=0,
        acq_index=acq_index,
        bin_mode=BinMode.APPEND,
        t0=pulse_delay,
    )

    inner = Schedule("inner", repetitions=1)
    inner.add(op)
    schedule.add(
        inner,
        control_flow=Loop(num_points),
        rel_time=acquisition_delay,
        ref_pt="start",
    )

    pulse_op_off = VoltageOffset(
        offset_path_I=0,
        offset_path_Q=0,
        port=port,
        clock=clock,
    )

    # Here rel_time = 4ns have to be add to order properly with the control flow.
    # Same think Idle pulse has to be added for loops
    # This has to be removed in a next version.
    schedule.add(pulse_op_off, rel_time=4e-9)
    schedule.add(IdlePulse(duration=4e-9))

    return schedule


def long_time_trace_with_qubit(
    qubit: DeviceElement,
    num_points: int,
    acq_index: int = 0,
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a simple schedule similar to a circuit layer to perform long trace acquisition.
    Wrapper function for :func:~quantify_scheduler.schedules.long_time_trace to use
    with a quantum device element..

    Parameters
    ----------
    qubit
        Device Element.
    num_points
        Number of points the output custom_long_trace contains.
        The total time of the custom_long_trace is then nb_pts*integration_time
    acq_index :
        The data register in which the acquisition is stored, by default 0.
        Describes the "when" information of the measurement, used to label or
        tag individual measurements in a large circuit. Typically corresponds
        to the setpoints of a schedule (e.g., tau in a T1 experiment).
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        The custom long Trace acquisition Schedule.
    """
    schedule = long_time_trace(
        pulse_amp=qubit.measure.pulse_amp(),
        pulse_delay=0,
        frequency=qubit.clock_freqs.readout(),
        acquisition_delay=qubit.measure.acq_delay(),
        integration_time=qubit.measure.integration_time(),
        port=qubit.ports.readout(),
        clock=qubit.name + ".ro",
        num_points=num_points,
        acq_index=acq_index,
        repetitions=repetitions,
    )
    return schedule
