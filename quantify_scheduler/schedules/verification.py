# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Schedules intended to verify (test) functionality of the system.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantify_scheduler import Schedule
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.pulse_library import IdlePulse, SquarePulse
from quantify_scheduler.resources import ClockResource

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def acquisition_staircase_sched(
    readout_pulse_amps: NDArray[np.ScalarType],
    readout_pulse_duration: float,
    readout_frequency: float,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    init_duration: float = 1e-6,
    acq_channel: int = 0,
    repetitions: int = 1,
) -> Schedule:
    """
    Generates a staircase program in which the amplitude of a readout pulse is varied.

    Schedule sequence
        .. centered:: Reset -- RO_pulse[i] -- Acq[i]

    Parameters
    ----------
    readout_pulse_amps
        amplitudes of the square readout pulse in Volts.
    readout_pulse_duration
        duration of the spectroscopy pulse in seconds.
    readout_frequency
        readout_frequency of the spectroscopy pulse and of the data acquisition in
        Hertz.
    acquisition_delay
        start of the data acquisition with respect to the start of the spectroscopy
        pulse in seconds.
    integration_time
        integration time of the data acquisition in seconds.
    port
        location on the device where the pulse should be applied.
    clock
        reference clock used to track the readout frequency.
    batched
        schedule to be run in batched mode in the hardware backend.
    init_duration :
        The relaxation time or dead time.
    acq_channel
        The acquisition channel to use for the acquisitions.
    repetitions
        The amount of times the Schedule will be repeated.

    Notes
    -----
    This schedule can be used to verify the binning and averaging functionality of
    weighted acquisition for readout modules such as a Qblox QRM or a ZI UHFQA.

    Because of the overlap between the readout pulse and the integration window, the
    change in readout pulse amplitude should show up immediately in the acquired signal.

    """
    sched = Schedule(name="Acquisition staircase", repetitions=repetitions)

    sched.add_resource(ClockResource(name=clock, freq=readout_frequency))

    # ensure readout_pulse_amps is an iterable when passing floats.
    readout_pulse_amps = np.asarray(readout_pulse_amps)
    readout_pulse_amps = readout_pulse_amps.reshape(readout_pulse_amps.shape or (1,))

    for acq_index, readout_pulse_amp in enumerate(readout_pulse_amps):

        sched.add(IdlePulse(duration=init_duration))

        pulse = sched.add(
            SquarePulse(
                duration=readout_pulse_duration,
                amp=readout_pulse_amp,
                port=port,
                clock=clock,
            ),
            label=f"SquarePulse_{acq_index}",
        )

        sched.add(
            SSBIntegrationComplex(
                duration=integration_time,
                port=port,
                clock=clock,
                acq_index=acq_index,
                acq_channel=acq_channel,
            ),
            ref_op=pulse,
            ref_pt="start",
            rel_time=acquisition_delay,
            label=f"Acquisition_{acq_index}",
        )

    return sched


def awg_staircase_sched(
    pulse_amps: NDArray[np.ScalarType],
    pulse_duration: float,
    readout_frequency: float,
    acquisition_delay: float,
    integration_time: float,
    mw_port: str,
    ro_port: str,
    mw_clock: str,
    ro_clock: str,
    init_duration: float = 1e-6,
    acq_channel: int = 0,
    repetitions: int = 1,
) -> Schedule:
    """
    Generates a staircase program in which the amplitude of a control pulse is varied.

    Schedule sequence
        .. centered:: Reset -- MW_pulse[i] -- Acq[i]

    Parameters
    ----------
    pulse_amps
        amplitudes of the square readout pulse in Volts.
    pulse_duration
        duration of the spectroscopy pulse in seconds.
    readout_frequency
        readout_frequency of the spectroscopy pulse and of the data acquisition in
        Hertz.
    acquisition_delay
        start of the data acquisition with respect to the start of the spectroscopy
        pulse in seconds.
    integration_time
        integration time of the data acquisition in seconds.
    mw_port
        location on the device where the pulse should be applied.
    ro_port
        location on the device where the signal should should be interpreted.
    ro_clock
        reference clock connected to hdawg used to track the readout frequency.
    mw_clock
        reference clock connected to uhfqa used to track the readout frequency.
    batched
        schedule to be run in batched mode in the hardware backend.
    init_duration :
        The relaxation time or dead time.
    acq_channel
        The acquisition channel to use for the acquisitions.
    repetitions
        The amount of times the Schedule will be repeated.


    Notes
    -----
    The control pulse is configured to be applied at the same frequency as the
    acquisition so that it shows up in the in the acquired signal.

    This schedule can be used to verify the binning and averaging functionality of
    weighted acquisition in combination with the synchronization between the readout
    module (e.g., Qblox QRM or ZI UHFQA) and the pulse generating module
    (e.g., Qblox QCM or ZI HDAWG).
    """
    sched = Schedule(name="AWG staircase", repetitions=repetitions)

    sched.add_resource(ClockResource(name=mw_clock, freq=readout_frequency))
    sched.add_resource(ClockResource(name=ro_clock, freq=readout_frequency))

    # ensure pulse_amps is an iterable when passing floats.
    pulse_amps = np.asarray(pulse_amps)
    pulse_amps = pulse_amps.reshape(pulse_amps.shape or (1,))

    for acq_index, pulse_amp in enumerate(pulse_amps):

        sched.add(IdlePulse(duration=init_duration))

        pulse = sched.add(
            SquarePulse(
                duration=pulse_duration,
                amp=pulse_amp,
                port=mw_port,
                clock=mw_clock,
            ),
            label=f"SquarePulse_{acq_index}",
        )

        sched.add(
            SSBIntegrationComplex(
                duration=integration_time,
                port=ro_port,
                clock=ro_clock,
                acq_index=acq_index,
                acq_channel=acq_channel,
            ),
            ref_op=pulse,
            ref_pt="start",
            rel_time=acquisition_delay,
            label=f"Acquisition_{acq_index}",
        )

    return sched
