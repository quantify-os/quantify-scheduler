# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard pulse-level operations for use with the quantify_scheduler."""

from __future__ import annotations

import numpy as np

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import (
    NumericalPulse,
    ReferenceMagnitude,
)
from quantify_scheduler.resources import BasebandClockResource


class LatchReset(Operation):
    """
    Operation that resets the feedback trigger addresses from the hardware.

    Currently only implemented for Qblox backend, refer to
    :class:`~quantify_scheduler.backends.qblox.operation_handling.virtual.ResetFeedbackTriggersStrategy`
    for more details.
    """

    def __init__(
        self,
        portclock: tuple[str, str],
        t0: float = 0,
        duration: float = 4e-9,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "name": self.__class__.__name__,
                "wf_func": None,
                "t0": t0,
                "port": portclock[0],
                "clock": portclock[1],
                "duration": duration,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"t0='{pulse_info['t0']}', "
            f"port='{pulse_info['port']}'"
            f"clock='{pulse_info['clock']}'"
            f"duration='{pulse_info['duration']}'"
            f")"
        )


class SimpleNumericalPulse(NumericalPulse):
    """
    Wrapper on top of NumericalPulse to provide a simple interface for creating a pulse
    where the samples correspond 1:1 to the produced waveform, without needing to specify
    the time samples.


    Parameters
    ----------
    samples
        An array of (possibly complex) values specifying the shape of the pulse.
    port
        The port that the pulse should be played on.
    clock
        Clock used to (de)modulate the pulse.
        By default the baseband clock.
    reference_magnitude
        Scaling value and unit for the unitless samples. Uses settings in
        hardware config if not provided.
    t0
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.


    Example
    -------

    .. jupyter-execute::

        from quantify_scheduler.backends.qblox.operations.pulse_library import SimpleNumericalPulse
        from quantify_scheduler import Schedule

        waveform = [0.1,0.2,0.2,0.3,0.5,0.4]

        schedule = Schedule("")
        schedule.add(SimpleNumericalPulse(waveform, port="q0:out"))


    """

    def __init__(
        self,
        samples: np.ndarray | list,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        reference_magnitude: ReferenceMagnitude | None = None,
        t0: float = 0,
    ) -> None:

        # Append samples with one value which will be truncated away by the interpolation.
        samples = np.append(samples, 0)

        t_samples = np.arange(len(samples)) / constants.SAMPLING_RATE

        super().__init__(
            samples=samples,
            t_samples=t_samples,
            port=port,
            clock=clock,
            reference_magnitude=reference_magnitude,
            t0=t0,
            interpolation="linear",
        )
