# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pulse compensation operations for use with the quantify_scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.operations.operation import Operation

if TYPE_CHECKING:
    from quantify_scheduler.schedules.schedule import Schedule

Port = str
"""Port on the hardware; this is an alias to str."""


class PulseCompensation(Operation):
    """
    Apply pulse compensation to an operation or schedule.

    Inserts a pulse at the end of the operation or schedule set in ``body`` for each port.
    The compensation pulses are calculated so that the integral of all pulses
    (including the compensation pulses) are zero for each port.
    Moreover, the compensating pulses are square pulses, and start just after the last
    pulse on each port individually, and their maximum amplitude is the one
    specified in the ``max_compensation_amp``. Their duration is divisible by ``duration_grid``.
    The clock is assumed to be the baseband clock; any other clock is not allowed.

    Parameters
    ----------
    body
        Operation to be pulse-compensated
    max_compensation_amp
        Dictionary for each port the maximum allowed amplitude for the compensation pulse.
    time_grid
        Grid time of the duration of the compensation pulse.
    sampling_rate
        Sampling rate for pulse integration calculation.
    """

    def __init__(
        self,
        body: Operation | Schedule,
        max_compensation_amp: dict[Port, float],
        time_grid: float,
        sampling_rate: float,
    ) -> None:
        super().__init__(name="PulseCompensation")
        self.data.update(
            {
                "pulse_compensation_info": {
                    "body": body,
                    "max_compensation_amp": max_compensation_amp,
                    "time_grid": time_grid,
                    "sampling_rate": sampling_rate,
                },
            }
        )
        self._update()

    @property
    def body(self) -> Operation | Schedule:
        """Body of a pulse compensation."""
        return self.data["pulse_compensation_info"]["body"]

    @body.setter
    def body(self, value: Operation | Schedule) -> None:
        """Body of a pulse compensation."""
        self.data["pulse_compensation_info"]["body"] = value

    @property
    def max_compensation_amp(self) -> dict[Port, float]:
        """For each port the maximum allowed amplitude for the compensation pulse."""
        return self.data["pulse_compensation_info"]["max_compensation_amp"]

    @property
    def time_grid(self) -> float:
        """Grid time of the duration of the compensation pulse."""
        return self.data["pulse_compensation_info"]["time_grid"]

    @property
    def sampling_rate(self) -> float:
        """Sampling rate for pulse integration calculation."""
        return self.data["pulse_compensation_info"]["sampling_rate"]
