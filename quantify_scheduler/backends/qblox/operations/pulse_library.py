# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard pulse-level operations for use with the quantify_scheduler."""

from __future__ import annotations

from quantify_scheduler import Operation


class LatchReset(Operation):
    """
    Operation that resets the feedback trigger addresses from the hardware.

    Currently only implemented for Qblox backend, refer to
    :class:`~quantify_scheduler.backends.qblox.operation_handling.virtual.ResetFeedbackTriggersStrategy`
    for more details.
    """

    def __init__(
        self,
        t0: float = 0,
        duration: float = 4e-9,
        portclocks: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        if portclocks is None:
            portclocks = []
        self.data["pulse_info"] = [
            {
                "name": self.__class__.__name__,
                "wf_func": None,
                "t0": t0,
                "port": portclock[0],
                "clock": portclock[1],
                "duration": duration,
            }
            for portclock in portclocks
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
