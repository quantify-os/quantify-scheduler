# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard pulse-level operations for use with the quantify_scheduler."""
# pylint: disable= too-many-arguments, too-many-ancestors
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from quantify_scheduler import Operation
from quantify_scheduler.helpers.deprecation import deprecated_arg_alias
from quantify_scheduler.resources import BasebandClockResource

if TYPE_CHECKING:
    from quantify_scheduler.operations.pulse_library import ReferenceMagnitude


class VoltageOffset(Operation):
    """
    Operation that represents setting a constant offset to the output voltage.

    Please refer to :ref:`sec-qblox-offsets-long-voltage-offsets` in the reference guide
    for more details.

    Parameters
    ----------
    offset_path_I : float
        Offset of path I.
    offset_path_Q : float
        Offset of path Q.
    port : str
        Port of the voltage offset.
    clock : str, optional
        Clock used to modulate the voltage offset. By default a BasebandClock is used.
    duration : float, optional
        (deprecated) The time to hold the offset for (in seconds).
    t0 : float, optional
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule.
    reference_magnitude :
        Scaling value and unit for the unitless amplitude. Uses settings in
        hardware config if not provided.
    """

    @deprecated_arg_alias(
        "0.20.0", offset_path_0="offset_path_I", offset_path_1="offset_path_Q"
    )
    def __init__(
        self,
        offset_path_I: float,
        offset_path_Q: float,
        port: str,
        clock: str = BasebandClockResource.IDENTITY,
        duration: float = 0.0,
        t0: float = 0,
        reference_magnitude: ReferenceMagnitude | None = None,
    ) -> None:
        if duration != 0.0:
            warnings.warn(
                "The duration parameter will be removed in quantify-scheduler >= "
                "0.20.0, and the duration will be 0.0 by default.",
                FutureWarning,
            )
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "t0": t0,
                "offset_path_I": offset_path_I,
                "offset_path_Q": offset_path_Q,
                "clock": clock,
                "port": port,
                "duration": duration,
                "reference_magnitude": reference_magnitude,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)
