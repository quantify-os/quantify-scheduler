# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains the control flow operations for the Qblox backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation as _ConditionalOperation,
)

if TYPE_CHECKING:
    from quantify_scheduler.operations.operation import Operation
    from quantify_scheduler.schedules.schedule import Schedule


class ConditionalOperation(_ConditionalOperation):
    """
    Conditional over another operation.

    If a preceding thresholded acquisition on ``qubit_name`` results in a "1", the
    body will be executed, otherwise it will generate a wait time that is
    equal to the time of the subschedule, to ensure the absolute timing of later
    operations remains consistent.

    Parameters
    ----------
    body
        Operation to be conditionally played
    qubit_name
        Name of the qubit on which the body will be conditioned
    t0
        Time offset, by default 0

    Example
    -------

    A conditional reset can be implemented as follows:

    .. admonition:: example

        .. jupyter-execute

            # relevant imports
            from quantify_scheduler import Schedule
            from quantify_scheduler.backends.qblox.operations import ConditionalOperation
            from quantify_scheduler.operations.gate_library import Measure, X

            # define conditional reset as a Schedule
            conditional_reset = Schedule("conditional reset")
            conditional_reset.add(Measure("q0", feedback_trigger_label="q0"))
            conditional_reset.add(
                ConditionalOperation(body=X("q0"), qubit_name="q0"),
                rel_time=364e-9,
            )

    """

    def __init__(
        self,
        body: Operation | Schedule,
        qubit_name: str,
        t0: float = 0.0,
    ) -> None:
        super().__init__(
            body,
            qubit_name,
            t0,
            hardware_buffer_time=constants.MIN_TIME_BETWEEN_OPERATIONS * 1e-9,
        )
