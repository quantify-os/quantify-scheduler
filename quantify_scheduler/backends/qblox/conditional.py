# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing logic to handle conditional playback."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.operation_handling.base import (
        IOperationStrategy,
    )


@dataclass
class ConditionalManager:
    """Class to manage a conditional control flow."""

    enable_conditional: list = field(default_factory=list)
    """Reference to initial `FEEDBACK_SET_COND` instruction."""
    num_real_time_instructions: int = 0
    """Number of real time instructions."""
    start_time: int = 0
    """Start time of conditional playback."""
    end_time: int = 0
    """End time of conditional playback."""

    def update(self, operation: IOperationStrategy) -> None:
        """
        Update the conditional manager.

        Parameters
        ----------
        operation : IOperationStrategy
            Operation whose information is used to update the conditional manager.
        time :
            Timing

        """
        if operation.operation_info.is_real_time_io_operation:
            self.num_real_time_instructions += 1

    def reset(self) -> None:
        """Reset the conditional manager."""
        self.num_real_time_instructions = 0

    @property
    def duration(self) -> int:
        """Duration of the conditional playback."""
        return self.end_time - self.start_time


class FeedbackTriggerOperator(Enum):
    """Enum for feedback trigger operations."""

    OR = 0
    """Any selected counters exceed their thresholds."""
    NOR = 1
    """No selected counters exceed their thresholds."""
    AND = 2
    """All selected counters exceed their thresholds."""
    NAND = 3
    """Any selected counters do not exceed their thresholds."""
    XOR = 4
    """An odd number of selected counters exceed their thresholds."""
    XNOR = 5
    """An even number of selected counters exceed their thresholds."""

    def __invert__(self) -> FeedbackTriggerOperator:
        """Define the negation operation for the enum."""
        negation_map = {
            FeedbackTriggerOperator.OR: FeedbackTriggerOperator.NOR,
            FeedbackTriggerOperator.NOR: FeedbackTriggerOperator.OR,
            FeedbackTriggerOperator.AND: FeedbackTriggerOperator.NAND,
            FeedbackTriggerOperator.NAND: FeedbackTriggerOperator.AND,
            FeedbackTriggerOperator.XOR: FeedbackTriggerOperator.XNOR,
            FeedbackTriggerOperator.XNOR: FeedbackTriggerOperator.XOR,
        }
        return negation_map[self]


@dataclass
class FeedbackTriggerCondition:
    """Contains all information needed to enable conditional playback."""

    enable: bool
    """Enable/disable conditional playback."""

    operator: FeedbackTriggerOperator
    """
    Specifies the logic to apply on the triggers that are selected by the mask.
    See :class:`~FeedbackTriggerOperator` for more information.
    """

    addresses: InitVar[Sequence[int]]
    """
    Sequence of trigger addresses to condition on. Addresses may
    range from 1 to 15.
    """

    mask: int = field(init=False)
    """
    Represents a bitwise mask in base-10. It dictates which trigger addresses
    will be monitored. For example, to track addresses 0 and 3, the mask would
    be 1001 in binary, which is 17 in base-10. This mask together with the
    operator will determine the conditional operation.
    """

    def __post_init__(self, addresses: Sequence[int]) -> None:
        """
        Compute the mask that selects the addresses to be used.

        This method is automatically invoked during the object's initialization
        to set the `mask` attribute.

        Example:
        If we want to create a mask for the addresses 1 and 3, we have
        the mask in binary `0101`, which corresponds to 5 in decimal notation.

        Parameters
        ----------
        duration : int
            Duration of the conditional playback in ns (at least 4).
        addresses : Sequence[int]
            List of addresses used for computing the mask.

        """
        self.mask = sum(2 ** (address - 1) for address in addresses)
