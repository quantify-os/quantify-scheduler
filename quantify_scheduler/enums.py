# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Enums for quantify-scheduler."""

from enum import Enum, unique


class StrEnum(Enum):
    """Enum that can be directly serialized to string."""

    def __str__(self):
        return self.value


@unique
class BinMode(StrEnum):
    """
    The acquisition protocol bin mode enum type.

    Used to set the bin type to
    append or average respectively.

    BinMode ``APPEND`` uses a list where every new
    result will be appended to the list.

    BinMode ``AVERAGE`` incrementally stores the weighted
    average result.
    """

    APPEND = "append"
    AVERAGE = "average"
