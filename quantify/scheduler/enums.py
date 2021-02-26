# -----------------------------------------------------------------------------
# Description:    Enums for quantify-scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------

from enum import Enum, unique


@unique
class BinMode(str, Enum):
    """
    The acquisition protocol bin mode enum type.

    Used to set the bin type to
    append or average respectively.

    BinMode `APPEND` uses a list where every new
    result will be appended to the list.

    BinMode `AVERAGE` incrementally stores the weighted
    average result.
    """

    APPEND = "append"
    AVERAGE = "average"
