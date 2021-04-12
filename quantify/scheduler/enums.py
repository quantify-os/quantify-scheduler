# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Enums for quantify-scheduler."""

from enum import Enum, unique


@unique
class SignalModeType(str, Enum):

    """
    The signal output enum type.

    Used to set the output signal type to a
    modulated or real respectively.
    """

    COMPLEX = "complex"
    REAL = "real"


@unique
class ReferenceSourceType(str, Enum):

    """
    The reference source enum type.

    Used to set the source trigger type to
    internal or external respectively.
    """

    NONE = "none"
    INTERNAL = "int"
    EXTERNAL = "ext"


@unique
class ModulationModeType(str, Enum):

    """
    The modulation mode enum type.

    Used to set the modulation type to
    None, premodulation or hardware modulation
    respectively.
    """

    NONE = "none"
    PREMODULATE = "premod"
    MODULATE = "modulate"


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
