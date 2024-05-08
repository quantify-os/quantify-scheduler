# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Enums used by Qblox backend."""
from __future__ import annotations

from enum import Enum


class ChannelMode(str, Enum):
    """Enum for the channel mode of the Sequencer."""

    COMPLEX = "complex"
    REAL = "real"
    DIGITAL = "digital"


class QbloxFilterConfig(str, Enum):
    """Configuration of a filter."""

    BYPASSED = "bypassed"
    ENABLED = "enabled"
    DELAY_COMP = "delay_comp"


class QbloxFilterMarkerDelay(str, Enum):
    """Marker delay setting of a filter."""

    BYPASSED = "bypassed"
    DELAY_COMP = "delay_comp"


class DistortionCorrectionLatencyEnum(int, Enum):
    """Settings related to distortion corrections."""

    NO_DELAY_COMP = 0
    """Setting for no distortion correction delay compensation"""
    BT = 1
    """Setting for delay compensation equal to bias tee correction"""
    EXP0 = 2
    """Setting for delay compensation equal to exponential overshoot or undershoot correction"""
    EXP1 = 4
    """Setting for delay compensation equal to exponential overshoot or undershoot correction"""
    EXP2 = 8
    """Setting for delay compensation equal to exponential overshoot or undershoot correction"""
    EXP3 = 16
    """Setting for delay compensation equal to exponential overshoot or undershoot correction"""
    FIR = 32
    """Setting for delay compensation equal to FIR filter"""

    def __int__(self) -> int:
        """Enable direct conversion to int."""
        return self.value

    def __index__(self) -> int:
        """Support index operations."""
        return self.value

    def __and__(self, other: DistortionCorrectionLatencyEnum | int) -> int:
        """Support bitwise AND operations."""
        if isinstance(other, Enum):
            return self.value & other.value
        return self.value & other

    def __rand__(self, other: DistortionCorrectionLatencyEnum | int) -> int:
        """Support bitwise AND operations, other order."""
        return self.__and__(other)

    def __or__(self, other: DistortionCorrectionLatencyEnum | int) -> int:
        """Support bitwise OR operations."""
        if isinstance(other, Enum):
            return self.value | other.value
        return self.value | other

    def __ror__(self, other: DistortionCorrectionLatencyEnum | int) -> int:
        """Support bitwise OR operations, other order."""
        return self.__or__(other)
