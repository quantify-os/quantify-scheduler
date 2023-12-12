# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Enums used by Qblox backend."""
from enum import Enum


class ChannelMode(str, Enum):
    """Enum for the channel mode of the Sequencer."""

    COMPLEX = "complex"
    REAL = "real"
    DIGITAL = "digital"
