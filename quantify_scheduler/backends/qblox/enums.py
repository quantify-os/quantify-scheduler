# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Enums used by Qblox backend."""
from enum import Enum


class IoMode(str, Enum):
    """Enum for the IO mode of the Sequencer."""

    COMPLEX = "complex"
    REAL = "real"
    IMAG = "imag"
    DIGITAL = "digital"
