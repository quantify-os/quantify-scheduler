# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing definitions related to stitched pulses."""
from __future__ import annotations

import warnings

from quantify_core.utilities import deprecated

from quantify_scheduler.backends.qblox.operations import (
    stitched_pulse as qblox_stitched_pulse,
)
from quantify_scheduler.operations.operation import Operation


class StitchedPulse(Operation):
    """Deprecated StitchedPulse."""

    def __new__(cls, *args, **kwargs) -> qblox_stitched_pulse.StitchedPulse:
        """Return StitchedPulse from the new location."""
        warnings.warn(
            (
                "Class quantify_scheduler.operations.stitched_pulse.StitchedPulse is "
                "deprecated and will be removed in quantify-scheduler-0.20.0. Use "
                "quantify_scheduler.backends.qblox.operations.stitched_pulse.StitchedPulse "
                "instead."
            ),
            FutureWarning,
        )
        return qblox_stitched_pulse.StitchedPulse(*args, **kwargs)


@deprecated("0.20.0", qblox_stitched_pulse.convert_to_numerical_pulse)
def convert_to_numerical_pulse():
    """Deprecated convert_to_numerical_pulse."""


class StitchedPulseBuilder:
    """Deprecated StitchedPulseBuilder."""

    def __new__(cls, *args, **kwargs) -> qblox_stitched_pulse.StitchedPulseBuilder:
        """Return StitchedPulseBuilder from the new location."""
        warnings.warn(
            (
                "Class "
                "quantify_scheduler.operations.stitched_pulse.StitchedPulseBuilder is "
                "deprecated and will be removed in quantify-scheduler-0.20.0. Use "
                "quantify_scheduler.backends.qblox.operations.stitched_pulse.StitchedPulseBuilder "
                "instead."
            ),
            FutureWarning,
        )
        return qblox_stitched_pulse.StitchedPulseBuilder(*args, **kwargs)
