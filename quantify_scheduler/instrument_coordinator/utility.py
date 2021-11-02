# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Utility functions for the instrument coordinator and components."""

import logging
from typing import Any

from qcodes.instrument.base import InstrumentBase

logger = logging.getLogger(__name__)


def lazy_set(instrument: InstrumentBase, parameter_name: str, val: Any) -> None:
    """
    Sets the value of a QCoDeS parameter only if it is different from the value in
    cache.

    Parameters
    ----------
    instrument:
        The QCoDeS instrument to set the parameter on.
    parameter_name:
        Name of the parameter to set.
    val:
        Value to set it to. If the value is `None` this always gets set, due to it being
        the same as the initial value of the cache.
    """
    parameter = instrument.parameters[parameter_name]
    if parameter.cache() != val or val is None:
        parameter.set(val)
    else:
        logger.info(
            f"Lazy set skipped setting parameter {instrument.name}.{parameter_name}"
        )
