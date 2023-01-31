# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utility functions for the instrument coordinator and components."""

import logging
from typing import Any

from qcodes.instrument.base import InstrumentBase
from qcodes.instrument import Parameter

logger = logging.getLogger(__name__)


def search_settable_param(
    instrument: InstrumentBase, nested_parameter_name: str
) -> Parameter:
    """
    Searches for a settable parameter of an instrument when it is in a nested
    hierarchical form such as instrument.submodule_1.channel_1.parameter

    Parameters
    ----------
    instrument:
        The root QCoDeS instrument where the parameter resides.
    nested_parameter_name:
        Hierarchical nested parameter name.

    Returns
    -------
    Parameter:

    """
    root_param = instrument
    split_params = nested_parameter_name.split(".")

    def _search_next_level(child_parameter_name, root_attr_dicts_list):
        for root_attr_dict in root_attr_dicts_list:
            if callable(child_parameter_name):
                return child_parameter_name
            if child_parameter_name in root_attr_dict:
                return root_attr_dict.get(child_parameter_name)

        return None

    # Search for the parameter within the parameter, function
    # or submodule delegate_attrs_dict of the instrument
    for child_parameter_name in split_params:
        root_attr_dicts_list = [
            root_param.parameters,
            root_param.submodules,
            root_param.functions,
        ]
        root_param = _search_next_level(child_parameter_name, root_attr_dicts_list)
        if root_param is None:
            break

    if not (isinstance(root_param, Parameter) or callable(root_param)):
        raise ValueError(
            f"Could not find settable parameter "
            f'"{nested_parameter_name}" in instrument "{instrument}"'
        )

    return root_param


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
        Value to set it to.
    """
    parameter = search_settable_param(
        instrument=instrument, nested_parameter_name=parameter_name
    )
    # parameter.cache() throws for non-gettable parameters if the cache is invalid. This order prevents the exception.
    if not parameter.cache.valid or parameter.cache() != val:
        parameter.set(val)
    else:
        logger.info(
            f"Lazy set skipped setting parameter {instrument.name}.{parameter_name}"
        )


def check_already_existing_acq_index(index, acqs):
    if index in acqs:
        acq_channel, acq_idx = index
        raise RuntimeError(
            f"Attempting to gather acquisitions. "
            f"Acquisition {acq_channel=}, {acq_idx=} is already stored, "
            f"make sure for an acq_channel, acq_index corresponds to not more than one acquisition."
        )
