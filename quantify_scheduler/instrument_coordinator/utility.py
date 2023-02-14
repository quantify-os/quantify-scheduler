# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utility functions for the instrument coordinator and components."""

import logging
from typing import Any

import numpy as np
import xarray

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


def check_already_existing_acquisition(
    new_dataset: xarray.Dataset, current_dataset: xarray.Dataset
):
    """
    Checks whether there is any data which is at the same coordinate in
    `new_dataset` and `current_dataset`. If there is, it will raise an error.

    Parameters
    ----------
    new_dataset
        New dataset.
    current_dataset
        Current dataset.
    """
    conflicting_indices_str = []
    for acq_channel, _data_array in new_dataset.items():
        if acq_channel in current_dataset:
            # The return values are two `DataArray`s with only coordinates which are common in the inputs.
            common_0, common_1 = xarray.align(
                new_dataset[acq_channel], current_dataset[acq_channel], join="inner"
            )
            # We need to check if the values are `math.nan`, because if they are,
            # that means there is no value at that position (xarray standard).
            mask_func = lambda x, y: 0 if (np.isnan(x) or np.isnan(y)) else 1
            conflict_mask = xarray.apply_ufunc(
                mask_func, common_0, common_1, vectorize=True
            )
            for conflict in conflict_mask:
                if conflict.values == [1]:
                    conflicting_coords = [("acq_channel", acq_channel)]
                    conflicting_coords += [
                        (dim, conflict[dim].values) for dim in conflict.coords
                    ]
                    coords_str = [f"{dim}={coord}" for dim, coord in conflicting_coords]
                    conflicting_indices_str.append("; ".join(coords_str))

    if conflicting_indices_str:
        conflicting_indices_str = "\n".join(conflicting_indices_str)
        raise RuntimeError(
            f"Attempting to gather acquisitions. "
            f"Make sure an acq_channel, acq_index corresponds to not more than one acquisition.\n"
            f"The following indices are defined multiple times.\n"
            f"{conflicting_indices_str}"
        )
