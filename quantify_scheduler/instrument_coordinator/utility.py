# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utility functions for the instrument coordinator and components."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray
from qcodes.parameters.parameter import Parameter

if TYPE_CHECKING:
    from qcodes.instrument.base import InstrumentBase

logger = logging.getLogger(__name__)


def search_settable_param(instrument: InstrumentBase, nested_parameter_name: str) -> Parameter:
    """
    Searches for a settable parameter in nested instrument hierarchies.

    For example `instrument.submodule_1.channel_1.parameter.`

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

    def _search_next_level(
        child_parameter_name: str | Parameter, root_attr_dicts_list: list
    ) -> Parameter | None:
        if callable(child_parameter_name):
            return child_parameter_name
        for root_attr_dict in root_attr_dicts_list:
            if child_parameter_name in root_attr_dict:
                return root_attr_dict.get(child_parameter_name)

        return

    # Search for the parameter within the parameter, function
    # or submodule delegate_attrs_dict of the instrument
    for child_parameter_name in split_params:
        # On the types: _search_next_level returns either None or an object that has the
        # parameters below. Types are omitted because of their complexity.
        root_attr_dicts_list = [
            root_param.parameters,  # type: ignore
            root_param.submodules,  # type: ignore
            root_param.functions,  # type: ignore
        ]
        root_param = _search_next_level(child_parameter_name, root_attr_dicts_list)
        if root_param is None:
            break

    if not (isinstance(root_param, Parameter) or callable(root_param)):
        raise ValueError(
            f"Could not find settable parameter "
            f'"{nested_parameter_name}" in instrument "{instrument}"'
        )

    # If the return type is not a Parameter, then we assume it is a structural subtype
    # (duck typing) of a Parameter.
    return root_param  # type: ignore


def parameter_value_same_as_cache(
    instrument: InstrumentBase, parameter_name: str, val: object
) -> bool:
    """
    Returns whether the value of a QCoDeS parameter is the same as the value in cache.

    Parameters
    ----------
    instrument:
        The QCoDeS instrument to set the parameter on.
    parameter_name:
        Name of the parameter to set.
    val:
        Value to set it to.

    Returns
    -------
    bool

    """
    parameter = search_settable_param(instrument=instrument, nested_parameter_name=parameter_name)
    # parameter.cache() throws for non-gettable parameters if the cache is invalid.
    # This order prevents the exception.
    return parameter.cache.valid and parameter.cache() == val


def lazy_set(instrument: InstrumentBase, parameter_name: str, val: object) -> None:
    """
    Set the value of a QCoDeS parameter only if it is different from the value in cache.

    Parameters
    ----------
    instrument:
        The QCoDeS instrument to set the parameter on.
    parameter_name:
        Name of the parameter to set.
    val:
        Value to set it to.

    """
    parameter = search_settable_param(instrument=instrument, nested_parameter_name=parameter_name)
    # parameter.cache() throws for non-gettable parameters if the cache is invalid.
    # This order prevents the exception.
    if not parameter_value_same_as_cache(instrument, parameter_name, val):
        parameter.set(val)
    else:
        logger.info(f"Lazy set skipped setting parameter {instrument.name}.{parameter_name}")


def check_already_existing_acquisition(
    new_dataset: xarray.Dataset, current_dataset: xarray.Dataset
) -> None:
    """
    Verifies non-overlapping data in new_dataset and current_dataset.

    If there is, it will raise an error.

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
            # The return values are two `DataArray`s with only coordinates
            # which are common in the inputs.
            common_0, common_1 = xarray.align(
                new_dataset[acq_channel], current_dataset[acq_channel], join="inner"
            )

            # We need to check if the values are `math.nan`, because if they are,
            # that means there is no value at that position (xarray standard).
            def mask_func(x: float, y: float) -> int:
                return 0 if np.isnan(x) or np.isnan(y) else 1

            conflict_mask = xarray.apply_ufunc(mask_func, common_0, common_1, vectorize=True)
            for conflict in conflict_mask:
                if conflict.values == [1]:
                    conflicting_coords = [("acq_channel", acq_channel)]
                    conflicting_coords += [(dim, conflict[dim].values) for dim in conflict.coords]
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
