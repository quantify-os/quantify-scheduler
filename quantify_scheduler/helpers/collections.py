# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helpers for various collections."""

import copy
from typing import Any

import numpy as np
import xxhash


def make_hash(obj: Any) -> int:
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    From: https://stackoverflow.com/questions/5884066/hashing-a-dictionary

    Parameters
    ----------
    obj
        Input collection.

    Returns
    -------
    :
        Hash.
    """
    new_hash = xxhash.xxh64()
    if isinstance(obj, (set, tuple, list)):
        return hash(tuple(make_hash(e) for e in obj))

    if isinstance(obj, np.ndarray):
        # numpy arrays behave funny for hashing
        new_hash.update(obj)
        val = new_hash.intdigest()
        new_hash.reset()
        return val

    if not isinstance(obj, dict):
        return hash(obj)

    new_o = copy.deepcopy(obj)
    for key, val in new_o.items():
        new_o[key] = make_hash(val)

    return hash(tuple(frozenset(sorted(new_o.items()))))


def without(dict_in: dict, keys: list) -> dict:
    """
    Utility that copies a dictionary excluding a specific list of keys.

    Parameters
    ----------
    dict_in
        Input dictionary.
    keys
        List of keys to exclude.

    Returns
    -------
    :
        Filtered dictionary.
    """
    if not isinstance(keys, list):
        keys = [keys]
    new_d = dict_in.copy()
    for key in keys:
        new_d.pop(key)
    return new_d
