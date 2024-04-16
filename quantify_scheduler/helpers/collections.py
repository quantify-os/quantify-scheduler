# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helpers for various collections."""
from __future__ import annotations

import copy
from collections import UserDict
from typing import Any, List, Iterable, Tuple

import numpy as np
import xxhash


def make_hash(obj: Any) -> int:
    """
    Make a hash from a dictionary, list, tuple or set to any level.

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

    tuple_of_hashes = ((key, make_hash(val)) for key, val in obj.items())
    return hash(frozenset(sorted(tuple_of_hashes)))


def without(dict_in: dict, keys: list) -> dict:
    """
    Copy a dictionary excluding a specific list of keys.

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


def find_inner_dicts_containing_key(d: dict, key: Any) -> List[dict]:
    """
    Generate a list of the first dictionaries encountered that contain a certain key.

    This is achieved by recursively traversing the nested structures until the key is
    found, which is then appended to a list.

    Parameters
    ----------
    d
        The dictionary to traverse.
    key
        The key to search for.

    Returns
    -------
    :
        A list containing all the inner dictionaries containing the specified key.
    """
    dicts_found = []
    if isinstance(d, dict):
        if key in d:
            dicts_found.append(d)
    for val in d.values():
        if isinstance(val, (dict, UserDict)):
            dicts_found.extend(find_inner_dicts_containing_key(val, key))
        elif isinstance(val, Iterable) and not isinstance(val, str):
            for i_item in val:
                try:
                    dicts_found.extend(find_inner_dicts_containing_key(i_item, key))
                # having a list that contains something other than a dict can cause an
                # AttributeError on d, but this should be ignored anyway
                except AttributeError:
                    continue
        else:
            continue
    return dicts_found


def find_all_port_clock_combinations(d: dict) -> List[Tuple[str, str]]:
    """
    Generate a list with all port-clock combinations found in a nested dictionary.

    Traversing the dictionary is done using the
    ``find_inner_dicts_containing_key`` function.

    Parameters
    ----------
    d
        The dictionary to traverse.

    Returns
    -------
    :
        A list containing tuples representing the port and clock combinations found
        in the dictionary.
    """
    port_clocks = []
    dicts_with_port = find_inner_dicts_containing_key(d, "port")
    for inner_dict in dicts_with_port:
        if "port" in inner_dict:
            port = inner_dict["port"]
            if port is None:
                continue
            if "clock" not in inner_dict:
                raise AttributeError(f"Port {inner_dict['port']} missing clock")
            clock = inner_dict["clock"]
            port_clocks.append((port, clock))
    return port_clocks


def find_port_clock_path(hardware_config: dict, port: str, clock: str) -> list | None:
    """
    Find the path to a port-clock combination in a nested dictionary.

    Parameters
    ----------
    hardware_config
        The (nested) hardware config dictionary to loop over.
    port
        The port to find.
    clock
        The clock to find.

    Returns
    -------
    :
        A list representing the keys to the port-clock combination in the hardware config.
        If the port-clock location is in a list, the list index is also included in this path.
    """

    def recursive_find(hardware_config, port, clock, path) -> list | None:
        for k, v in hardware_config.items():
            # If key is port, we are done
            if k == "port":
                if (
                    hardware_config["port"] == port
                    and hardware_config["clock"] == clock
                ):
                    return path

            # If value is list, append key to path and loop trough its elements.
            elif isinstance(v, list):
                path.append(k)  # Add list key to path.
                for i, sub_config in enumerate(v):
                    path.append(i)  # Add list element index to path.
                    if isinstance(sub_config, dict):
                        found_path = recursive_find(sub_config, port, clock, path)
                        if found_path:
                            return found_path
                    path.pop()  # Remove list index if port-clock not found in element.
                path.pop()  # Remove list key if port-clock not found in list.

            # If dict append its key. If port is not found delete it
            elif isinstance(v, dict):
                path.append(k)
                found_path = recursive_find(v, port, clock, path)
                if found_path:
                    return found_path
                path.pop()  # Remove dict key if port-clock not found in this dict.

    port_clock_path = recursive_find(hardware_config, port, clock, path=[])
    if port_clock_path is None:
        raise KeyError(
            f"The combination of {port=} and {clock=} could not be found in {hardware_config=}."
        )
    else:
        return port_clock_path
