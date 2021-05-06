# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Helper functions for Qblox backend."""

from typing import Any, Dict, Union, List, Iterable, Tuple
from collections import UserDict

import numpy as np

from quantify.scheduler.helpers.waveforms import exec_waveform_function

try:
    from qblox_instruments.build import __version__ as driver_version
except ImportError:
    driver_version = None

SUPPORTED_DRIVER_VERSIONS = ("0.3.2",)


class DriverVersionError(Exception):
    """
    Raise when the installed driver version is not supported
    """


def verify_qblox_instruments_version():
    """
    Verifies whether the installed version is supported by the qblox_backend.

    Raises
    ------
    DriverVersionError
        When an incorrect or no installation of qblox-instruments was found.
    """
    if driver_version is None:
        raise DriverVersionError(
            "Qblox DriverVersionError: qblox-instruments version check could not be "
            "performed. Either the package is not installed "
            "correctly or a version < 0.3.2 was found."
        )
    if driver_version not in SUPPORTED_DRIVER_VERSIONS:
        message = (
            f"Qblox DriverVersionError: Installed driver version {driver_version}"
            f" not supported by backend."
        )
        message += (
            f" Please install version {SUPPORTED_DRIVER_VERSIONS[0]}"
            if len(SUPPORTED_DRIVER_VERSIONS) == 1
            else f" Please install a supported version (currently supported: "
            f"{SUPPORTED_DRIVER_VERSIONS})"
        )
        message += " to continue to use this backend."
        raise DriverVersionError(message)


# pylint: disable=invalid-name
def find_inner_dicts_containing_key(d: Union[dict], key: Any) -> List[dict]:
    """
    Generates a list of the first dictionaries encountered that contain a certain key,
    in a complicated dictionary with nested dictionaries or Iterables.

    This is achieved by recursively traversing the nested structures until the key is
    found, which is then appended to a list.

    Parameters
    ----------
    d:
        The dictionary to traverse.
    key:
        The key to search for.

    Returns
    -------
    :
        A list containing all the inner dictionaries containing the specified key.
    """
    dicts_found = list()
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


# pylint: disable=invalid-name
def find_all_port_clock_combinations(d: dict) -> List[Tuple[str, str]]:
    """
    Generates a list with all port and clock combinations found in a dictionary with
    nested structures. Traversing the dictionary is done using the
    `find_inner_dicts_containing_key` function.

    Parameters
    ----------
    d:
        The dictionary to traverse.

    Returns
    -------
    :
        A list containing tuples representing the port and clock combinations found
        in the dictionary.
    """
    port_clocks = list()
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


def generate_waveform_data(data_dict: dict, sampling_rate: float) -> np.ndarray:
    """
    Generates an array using the parameters specified in `data_dict`.

    Parameters
    ----------
    data_dict:
        The dictionary that contains the values needed to parameterize the
        waveform. `data_dict['wf_func']` is then called to calculate the values.
    sampling_rate:
        The sampling rate used to generate the time axis values.

    Returns
    -------
    :
        The (possibly complex) values of the generated waveform
    """
    time_duration = data_dict["duration"]
    t = np.linspace(0, time_duration, int(time_duration * sampling_rate))

    wf_data = exec_waveform_function(data_dict["wf_func"], t, data_dict)

    return wf_data


def generate_waveform_names_from_uuid(uuid: Any) -> Tuple[str, str]:
    """
    Generates names for the I and Q parts of the complex waveform based on a unique
    identifier for the pulse/acquisition.

    Parameters
    ----------
    uuid
        A unique identifier for a pulse/acquisition.

    Returns
    -------
    :
        Name for the I waveform.
    :
        Name for the Q waveform.
    """
    return f"{str(uuid)}_I", f"{str(uuid)}_Q"


def _generate_waveform_dict(
    waveforms_complex: Dict[int, np.ndarray]
) -> Dict[str, dict]:
    """
    Takes a dictionary with complex waveforms and generates a new dictionary with
    real valued waveforms with a unique index, as required by the hardware.

    Parameters
    ----------
    waveforms_complex:
        Dictionary containing the complex waveforms. Keys correspond to a unique
        identifier, value is the complex waveform.

    Returns
    -------
    :
        A dictionary with as key the unique name for that waveform, as value another
        dictionary containing the real-valued data (list) as well as a unique index.
        Note that the index of the Q waveform is always the index of the I waveform
        +1.

    Examples
    --------

    .. code-block::

        complex_waveforms = {12345: np.array([1, 2])}
        _generate_waveform_dict(complex_waveforms)
        {
            "12345_I": {"data": [1, 2], "index": 0},
            "12345_Q": {"data": [0, 0], "index": 1}
        }
    """
    wf_dict = dict()
    for idx, (uuid, complex_data) in enumerate(waveforms_complex.items()):
        name_i, name_q = generate_waveform_names_from_uuid(uuid)
        to_add = {
            name_i: {"data": complex_data.real.tolist(), "index": 2 * idx},
            name_q: {"data": complex_data.imag.tolist(), "index": 2 * idx + 1},
        }
        wf_dict.update(to_add)
    return wf_dict
