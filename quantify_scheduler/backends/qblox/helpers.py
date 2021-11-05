# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Helper functions for Qblox backend."""

from collections import UserDict
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
from typing_extensions import Literal

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.helpers.waveforms import exec_waveform_function


# pylint: disable=invalid-name
def find_inner_dicts_containing_key(d: Union[dict], key: Any) -> List[dict]:
    """
    Generates a list of the first dictionaries encountered that contain a certain key,
    in a complicated dictionary with nested dictionaries or Iterables.

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
    d
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
    data_dict
        The dictionary that contains the values needed to parameterize the
        waveform. `data_dict['wf_func']` is then called to calculate the values.
    sampling_rate
        The sampling rate used to generate the time axis values.

    Returns
    -------
    :
        The (possibly complex) values of the generated waveform.
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
    uuid_I:
        Name for the I waveform.
    uuid_Q:
        Name for the Q waveform.
    """
    return f"{str(uuid)}_I", f"{str(uuid)}_Q"


def generate_uuid_from_wf_data(wf_data: np.ndarray, decimals: int = 12) -> str:
    """
    Creates a unique identifier from the waveform data, using a hash. Identical arrays
    yield identical strings within the same process.

    Parameters
    ----------
    wf_data:
        The data to generate the unique id for.
    decimals:
        The number of decimal places to consider.

    Returns
    -------
    :
        A unique identifier.
    """
    waveform_hash = hash(wf_data.round(decimals=decimals).tobytes())
    return str(waveform_hash)


def output_name_to_outputs(name: str) -> Union[Tuple[int], Tuple[int, int]]:
    """
    Finds the output path index associated with the output names specified in the
    config.

    For the baseband modules, these indices correspond directly to a physical output (
    e.g. index 0 corresponds to output 1 etc.).

    For the RF modules, index 0 and 2 correspond to path0 of output 1 and output 2
    respectively, and 1 and 3 to path1 of those outputs.

    Parameters
    ----------
    name
        name of the output channel. e.g. 'complex_output_0'.

    Returns
    -------
    :
        A tuple containing the indices of the physical (real) outputs.
    """
    return {
        "complex_output_0": (0, 1),
        "complex_output_1": (2, 3),
        "real_output_0": (0,),
        "real_output_1": (1,),
        "real_output_2": (2,),
        "real_output_3": (3,),
    }[name]


def output_mode_from_outputs(
    outputs: Union[Tuple[int], Tuple[int, int]]
) -> Literal["complex", "real", "imag"]:
    """
    Takes the specified outputs to use and extracts a "sequencer mode" from it.

    Modes:

    - ``"real"``: only path0 is used
    - ``"imag"``: only path1 is used
    - ``"complex"``: both path0 and path1 paths are used.

    Parameters
    ----------
    outputs
        The outputs the sequencer is supposed to use. Note that the outputs start from
        0, but the labels on the front panel start counting from 1. So the mapping
        differs n-1.

    Returns
    -------
    :
        The mode

    Raises
    ------
    RuntimeError
        The amount of outputs is more than 2, which is impossible for one sequencer.
    """
    if len(outputs) > 2:
        raise RuntimeError(
            f"Too many outputs specified for this channel. Given: {outputs}."
        )

    if len(outputs) == 2:
        assert (
            outputs[0] - outputs[1]
        ) ** 2 == 1, "Attempting to use two outputs that are not next to each other."
        if 1 in outputs:
            assert 2 not in outputs, (
                "Attempting to use output 1 and output 2 (2 and 3 on front panel) "
                "together, but they belong to different pairs."
            )
        return "complex"

    output = outputs[0]
    mode = "real" if output % 2 == 0 else "imag"
    return mode


def generate_waveform_dict(waveforms_complex: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Takes a dictionary with complex waveforms and generates a new dictionary with
    real valued waveforms with a unique index, as required by the hardware.

    Parameters
    ----------
    waveforms_complex
        Dictionary containing the complex waveforms. Keys correspond to a unique
        identifier, value is the complex waveform.

    Returns
    -------
    :
        A dictionary with as key the unique name for that waveform, as value another
        dictionary containing the real-valued data (list) as well as a unique index.
        Note that the index of the Q waveform is always the index of the I waveform
        +1.

    .. admonition:: Examples

        .. jupyter-execute::

            import numpy as np
            from quantify_scheduler.backends.qblox.helpers import generate_waveform_dict

            complex_waveforms = {12345: np.array([1, 2])}
            generate_waveform_dict(complex_waveforms)

            # {'12345_I': {'data': [1, 2], 'index': 0},
            # '12345_Q': {'data': [0, 0], 'index': 1}}
    """
    wf_dict = {}
    for idx, (uuid, complex_data) in enumerate(waveforms_complex.items()):
        name_i, name_q = generate_waveform_names_from_uuid(uuid)
        to_add = {
            name_i: {"data": complex_data.real.tolist(), "index": 2 * idx},
            name_q: {"data": complex_data.imag.tolist(), "index": 2 * idx + 1},
        }
        wf_dict.update(to_add)
    return wf_dict


def to_grid_time(time: float, grid_time_ns: int = constants.GRID_TIME) -> int:
    """
    Takes a float value representing a time in seconds as used by the schedule, and
    returns the integer valued time in nanoseconds that the sequencer uses.

    Parameters
    ----------
    time
        The time to convert.
    grid_time_ns
        The grid time to use in ns.

    Returns
    -------
    :
        The integer valued nanosecond time.
    """
    time_ns = int(round(time * 1e9))
    if time_ns % grid_time_ns != 0:
        raise ValueError(
            f"Attempting to use a time interval of {time_ns} ns. "
            f"Please ensure that the durations of operations and wait times between"
            f" operations are multiples of {grid_time_ns} ns."
        )
    return time_ns


def is_multiple_of_grid_time(
    time: float, grid_time_ns: int = constants.GRID_TIME
) -> bool:
    """
    Takes a time in seconds and converts it to the ns grid time that the Qblox hardware
    expects.

    Parameters
    ----------
    time:
        A time in seconds.
    grid_time_ns
        A grid time in ns.

    Returns
    -------
    :
        If it the time is a multiple of the grid time.
    """
    time_ns = int(round(time * 1e9))
    return time_ns % grid_time_ns == 0
