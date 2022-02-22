# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions for Qblox backend."""

from collections import UserDict
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
from typing_extensions import Literal

from quantify_core.utilities.general import without

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.helpers.waveforms import exec_waveform_function
from quantify_scheduler import Schedule

from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.operations.pulse_library import WindowOperation


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


def add_to_wf_dict_if_unique(
    wf_dict: Dict[str, Any], waveform: np.ndarray
) -> Tuple[Dict[str, Any], str, int]:
    """
    Adds a waveform to the waveform dictionary if it is not yet in there and returns the
    uuid and index. If it is already present it simply returns the uuid and index.

    Parameters
    ----------
    wf_dict:
        The waveform dict in the format expected by the sequencer.
    waveform:
        The waveform to add.

    Returns
    -------
    Dict[str, Any]
        The (updated) wf_dict.
    str
        The uuid of the waveform.
    int
        The index.
    """

    def generate_entry(name: str, data: np.ndarray, idx: int) -> Dict[str, Any]:
        return {name: {"data": data.tolist(), "index": idx}}

    if not np.isrealobj(waveform):
        raise RuntimeError("This function only accepts real arrays.")

    uuid = generate_uuid_from_wf_data(waveform)
    if uuid in wf_dict:
        index: int = wf_dict[uuid]["index"]
    else:
        index = len(wf_dict)
        wf_dict.update(generate_entry(uuid, waveform, len(wf_dict)))
    return wf_dict, uuid, index


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


def get_nco_phase_arguments(phase_deg: float) -> Tuple[int, int, int]:
    """
    Converts a phase in degrees to the int arguments the NCO phase instructions expect.

    Parameters
    ----------
    phase_deg
        The phase in degrees

    Returns
    -------
    :
        The three ints corresponding to the phase arguments (course, fine, ultra-fine).
    """
    phase_course: int = int(phase_deg // constants.NCO_PHASE_DEG_STEP_COURSE)
    assert phase_course <= constants.NCO_PHASE_NUM_STEP_COURSE

    remaining_phase = phase_deg % constants.NCO_PHASE_DEG_STEP_COURSE
    phase_fine: int = int(remaining_phase // constants.NCO_PHASE_DEG_STEP_FINE)
    assert phase_fine <= constants.NCO_PHASE_NUM_STEP_FINE

    remaining_phase = remaining_phase % constants.NCO_PHASE_DEG_STEP_FINE
    phase_ultra_fine: int = int(remaining_phase // constants.NCO_PHASE_DEG_STEP_U_FINE)
    assert phase_fine <= constants.NCO_PHASE_NUM_STEP_U_FINE

    return phase_course, phase_fine, phase_ultra_fine


def generate_port_clock_to_device_map(
    mapping: Dict[str, Any]
) -> Dict[Tuple[str, str], str]:
    """
    Generates a mapping which specifies which port-clock combinations belong to which
    device.

    .. note::
        The same device may contain multiple port-clock combinations, but each
        port-clock combination may only occur once.

    Parameters
    ----------
    mapping:
        The hardware mapping config.

    Returns
    -------
    :
        A dictionary with as key a tuple representing a port-clock combination, and
        as value the name of the device. Note that multiple port-clocks may point to
        the same device.
    """

    portclock_map = dict()
    for device_name, device_info in mapping.items():
        if not isinstance(device_info, dict):
            continue

        portclocks = find_all_port_clock_combinations(device_info)

        for portclock in portclocks:
            portclock_map[portclock] = device_name

    return portclock_map


# pylint: disable=too-many-locals
def _assign_pulse_and_acq_info_to_devices(
    schedule: Schedule,
    device_compilers: Dict[str, Any],
    portclock_mapping: Dict[Tuple[str, str], str],
):
    """
    Traverses the schedule and generates `OpInfo` objects for every pulse and
    acquisition, and assigns it to the correct `InstrumentCompiler`.

    Parameters
    ----------
    schedule
        The schedule to extract the pulse and acquisition info from.
    device_compilers
        Dictionary containing InstrumentCompilers as values and their names as keys.
    portclock_mapping
        A dictionary that maps tuples containing a port and a clock to names of
        instruments. The port and clock combinations are unique, but multiple portclocks
        can point to the same instrument.

    Raises
    ------
    RuntimeError
        This exception is raised then the function encountered an operation that has no
        pulse or acquisition info assigned to it.
    KeyError
        This exception is raised when attempting to assign a pulse with a port-clock
        combination that is not defined in the hardware configuration.
    KeyError
        This exception is raised when attempting to assign an acquisition with a
        port-clock combination that is not defined in the hardware configuration.
    """

    for op_timing_constraint in schedule.schedulables.values():
        op_hash = op_timing_constraint["operation_repr"]
        op_data = schedule.operations[op_hash]

        if isinstance(op_data, WindowOperation):
            continue

        if not op_data.valid_pulse and not op_data.valid_acquisition:
            raise RuntimeError(
                f"Operation {op_hash} is not a valid pulse or acquisition. Please check"
                f" whether the device compilation been performed successfully. "
                f"Operation data: {repr(op_data)}"
            )

        operation_start_time = op_timing_constraint["abs_time"]
        for pulse_data in op_data.data["pulse_info"]:
            if "t0" in pulse_data:
                pulse_start_time = operation_start_time + pulse_data["t0"]
            else:
                pulse_start_time = operation_start_time

            port = pulse_data["port"]
            clock = pulse_data["clock"]
            if port is None:
                continue  # ignore idle pulses

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=pulse_data,
                timing=pulse_start_time,
            )

            if (port, clock) not in portclock_mapping:
                raise KeyError(
                    f"Could not assign pulse data to device. The combination"
                    f" of port {port} and clock {clock} could not be found "
                    f"in hardware configuration.\n\nAre both the port and clock "
                    f"specified in the hardware configuration?\n\nRelevant operation:\n"
                    f"{combined_data}."
                )
            dev = portclock_mapping[(port, clock)]
            device_compilers[dev].add_pulse(port, clock, pulse_info=combined_data)

        for acq_data in op_data.data["acquisition_info"]:
            if "t0" in acq_data:
                acq_start_time = operation_start_time + acq_data["t0"]
            else:
                acq_start_time = operation_start_time
            port = acq_data["port"]
            clock = acq_data["clock"]
            if port is None:
                continue

            hashed_dict = without(acq_data, ["t0", "waveforms"])
            hashed_dict["waveforms"] = list()
            for acq in acq_data["waveforms"]:
                hashed_dict["waveforms"].append(without(acq, ["t0"]))

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=acq_data,
                timing=acq_start_time,
            )
            if (port, clock) not in portclock_mapping:
                raise KeyError(
                    f"Could not assign acquisition data to device. The combination"
                    f" of port {port} and clock {clock} could not be found "
                    f"in hardware configuration.\n\nAre both the port and clock "
                    f"specified in the hardware configuration?\n\nRelevant operation:\n"
                    f"{combined_data}."
                )
            dev = portclock_mapping[(port, clock)]
            device_compilers[dev].add_acquisition(port, clock, acq_info=combined_data)


def assign_pulse_and_acq_info_to_devices(
    schedule: Schedule,
    mapping: Dict[str, Any],
    device_compilers: Dict[str, Any],
):
    """
    Traverses the schedule and generates `OpInfo` objects for every pulse and
    acquisition, and assigns it to the correct `InstrumentCompiler`.

    Parameters
    ----------
    schedule
        The schedule to extract the pulse and acquisition info from.
    mapping
        The hardware mapping config.
    device_compilers
        Dictionary containing InstrumentCompilers as values and their names as keys.


    Raises
    ------
    RuntimeError
        This exception is raised then the function encountered an operation that has no
        pulse or acquisition info assigned to it.
    KeyError
        This exception is raised when attempting to assign a pulse with a port-clock
        combination that is not defined in the hardware configuration.
    KeyError
        This exception is raised when attempting to assign an acquisition with a
        port-clock combination that is not defined in the hardware configuration.
    """

    portclock_mapping = generate_port_clock_to_device_map(mapping)
    _assign_pulse_and_acq_info_to_devices(schedule, device_compilers, portclock_mapping)
