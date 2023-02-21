# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions for Qblox backend."""
import dataclasses
import re
import warnings
from copy import deepcopy
from collections import UserDict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np

from quantify_core.utilities.general import without

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.helpers.waveforms import exec_waveform_function
from quantify_scheduler.schedules.schedule import AcquisitionMetadata
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_acquisition_protocols,
)
from quantify_scheduler import Schedule

from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.operations.pulse_library import WindowOperation


# pylint: disable=invalid-name
def find_inner_dicts_containing_key(d: dict, key: Any) -> List[dict]:
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
        The (possibly complex) values of the generated waveform. The number of values is
        determined by rounding to the nearest integer.
    """
    time_duration = data_dict["duration"]
    t = np.linspace(
        start=0, stop=time_duration, num=int(np.round(time_duration * sampling_rate))
    )

    wf_data = exec_waveform_function(
        wf_func=data_dict["wf_func"], t=t, pulse_info=data_dict
    )

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


def output_name_to_outputs(name: str) -> Optional[Union[Tuple[int], Tuple[int, int]]]:
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
    if "output" not in name:
        return None

    return {
        "complex_output_0": (0, 1),
        "complex_output_1": (2, 3),
        "real_output_0": (0,),
        "real_output_1": (1,),
        "real_output_2": (2,),
        "real_output_3": (3,),
    }[name]


def input_name_to_inputs(name: str) -> Union[Tuple[int], Tuple[int, int]]:
    """
    Finds the input path index associated with the input names specified in the
    config.

    For the baseband modules, these indices correspond directly to a physical input (
    e.g. index 0 corresponds to input 1 etc.).

    For the RF modules, index 0 corresponds to path0 of input 1 and path 1 of input 1.

    Parameters
    ----------
    name
        name of the input channel. e.g. 'real_input_0'.

    Returns
    -------
    :
        A tuple containing the indices of the physical (real) inputs.
    """
    if "input" not in name:
        return None

    return {
        "complex_input_0": (0, 1),
        "real_input_0": (0,),
        "real_input_1": (1,),
    }[name]


def io_mode_from_ios(
    io: Union[Tuple[int], Tuple[int, int]]
) -> Literal["complex", "real", "imag"]:
    """
    Takes the specified outputs to use and extracts a "sequencer mode" from it.

    Modes:

    - ``"real"``: only path0 is used
    - ``"imag"``: only path1 is used
    - ``"complex"``: both path0 and path1 paths are used.

    Parameters
    ----------
    io
        The io the sequencer is supposed to use. Note that the outputs start from
        0, but the labels on the front panel start counting from 1. So the mapping
        differs n-1.

    Returns
    -------
    :
        The mode

    Raises
    ------
    RuntimeError
        The amount of ios is more than 2, which is impossible for one sequencer.
    """
    if len(io) > 2:
        raise RuntimeError(f"Too many io specified for this channel. Given: {io}.")

    if len(io) == 2:
        assert (
            io[0] - io[1]
        ) ** 2 == 1, "Attempting to use two outputs that are not next to each other."
        if 1 in io:
            assert 2 not in io, (
                "Attempting to use output 1 and output 2 (2 and 3 on front panel) "
                "together, but they belong to different pairs."
            )
        return "complex"

    output = io[0]
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
    dict[str, dict]
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


def get_nco_phase_arguments(phase_deg: float) -> int:
    """
    Converts a phase in degrees to the int arguments the NCO phase instructions expect.
    We take `phase_deg` modulo 360 to account for negative phase and phase larger than
    360.

    Parameters
    ----------
    phase_deg
        The phase in degrees

    Returns
    -------
    :
        The int corresponding to the phase argument.
    """
    phase_deg %= 360
    return round(phase_deg * constants.NCO_PHASE_STEPS_PER_DEG)


def get_nco_set_frequency_arguments(frequency_hz: float) -> int:
    """
    Converts a frequency in Hz to the int argument the NCO set_freq instruction expects.

    Parameters
    ----------
    frequency_hz
        The frequency in Hz.

    Returns
    -------
    :
        The frequency expressed in steps for the NCO set_freq instruction.

    Raises
    ------
    ValueError
        If the frequency_hz is out of range.
    """
    frequency_steps = round(frequency_hz * constants.NCO_FREQ_STEPS_PER_HZ)

    if (
        frequency_steps < -constants.NCO_FREQ_LIMIT_STEPS
        or frequency_steps > constants.NCO_FREQ_LIMIT_STEPS
    ):
        min_max_frequency_in_hz = (
            constants.NCO_FREQ_LIMIT_STEPS / constants.NCO_FREQ_STEPS_PER_HZ
        )
        raise ValueError(
            f"Attempting to set NCO frequency. "
            f"The frequency must be between and including "
            f"-{min_max_frequency_in_hz:e} Hz and {min_max_frequency_in_hz:e} Hz. "
            f"Got {frequency_hz:e} Hz."
        )

    return frequency_steps


@dataclasses.dataclass
class Frequencies:
    clock: Optional[float] = None
    LO: Optional[float] = None
    IF: Optional[float] = None


def determine_clock_lo_interm_freqs(
    clock_freq: float,
    lo_freq: Union[float, None],
    interm_freq: Union[float, None],
    downconverter_freq: Optional[float] = None,
    mix_lo: bool = True,
) -> Frequencies:
    """
    .. warning::
        Using `downconverter_freq` requires custom Qblox hardware, do not use otherwise.

    From known frequency for the local oscillator or known intermodulation frequency,
    determine any missing frequency, after optionally applying `downconverter_freq` to
    the clock frequency.

    If `mix_lo` is ``True``, the following relation is obeyed:
    :math:`f_{RF} = f_{LO} + f_{IF}`.

    If `mix_lo` is ``False``, :math:`f_{RF} = f_{LO}` is upheld.

    Parameters
    ----------
    clock_freq : float
        Frequency of the clock.
    lo_freq : Union[float, None]
        Frequency of the local oscillator (LO).
    interm_freq : Union[float, None]
        Intermodulation frequency (IF), the frequency of the numerically controlled
        oscillator (NCO).
    downconverter_freq : Optional[float]
        Frequency for downconverting the clock frequency, using:
        :math:`f_\mathrm{out} = f_\mathrm{downconverter} - f_\mathrm{in}`.
    mix_lo : bool
        Flag indicating whether IQ mixing is enabled with the LO.

    Returns
    -------
    :
        :class:`.Frequencies` object containing the determined LO and IF frequencies and
        the optionally downconverted clock frequency.

    Warns
    -----
    ValueWarning
        In case `downconverter_freq` is set equal to 0, warns to unset via
        ``null``/``None`` instead.
    Raises
    ------
    ValueError
        In case `downconverter_freq` is less than 0.
    ValueError
        In case `downconverter_freq` is less than `clock_freq`.
    """

    def _downconvert_clock(downconverter_freq: float, clock_freq: float) -> float:
        if downconverter_freq == 0:
            warnings.warn(
                "Downconverter frequency 0 supplied. To unset 'downconverter_freq', "
                "set to `null` (json) / `None` instead in hardware configuration.",
                RuntimeWarning,
            )

        if downconverter_freq < 0:
            raise ValueError(
                f"Downconverter frequency must be positive ({downconverter_freq=:e})"
            )

        if downconverter_freq < clock_freq:
            raise ValueError(
                f"Downconverter frequency must be greater than clock frequency "
                f"({downconverter_freq=:e}, {clock_freq=:e})"
            )

        return downconverter_freq - clock_freq

    freqs = Frequencies(clock=clock_freq, LO=lo_freq, IF=interm_freq)

    if downconverter_freq is not None:
        freqs.clock = _downconvert_clock(
            downconverter_freq=downconverter_freq,
            clock_freq=clock_freq,
        )

    if not mix_lo:
        freqs.LO = freqs.clock
        freqs.IF = None
    else:
        if interm_freq is not None:
            freqs.LO = freqs.clock - interm_freq

        if lo_freq is not None:
            freqs.IF = freqs.clock - lo_freq

    return freqs


def generate_port_clock_to_device_map(
    hardware_cfg: Dict[str, Any]
) -> Dict[Tuple[str, str], str]:
    """
    Generates a mapping that specifies which port-clock combinations belong to which
    device.

    .. note::
        The same device may contain multiple port-clock combinations, but each
        port-clock combination may only occur once.

    Parameters
    ----------
    hardware_cfg:
        The hardware config dictionary.

    Returns
    -------
    :
        A dictionary with as key a tuple representing a port-clock combination, and
        as value the name of the device. Note that multiple port-clocks may point to
        the same device.
    """

    portclock_map = {}
    for device_name, device_info in hardware_cfg.items():
        if not isinstance(device_info, dict):
            continue

        portclocks = find_all_port_clock_combinations(device_info)

        for portclock in portclocks:
            portclock_map[portclock] = device_name

    return portclock_map


# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
def assign_pulse_and_acq_info_to_devices(
    schedule: Schedule,
    device_compilers: Dict[str, Any],
    hardware_cfg: Dict[str, Any],
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
    hardware_cfg
        The hardware config dictionary.

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

    portclock_mapping = generate_port_clock_to_device_map(hardware_cfg)

    for schedulable in schedule.schedulables.values():
        op_hash = schedulable["operation_repr"]
        op_data = schedule.operations[op_hash]

        if isinstance(op_data, WindowOperation):
            continue

        if not op_data.valid_pulse and not op_data.valid_acquisition:
            raise RuntimeError(
                f"Operation {op_hash} is not a valid pulse or acquisition. Please check"
                f" whether the device compilation been performed successfully. "
                f"Operation data: {repr(op_data)}"
            )

        operation_start_time = schedulable["abs_time"]
        for pulse_data in op_data.data["pulse_info"]:
            if "t0" in pulse_data:
                pulse_start_time = operation_start_time + pulse_data["t0"]
            else:
                pulse_start_time = operation_start_time

            port = pulse_data["port"]
            clock = pulse_data["clock"]

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=pulse_data,
                timing=pulse_start_time,
            )

            if port is None:
                # Distribute clock operations to all sequencers utilizing that clock
                for (map_port, map_clock), device_name in portclock_mapping.items():
                    if map_clock == clock:
                        device_compilers[device_name].add_pulse(
                            port=map_port, clock=clock, pulse_info=combined_data
                        )
            else:
                if (port, clock) not in portclock_mapping:
                    raise KeyError(
                        f"Could not assign pulse data to device. The combination "
                        f"of port {port} and clock {clock} could not be found "
                        f"in hardware configuration.\n\nAre both the port and clock "
                        f"specified in the hardware configuration?\n\n"
                        f"Relevant operation:\n{combined_data}."
                    )
                device_name = portclock_mapping[(port, clock)]
                device_compilers[device_name].add_pulse(
                    port=port, clock=clock, pulse_info=combined_data
                )

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
            hashed_dict["waveforms"] = []
            for acq in acq_data["waveforms"]:
                hashed_dict["waveforms"].append(without(acq, ["t0"]))

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=acq_data,
                timing=acq_start_time,
            )

            if (port, clock) not in portclock_mapping:
                raise KeyError(
                    f"Could not assign acquisition data to device. The combination "
                    f"of port {port} and clock {clock} could not be found "
                    f"in hardware configuration.\n\nAre both the port and clock "
                    f"specified in the hardware configuration?\n\n"
                    f"Relevant operation:\n{combined_data}."
                )
            device_name = portclock_mapping[(port, clock)]
            device_compilers[device_name].add_acquisition(
                port=port, clock=clock, acq_info=combined_data
            )


def convert_hw_config_to_portclock_configs_spec(
    hw_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converts possibly old hardware configs to the new format introduced by
    the new dynamic sequencer allocation feature.

    Manual assignment between sequencers and port-clock combinations under each output
    is removed, and instead only a list of port-clock configurations is specified,
    under the new ``"portclock_configs"`` key.

    Furthermore, we scan for ``"latency_correction"`` defined at sequencer or
    portclock_configs level and store under ``"port:clock"`` under toplevel
    ``"latency_corrections"`` key.

    Parameters
    ----------
    hw_config
        The hardware config to be upgraded to the new specification.

    Returns
    -------
    :
        A hardware config compatible with the specification required by the new
        dynamic sequencer allocation feature.

    """

    def _update_hw_config(nested_dict, max_depth=4):
        if max_depth == 0:
            return
        # List is needed because the dictionary keys are changed during recursion
        for key, value in list(nested_dict.items()):
            if isinstance(key, str) and re.match(r"^seq\d+$", key):
                nested_dict["portclock_configs"] = nested_dict.get(
                    "portclock_configs", []
                )
                # Move latency_corrections to parent level of hw_config
                if "latency_correction" in value.keys():
                    hw_config["latency_corrections"] = hw_config.get(
                        "latency_corrections", {}
                    )
                    latency_correction_key = f"{value['port']}-{value['clock']}"
                    hw_config["latency_corrections"][latency_correction_key] = value[
                        "latency_correction"
                    ]
                    del value["latency_correction"]

                nested_dict["portclock_configs"].append(value)
                del nested_dict[key]

            elif isinstance(value, dict):
                _update_hw_config(value, max_depth - 1)

    hw_config = deepcopy(hw_config)
    _update_hw_config(hw_config)

    return hw_config


def calc_from_units_volt(
    voltage_range, name: str, param_name: str, cfg: Dict[str, Any]
) -> Optional[float]:
    """
    Helper method to calculate the offset from mV or V.
    Then compares to given voltage range, and throws a ValueError if out of bounds.

    Parameters
    ----------
    voltage_range
        The range of the voltage levels of the device used.
    name
        The name of the device used.
    param_name
        The name of the current parameter the method is used for.
    cfg
        The hardware config of the device used.

    Returns
    -------
    :
        The normalized offsets.

    Raises
    ------
    RuntimeError
        When a unit range is given that is not supported, or a value is given that falls
        outside the allowed range.

    """
    offset_in_config = cfg.get(param_name, None)  # Always in volts
    if offset_in_config is None:
        return None

    conversion_factor = 1
    if voltage_range.units == "mV":
        conversion_factor = 1e3
    elif voltage_range.units != "V":
        raise RuntimeError(
            f"Parameter {param_name} of {name} specifies "
            f"the units {voltage_range.units}, but the Qblox "
            f"backend only supports mV and V."
        )

    calculated_offset = offset_in_config * conversion_factor
    if (
        calculated_offset < voltage_range.min_val
        or calculated_offset > voltage_range.max_val
    ):
        raise ValueError(
            f"Attempting to set {param_name} of {name} to "
            f"{offset_in_config} V. {param_name} has to be between "
            f"{voltage_range.min_val / conversion_factor} and "
            f"{voltage_range.max_val / conversion_factor} V!"
        )

    return calculated_offset


def extract_acquisition_metadata_from_acquisitions(
    acquisitions: List[OpInfo], repetitions: int
) -> AcquisitionMetadata:
    """
    Variant of
    :func:`~quantify_scheduler.helpers.schedule.extract_acquisition_metadata_from_acquisition_protocols`
    for use with the Qblox backend.
    """
    return extract_acquisition_metadata_from_acquisition_protocols(
        acquisition_protocols=[acq.data for acq in acquisitions],
        repetitions=repetitions,
    )


def single_scope_mode_acquisition_raise(sequencer_0, sequencer_1, module_name):
    """
    Raises an error stating that only one scope mode acquisition can be used per module.

    Parameters
    ----------
    sequencer_0
        First sequencer which attempts to use the scope mode acquisition.
    sequencer_1
        Second sequencer which attempts to use the scope mode acquisition.
    module_name
        Name of the module.

    Raises
    ------
    ValueError
        Always raises the error message.
    """
    raise ValueError(
        f"Both sequencer '{sequencer_0}' and '{sequencer_1}' "
        f"of '{module_name}' attempts to perform scope mode acquisitions. "
        f"Only one sequencer per device can "
        f"trigger raw trace capture.\n\nPlease ensure that "
        f"only one port-clock combination performs "
        f"raw trace acquisition per instrument."
    )
