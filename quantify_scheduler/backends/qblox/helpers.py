# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions for Qblox backend."""
import math
import dataclasses
import re
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from quantify_core.utilities import deprecated
from quantify_core.utilities.general import without

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.enums import IoMode
from quantify_scheduler.helpers.waveforms import exec_waveform_function
from quantify_scheduler.schedules.schedule import AcquisitionMetadata
from quantify_scheduler.helpers.collections import (
    find_all_port_clock_combinations,
    find_port_clock_path,
)
from quantify_scheduler.helpers.importers import export_python_object_to_path_string
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_acquisition_protocols,
)
from quantify_scheduler import Schedule

from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.operations.pulse_library import WindowOperation
from quantify_scheduler.backends.graph_compilation import CompilationConfig


def generate_waveform_data(
    data_dict: dict, sampling_rate: float, duration: Optional[float] = None
) -> np.ndarray:
    """Generates an array using the parameters specified in ``data_dict``.

    Parameters
    ----------
    data_dict : dict
        The dictionary that contains the values needed to parameterize the
        waveform. ``data_dict['wf_func']`` is then called to calculate the values.
    sampling_rate : float
        The sampling rate used to generate the time axis values.
    duration : float or None, optional
        The duration of the waveform in seconds. This parameter can be used if
        ``data_dict`` does not contain a ``'duration'`` key. By default None.

    Returns
    -------
    wf_data : np.ndarray
        The (possibly complex) values of the generated waveform. The number of values is
        determined by rounding to the nearest integer.

    Raises
    ------
    TypeError
        If ``data_dict`` does not contain a ``'duration'`` entry and ``duration is
        None``.
    """
    if duration is None:
        try:
            duration = data_dict["duration"]
        except KeyError as exc:
            raise TypeError(
                "Parameter 'duration' has value None. If 'data_dict' does not contain "
                "'duration', the function parameter can be used instead."
            ) from exc

    num_samples = round(duration * sampling_rate)
    t = np.arange(start=0, stop=num_samples, step=1) / sampling_rate

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

    def find_first_free_wf_index():
        index = 0
        reserved_indices = [wf_dict[uuid]["index"] for uuid in wf_dict]
        while index in reserved_indices:
            index += 1
        return index

    if not np.isrealobj(waveform):
        raise RuntimeError("This function only accepts real arrays.")

    uuid = generate_uuid_from_wf_data(waveform)
    if uuid in wf_dict:
        index: int = wf_dict[uuid]["index"]
    else:
        index: int = find_first_free_wf_index()
        wf_dict.update(generate_entry(uuid, waveform, index))
    return index


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
        "digital_output_0": (0,),
        "digital_output_1": (1,),
        "digital_output_2": (2,),
        "digital_output_3": (3,),
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


def get_io_info(
    io: str,
) -> Tuple[
    IoMode, Union[Tuple[int], Tuple[int, int]], Union[Tuple[int], Tuple[int, int]]
]:
    """
    Takes the specified io to use and extracts a "sequencer mode" from it.
    Also returns the connected inputs and outputs.

    Modes:

    - ``"real"``: only path0 is used
    - ``"imag"``: only path1 is used
    - ``"complex"``: both path0 and path1 paths are used.
    - ``"digital"``: the digital inputs and outputs are used.

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
    :
        The connected outputs
    :
        The connected inputs

    Raises
    ------
    RuntimeError
        The amount of ios is more than 2, which is impossible for one sequencer.
    """
    connected_outputs = output_name_to_outputs(io)
    connected_inputs = input_name_to_inputs(io)
    if "digital" in io:
        return IoMode.DIGITAL, connected_outputs, connected_inputs
    inputs_or_outputs = connected_outputs or connected_inputs

    if len(inputs_or_outputs) > 2:
        raise RuntimeError(
            f"Too many io specified for this channel. Given: {inputs_or_outputs}."
        )

    if len(inputs_or_outputs) == 2:
        assert (
            inputs_or_outputs[0] - inputs_or_outputs[1]
        ) ** 2 == 1, "Attempting to use two outputs that are not next to each other."
        if 1 in inputs_or_outputs:
            assert 2 not in inputs_or_outputs, (
                "Attempting to use output 1 and output 2 (2 and 3 on front panel) "
                "together, but they belong to different pairs."
            )
        return IoMode.COMPLEX, connected_outputs, connected_inputs

    output = inputs_or_outputs[0]
    mode = IoMode.REAL if output % 2 == 0 else IoMode.IMAG
    return mode, connected_outputs, connected_inputs


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
    clock: float
    LO: Optional[float] = None
    IF: Optional[float] = None

    def __post_init__(self):
        if self.LO is not None and math.isnan(self.LO):
            self.LO = None
        if self.IF is not None and math.isnan(self.IF):
            self.IF = None

    def validate(self):
        if self.clock is None or math.isnan(self.clock):
            raise ValueError(f"Clock frequency must be specified ({self.clock=}).")
        for freq in [self.LO, self.IF]:
            if freq is not None and math.isnan(freq):
                raise ValueError(
                    f"Frequencies must be a number or None, not NaN ({self.LO=}, {self.IF=})."
                )


def determine_clock_lo_interm_freqs(
    freqs: Frequencies,
    downconverter_freq: Optional[float] = None,
    mix_lo: bool = True,
) -> Frequencies:
    """
    From known frequency for the local oscillator or known intermodulation frequency,
    determine any missing frequency, after optionally applying `downconverter_freq` to
    the clock frequency.

    If `mix_lo` is ``True``, the following relation is obeyed:
    :math:`f_{RF} = f_{LO} + f_{IF}`.

    If `mix_lo` is ``False``, :math:`f_{RF} = f_{LO}` is upheld.

    .. warning::
        Using `downconverter_freq` requires custom Qblox hardware, do not use otherwise.

    Parameters
    ----------
    freqs : Frequencies
        Frequencies object containing clock, local oscillator (LO) and
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
    RuntimeWarning
        In case `downconverter_freq` is set equal to 0, warns to unset via
        ``null``/``None`` instead.
    RuntimeWarning
        In case LO is overridden to clock due to `mix_lo` being `False`
    Raises
    ------
    ValueError
        In case `downconverter_freq` is less than 0.
    ValueError
        In case `downconverter_freq` is less than `clock_freq`.
    ValueError
        In case `mix_lo` is `True` and neither LO frequency nor IF has been supplied.
    ValueError
        In case `mix_lo` is `True` and both LO frequency and IF have been supplied and do not adhere to
        :math:`f_{RF} = f_{LO} + f_{IF}`.

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

    freqs.validate()

    if downconverter_freq is not None:
        freqs.clock = _downconvert_clock(
            downconverter_freq=downconverter_freq,
            clock_freq=freqs.clock,
        )
    if not mix_lo:
        if freqs.LO is not None and not math.isclose(freqs.LO, freqs.clock):
            warnings.warn(
                f"Overriding {freqs.LO=} to {freqs.clock=} due to mix_lo=False."
            )
        freqs.LO = freqs.clock
    else:
        if freqs.LO is None and freqs.IF is None:
            raise ValueError(
                f"Frequency settings underconstrained for {freqs.clock=}."
                f" Neither LO nor IF supplied ({freqs.LO=}, {freqs.IF=})."
            )
        elif freqs.LO is not None and freqs.IF is not None:
            if not math.isclose(freqs.LO + freqs.IF, freqs.clock):
                raise ValueError(
                    f"Frequency settings overconstrained."
                    f" {freqs.clock=} must be equal to {freqs.LO=}+{freqs.IF=} if both are supplied."
                )
        elif freqs.LO is None and freqs.IF is not None:
            freqs.LO = freqs.clock - freqs.IF
        elif freqs.LO is not None and freqs.IF is None:
            freqs.IF = freqs.clock - freqs.LO

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
            # Check whether start time aligns with grid time
            try:
                _ = to_grid_time(pulse_start_time)
            except ValueError as exc:
                raise ValueError(
                    f"An operation start time of {pulse_start_time * 1e9} ns does not "
                    f"align with a grid time of {constants.GRID_TIME} ns. Please make "
                    f"sure the start time of all operations is a multiple of "
                    f"{constants.GRID_TIME} ns.\n\nOffending operation:"
                    f"\n{repr(op_data)}."
                ) from exc

            if pulse_data.get("reference_magnitude", None) is not None:
                warnings.warn(
                    "reference_magnitude parameter not implemented. This parameter will be ignored.",
                    RuntimeWarning,
                )

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
                if "t0" in acq:
                    # TODO 'without' will raise a KeyError if the key is not already
                    # present. Keep only the else-part and update the requirements when
                    # quantify-core!438 is in the latest release.
                    hashed_dict["waveforms"].append(without(acq, ["t0"]))
                else:
                    hashed_dict["waveforms"].append(acq)

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


@deprecated(
    "0.17.0",
    "`convert_hw_config_to_portclock_configs_spec` will be removed in a future "
    "version.",
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


def generate_hardware_config(compilation_config: CompilationConfig):
    """
    Extract the old-style Qblox hardware config from the CompilationConfig.

    Parameters
    ----------
    config: CompilationConfig
        CompilationConfig from which hardware config is extracted.

    Returns
    -------
    hardware_config : dict
        Qblox hardware configuration.

    Raises
    ------
    KeyError
        If the CompilationConfig.connectivity does not contain a hardware config.
    ValueError
        If a value is specified in both the hardware options and the hardware config.
    ValueError
        If setting a power_scaling option is not supported in the given configuration.
    RuntimeError
        If no external local oscillator is found in the generated Qblox hardware configuration.
    """

    def _recursive_digital_io_search(nested_dict, max_depth=3):
        if max_depth == 0:
            return
        for k in nested_dict:
            if k.startswith("digital"):
                nested_dict[k]["portclock_configs"][0]["clock"] = "digital"
            elif isinstance(nested_dict[k], Dict):
                _recursive_digital_io_search(nested_dict[k], max_depth - 1)

    if not isinstance(
        compilation_config.hardware_compilation_config.connectivity, Dict
    ):
        raise KeyError(
            f"CompilationConfig.connectivity does not contain a "
            f"hardware config dict:\n {compilation_config.hardware_compilation_config.connectivity=}"
        )

    hardware_config = deepcopy(
        compilation_config.hardware_compilation_config.connectivity
    )
    hardware_description = (
        compilation_config.hardware_compilation_config.hardware_description
    )
    hardware_options = compilation_config.hardware_compilation_config.hardware_options

    # Add digital clock to digital IO's, so that users don't have to specify it.
    _recursive_digital_io_search(hardware_config)

    port_clocks = find_all_port_clock_combinations(hardware_config)

    hardware_config["backend"] = export_python_object_to_path_string(
        compilation_config.hardware_compilation_config.backend
    )

    if hardware_description is not None:
        # Add info from hardware description to hardware config
        for instr_name, instr_description in hardware_description.items():
            if hardware_config.get(instr_name) is None:
                # Initialize instrument config dict
                hardware_config[instr_name] = {}
            instr_config = hardware_config[instr_name]
            if instr_description.hardware_type == "Qblox":
                # Set the instrument_type in the instrument config:
                legacy_instr_type = instr_config.get("instrument_type")
                if instr_description.instrument_type is None:
                    pass
                elif legacy_instr_type is None:
                    instr_config["instrument_type"] = instr_description.instrument_type
                elif legacy_instr_type != instr_description.instrument_type:
                    raise ValueError(
                        f"Trying to set instrument type for instrument: {instr_name} to"
                        f" {instr_description.instrument_type} from the hardware description while"
                        f" it has previously been set to {legacy_instr_type} in the hardware"
                        f" config. To avoid conflicting settings, please make sure this"
                        f" value is only set in one place."
                    )
                # Set the instrument reference source in the instrument config:
                legacy_ref = instr_config.get("ref")
                if instr_description.ref is None:
                    pass
                elif legacy_ref is None:
                    instr_config["ref"] = instr_description.ref
                elif legacy_ref != instr_description.ref:
                    raise ValueError(
                        f"Trying to set reference source for instrument: {instr_name} to"
                        f" {instr_description.ref} from the hardware description while"
                        f" it has previously been set to {legacy_ref} in the hardware"
                        f" config. To avoid conflicting settings, please make sure this"
                        f" value is only set in one place."
                    )
                # Set the sequence_to_file option in the instrument config:
                legacy_seq_to_file = instr_config.get("sequence_to_file")
                if instr_description.sequence_to_file is None:
                    pass
                elif legacy_seq_to_file is None:
                    instr_config[
                        "sequence_to_file"
                    ] = instr_description.sequence_to_file
                elif legacy_seq_to_file != instr_description.sequence_to_file:
                    raise ValueError(
                        f"Trying to set `sequence_to_file` for instrument: {instr_name} to"
                        f" {instr_description.sequence_to_file} from the hardware description while"
                        f" it has previously been set to {legacy_seq_to_file} in the hardware"
                        f" config. To avoid conflicting settings, please make sure this"
                        f" value is only set in one place."
                    )
                if instr_description.instrument_type == "Cluster":
                    for (
                        module_slot_idx,
                        module_description,
                    ) in instr_description.modules.items():
                        module_name = f"{instr_name}_module{module_slot_idx}"
                        if instr_config.get(module_name) is None:
                            # Initialize module config dict
                            instr_config[module_name] = {}
                        module_config = hardware_config[instr_name][module_name]
                        # Set the instrument_type in the module config:
                        legacy_mod_type = module_config.get("instrument_type")
                        if instr_description.instrument_type is None:
                            pass
                        elif legacy_mod_type is None:
                            module_config[
                                "instrument_type"
                            ] = module_description.module_type
                        elif legacy_mod_type != module_description.module_type:
                            raise ValueError(
                                f"Trying to set instrument type for {module_name} to"
                                f" {module_description.module_type} from the hardware description while "
                                f" it has previously been set to {legacy_mod_type} in the hardware"
                                f" config. To avoid conflicting settings, please make sure this"
                                f" value is only set in one place."
                            )
                        # Set the sequence_to_file option in the module config:
                        legacy_mod_seq_to_file = module_config.get("sequence_to_file")
                        if module_description.sequence_to_file is None:
                            pass
                        elif legacy_mod_seq_to_file is None:
                            module_config[
                                "sequence_to_file"
                            ] = module_description.sequence_to_file
                        elif (
                            legacy_mod_seq_to_file != instr_description.sequence_to_file
                        ):
                            raise ValueError(
                                f"Trying to set `sequence_to_file` for instrument: {module_name} to"
                                f" {module_description.sequence_to_file} from the hardware description while"
                                f" it has previously been set to {legacy_mod_seq_to_file} in the hardware"
                                f" config. To avoid conflicting settings, please make sure this"
                                f" value is only set in one place."
                            )
            elif instr_description.hardware_type == "LocalOscillator":
                instr_config["instrument_type"] = instr_description.hardware_type

                # Set the lo power in the lo config:
                legacy_lo_power = instr_config.get(instr_description.power_param)
                if instr_description.power_param is None:
                    pass
                elif legacy_lo_power is None:
                    instr_config[
                        instr_description.power_param
                    ] = instr_description.power
                elif legacy_lo_power != instr_description.power:
                    raise ValueError(
                        f"Trying to set the power for Local Oscillator: {instr_name} to"
                        f" {instr_description.power} from the hardware description while"
                        f" it has previously been set to {legacy_lo_power} in the hardware"
                        f" config. To avoid conflicting settings, please make sure this"
                        f" value is only set in one place."
                    )

    if hardware_options is not None:
        # Add latency corrections from hardware options to hardware config
        latency_corrections = hardware_options.dict()["latency_corrections"]
        legacy_latency_corrections = hardware_config.get("latency_corrections")

        if latency_corrections is None:
            pass
        elif legacy_latency_corrections is None:
            hardware_config["latency_corrections"] = latency_corrections
        elif legacy_latency_corrections != latency_corrections:
            raise ValueError(
                f"Trying to set latency corrections to {latency_corrections} from "
                f"the hardware options while it has previously been set to "
                f"{legacy_latency_corrections} in the hardware config. To avoid conflicting "
                f"settings, please make sure these corrections are only set in one place."
            )

        # Add distortion corrections from hardware options to hardware config
        distortion_corrections = hardware_options.dict()["distortion_corrections"]
        legacy_distortion_corrections = hardware_config.get("distortion_corrections")

        if distortion_corrections is None:
            pass
        elif legacy_distortion_corrections is None:
            hardware_config["distortion_corrections"] = distortion_corrections
        elif legacy_distortion_corrections != distortion_corrections:
            raise ValueError(
                f"Trying to set distortion corrections to {distortion_corrections} from "
                f"the hardware options while it has previously been set to "
                f"{legacy_distortion_corrections} in the hardware config. To avoid conflicting "
                f"settings, please make sure these corrections are only set in one place."
            )

        if hardware_options.modulation_frequencies is not None:
            for port, clock in port_clocks:
                if (
                    pc_mod_freqs := hardware_options.modulation_frequencies.get(
                        f"{port}-{clock}"
                    )
                ) is None:
                    # No modulation frequencies to set for this port-clock.
                    continue
                # Find path to port-clock combination in the hardware config, e.g.,
                # ["cluster0", "cluster0_module1", "complex_output_0", "portclock_configs", 1]
                pc_path = find_port_clock_path(
                    hardware_config=hardware_config, port=port, clock=clock
                )
                # Extract the port-clock config dict:
                pc_config = hardware_config
                for key in pc_path:
                    pc_config = pc_config[key]

                legacy_interm_freq = pc_config.get("interm_freq", "not_present")
                # Using default="not_present" because IF=None is also a valid setting
                if legacy_interm_freq == "not_present":
                    # Set the interm_freq in the port-clock config.
                    pc_config["interm_freq"] = pc_mod_freqs.interm_freq
                elif legacy_interm_freq != pc_mod_freqs.interm_freq:
                    raise ValueError(
                        f"Trying to set IF for {port=}, {clock=} to"
                        f" {pc_mod_freqs.interm_freq} from the hardware options while it"
                        f" has previously been set to {legacy_interm_freq} in the hardware"
                        f" config. To avoid conflicting settings, please make sure this"
                        f" value is only set in one place."
                    )

                # Extract instrument config and I/O channel config dicts:
                instr_config = hardware_config
                # Exclude ["complex_output/input_X", "portclock_configs", i]:
                for key in pc_path[:-3]:
                    instr_config = instr_config[key]
                io_config = instr_config[pc_path[-3]]

                # If RF module, set the lo frequency in the I/O config:
                if "RF" in instr_config["instrument_type"]:
                    legacy_lo_freq = io_config.get("lo_freq", "not_present")
                    # Using default="not_present" because lo_freq=None is also a valid setting
                    if legacy_lo_freq == "not_present":
                        io_config["lo_freq"] = pc_mod_freqs.lo_freq
                    elif legacy_lo_freq != pc_mod_freqs.lo_freq:
                        raise ValueError(
                            f"Trying to set frequency for {lo_name} to"
                            f" {pc_mod_freqs.lo_freq} from the hardware options while"
                            f" it has previously been set to {legacy_lo_freq} in"
                            f" the hardware config. To avoid conflicting settings,"
                            f" please make sure this value is only set in one place."
                        )
                # Else, set the lo frequency in the external lo config:
                else:
                    lo_name: str = io_config["lo_name"]
                    if (lo_config := hardware_config.get(lo_name)) is None:
                        raise RuntimeError(
                            f"External local oscillator '{lo_name}' set to "
                            f"be used for {port=} and {clock=} not found! Make "
                            f"sure it is present in the hardware configuration."
                        )
                    legacy_lo_freq = lo_config.get("frequency", "not_present")
                    # Using default="not_present" because lo_freq=None is also a valid setting
                    if legacy_lo_freq == "not_present":
                        lo_config["frequency"] = pc_mod_freqs.lo_freq
                    elif legacy_lo_freq != pc_mod_freqs.lo_freq:
                        raise ValueError(
                            f"Trying to set frequency for {lo_name} to"
                            f" {pc_mod_freqs.lo_freq} from the hardware options while"
                            f" it has previously been set to {legacy_lo_freq} in"
                            f" the hardware config. To avoid conflicting settings,"
                            f" please make sure this value is only set in one place."
                        )

        mixer_corrections = hardware_options.mixer_corrections
        if mixer_corrections is not None:
            for port, clock in port_clocks:
                if (pc_mix_corr := mixer_corrections.get(f"{port}-{clock}")) is None:
                    # No mixer corrections to set for this port-clock.
                    continue
                pc_mix_corr = {
                    "mixer_amp_ratio": pc_mix_corr.amp_ratio,
                    "mixer_phase_error_deg": pc_mix_corr.phase_error,
                    "dc_mixer_offset_I": pc_mix_corr.dc_offset_i,
                    "dc_mixer_offset_Q": pc_mix_corr.dc_offset_q,
                }
                # Find path to port-clock combination in the hardware config, e.g.,
                # ["cluster0", "cluster0_module1", "complex_output_0", "portclock_configs", 1]
                pc_path = find_port_clock_path(
                    hardware_config=hardware_config, port=port, clock=clock
                )
                # Extract the I/O channel config dict:
                io_config = hardware_config
                # Exclude ["portclock_configs", i]:
                for key in pc_path[:-2]:
                    io_config = io_config[key]
                pc_config = io_config["portclock_configs"][pc_path[-1]]

                # Add mixer corrections from hardware options to I/O channel config
                legacy_mix_corr = {
                    "mixer_amp_ratio": pc_config.get("mixer_amp_ratio"),
                    "mixer_phase_error_deg": pc_config.get("mixer_phase_error_deg"),
                    "dc_mixer_offset_I": io_config.get("dc_mixer_offset_I"),
                    "dc_mixer_offset_Q": io_config.get("dc_mixer_offset_Q"),
                }
                if all(v is None for v in legacy_mix_corr.values()):
                    pc_config["mixer_amp_ratio"] = pc_mix_corr["mixer_amp_ratio"]
                    pc_config["mixer_phase_error_deg"] = pc_mix_corr[
                        "mixer_phase_error_deg"
                    ]
                    io_config["dc_mixer_offset_I"] = pc_mix_corr["dc_mixer_offset_I"]
                    io_config["dc_mixer_offset_Q"] = pc_mix_corr["dc_mixer_offset_Q"]
                elif legacy_mix_corr != pc_mix_corr:
                    raise ValueError(
                        f"Trying to set mixer corrections for channel={pc_path[:-2]} to "
                        f"{pc_mix_corr} from the hardware options while it has previously "
                        f"been set to {legacy_mix_corr} in the hardware config. To avoid "
                        f"conflicting settings, please make sure these corrections are "
                        f"only set in one place."
                    )

        power_scaling = hardware_options.power_scaling
        if power_scaling is not None:
            for port, clock in port_clocks:
                if (pc_power_scaling := power_scaling.get(f"{port}-{clock}")) is None:
                    # No power scaling parameters to set for this port-clock.
                    continue
                # Find path to port-clock combination in the hardware config, e.g.,
                # ["cluster0", "cluster0_module1", "complex_output_0", "portclock_configs", 1]
                pc_path = find_port_clock_path(
                    hardware_config=hardware_config, port=port, clock=clock
                )
                # Extract instrument config and I/O channel config.
                instr_config = hardware_config
                # Exclude ["complex_output/input_X", "portclock_configs", i]:
                for key in pc_path[:-3]:
                    instr_config = instr_config[key]
                io_name = pc_path[-3]
                io_config = instr_config[io_name]
                instr_type = instr_config["instrument_type"]

                if not (io_name.startswith("complex") or io_name.startswith("real")):
                    raise KeyError(
                        f"The name of i/o channel {pc_path[:-2]} used for {port=} and {clock=} must start "
                        f"with either 'real' or 'complex'."
                    )
                # Different modules/channels support different power scaling settings
                supported_options = []
                if instr_type in ["QRM_RF", "QCM_RF"]:
                    if io_name.startswith("complex_output"):
                        supported_options.append("output_att")
                    if instr_type == "QRM_RF" and io_name.startswith("complex"):
                        # Note: input_att might also be set for complex_outputs in some cases
                        supported_options.append("input_att")
                elif instr_type == "QRM":
                    supported_options.append("input_gain")

                for option, value in pc_power_scaling.dict().items():
                    if (option not in supported_options) and (value is not None):
                        raise ValueError(
                            f"Setting the '{option}' for {io_name} of "
                            f"{instr_type=} is not supported in the Qblox backend."
                        )

                # Set the output_att in the I/O channel config:
                legacy_output_att = io_config.get("output_att")
                if pc_power_scaling.output_att is None:
                    pass
                elif legacy_output_att is None:
                    io_config["output_att"] = pc_power_scaling.output_att
                elif legacy_output_att != pc_power_scaling.output_att:
                    raise ValueError(
                        f"Trying to set output attenuation for channel={pc_path[:-2]} to"
                        f" {pc_power_scaling.output_att} from the hardware options while "
                        f" it has previously been set to {legacy_output_att} in the hardware"
                        f" config. To avoid conflicting settings, please make sure this"
                        f" value is only set in one place."
                    )
                # Set the input_att in the I/O channel config:
                legacy_input_att = io_config.get("input_att")
                if pc_power_scaling.input_att is None:
                    pass
                elif legacy_input_att is None:
                    io_config["input_att"] = pc_power_scaling.input_att
                elif legacy_input_att != pc_power_scaling.input_att:
                    raise ValueError(
                        f"Trying to set input attenuation for channel={pc_path[:-2]} to"
                        f" {pc_power_scaling.input_att} from the hardware options while "
                        f" it has previously been set to {legacy_input_att} in the hardware"
                        f" config. To avoid conflicting settings, please make sure this"
                        f" value is only set in one place."
                    )
                # Set the input_gain in the I/O channel config:
                if io_name.startswith("complex_output"):
                    legacy_input_gain = (
                        io_config.get("input_gain_I"),
                        io_config.get("input_gain_Q"),
                    )

                    if pc_power_scaling.input_gain is None:
                        pass
                    elif legacy_input_gain == (None, None):
                        io_config["input_gain_I"] = pc_power_scaling.input_gain[0]
                        io_config["input_gain_Q"] = pc_power_scaling.input_gain[1]
                    elif legacy_input_gain != pc_power_scaling.input_gain:
                        raise ValueError(
                            f"Trying to set input gain for channel={pc_path[:-2]} to"
                            f" {pc_power_scaling.input_gain} from the hardware options while "
                            f" it has previously been set to {legacy_input_gain} in the hardware"
                            f" config. To avoid conflicting settings, please make sure this"
                            f" value is only set in one place."
                        )
                elif io_name.startswith("real_output"):
                    if io_name == "real_output_0":
                        legacy_input_gain = io_config.get("input_gain_0")
                    elif io_name == "real_output_1":
                        legacy_input_gain = io_config.get("input_gain_1")

                    if pc_power_scaling.input_gain is None:
                        pass
                    elif legacy_input_gain is None:
                        if io_name == "real_output_0":
                            io_config["input_gain_0"] = pc_power_scaling.input_gain
                        elif io_name == "real_output_1":
                            io_config["input_gain_1"] = pc_power_scaling.input_gain
                    elif legacy_input_gain != pc_power_scaling.input_gain:
                        raise ValueError(
                            f"Trying to set input gain for channel={pc_path[:-2]} to"
                            f" {pc_power_scaling.input_gain} from the hardware options while "
                            f" it has previously been set to {legacy_input_gain} in the hardware"
                            f" config. To avoid conflicting settings, please make sure this"
                            f" value is only set in one place."
                        )

    return hardware_config


def is_within_grid_time(a, b):
    tolerance = 0.5e-9 * constants.GRID_TIME
    return math.isclose(a, b, abs_tol=tolerance, rel_tol=0)
