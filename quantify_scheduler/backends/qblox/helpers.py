# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions for Qblox backend."""
import dataclasses
import math
import re
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from quantify_core.utilities import deprecated
from quantify_core.utilities.general import without

from quantify_scheduler import Schedule
from quantify_scheduler.backends.graph_compilation import CompilationConfig
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.enums import IoMode
from quantify_scheduler.backends.types.common import Connectivity
from quantify_scheduler.backends.types.qblox import (
    ComplexChannelDescription,
    ComplexInputGain,
    OpInfo,
    RealInputGain,
)
from quantify_scheduler.helpers.collections import (
    find_all_port_clock_combinations,
    find_port_clock_path,
)
from quantify_scheduler.helpers.importers import export_python_object_to_path_string
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_acquisition_protocols,
    _extract_port_clocks_used,
)
from quantify_scheduler.helpers.waveforms import exec_waveform_function
from quantify_scheduler.operations.pulse_library import WindowOperation
from quantify_scheduler.schedules.schedule import AcquisitionMetadata


def generate_waveform_data(
    data_dict: dict, sampling_rate: float, duration: Optional[float] = None
) -> np.ndarray:
    """
    Generates an array using the parameters specified in ``data_dict``.

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


def output_name_to_output_indices(
    output_name: str,
) -> Optional[Union[Tuple[int], Tuple[int, int]]]:
    """
    Return the output indices associated with the output name specified in the hardware config.

    For the baseband modules, output index 'n' corresponds to physical module output 'n+1'.

    For RF modules, output indices '0' and '1' (or: '2' and '3') correspond to 'path0' and 'path1' of some sequencer, and both these paths are routed to the **same** physical module output '1' (or: '2').

    Parameters
    ----------
    output_name
        Hardware config output name (e.g. 'complex_output_0').

    Returns
    -------
    :
        A tuple containing output indices corresponding to certain physical module outputs.
    """
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
    }[output_name]


def input_name_to_input_indices(input_name: str) -> Union[Tuple[int], Tuple[int, int]]:
    """
    Return the input indices associated with the input name specified in the
    hardware config.

    For the baseband modules, input index 'n' corresponds to physical module input 'n+1'.

    For RF modules, input indices '0' and '1' correspond to 'path0' and 'path1' of some sequencer, and both paths are connected to physical module input '1'.

    Parameters
    ----------
    input_name
        Hardware config input name (e.g. 'complex_input_0').

    Returns
    -------
    :
        A tuple containing input indices corresponding to certain physical module inputs.
    """
    return {
        "complex_input_0": (0, 1),
        "real_input_0": (0,),
        "real_input_1": (1,),
    }[input_name]


def get_io_info(
    io_name: str,
) -> Tuple[
    IoMode, Union[Tuple[int], Tuple[int, int]], Union[Tuple[int], Tuple[int, int]]
]:
    """
    Return a "sequencer mode" based on the paths used by the sequencer, as well as the
    input or output indices associated to the io name the sequencer is supposed to use.

    Sequencer modes:

    - :attr:`.IoMode.REAL`: only path0 is used.
    - :attr:`.IoMode.IMAG`: only path1 is used.
    - :attr:`.IoMode.COMPLEX`: both path0 and path1 paths are used.
    - :attr:`.IoMode.DIGITAL`: the digital outputs are used.

    Parameters
    ----------
    io_name
        The io name from the hardware config that the sequencer is supposed to use.

    Returns
    -------
    :
        The sequencer mode
    :
        The output indices
    :
        The input indices

    """
    connected_outputs, connected_inputs = None, None

    if "output" in io_name:
        connected_outputs = output_name_to_output_indices(io_name)
    elif "input" in io_name:
        connected_inputs = input_name_to_input_indices(io_name)
    else:
        raise ValueError(f"Input/output name '{io_name}' is not valid")

    if "digital" in io_name:
        sequencer_mode = IoMode.DIGITAL
    elif "complex" in io_name:
        sequencer_mode = IoMode.COMPLEX
    elif "real" in io_name:
        io_idx = (
            connected_outputs[0]
            if connected_outputs is not None
            else connected_inputs[0]
        )
        sequencer_mode = IoMode.REAL if io_idx in (0, 2) else IoMode.IMAG

    return sequencer_mode, connected_outputs, connected_inputs


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
    Convert time value in s to time in ns, and verify that it is aligned with grid time.

    Takes a float value representing a time in seconds as used by the schedule, and
    returns the integer valued time in nanoseconds that the sequencer uses.

    The time value needs to be aligned with grid time, i.e., needs to be a multiple
    of :data:`~.constants.GRID_TIME`, within a tolerance of 1 picosecond.

    Parameters
    ----------
    time
        A time value in seconds.
    grid_time_ns
        The grid time to use in nanoseconds.

    Returns
    -------
    :
        The integer valued nanosecond time.

    Raises
    ------
    ValueError
        If `time` is not a multiple of :data:`~.constants.GRID_TIME` within the tolerance.
    """
    time_ns_float = time * 1e9
    time_ns = int(round(time_ns_float))

    tolerance = 1.1e-3  # Slightly more to compensate for float repr, allowing for 1 ps
    if (
        not math.isclose(
            time_ns_float, time_ns, abs_tol=tolerance, rel_tol=0
        )  # rel_tol=0 results in: abs(a-b) <= max(0, abs_tol)
        or time_ns % grid_time_ns != 0
    ):
        raise ValueError(
            f"Attempting to use a time value of {time_ns_float} ns."
            f" Please ensure that the durations of operations and wait times between"
            f" operations are multiples of {grid_time_ns} ns"
            f" (tolerance: {tolerance:.0e} ns)."  # Intentionally not showing digits
        )

    return time_ns


def is_multiple_of_grid_time(
    time: float, grid_time_ns: int = constants.GRID_TIME
) -> bool:
    """
    Determine whether a time value in seconds is a multiple of the grid time.

    Within a tolerance as defined by
    :meth:`~quantify_scheduler.backends.qblox.helpers.to_grid_time`.

    Parameters
    ----------
    time
        A time value in seconds.
    grid_time_ns
        The grid time to use in nanoseconds.

    Returns
    -------
    :
        `True` if `time` is a multiple of the grid time, `False` otherwise.
    """
    try:
        _ = to_grid_time(time=time, grid_time_ns=grid_time_ns)
    except ValueError:
        return False

    return True


def is_within_half_grid_time(a, b, grid_time_ns: int = constants.GRID_TIME):
    """
    Determine whether two time values in seconds are within half grid time of each other.

    Parameters
    ----------
    a
        A time value in seconds.
    b
        A time value in seconds.
    grid_time_ns
        The grid time to use in nanoseconds.

    Returns
    -------
    :
        `True` if `a` and `b`  are less than half grid time apart, `False` otherwise.
    """
    tolerance = 0.5e-9 * grid_time_ns
    within_half_grid_time = math.isclose(
        a, b, abs_tol=tolerance, rel_tol=0
    )  # rel_tol=0 results in: abs(a-b) <= max(0, abs_tol)

    return within_half_grid_time


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
    """Holds and validates frequencies."""

    clock: float
    LO: Optional[float] = None
    IF: Optional[float] = None

    def __post_init__(self):
        if self.LO is not None and math.isnan(self.LO):
            self.LO = None
        if self.IF is not None and math.isnan(self.IF):
            self.IF = None

    def validate(self):
        """Validates frequencies."""
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
    r"""
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

    Here, device means a top-level entry in the hardware config, e.g. a Cluster,
    not which module within the Cluster.

    Each port-clock combination may only occur once.

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

    Raises
    ------
    ValueError
        If a port-clock combination occurs multiple times in the hardware configuration.
    """
    portclock_map = {}
    for device_name, device_info in hardware_cfg.items():
        if not isinstance(device_info, dict):
            continue

        for portclock in find_all_port_clock_combinations(device_info):
            if portclock in portclock_map:
                raise ValueError(
                    f"Port-clock combination '{portclock[0]}-{portclock[1]}'"
                    f" occurs multiple times in the hardware configuration;"
                    f" each port-clock combination may only occur once. When using"
                    f" the same port-clock combination for output and input, assigning"
                    f" only the output suffices."
                )
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


def generate_hardware_config(schedule: Schedule, compilation_config: CompilationConfig):
    """
    Extract the old-style Qblox hardware config from the CompilationConfig.

    Only the port-clock combinations that are used in the schedule are included in the
    old-style hardware config.

    Parameters
    ----------
    schedule: Schedule
        Schedule from which the port-clock combinations are extracted.
    compilation_config : CompilationConfig
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

    def _propagate_channel_description_settings(
        config: dict,
        description: Any,
    ):
        for key in [
            "complex_output_0",
            "complex_output_1",
            "complex_input_0",
            "real_output_0",
            "real_output_1",
            "real_output_2",
            "real_output_3",
            "real_input_0",
            "real_input_1",
        ]:
            if (io_description := getattr(description, key, None)) is None:
                # No channel description to set
                continue
            if key not in config:
                # The channel is not present in the existing instrument config,
                # most likely because it is not specified in the connectivity
                continue

            config[key][
                "marker_debug_mode_enable"
            ] = io_description.marker_debug_mode_enable
            if "complex" in key:
                config[key]["mix_lo"] = io_description.mix_lo
                config[key]["downconverter_freq"] = io_description.downconverter_freq

    hardware_description = (
        compilation_config.hardware_compilation_config.hardware_description
    )
    hardware_options = compilation_config.hardware_compilation_config.hardware_options
    connectivity = compilation_config.hardware_compilation_config.connectivity

    hardware_config = {}
    if isinstance(connectivity, dict):
        hardware_config = connectivity

    hardware_config["backend"] = export_python_object_to_path_string(
        compilation_config.hardware_compilation_config.backend
    )

    port_clocks = _extract_port_clocks_used(schedule=schedule)

    # Add connectivity information to the hardware config:
    if isinstance(connectivity, Connectivity):
        connectivity_graph = (
            compilation_config.hardware_compilation_config.connectivity.graph
        )
        for port, clock in port_clocks:
            # Find all nodes connected to quantum device port
            connected_nodes = {}
            for node in connectivity_graph:
                if port in node:
                    connected_nodes = connectivity_graph[node]
                    break

            for connected_node in connected_nodes:
                io_path = connected_node.split(sep=".")
                instrument = io_path[0]

                lo_name = None
                if hardware_description[instrument].instrument_type == "IQMixer":
                    # Find which lo is used for this IQ mixer
                    lo_name = list(connectivity_graph[instrument + ".lo"])[0].split(
                        sep="."
                    )[0]
                    # Find which instrument is connected to if port
                    io_path = list(connectivity_graph[instrument + ".if"])[0].split(
                        sep="."
                    )

                if hardware_description[io_path[0]].instrument_type == "Cluster":
                    # Format the io_path to match the hardware config:
                    # e.g., ["cluster0", "cluster0_module1", "complex_output_0"]
                    io_path[1] = f"{io_path[0]}_{io_path[1]}"

                # Set port-clock combination in io config:
                instr_config: dict = hardware_config
                for key in io_path[:-1]:
                    if key not in instr_config:
                        instr_config[key] = {}
                    instr_config = instr_config[key]

                instrument_io = io_path[-1]
                if instrument_io not in instr_config:
                    instr_config[instrument_io] = {"portclock_configs": []}
                instr_config[instrument_io]["portclock_configs"].append(
                    {"port": port, "clock": clock}
                )
                if lo_name is not None:
                    instr_config[instrument_io]["lo_name"] = lo_name

    # Add info from hardware description to hardware config:
    if hardware_description is not None:
        for instr_name, instr_description in hardware_description.items():
            if instr_description.instrument_type == "IQMixer":
                # Skip IQMixer instruments, as they are not present in the old-style
                # hardware config
                continue

            if hardware_config.get(instr_name) is None:
                # Initialize instrument config dict
                hardware_config[instr_name] = {}
            instr_config = hardware_config[instr_name]

            for key in [
                "instrument_type",
                "ref",
                "sequence_to_file",
            ]:
                try:
                    instr_config[key] = getattr(instr_description, key)
                except AttributeError:
                    pass

            # Propagate I/O channel description settings for Pulsars
            _propagate_channel_description_settings(
                config=instr_config, description=instr_description
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

                    for key in ["instrument_type", "sequence_to_file"]:
                        try:
                            module_config[key] = getattr(module_description, key)
                        except AttributeError:
                            pass

                    # Propagate I/O channel description settings for Cluster modules
                    _propagate_channel_description_settings(
                        config=module_config, description=module_description
                    )

            if instr_description.instrument_type == "LocalOscillator":
                instr_config[instr_description.power_param] = instr_description.power

    if hardware_options is not None:
        if hardware_options.latency_corrections is not None:
            hardware_config["latency_corrections"] = hardware_options.model_dump()[
                "latency_corrections"
            ]
        if hardware_options.distortion_corrections is not None:
            hardware_config["distortion_corrections"] = hardware_options.model_dump()[
                "distortion_corrections"
            ]

        # Set Hardware Options for all port-clock combinations in the Schedule:
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

                # Set the interm_freq in the portclock config:
                pc_config["interm_freq"] = pc_mod_freqs.interm_freq

                # Extract instrument config and I/O channel config dicts:
                instr_config = hardware_config
                # Exclude ["complex_output/input_X", "portclock_configs", i]:
                for key in pc_path[:-3]:
                    instr_config = instr_config[key]
                io_config = instr_config[pc_path[-3]]

                # If RF module, set the lo frequency in the I/O config:
                if "RF" in instr_config["instrument_type"]:
                    io_config["lo_freq"] = pc_mod_freqs.lo_freq
                # Else, set the lo frequency in the external lo config:
                else:
                    lo_name: str = io_config["lo_name"]
                    if (lo_config := hardware_config.get(lo_name)) is None:
                        raise RuntimeError(
                            f"External local oscillator '{lo_name}' set to "
                            f"be used for {port=} and {clock=} not found! Make "
                            f"sure it is present in the hardware configuration."
                        )
                    lo_config["frequency"] = pc_mod_freqs.lo_freq

        mixer_corrections = hardware_options.mixer_corrections
        if mixer_corrections is not None:
            for port, clock in port_clocks:
                if (pc_mix_corr := mixer_corrections.get(f"{port}-{clock}")) is None:
                    # No mixer corrections to set for this port-clock.
                    continue

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

                for config, hw_config_key, hw_compilation_config_value in [
                    (pc_config, "mixer_amp_ratio", pc_mix_corr.amp_ratio),
                    (pc_config, "mixer_phase_error_deg", pc_mix_corr.phase_error),
                    (io_config, "dc_mixer_offset_I", pc_mix_corr.dc_offset_i),
                    (io_config, "dc_mixer_offset_Q", pc_mix_corr.dc_offset_q),
                ]:
                    config[hw_config_key] = hw_compilation_config_value

        input_gain = hardware_options.input_gain
        if input_gain is not None:
            for port, clock in port_clocks:
                if (pc_input_gain := input_gain.get(f"{port}-{clock}")) is None:
                    # No input gain parameters to set for this port-clock.
                    continue

                # Find path to port-clock combination in the hardware config, e.g.,
                # ["cluster0", "cluster0_module1", "complex_output_0", "portclock_configs", 1]
                pc_path = find_port_clock_path(
                    hardware_config=hardware_config, port=port, clock=clock
                )
                # Extract I/O channel config:
                io_config = hardware_config
                # Exclude ["portclock_configs", i]:
                for key in pc_path[:-2]:
                    io_config = io_config[key]
                io_name = pc_path[-3]

                if not (io_name.startswith("complex") or io_name.startswith("real")):
                    raise KeyError(
                        f"The name of i/o channel {pc_path[:-2]} used for {port=} and {clock=} must start "
                        f"with either 'real' or 'complex'."
                    )

                # Set the input_gain in the I/O channel config:
                if isinstance(pc_input_gain, ComplexInputGain):
                    io_config["input_gain_I"] = pc_input_gain.gain_I
                    io_config["input_gain_Q"] = pc_input_gain.gain_Q
                elif isinstance(pc_input_gain, RealInputGain):
                    if io_name == "real_output_0":
                        io_config["input_gain_0"] = pc_input_gain.gain
                    elif io_name == "real_output_1":
                        io_config["input_gain_1"] = pc_input_gain.gain

        output_att = hardware_options.output_att
        if output_att is not None:
            for port, clock in port_clocks:
                if (pc_output_att := output_att.get(f"{port}-{clock}")) is None:
                    # No output attenuation parameters to set for this port-clock.
                    continue

                # Find path to port-clock combination in the hardware config, e.g.,
                # ["cluster0", "cluster0_module1", "complex_output_0", "portclock_configs", 1]
                pc_path = find_port_clock_path(
                    hardware_config=hardware_config, port=port, clock=clock
                )
                # Extract I/O channel config:
                io_config = hardware_config
                # Exclude ["portclock_configs", i]:
                for key in pc_path[:-2]:
                    io_config = io_config[key]

                # Set the output_att in the I/O channel config:
                io_config["output_att"] = pc_output_att

        input_att = hardware_options.input_att
        if input_att is not None:
            for port, clock in port_clocks:
                if (pc_input_att := input_att.get(f"{port}-{clock}")) is None:
                    # No input attenuation parameters to set for this port-clock.
                    continue

                # Find path to port-clock combination in the hardware config, e.g.,
                # ["cluster0", "cluster0_module1", "complex_output_0", "portclock_configs", 1]
                pc_path = find_port_clock_path(
                    hardware_config=hardware_config, port=port, clock=clock
                )
                # Extract I/O channel config:
                io_config = hardware_config
                # Exclude ["portclock_configs", i]:
                for key in pc_path[:-2]:
                    io_config = io_config[key]

                # Set the input_att in the I/O channel config:
                io_config["input_att"] = pc_input_att

        sequencer_options = hardware_options.sequencer_options
        if sequencer_options is not None:
            for port, clock in port_clocks:
                if (
                    pc_sequencer_options := sequencer_options.get(f"{port}-{clock}")
                ) is None:
                    # No sequencer_options to set for this port-clock.
                    continue

                # Find path to port-clock combination in the hardware config, e.g.,
                # ["cluster0", "cluster0_module1", "complex_output_0", "portclock_configs", 1]
                pc_path = find_port_clock_path(
                    hardware_config=hardware_config, port=port, clock=clock
                )
                # Extract the port-clock config:
                pc_config = hardware_config
                for key in pc_path:
                    pc_config = pc_config[key]

                pc_config["ttl_acq_threshold"] = pc_sequencer_options.ttl_acq_threshold
                pc_config[
                    "init_offset_awg_path_0"
                ] = pc_sequencer_options.init_offset_awg_path_0
                pc_config[
                    "init_offset_awg_path_1"
                ] = pc_sequencer_options.init_offset_awg_path_1
                pc_config[
                    "init_gain_awg_path_0"
                ] = pc_sequencer_options.init_gain_awg_path_0
                pc_config[
                    "init_gain_awg_path_1"
                ] = pc_sequencer_options.init_gain_awg_path_1
                pc_config["qasm_hook_func"] = pc_sequencer_options.qasm_hook_func

    # Add digital clock to digital IO's, so that users don't have to specify it.
    _recursive_digital_io_search(hardware_config)

    return hardware_config
