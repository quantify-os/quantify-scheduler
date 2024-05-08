# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions for Qblox backend."""
from __future__ import annotations

import dataclasses
import math
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.enums import ChannelMode
from quantify_scheduler.backends.types.qblox import (
    ComplexChannelDescription,
    ComplexInputGain,
    OpInfo,
    QbloxHardwareDistortionCorrection,
    RealChannelDescription,
    RealInputGain,
)
from quantify_scheduler.helpers.collections import (
    find_all_port_clock_combinations,
    find_port_clock_path,
)
from quantify_scheduler.helpers.schedule import _extract_port_clocks_used
from quantify_scheduler.helpers.waveforms import exec_waveform_function
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import WindowOperation
from quantify_scheduler.resources import DigitalClockResource
from quantify_scheduler.schedules.schedule import Schedule, ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import CompilationConfig
    from quantify_scheduler.backends.qblox.instrument_compilers import ClusterCompiler


def generate_waveform_data(
    data_dict: dict, sampling_rate: float, duration: float | None = None
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


def generate_waveform_names_from_uuid(uuid: Any) -> tuple[str, str]:
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
    wf_dict: dict[str, Any], waveform: np.ndarray
) -> tuple[dict[str, Any], str, int]:
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

    def generate_entry(name: str, data: np.ndarray, idx: int) -> dict[str, Any]:
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


def generate_waveform_dict(waveforms_complex: dict[str, np.ndarray]) -> dict[str, dict]:
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
        If ``time`` is not a multiple of :data:`~.constants.GRID_TIME` within the tolerance.
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
        ``True`` if ``time`` is a multiple of the grid time, ``False`` otherwise.
    """
    try:
        _ = to_grid_time(time=time, grid_time_ns=grid_time_ns)
    except ValueError:
        return False

    return True


def get_nco_phase_arguments(phase_deg: float) -> int:
    """
    Converts a phase in degrees to the int arguments the NCO phase instructions expect.
    We take ``phase_deg`` modulo 360 to account for negative phase and phase larger than
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
    LO: float | None = None
    IF: float | None = None

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
    downconverter_freq: float | None = None,
    mix_lo: bool = True,
) -> Frequencies:
    r"""
    From known frequency for the local oscillator or known intermodulation frequency,
    determine any missing frequency, after optionally applying ``downconverter_freq`` to
    the clock frequency.

    If ``mix_lo`` is ``True``, the following relation is obeyed:
    :math:`f_{RF} = f_{LO} + f_{IF}`.

    If ``mix_lo`` is ``False``, :math:`f_{RF} = f_{LO}` is upheld.

    .. warning::
        Using ``downconverter_freq`` requires custom Qblox hardware, do not use otherwise.

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
        In case ``downconverter_freq`` is set equal to 0, warns to unset via
        ``null``/``None`` instead.
    RuntimeWarning
        In case LO is overridden to clock due to ``mix_lo`` being `False`

    Raises
    ------
    ValueError
        In case ``downconverter_freq`` is less than 0.
    ValueError
        In case ``downconverter_freq`` is less than ``clock_freq``.
    ValueError
        In case ``mix_lo`` is ``True`` and neither LO frequency nor IF has been supplied.
    ValueError
        In case ``mix_lo`` is ``True`` and both LO frequency and IF have been supplied and do not adhere to
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
    hardware_cfg: dict[str, Any]
) -> dict[tuple[str, str], str]:
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


def _get_list_of_operations_for_op_info_creation(
    operation: Operation | Schedule,
    time_offset: float,
    accumulator: list[tuple[float, Operation]],
) -> None:
    if isinstance(operation, ScheduleBase):
        for schedulable in operation.schedulables.values():
            abs_time = schedulable["abs_time"]
            inner_operation = operation.operations[schedulable["operation_id"]]
            _get_list_of_operations_for_op_info_creation(
                inner_operation, time_offset + abs_time, accumulator
            )
    else:
        accumulator.append((time_offset, operation))


def assign_pulse_and_acq_info_to_devices(
    schedule: Schedule,
    device_compilers: dict[str, ClusterCompiler],
    hardware_cfg: dict[str, Any],
):
    """
    Traverses the schedule and generates `OpInfo` objects for every pulse and
    acquisition, and assigns it to the correct `ClusterCompiler`.

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

    list_of_operations: list[tuple[float, Operation]] = list()
    _get_list_of_operations_for_op_info_creation(schedule, 0, list_of_operations)

    for operation_start_time, op_data in list_of_operations:
        # FIXME #461 Help the type checker. Schedule should have been flattened at this
        # point.
        assert isinstance(op_data, Operation)

        if isinstance(op_data, WindowOperation):
            continue

        if not op_data.valid_pulse and not op_data.valid_acquisition:
            raise RuntimeError(
                f"Operation is not a valid pulse or acquisition. Please check"
                f" whether the device compilation been performed successfully. "
                f"Operation data: {repr(op_data)}"
            )

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
                    if (combined_data.name == "LatchReset") or map_clock == clock:
                        device_compilers[device_name].add_op_info(
                            port=map_port, clock=map_clock, op_info=combined_data
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
                device_compilers[device_name].add_op_info(
                    port=port, clock=clock, op_info=combined_data
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
            device_compilers[device_name].add_op_info(
                port=port, clock=clock, op_info=combined_data
            )


def calc_from_units_volt(
    voltage_range, name: str, param_name: str, cfg: dict[str, Any]
) -> float | None:
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
    offset_in_config = cfg.get(param_name)  # Always in volts
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


def _generate_legacy_hardware_config(
    schedule: Schedule, compilation_config: CompilationConfig
) -> dict[str, Any]:
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

    def _recursive_digital_channel_search(nested_dict, max_depth=3):
        if max_depth == 0:
            return
        for k in nested_dict:
            if k.startswith(ChannelMode.DIGITAL):
                if "clock" not in nested_dict[k]["portclock_configs"][0]:
                    nested_dict[k]["portclock_configs"][0][
                        "clock"
                    ] = DigitalClockResource.IDENTITY
            elif isinstance(nested_dict[k], dict):
                _recursive_digital_channel_search(nested_dict[k], max_depth - 1)

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
            "digital_output_0",
            "digital_output_1",
            "digital_output_2",
            "digital_output_3",
        ]:
            if (channel_description := getattr(description, key, None)) is None:
                # No channel description to set
                continue
            if key not in config:
                # The channel is not present in the existing instrument config,
                # most likely because it is not specified in the connectivity
                continue

            config[key][
                "distortion_correction_latency_compensation"
            ] = channel_description.distortion_correction_latency_compensation
            if ChannelMode.DIGITAL not in key:
                config[key][
                    "marker_debug_mode_enable"
                ] = channel_description.marker_debug_mode_enable
            if ChannelMode.COMPLEX in key:
                config[key]["mix_lo"] = channel_description.mix_lo
                config[key][
                    "downconverter_freq"
                ] = channel_description.downconverter_freq

    hardware_description = (
        compilation_config.hardware_compilation_config.hardware_description
    )
    hardware_options = compilation_config.hardware_compilation_config.hardware_options
    connectivity = compilation_config.hardware_compilation_config.connectivity

    if isinstance(connectivity, dict):
        if "graph" in connectivity:
            raise KeyError(
                "Connectivity contains a dictionary including a 'graph' key, most likely"
                " because the networkx Graph object could not be parsed correctly."
            )
        # The connectivity contains the full old-style hardware config
        _recursive_digital_channel_search(connectivity)
        return connectivity

    hardware_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile"
    }

    port_clocks = _extract_port_clocks_used(operation=schedule)

    # Add connectivity information to the hardware config:
    connectivity_graph = (
        compilation_config.hardware_compilation_config.connectivity.graph
    )
    for port, clock in sorted(port_clocks):
        # Find all nodes connected to quantum device port
        connected_nodes = {}
        for node in connectivity_graph:
            if port in node:
                connected_nodes = connectivity_graph[node]
                break

        for connected_node in connected_nodes:
            channel_path = connected_node.split(sep=".")
            instrument = channel_path[0]

            lo_name = None
            if hardware_description[instrument].instrument_type == "IQMixer":
                # Find which lo is used for this IQ mixer
                lo_name = list(connectivity_graph[instrument + ".lo"])[0].split(
                    sep="."
                )[0]
                # Find which instrument is connected to if port
                channel_path = list(connectivity_graph[instrument + ".if"])[0].split(
                    sep="."
                )

            if hardware_description[channel_path[0]].instrument_type == "Cluster":
                # Format the channel_path to match the hardware config:
                # e.g., ["cluster0", "cluster0_module1", "complex_output_0"]
                channel_path[1] = f"{channel_path[0]}_{channel_path[1]}"

            # Set port-clock combination in channel config:
            instr_config: dict = hardware_config
            for key in channel_path[:-1]:
                if key not in instr_config:
                    instr_config[key] = {}
                instr_config = instr_config[key]

            instrument_channel = channel_path[-1]
            if instrument_channel not in instr_config:
                instr_config[instrument_channel] = {"portclock_configs": []}
            instr_config[instrument_channel]["portclock_configs"].append(
                {"port": port, "clock": clock}
            )
            if lo_name is not None:
                instr_config[instrument_channel]["lo_name"] = lo_name

    # Add info from hardware description to hardware config:
    for instr_name, instr_description in hardware_description.items():
        if instr_description.instrument_type not in [
            "Cluster",
            "LocalOscillator",
        ]:
            # Only generate hardware config entries for supported instruments,
            # while allowing them in the HardwareCompilationConfig to be used by
            # different compilation nodes.
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

                # Propagate channel description settings for Cluster modules
                _propagate_channel_description_settings(
                    config=module_config, description=module_description
                )

        if instr_description.instrument_type == "LocalOscillator":
            instr_config[instr_description.power_param] = instr_description.power

    # Add hardware options to hardware config
    if hardware_options.latency_corrections is not None:
        hardware_config["latency_corrections"] = hardware_options.model_dump()[
            "latency_corrections"
        ]
    if hardware_options.distortion_corrections is not None:
        hardware_config["distortion_corrections"] = {}
        used_keys = [f"{port}-{clock}" for port, clock in port_clocks]
        for key in hardware_options.distortion_corrections:
            if key not in used_keys:
                warnings.warn(
                    f"Distortion correction portclock {key} is not used in the schedule."
                )
            distortion_correction = hardware_options.distortion_corrections[key]
            if isinstance(distortion_correction, list):
                distortion_correction_list = []
                for dc in distortion_correction:
                    dc_dict = dc.model_dump()
                    dc_dict["correction_type"] = "qblox"
                    distortion_correction_list.append(dc_dict)
                # Set the distortion correction in the hardware config:
                hardware_config["distortion_corrections"][
                    key
                ] = distortion_correction_list
            else:
                distortion_correction_dict = distortion_correction.model_dump()
                if isinstance(distortion_correction, QbloxHardwareDistortionCorrection):
                    distortion_correction_dict["correction_type"] = "qblox"
                # Set the distortion correction in the hardware config:
                hardware_config["distortion_corrections"][
                    key
                ] = distortion_correction_dict

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

            # Extract instrument config and channel config dicts:
            instr_config = hardware_config
            # Exclude ["complex_output/input_X", "portclock_configs", i]:
            for key in pc_path[:-3]:
                instr_config = instr_config[key]
            channel_config = instr_config[pc_path[-3]]

            # If RF module, set the lo frequency in the channel config:
            if "RF" in instr_config["instrument_type"]:
                channel_config["lo_freq"] = pc_mod_freqs.lo_freq
            # Else, set the lo frequency in the external lo config:
            elif (lo_name := channel_config.get("lo_name")) is not None:
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
            # Extract the channel config dict:
            channel_config = hardware_config
            # Exclude ["portclock_configs", i]:
            for key in pc_path[:-2]:
                channel_config = channel_config[key]
            pc_config = channel_config["portclock_configs"][pc_path[-1]]

            for config, hw_config_key, hw_compilation_config_value in [
                (pc_config, "mixer_amp_ratio", pc_mix_corr.amp_ratio),
                (pc_config, "mixer_phase_error_deg", pc_mix_corr.phase_error),
                (channel_config, "dc_mixer_offset_I", pc_mix_corr.dc_offset_i),
                (channel_config, "dc_mixer_offset_Q", pc_mix_corr.dc_offset_q),
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
            # Extract channel config:
            channel_config = hardware_config
            # Exclude ["portclock_configs", i]:
            for key in pc_path[:-2]:
                channel_config = channel_config[key]
            channel_name = pc_path[-3]

            # Set the input_gain in the channel config:
            if isinstance(pc_input_gain, ComplexInputGain):
                channel_config["input_gain_I"] = pc_input_gain.gain_I
                channel_config["input_gain_Q"] = pc_input_gain.gain_Q
            elif isinstance(pc_input_gain, RealInputGain):
                if channel_name == "real_output_0":
                    channel_config["input_gain_0"] = pc_input_gain
                elif channel_name == "real_output_1":
                    channel_config["input_gain_1"] = pc_input_gain

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
            # Extract channel config:
            channel_config = hardware_config
            # Exclude ["portclock_configs", i]:
            for key in pc_path[:-2]:
                channel_config = channel_config[key]

            # Set the output_att in the channel config:
            channel_config["output_att"] = pc_output_att

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
            # Extract channel config:
            channel_config = hardware_config
            # Exclude ["portclock_configs", i]:
            for key in pc_path[:-2]:
                channel_config = channel_config[key]

            # Set the input_att in the channel channel config:
            channel_config["input_att"] = pc_input_att

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
            pc_config["init_offset_awg_path_I"] = (
                pc_sequencer_options.init_offset_awg_path_I
            )
            pc_config["init_offset_awg_path_Q"] = (
                pc_sequencer_options.init_offset_awg_path_Q
            )
            pc_config["init_gain_awg_path_I"] = (
                pc_sequencer_options.init_gain_awg_path_I
            )
            pc_config["init_gain_awg_path_Q"] = (
                pc_sequencer_options.init_gain_awg_path_Q
            )
            pc_config["qasm_hook_func"] = pc_sequencer_options.qasm_hook_func

    # Add digital clock to digital channels, so that users don't have to specify it.
    _recursive_digital_channel_search(hardware_config)

    return hardware_config


def find_channel_names(instrument_config: dict[str, Any]) -> list[str]:
    """Find all channel names within this Qblox instrument config dict."""
    channel_names = []
    for channel_name, channel_cfg in instrument_config.items():
        try:
            if "portclock_configs" in channel_cfg.keys():
                channel_names.append(channel_name)
        except AttributeError:
            pass

    return channel_names


def _preprocess_legacy_hardware_config(
    hardware_config: dict[str, Any]
) -> dict[str, Any]:
    """Modify a legacy hardware config into a form that is compatible with the current backend."""

    def _modify_inner_dicts(
        config: dict[str, Any], target_key: str, value_modifier: Callable
    ) -> dict[str, Any]:
        for key, value in config.items():
            if key == target_key:
                config[key] = value_modifier(value)
            elif isinstance(config[key], dict):
                config[key] = _modify_inner_dicts(
                    config=value, target_key=target_key, value_modifier=value_modifier
                )

        return config

    def _replace_deprecated_portclock_keys(
        portclock_configs: list[dict],
    ) -> list[dict]:
        for portclock_config in portclock_configs:
            for deprecated_key, updated_key in {
                "init_offset_awg_path_0": "init_offset_awg_path_I",
                "init_offset_awg_path_1": "init_offset_awg_path_Q",
                "init_gain_awg_path_0": "init_gain_awg_path_I",
                "init_gain_awg_path_1": "init_gain_awg_path_Q",
            }.items():
                if deprecated_key in portclock_config:
                    warnings.warn(
                        f"'{deprecated_key}' is deprecated and will be removed from the public "
                        f"interface in quantify-scheduler >= 0.20.0. Please use "
                        f"'{updated_key}' instead.",
                        FutureWarning,
                    )
                    portclock_config[updated_key] = portclock_config[deprecated_key]
                    del portclock_config[deprecated_key]

        return portclock_configs

    # Preprocessed config is a deepcopy of original config
    return _modify_inner_dicts(
        config=deepcopy(hardware_config),
        target_key="portclock_configs",
        value_modifier=_replace_deprecated_portclock_keys,
    )


def _generate_new_style_hardware_compilation_config(
    old_style_config: dict,
) -> dict:
    """
    Generate a new-style QbloxHardwareCompilationConfig from an old-style hardware config.

    Parameters
    ----------
    old_style_config
        Old-style hardware config.

    Returns
    -------
    dict
        New-style hardware compilation config dictionary.

    """

    def _convert_complex_channel_config(
        cluster_name: str,
        module_slot_idx: int,
        channel_name: str,
        old_channel_config: dict,
        new_style_config: dict,
    ) -> None:
        """Add information from old-style complex channel config to new-style config."""
        new_style_config["hardware_description"][cluster_name]["modules"][
            module_slot_idx
        ][channel_name] = {}
        port_name = f"{cluster_name}.module{module_slot_idx}.{channel_name}"
        for (
            channel_cfg_key,
            channel_cfg_value,
        ) in old_channel_config.items():
            # Find attached port-clock combinations:
            channel_port_clocks = [
                f"{pc_cfg['port']}-{pc_cfg['clock']}"
                for pc_cfg in old_channel_config["portclock_configs"]
            ]
            if channel_cfg_key in [
                "marker_debug_mode_enable",
                "mix_lo",
                "downconverter_freq",
            ]:
                new_style_config["hardware_description"][cluster_name]["modules"][
                    module_slot_idx
                ][channel_name][channel_cfg_key] = channel_cfg_value
            elif channel_cfg_key == "lo_name":
                # Add IQ mixer to the hardware_description:
                new_style_config["hardware_description"][
                    f"iq_mixer_{channel_cfg_value}"
                ] = {"instrument_type": "IQMixer"}
                # Add LO and IQ mixer to connectivity graph:
                new_style_config["connectivity"]["graph"].extend(
                    [
                        (
                            port_name,
                            f"iq_mixer_{channel_cfg_value}.if",
                        ),
                        (
                            f"{channel_cfg_value}.output",
                            f"iq_mixer_{channel_cfg_value}.lo",
                        ),
                    ]
                )
                # Overwrite port_name to IQ mixer RF output:
                port_name = f"iq_mixer_{channel_cfg_value}.rf"
                if "frequency" in old_style_config[channel_cfg_value]:
                    # Set lo_freq for all port-clock combinations (external LO)
                    for port_clock in channel_port_clocks:
                        new_style_config["hardware_options"]["modulation_frequencies"][
                            port_clock
                        ]["lo_freq"] = old_style_config[channel_cfg_value]["frequency"]
            elif channel_cfg_key == "lo_freq":
                # Set lo_freq for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["modulation_frequencies"][
                        port_clock
                    ]["lo_freq"] = channel_cfg_value
            elif channel_cfg_key == "dc_mixer_offset_I":
                # Set mixer offsets for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["mixer_corrections"][
                        port_clock
                    ]["dc_offset_i"] = channel_cfg_value
            elif channel_cfg_key == "dc_mixer_offset_Q":
                # Set mixer offsets for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["mixer_corrections"][
                        port_clock
                    ]["dc_offset_q"] = channel_cfg_value
            elif channel_cfg_key == "input_gain_I":
                # Set input gains for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["input_gain"][port_clock][
                        "gain_I"
                    ] = channel_cfg_value
            elif channel_cfg_key == "input_gain_Q":
                # Set input gains for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["input_gain"][port_clock][
                        "gain_Q"
                    ] = channel_cfg_value
            elif channel_cfg_key == "output_att":
                # Set output attenuation for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["output_att"][
                        port_clock
                    ] = channel_cfg_value
            elif channel_cfg_key == "input_att":
                # Set input attenuation for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["input_att"][
                        port_clock
                    ] = channel_cfg_value
            elif channel_cfg_key == "portclock_configs":
                # Add connectivity information to connectivity graph:
                for portclock_cfg in channel_cfg_value:
                    new_style_config["connectivity"]["graph"].append(
                        (
                            port_name,
                            f"{portclock_cfg['port']}",
                        )
                    )
                    port_clock = (
                        f"{portclock_cfg.pop('port')}-{portclock_cfg.pop('clock')}"
                    )
                    if "interm_freq" in portclock_cfg:
                        # Set intermodulation freqs from portclock config:
                        new_style_config["hardware_options"]["modulation_frequencies"][
                            port_clock
                        ]["interm_freq"] = portclock_cfg.pop("interm_freq")
                    if "mixer_amp_ratio" in portclock_cfg:
                        # Set intermodulation freqs from portclock config:
                        new_style_config["hardware_options"]["mixer_corrections"][
                            port_clock
                        ]["amp_ratio"] = portclock_cfg.pop("mixer_amp_ratio")
                    if "mixer_phase_error_deg" in portclock_cfg:
                        # Set intermodulation freqs from portclock config:
                        new_style_config["hardware_options"]["mixer_corrections"][
                            port_clock
                        ]["phase_error"] = portclock_cfg.pop("mixer_phase_error_deg")
                    if portclock_cfg != {}:
                        # Set remaining portclock config parameters to sequencer options:
                        new_style_config["hardware_options"]["sequencer_options"][
                            port_clock
                        ] = portclock_cfg

    def _convert_real_channel_config(
        cluster_name: str,
        module_slot_idx: int,
        channel_name: str,
        old_channel_config: dict,
        new_style_config: dict,
    ) -> None:
        """Add information from old-style real channel config to new-style config."""
        new_style_config["hardware_description"][cluster_name]["modules"][
            module_slot_idx
        ][channel_name] = {}
        port_name = f"{cluster_name}.module{module_slot_idx}.{channel_name}"
        for (
            channel_cfg_key,
            channel_cfg_value,
        ) in old_channel_config.items():
            # Find attached port-clock combinations:
            channel_port_clocks = [
                f"{pc_cfg['port']}-{pc_cfg['clock']}"
                for pc_cfg in old_channel_config["portclock_configs"]
            ]
            if channel_cfg_key == "marker_debug_mode_enable":
                new_style_config["hardware_description"][cluster_name]["modules"][
                    module_slot_idx
                ][channel_name][channel_cfg_key] = channel_cfg_value
            elif channel_cfg_key in ("input_gain_0", "input_gain_1"):
                # Set input gains for all port-clock combinations
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["input_gain"][
                        port_clock
                    ] = channel_cfg_value
            elif channel_cfg_key == "lo_name":
                # Add IQ mixer to the hardware_description:
                new_style_config["hardware_description"][
                    f"iq_mixer_{channel_cfg_value}"
                ] = {"instrument_type": "IQMixer"}
                # Add LO and IQ mixer to connectivity graph:
                new_style_config["connectivity"]["graph"].extend(
                    [
                        (
                            port_name,
                            f"iq_mixer_{channel_cfg_value}.if",
                        ),
                        (
                            f"{channel_cfg_value}.output",
                            f"iq_mixer_{channel_cfg_value}.lo",
                        ),
                    ]
                )
                # Overwrite port_name to IQ mixer RF output:
                port_name = f"iq_mixer_{channel_cfg_value}.rf"
                if "frequency" in old_style_config[channel_cfg_value]:
                    # Set lo_freq for all port-clock combinations (external LO)
                    for port_clock in channel_port_clocks:
                        new_style_config["hardware_options"]["modulation_frequencies"][
                            port_clock
                        ]["lo_freq"] = old_style_config[channel_cfg_value]["frequency"]
            elif channel_cfg_key == "portclock_configs":
                # Add connectivity information to connectivity graph:
                for portclock_cfg in channel_cfg_value:
                    new_style_config["connectivity"]["graph"].append(
                        (
                            port_name,
                            f"{portclock_cfg['port']}",
                        )
                    )
                    if "init_gain_awg_path_I" in portclock_cfg:
                        # Set init gain from portclock config:
                        new_style_config["hardware_options"]["sequencer_options"][
                            port_clock
                        ]["init_gain_awg_path_I"] = portclock_cfg.pop(
                            "init_gain_awg_path_I"
                        )
                    if "init_gain_awg_path_Q" in portclock_cfg:
                        # Set init gain from portclock config:
                        new_style_config["hardware_options"]["sequencer_options"][
                            port_clock
                        ]["init_gain_awg_path_Q"] = portclock_cfg.pop(
                            "init_gain_awg_path_Q"
                        )

    def _convert_digital_channel_config(
        cluster_name: str,
        module_slot_idx: int,
        channel_name: str,
        old_channel_config: dict,
        new_style_config: dict,
    ) -> None:
        new_style_config["hardware_description"][cluster_name]["modules"][
            module_slot_idx
        ][channel_name] = {}
        for (
            channel_cfg_key,
            channel_cfg_value,
        ) in old_channel_config.items():
            if channel_cfg_key == "portclock_configs":
                # Add connectivity information to connectivity graph:
                for portclock_cfg in channel_cfg_value:
                    new_style_config["connectivity"]["graph"].append(
                        (
                            f"{cluster_name}.module{module_slot_idx}.{channel_name}",
                            f"{portclock_cfg['port']}",
                        )
                    )

    def _convert_cluster_module_config(
        cluster_name: str,
        module_slot_idx: int,
        old_module_config: dict,
        new_style_config: dict,
    ) -> None:
        """Add information from old-style Cluster module config to new-style config."""
        new_style_config["hardware_description"][cluster_name]["modules"][
            module_slot_idx
        ] = {}
        for module_cfg_key, module_cfg_value in old_module_config.items():
            if module_cfg_key in ["instrument_type", "sequence_to_file"]:
                new_style_config["hardware_description"][cluster_name]["modules"][
                    module_slot_idx
                ][module_cfg_key] = module_cfg_value
            elif module_cfg_key.startswith("complex_"):
                # Portclock configs dict must be last item in dict for correct conversion
                old_channel_config = {
                    k: v
                    for k, v in module_cfg_value.items()
                    if k != "portclock_configs"
                }
                old_channel_config["portclock_configs"] = module_cfg_value[
                    "portclock_configs"
                ]
                _convert_complex_channel_config(
                    cluster_name=cluster_name,
                    module_slot_idx=module_slot_idx,
                    channel_name=module_cfg_key,
                    old_channel_config=old_channel_config,
                    new_style_config=new_style_config,
                )
                # Remove channel description if only default values are set
                parsed_channel_description = ComplexChannelDescription.model_validate(
                    new_style_config["hardware_description"][cluster_name]["modules"][
                        module_slot_idx
                    ][module_cfg_key]
                )
                if (
                    parsed_channel_description.model_dump()
                    == ComplexChannelDescription().model_dump()
                ):
                    new_style_config["hardware_description"][cluster_name]["modules"][
                        module_slot_idx
                    ].pop(module_cfg_key)
            elif module_cfg_key.startswith("real_"):
                # Portclock configs dict must be last item in dict for correct conversion
                old_channel_config = {
                    k: v
                    for k, v in module_cfg_value.items()
                    if k != "portclock_configs"
                }
                old_channel_config["portclock_configs"] = module_cfg_value[
                    "portclock_configs"
                ]
                _convert_real_channel_config(
                    cluster_name=cluster_name,
                    module_slot_idx=module_slot_idx,
                    channel_name=module_cfg_key,
                    old_channel_config=module_cfg_value,
                    new_style_config=new_style_config,
                )
                # Remove channel description if only default values are set
                parsed_channel_description = RealChannelDescription.model_validate(
                    new_style_config["hardware_description"][cluster_name]["modules"][
                        module_slot_idx
                    ][module_cfg_key]
                )
                if (
                    parsed_channel_description.model_dump()
                    == RealChannelDescription().model_dump()
                ):
                    new_style_config["hardware_description"][cluster_name]["modules"][
                        module_slot_idx
                    ].pop(module_cfg_key)
            elif module_cfg_key.startswith("digital_"):
                _convert_digital_channel_config(
                    cluster_name=cluster_name,
                    module_slot_idx=module_slot_idx,
                    channel_name=module_cfg_key,
                    old_channel_config=module_cfg_value,
                    new_style_config=new_style_config,
                )

    def _convert_cluster_config(
        cluster_name: str, old_cluster_config: dict, new_style_config: dict
    ) -> None:
        """Add information from old-style Cluster config to new-style config."""
        new_style_config["hardware_description"][cluster_name] = {
            "instrument_type": "Cluster",
            "modules": {},
        }
        for cluster_cfg_key, cluster_cfg_value in old_cluster_config.items():
            if cluster_cfg_key in ["ref", "sequence_to_file"]:
                new_style_config["hardware_description"][cluster_name][
                    cluster_cfg_key
                ] = cluster_cfg_value
            elif "module" in cluster_cfg_key:
                _convert_cluster_module_config(
                    cluster_name=cluster_name,
                    module_slot_idx=int(cluster_cfg_key.split(sep="module")[1]),
                    old_module_config=cluster_cfg_value,
                    new_style_config=new_style_config,
                )

    # Update deprecated keys and avoid modifying the original config
    old_style_config = _preprocess_legacy_hardware_config(old_style_config)

    # Initialize new-style hardware compilation config dictionary
    new_style_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {},
        "hardware_options": defaultdict(lambda: defaultdict(dict)),
        "connectivity": {"graph": []},
    }

    # Loop over old-style config and populate new-style input dicts
    for hw_cfg_key, hw_cfg_value in old_style_config.items():
        if hw_cfg_key == "backend":
            pass
        elif hw_cfg_key in ["latency_corrections", "distortion_corrections"]:
            new_style_config["hardware_options"][hw_cfg_key] = hw_cfg_value
        elif "instrument_type" not in hw_cfg_value:
            warnings.warn(
                f"Skipping hardware config entry '{hw_cfg_key}' because it does not specify an instrument type."
            )
        elif hw_cfg_value["instrument_type"] == "Cluster":
            _convert_cluster_config(
                cluster_name=hw_cfg_key,
                old_cluster_config=hw_cfg_value,
                new_style_config=new_style_config,
            )
        elif hw_cfg_value["instrument_type"] == "LocalOscillator":
            new_style_config["hardware_description"][hw_cfg_key] = {}
            for lo_cfg_key, lo_cfg_value in hw_cfg_value.items():
                if lo_cfg_key in ["instrument_type", "power"]:
                    new_style_config["hardware_description"][hw_cfg_key][
                        lo_cfg_key
                    ] = lo_cfg_value
                elif lo_cfg_key == "frequency":
                    pass
                else:
                    raise KeyError(
                        f"Unexpected key {lo_cfg_key} in LocalOscillator config."
                    )
        else:
            raise ValueError(
                f"Unexpected instrument_type {hw_cfg_value['instrument_type']} in old-style hardware config."
            )
    return new_style_config
