# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions for Qblox backend."""
from __future__ import annotations

import dataclasses
import math
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.types.qblox import (
    BoundedParameter,
    ComplexChannelDescription,
    DigitalChannelDescription,
    OpInfo,
    RealChannelDescription,
)
from quantify_scheduler.helpers.schedule import _extract_port_clocks_used
from quantify_scheduler.helpers.waveforms import exec_waveform_function
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    ControlFlowOperation,
    LoopOperation,
)
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import WindowOperation
from quantify_scheduler.schedules.schedule import Schedule, ScheduleBase

if TYPE_CHECKING:
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
    try:
        duration_validated = duration or data_dict["duration"]
    except KeyError as exc:
        raise TypeError(
            "Parameter 'duration' has value None. If 'data_dict' does not contain "
            "'duration', the function parameter can be used instead."
        ) from exc

    num_samples = round(duration_validated * sampling_rate)
    t = np.arange(start=0, stop=num_samples, step=1) / sampling_rate

    wf_data = exec_waveform_function(wf_func=data_dict["wf_func"], t=t, pulse_info=data_dict)

    return wf_data


def generate_waveform_names_from_uuid(uuid: object) -> tuple[str, str]:
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


def add_to_wf_dict_if_unique(wf_dict: dict[str, Any], waveform: np.ndarray) -> int:
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
    dict[str, Any]
        The (updated) wf_dict.
    str
        The uuid of the waveform.
    int
        The index.

    """

    def generate_entry(name: str, data: np.ndarray, idx: int) -> dict[str, Any]:
        return {name: {"data": data.tolist(), "index": idx}}

    def find_first_free_wf_index() -> int:
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

    tolerance = constants.GRID_TIME_TOLERANCE_TIME
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
            f" (tolerance: {tolerance:.0e} ns). If you think this is a mistake, try "
            "increasing the tolerance by setting e.g.:"
            f" `quantify_scheduler.backends.qblox.constants.GRID_TIME_TOLERANCE_TIME = 0.1e-3` "
            "at the top of your script."
        )

    return time_ns


def is_multiple_of_grid_time(time: float, grid_time_ns: int = constants.GRID_TIME) -> bool:
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
        min_max_frequency_in_hz = constants.NCO_FREQ_LIMIT_STEPS / constants.NCO_FREQ_STEPS_PER_HZ
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

    def __post_init__(self) -> None:
        if self.clock is None or math.isnan(self.clock):
            raise ValueError(f"Clock frequency must be specified ({self.clock=}).")
        if self.LO is not None and math.isnan(self.LO):
            self.LO = None
        if self.IF is not None and math.isnan(self.IF):
            self.IF = None


@dataclasses.dataclass(frozen=True)
class ValidatedFrequencies:
    """Simple dataclass that holds immutable frequencies after validation."""

    clock: float
    LO: float
    IF: float


def determine_clock_lo_interm_freqs(
    freqs: Frequencies,
    downconverter_freq: float | None = None,
    mix_lo: bool | None = True,
) -> ValidatedFrequencies:
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
        :class:`.ValidatedFrequencies` object containing the determined LO and IF
        frequencies and the optionally downconverted clock frequency.

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
        In case ``mix_lo`` is ``True`` and both LO frequency
        and IF have been supplied and do not adhere to
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
            raise ValueError(f"Downconverter frequency must be positive ({downconverter_freq=:e})")

        if downconverter_freq < clock_freq:
            raise ValueError(
                f"Downconverter frequency must be greater than clock frequency "
                f"({downconverter_freq=:e}, {clock_freq=:e})"
            )

        return downconverter_freq - clock_freq

    if downconverter_freq is not None:
        freqs.clock = _downconvert_clock(
            downconverter_freq=downconverter_freq,
            clock_freq=freqs.clock,
        )
    if not mix_lo:
        if freqs.LO is not None and not math.isclose(freqs.LO, freqs.clock):
            warnings.warn(f"Overriding {freqs.LO=} to {freqs.clock=} due to mix_lo=False.")
        freqs.LO = freqs.clock
        if freqs.IF is None:
            raise ValueError(
                f"Frequency settings underconstrained for {freqs.clock=}. "
                "If mix_lo=False is specified, the IF must also be supplied "
                f"({freqs.IF=})."
            )
    elif freqs.LO is None and freqs.IF is None:
        raise ValueError(
            f"Frequency settings underconstrained for {freqs.clock=}."
            f" Neither LO nor IF supplied ({freqs.LO=}, {freqs.IF=})."
        )
    elif freqs.LO is not None and freqs.IF is not None:
        if not math.isclose(freqs.LO + freqs.IF, freqs.clock):
            raise ValueError(
                f"Frequency settings overconstrained."
                f" {freqs.clock=} must be equal to "
                f"{freqs.LO=}+{freqs.IF=} when both are supplied."
            )
    elif freqs.LO is None and freqs.IF is not None:
        freqs.LO = freqs.clock - freqs.IF
    elif freqs.LO is not None and freqs.IF is None:
        freqs.IF = freqs.clock - freqs.LO

    return ValidatedFrequencies(clock=freqs.clock, LO=freqs.LO, IF=freqs.IF)  # type: ignore


def generate_port_clock_to_device_map(
    device_compilers: dict[str, Any],
) -> dict[str, str]:
    """
    Generates a mapping that specifies which port-clock combinations belong to which
    device.

    Here, device means a top-level entry in the hardware config, e.g. a Cluster,
    not which module within the Cluster.

    Each port-clock combination may only occur once.

    Parameters
    ----------
    device_compilers:
        Dictionary containing compiler configs.


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
    for device_name, device_compiler in device_compilers.items():
        if hasattr(device_compiler, "portclock_to_path"):
            for portclock in device_compiler.portclock_to_path:
                portclock_map[portclock] = device_name

    return portclock_map


class LoopBegin(Operation):
    """
    Operation to indicate the beginning of a loop.

    Parameters
    ----------
    repetitions : int
        number of repetitions
    t0 : float, optional
        time offset, by default 0

    """

    def __init__(self, repetitions: int, t0: float = 0) -> None:
        super().__init__(name="Loop")
        self.data.update(
            {
                "name": "Loop",
                "control_flow_info": {
                    "t0": t0,
                    "repetitions": repetitions,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        """
        Represent the Operation as string.

        Returns
        -------
        str
            description

        """
        return self._get_signature(self.data["control_flow_info"])


class ConditionalBegin(Operation):
    """
    Operation to indicate the beginning of a conditional.

    Parameters
    ----------
    qubit_name
        The name of the qubit to condition on.
    feedback_trigger_address
        Feedback trigger address
    t0
        Time offset, by default 0


    """

    def __init__(self, qubit_name: str, feedback_trigger_address: int, t0: float) -> None:
        class_name = self.__class__.__name__
        super().__init__(name=class_name)
        self.data.update(
            {
                "name": class_name,
                "control_flow_info": {
                    "qubit_name": qubit_name,
                    "t0": t0,
                    "feedback_trigger_address": feedback_trigger_address,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        """
        Represent the Operation as string.

        Returns
        -------
        str
            The string representation of this operation.

        """
        return self._get_signature(self.data["control_flow_info"])


def _get_control_flow_begin(
    control_flow_operation: ControlFlowOperation,
) -> Operation:
    assert isinstance(control_flow_operation, (LoopOperation, ConditionalOperation))

    port_clocks = _extract_port_clocks_used(control_flow_operation)
    if isinstance(control_flow_operation, LoopOperation):
        begin_operation: Operation = LoopBegin(
            control_flow_operation.data["control_flow_info"]["repetitions"],
            control_flow_operation.data["control_flow_info"]["t0"],
        )
    else:
        begin_operation = ConditionalBegin(
            control_flow_operation.data["control_flow_info"]["qubit_name"],
            control_flow_operation.data["control_flow_info"]["feedback_trigger_address"],
            control_flow_operation.data["control_flow_info"]["t0"],
        )
    begin_operation["pulse_info"] = [
        {
            "wf_func": None,
            "clock": clock,
            "port": port,
            "duration": 0,
            "control_flow_begin": True,
            **begin_operation["control_flow_info"],
        }
        for port, clock in port_clocks
    ]
    return begin_operation


class _ControlFlowReturn(Operation):
    """
    An operation that signals the end of the current control flow statement.

    Cannot be added to Schedule manually.

    Parameters
    ----------
    t0 : float, optional
        time offset, by default 0

    """

    def __init__(self, t0: float = 0) -> None:
        super().__init__(name="ControlFlowReturn")
        self.data.update(
            {
                "name": "ControlFlowReturn",
                "control_flow_info": {
                    "t0": t0,
                    "duration": 0.0,
                    "return_stack": True,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        return self._get_signature(self.data["control_flow_info"])


def _get_control_flow_end(
    control_flow_operation: ControlFlowOperation,
) -> Operation:
    assert isinstance(control_flow_operation, (LoopOperation, ConditionalOperation))

    port_clocks = _extract_port_clocks_used(control_flow_operation)
    end_operation: Operation = _ControlFlowReturn()
    end_operation["pulse_info"] = [
        {
            "wf_func": None,
            "clock": clock,
            "port": port,
            "duration": 0,
            "control_flow_end": True,
            **end_operation["control_flow_info"],
        }
        for port, clock in port_clocks
    ]
    return end_operation


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
    elif isinstance(operation, ControlFlowOperation):
        accumulator.append((to_grid_time(time_offset) * 1e-9, _get_control_flow_begin(operation)))
        _get_list_of_operations_for_op_info_creation(operation.body, time_offset, accumulator)
        assert operation.body.duration is not None
        accumulator.append(
            (
                to_grid_time(time_offset + operation.body.duration) * 1e-9,
                _get_control_flow_end(operation),
            )
        )
    else:
        accumulator.append((to_grid_time(time_offset) * 1e-9, operation))


def assign_pulse_and_acq_info_to_devices(
    schedule: Schedule,
    device_compilers: dict[str, ClusterCompiler],
) -> None:
    """
    Traverses the schedule and generates `OpInfo` objects for every pulse and
    acquisition, and assigns it to the correct `ClusterCompiler`.

    Parameters
    ----------
    schedule
        The schedule to extract the pulse and acquisition info from.
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
    portclock_mapping = generate_port_clock_to_device_map(device_compilers)

    list_of_operations: list[tuple[float, Operation]] = list()
    _get_list_of_operations_for_op_info_creation(schedule, 0, list_of_operations)
    list_of_operations.sort(key=lambda abs_time_and_op: abs_time_and_op[0])

    for operation_start_time, op_data in list_of_operations:
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
                    "reference_magnitude parameter not implemented. "
                    "This parameter will be ignored.",
                    RuntimeWarning,
                )

            port = pulse_data["port"]
            clock = pulse_data["clock"]
            portclock = f"{port}-{clock}"

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=pulse_data,
                timing=pulse_start_time,
            )

            if port is None:
                # Distribute clock operations to all sequencers utilizing that clock
                for map_portclock, device_name in portclock_mapping.items():
                    map_port, map_clock = map_portclock.split("-")
                    if (combined_data.name == "LatchReset") or map_clock == clock:
                        device_compilers[device_name].add_op_info(
                            port=map_port, clock=map_clock, op_info=combined_data
                        )
            else:
                if portclock not in portclock_mapping:
                    raise KeyError(
                        f"Could not assign pulse data to device. The combination "
                        f"of port {port} and clock {clock} could not be found "
                        f"in hardware configuration.\n\nAre both the port and clock "
                        f"specified in the hardware configuration?\n\n"
                        f"Relevant operation:\n{combined_data}."
                    )
                device_name = portclock_mapping[portclock]
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
            portclock = f"{port}-{clock}"

            if port is None:
                continue

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=acq_data,
                timing=acq_start_time,
            )

            if portclock not in portclock_mapping:
                raise KeyError(
                    f"Could not assign acquisition data to device. The combination "
                    f"of port {port} and clock {clock} could not be found "
                    f"in hardware configuration.\n\nAre both the port and clock "
                    f"specified in the hardware configuration?\n\n"
                    f"Relevant operation:\n{combined_data}."
                )
            device_name = portclock_mapping[portclock]
            device_compilers[device_name].add_op_info(port=port, clock=clock, op_info=combined_data)


def calc_from_units_volt(
    voltage_range: BoundedParameter, name: str, param_name: str, offset: float | None
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
        The name of the offset parameter this method is using.
    offset
        The value of the offset parameter this method is using.


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
    offset_in_arg = offset  # Always in volts
    if offset_in_arg is None:
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

    calculated_offset = offset_in_arg * conversion_factor
    if calculated_offset < voltage_range.min_val or calculated_offset > voltage_range.max_val:
        raise ValueError(
            f"Attempting to set {param_name} of {name} to "
            f"{offset_in_arg} V. {param_name} has to be between "
            f"{voltage_range.min_val / conversion_factor} and "
            f"{voltage_range.max_val / conversion_factor} V!"
        )

    return calculated_offset


def single_scope_mode_acquisition_raise(
    sequencer_0: int, sequencer_1: int, module_name: str
) -> None:
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


def _generate_new_style_hardware_compilation_config(  # noqa PLR0915 too many statements. Remove on duplication
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
        new_style_config["hardware_description"][cluster_name]["modules"][module_slot_idx][
            channel_name
        ] = {}
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
                new_style_config["hardware_description"][cluster_name]["modules"][module_slot_idx][
                    channel_name
                ][channel_cfg_key] = channel_cfg_value
            elif channel_cfg_key == "lo_name":
                # Add IQ mixer to the hardware_description:
                new_style_config["hardware_description"][f"iq_mixer_{channel_cfg_value}"] = {
                    "instrument_type": "IQMixer"
                }
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
                        new_style_config["hardware_options"]["modulation_frequencies"][port_clock][
                            "lo_freq"
                        ] = old_style_config[channel_cfg_value]["frequency"]
            elif channel_cfg_key == "lo_freq":
                # Set lo_freq for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["modulation_frequencies"][port_clock][
                        "lo_freq"
                    ] = channel_cfg_value
            elif channel_cfg_key == "auto_lo_cal":
                # Set auto_lo_cal for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["mixer_corrections"][port_clock][
                        "auto_lo_cal"
                    ] = channel_cfg_value
            elif channel_cfg_key == "dc_mixer_offset_I":
                # Set mixer offsets for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["mixer_corrections"][port_clock][
                        "dc_offset_i"
                    ] = channel_cfg_value
            elif channel_cfg_key == "dc_mixer_offset_Q":
                # Set mixer offsets for all port-clock combinations (RF modules)
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["mixer_corrections"][port_clock][
                        "dc_offset_q"
                    ] = channel_cfg_value
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
                    port_clock = f"{portclock_cfg.pop('port')}-{portclock_cfg.pop('clock')}"
                    if "interm_freq" in portclock_cfg:
                        # Set intermodulation freqs from portclock config:
                        new_style_config["hardware_options"]["modulation_frequencies"][port_clock][
                            "interm_freq"
                        ] = portclock_cfg.pop("interm_freq")
                    if "mixer_amp_ratio" in portclock_cfg:
                        # Set intermodulation freqs from portclock config:
                        new_style_config["hardware_options"]["mixer_corrections"][port_clock][
                            "amp_ratio"
                        ] = portclock_cfg.pop("mixer_amp_ratio")
                    if "auto_sideband_cal" in portclock_cfg:
                        # Set auto_sideband_cal from portclock config:
                        new_style_config["hardware_options"]["mixer_corrections"][port_clock][
                            "auto_sideband_cal"
                        ] = portclock_cfg.pop("auto_sideband_cal")
                    if "mixer_phase_error_deg" in portclock_cfg:
                        # Set intermodulation freqs from portclock config:
                        new_style_config["hardware_options"]["mixer_corrections"][port_clock][
                            "phase_error"
                        ] = portclock_cfg.pop("mixer_phase_error_deg")
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
        new_style_config["hardware_description"][cluster_name]["modules"][module_slot_idx][
            channel_name
        ] = {}
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
            if channel_cfg_key in ["marker_debug_mode_enable", "mix_lo"]:
                new_style_config["hardware_description"][cluster_name]["modules"][module_slot_idx][
                    channel_name
                ][channel_cfg_key] = channel_cfg_value
            elif channel_cfg_key in ("input_gain_0", "input_gain_1"):
                # Set input gains for all port-clock combinations
                for port_clock in channel_port_clocks:
                    new_style_config["hardware_options"]["input_gain"][
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
                    port_clock = f"{portclock_cfg.pop('port')}-{portclock_cfg.pop('clock')}"

                    if "interm_freq" in portclock_cfg:
                        # Set intermodulation freqs from portclock config:
                        new_style_config["hardware_options"]["modulation_frequencies"][port_clock][
                            "interm_freq"
                        ] = portclock_cfg.pop("interm_freq")
                    if "init_gain_awg_path_I" in portclock_cfg:
                        # Set init gain from portclock config:
                        new_style_config["hardware_options"]["sequencer_options"][port_clock][
                            "init_gain_awg_path_I"
                        ] = portclock_cfg.pop("init_gain_awg_path_I")
                    if "init_gain_awg_path_Q" in portclock_cfg:
                        # Set init gain from portclock config:
                        new_style_config["hardware_options"]["sequencer_options"][port_clock][
                            "init_gain_awg_path_Q"
                        ] = portclock_cfg.pop("init_gain_awg_path_Q")
                    if any(
                        option in portclock_cfg
                        for option in [
                            "init_gain_awg_path_I",
                            "init_gain_awg_path_Q",
                            "init_offset_awg_path_I",
                            "init_offset_awg_path_Q",
                            "qasm_hook_func",
                            "ttl_acq_threshold",
                        ]
                    ):
                        # Set remaining portclock config parameters to sequencer options:
                        new_style_config["hardware_options"]["sequencer_options"][
                            port_clock
                        ] = portclock_cfg

            if any("optical_control" in pc for pc in channel_port_clocks):
                channel_mixer = "OpticalModulator"
                mixer_tag = "optical_mod"
                mixer_output_tag = "out"
            else:
                channel_mixer = "IQMixer"
                mixer_tag = "iq_mixer"
                mixer_output_tag = "rf"

            if channel_cfg_key == "lo_name":
                # Add optical/iq mixer to the hardware_description
                new_style_config["hardware_description"][f"{mixer_tag}_{channel_cfg_value}"] = {
                    "instrument_type": channel_mixer
                }
                # Add LO and mixer to connectivity graph:
                new_style_config["connectivity"]["graph"].extend(
                    [
                        (
                            port_name,
                            f"{mixer_tag}_{channel_cfg_value}.if",
                        ),
                        (
                            f"{channel_cfg_value}.output",
                            f"{mixer_tag}_{channel_cfg_value}.lo",
                        ),
                    ]
                )
                # Overwrite port_name to mixer output:
                port_name = f"{mixer_tag}_{channel_cfg_value}.{mixer_output_tag}"
                if "frequency" in old_style_config[channel_cfg_value]:
                    # Set lo_freq for all port-clock combinations (external LO)
                    for port_clock in channel_port_clocks:
                        new_style_config["hardware_options"]["modulation_frequencies"][port_clock][
                            "lo_freq"
                        ] = old_style_config[channel_cfg_value]["frequency"]

    def _convert_digital_channel_config(
        cluster_name: str,
        module_slot_idx: int,
        channel_name: str,
        old_channel_config: dict,
        new_style_config: dict,
    ) -> None:
        new_style_config["hardware_description"][cluster_name]["modules"][module_slot_idx][
            channel_name
        ] = {}
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

                    port_clock = f"{portclock_cfg.pop('port')}-{portclock_cfg.pop('clock')}"
                    if "in_threshold_primary" in portclock_cfg:
                        # Set init gain from portclock config:
                        new_style_config["hardware_options"]["digitization_thresholds"][port_clock][
                            "in_threshold_primary"
                        ] = portclock_cfg.pop("in_threshold_primary")

    def _convert_cluster_module_config(
        cluster_name: str,
        module_slot_idx: int,
        old_module_config: dict,
        new_style_config: dict,
    ) -> None:
        """Add information from old-style Cluster module config to new-style config."""
        new_style_config["hardware_description"][cluster_name]["modules"][module_slot_idx] = {}
        for module_cfg_key, module_cfg_value in old_module_config.items():
            if module_cfg_key in ["instrument_type", "sequence_to_file"]:
                new_style_config["hardware_description"][cluster_name]["modules"][module_slot_idx][
                    module_cfg_key
                ] = module_cfg_value
            elif module_cfg_key.startswith("complex_"):
                # Portclock configs dict must be last item in dict for correct conversion
                old_channel_config = {
                    k: v for k, v in module_cfg_value.items() if k != "portclock_configs"
                }
                old_channel_config["portclock_configs"] = module_cfg_value["portclock_configs"]
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
                    k: v for k, v in module_cfg_value.items() if k != "portclock_configs"
                }
                old_channel_config["portclock_configs"] = module_cfg_value["portclock_configs"]
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
                if parsed_channel_description.model_dump() == RealChannelDescription().model_dump():
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
                # Remove channel description if only default values are set
                parsed_channel_description = DigitalChannelDescription.model_validate(
                    new_style_config["hardware_description"][cluster_name]["modules"][
                        module_slot_idx
                    ][module_cfg_key]
                )
                if (
                    parsed_channel_description.model_dump()
                    == DigitalChannelDescription().model_dump()
                ):
                    new_style_config["hardware_description"][cluster_name]["modules"][
                        module_slot_idx
                    ].pop(module_cfg_key)

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

    warnings.warn(
        "The hardware configuration dictionary is deprecated and will not be supported in "
        "quantify-scheduler >= 1.0.0. Please use a `HardwareCompilationConfig` instead. For "
        "more information on how to migrate from old- to new-style hardware specification, "
        "please visit "
        "https://quantify-os.org/docs/quantify-scheduler/dev/examples/hardware_config_migration.html"  # noqa: E501 Line too long
        " in the documentation.",
        FutureWarning,
    )

    old_style_config = deepcopy(old_style_config)

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
                f"Skipping hardware config entry '{hw_cfg_key}' "
                f"because it does not specify an instrument type."
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
                    new_style_config["hardware_description"][hw_cfg_key][lo_cfg_key] = lo_cfg_value
                elif lo_cfg_key == "frequency":
                    pass
                else:
                    raise KeyError(f"Unexpected key {lo_cfg_key} in LocalOscillator config.")
        else:
            raise ValueError(
                f"Unexpected instrument_type {hw_cfg_value['instrument_type']} "
                f"in old-style hardware config."
            )
    return new_style_config


def is_square_pulse(operation: Operation | Schedule) -> bool:
    """
    Check if the operation is a square pulse.

    Parameters
    ----------
    operation:
        The operation to check.

    Returns
    -------
    :
        True if the operation is a square pulse, False otherwise.

    """
    for pulse_info in operation.data["pulse_info"]:
        if pulse_info["wf_func"] != "quantify_scheduler.waveforms.square":
            return False
    return True


def convert_qtm_fine_delay_to_int(fine_delay: float) -> int:
    """Convert a fine delay value in seconds to an integer value for Q1ASM."""
    fine_delay_int = round(fine_delay * 128e9)
    if (
        not 0
        <= fine_delay_int
        <= constants.MAX_QTM_FINE_DELAY_NS * constants.QTM_FINE_DELAY_INT_TO_NS_RATIO
    ):
        raise ValueError(
            f"Fine delay value {fine_delay} s is outside of the hardware supported "
            f"range of (0, {constants.MAX_QTM_FINE_DELAY_NS}) ns."
        )
    return fine_delay_int
