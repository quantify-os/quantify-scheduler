# -----------------------------------------------------------------------------
# Description:    Compiler backend for Qblox hardware.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations

import inspect
import os
import sys
import warnings
import json
from columnar import columnar
from columnar.exceptions import TableOverflowError
from qcodes.utils.helpers import NumpyJSONEncoder
from abc import ABCMeta, abstractmethod
from collections import UserDict, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple, Union, Set, Callable

import numpy as np
from dataclasses_json import DataClassJsonMixin
from quantify.data.handling import get_datadir, gen_tuid
from quantify.utilities.general import make_hash, without, import_func_from_string

if TYPE_CHECKING:
    from quantify.scheduler.types import Schedule


# ---------- utility functions ----------
def _sanitize_file_name(filename: str) -> str:
    """
    Takes a str representing a filename and removes invalid characters
    by replacing it with an underscore character.

    e.g. invalid:file?name.json -> invalid_file_name.json

    Characters that are considered invalid: ',<>:"/\\|!?* '

    Parameters
    ----------
    filename
        The str representing the filename

    Returns
    -------
        str
            The sanitized filename
    """
    invalid = ',<>:"/\\|!?* '
    sanitized_fn = filename
    for char in invalid:
        sanitized_fn = sanitized_fn.replace(char, "_")
    return sanitized_fn


def modulate_waveform(
    t: np.ndarray, envelope: np.ndarray, freq: float, t0: float = 0
) -> np.ndarray:
    """
    Generates a (single sideband) modulated waveform from a given envelope by
    multiplying it with a complex exponential.

    .. math::
        z_{mod} (t) = z (t) \cdot e^{2\pi i f (t+t_0)}

    The signs are chosen such that the frequencies follow the relation RF = LO + IF for
    LO, IF > 0.

    Parameters
    ----------
    t: np.ndarray
        A numpy array with time values
    envelope: np.ndarray
        The complex-valued envelope of the modulated waveform
    freq: float
        The frequency of the modulation
    t0: float
        Time offset for the modulation

    Returns
    -------
        np.ndarray
            The modulated waveform
    """
    modulation = np.exp(1.0j * 2 * np.pi * freq * (t + t0))
    return envelope * modulation


def apply_mixer_skewness_corrections(
    waveform: np.ndarray, amplitude_ratio: float, phase_shift: float
) -> np.ndarray:
    """
    Takes a waveform and applies a correction for amplitude imbalances and
    phase errors when using an IQ mixer from previously calibrated values.

    Phase correction is done using:

    .. math::
        Re(z_{corrected}) (t) = Re(z (t)) + Im(z (t)) \tan(\phi)
        Im(z_{corrected}) (t) = Im(z (t)) / \cos(\phi)

    The amplitude correction is achieved by rescaling the waveforms back to their
    original amplitudes and multiplying or dividing the I and Q signals respectively by
    the square root of the amplitude ratio.

    Parameters
    ----------
    waveform: np.ndarray
        The complex valued waveform on which the correction will be applied.
    amplitude_ratio: float
        The ratio between the amplitudes of I and Q that is used to correct
        for amplitude imbalances between the different paths in the IQ mixer.
    phase_shift: float
        The phase error (in deg) used to correct the phase between I and Q.

    Returns
    -------
        np.ndarray
            The complex valued waveform with the applied phase and amplitude
            corrections.
    """

    def calc_corrected_re(wf: np.ndarray, alpha: float, phi: float):
        original_amp = np.max(np.abs(wf.real))
        wf_re = wf.real + wf.imag * np.tan(phi)
        wf_re = wf_re / np.max(np.abs(wf_re))
        return wf_re * original_amp * np.sqrt(alpha)

    def calc_corrected_im(wf: np.ndarray, alpha: float, phi: float):
        original_amp = np.max(np.abs(wf.imag))
        wf_im = wf.imag / np.cos(phi)
        wf_im = wf_im / np.max(np.abs(wf_im))
        return wf_im * original_amp / np.sqrt(alpha)

    corrected_re = calc_corrected_re(waveform, amplitude_ratio, np.deg2rad(phase_shift))
    corrected_im = calc_corrected_im(waveform, amplitude_ratio, np.deg2rad(phase_shift))
    return corrected_re + 1.0j * corrected_im


def _generate_waveform_data(data_dict: dict, sampling_rate: float) -> np.ndarray:
    """
    Generates an array using the parameters specified in `data_dict`.

    Parameters
    ----------
    data_dict: dict
        The dictionary that contains the values needed to parameterize the
        waveform. `data_dict['wf_func']` is then called to calculate the values.
    sampling_rate: float
        The sampling rate used to generate the time axis values.

    Returns
    -------
        np.ndarray
            The (possibly complex) values of the generated waveform
    """
    t = np.arange(0, 0 + data_dict["duration"], 1 / sampling_rate)

    func = import_func_from_string(data_dict["wf_func"])
    par_map = inspect.signature(func).parameters.keys()

    data_dict_keys = set(data_dict.keys())
    parameters_supplied = data_dict_keys.intersection(set(par_map))

    kwargs = {key: data_dict[key] for key in parameters_supplied}
    wf_data = func(t=t, **kwargs)
    if not isinstance(wf_data, np.ndarray):
        wf_data = np.array(wf_data)

    return wf_data


def generate_ext_local_oscillators(
    total_play_time: float, hardware_cfg: Dict[str, Any]
) -> Dict[str, LocalOscillator]:
    """
    Traverses the `hardware_cfg` dict and extracts the used local oscillators.
    `LocalOscillator` objects are instantiated for each LO and the `lo_freq` is
    assigned if specified.

    Parameters
    ----------
    total_play_time: float
        Total time the schedule is played for, not counting repetitions.
    hardware_cfg: dict
        Hardware mapping dictionary

    Returns
    -------
        Dict[str, LocalOscillator]
            A dictionary with the names of the devices as keys and compiler
            objects for the local oscillators as values.
    """
    # TODO more generic with get_inner_dicts_containing_key?
    lo_dict = dict()
    for key, device in hardware_cfg.items():
        if not isinstance(device, dict):  # is not a device
            continue

        for io, io_cfg in device.items():
            if not isinstance(io_cfg, dict):  # is not a in/output
                continue

            if "lo_name" in io_cfg:
                lo_name = io_cfg["lo_name"]
                if lo_name not in lo_dict:
                    lo_obj = LocalOscillator(
                        lo_name,
                        total_play_time,
                    )
                    lo_dict[lo_name] = lo_obj

                if "lo_freq" in io_cfg:
                    lo_dict[lo_name].assign_frequency(io_cfg["lo_freq"])

    return lo_dict


def _calculate_total_play_time(schedule: Schedule) -> float:
    """
    Calculates the total time the schedule has to be executed on the hardware, not
    accounting for repetitions. Effectively, this is the maximum of the end times of
    the pulses and acquisitions.

    Parameters
    ----------
    schedule: Schedule
        The quantify schedule object of which we want the total execution time

    Returns
    -------
        float
            Total play time in seconds
    """
    max_found: float = 0.0
    for time_constraint in schedule.timing_constraints:
        pulse_id = time_constraint["operation_hash"]
        operation = schedule.operations[pulse_id]
        end_time = operation.duration + time_constraint["abs_time"]

        if end_time > max_found:
            max_found = end_time

    return max_found


def find_inner_dicts_containing_key(d: Union[Dict, UserDict], key: Any) -> List[dict]:
    """
    Generates a list of the first dictionaries encountered that contain a certain key,
    in a complicated dictionary with nested dictionaries or Iterables.

    This is achieved by recursively traversing the nested structures until the key is
    found, which is then appended to a list.

    Parameters
    ----------
    d: Union[Dict, UserDict]
        The dictionary to traverse.
    key: Any
        The key to search for.

    Returns
    -------
        List[dict]
            A list containing all the inner dictionaries containing the specified key.
    """
    dicts_found = list()
    if isinstance(d, dict):
        if key in d:
            dicts_found.append(d)
    for val in d.values():
        if isinstance(val, dict) or isinstance(val, UserDict):
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


def find_all_port_clock_combinations(d: Union[Dict, UserDict]) -> List[Tuple[str, str]]:
    """
    Generates a list with all port and clock combinations found in a dictionary with
    nested structures. Traversing the dictionary is done using the
    `find_inner_dicts_containing_key` function.

    Parameters
    ----------
    d: Union[Dict, UserDict]
        The dictionary to traverse.

    Returns
    -------
        List[Tuple[str, str]]
            A list containing tuples representing the port and clock combinations found
            in the dictionary.
    """
    port_clocks = list()
    dicts_with_port = find_inner_dicts_containing_key(d, "port")
    for d in dicts_with_port:
        if "port" in d:
            port = d["port"]
            if port is None:
                continue
            if "clock" not in d:
                raise AttributeError(f"Port {d['port']} missing clock")
            clock = d["clock"]
            port_clocks.append((port, clock))
    return port_clocks


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
    mapping: Dict[str, Any]
        The hardware mapping config.

    Returns
    -------
        Dict[Tuple[str, str], str]
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


def find_abs_time_from_operation_hash(schedule: Schedule, op_hash: int) -> float:
    """
    Utility function to find the "abs_time" in the `timing_constraints` of the schedule
    from a given "operation_hash".

    Parameters
    ----------
    schedule: Schedule
        The schedule that contains the operation to get the "abs_time" from.
    op_hash: int
        The operation hash of the operation that we want to get the "abs_time" from.

    Returns
    -------
    float
        The absolute start time of the operation.
    """
    timing_constraints = schedule.timing_constraints
    for tc in timing_constraints:
        if tc["operation_hash"] == op_hash:
            return tc["abs_time"]


def _generate_waveform_dict(
    waveforms_complex: Dict[int, np.ndarray]
) -> Dict[str, dict]:
    """
    Takes a dictionary with complex waveforms and generates a new dictionary with
    real valued waveforms with a unique index, as required by the hardware.

    Parameters
    ----------
    waveforms_complex: Dict[int, np.ndarray]
        Dictionary containing the complex waveforms. Keys correspond to a unique
        identifier, value is the complex waveform.

    Returns
    -------
    Dict[str, dict]
        A dictionary with as key the unique name for that waveform, as value another
        dictionary containing the real-valued data (list) as well as a unique index.
        Note that the index of the Q waveform is always the index of the I waveform
        +1.

    Examples
    --------
    >>> complex_waveforms = {12345: np.array([1, 2])}
    >>> _generate_waveform_dict(complex_waveforms)
    {
        "12345_I": {"data": [1, 2], "index": 0},
        "12345_Q": {"data": [0, 0], "index": 1}
    }
    """
    wf_dict = dict()
    for idx, (uuid, complex_data) in enumerate(waveforms_complex.items()):
        name_i, name_q = Pulsar_sequencer_base.generate_waveform_names_from_uuid(uuid)
        to_add = {
            name_i: {"data": complex_data.real.tolist(), "index": 2 * idx},
            name_q: {"data": complex_data.imag.tolist(), "index": 2 * idx + 1},
        }
        wf_dict.update(to_add)
    return wf_dict


# ---------- classes ----------
class InstrumentCompiler(metaclass=ABCMeta):
    """
    Abstract base class that defines a generic instrument compiler. The subclasses that
    inherit from this are meant to implement the compilation steps needed to compile the
    lists of `OpInfo` representing the pulse and acquisition info to device specific
    instructions.

    For each device that needs to be part of the compilation process such a
    `InstrumentCompiler` should be implemented.
    """

    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Optional[Dict[str, Any]] = None,
    ):
        """
        Constructor for an InstrumentCompiler object.

        Parameters
        ----------
        name: str
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time: str
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        hw_mapping: Optional[Dict[str, Any]]
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        """
        self.name = name
        self.total_play_time = total_play_time
        self.hw_mapping = hw_mapping
        self._pulses = defaultdict(list)
        self._acquisitions = defaultdict(list)

    def add_pulse(self, port: str, clock: str, pulse_info: OpInfo):
        """
        Assigns a certain pulse to this device.

        Parameters
        ----------
        port: str
            The port the pulse needs to be sent to.
        clock: str
            The clock for modulation of the pulse. Can be a BasebandClock.
        pulse_info: OpInfo
            Data structure containing all the information regarding this specific pulse
            operation.

        Returns
        -------

        """
        self._pulses[(port, clock)].append(pulse_info)

    def add_acquisition(self, port: str, clock: str, acq_info: OpInfo):
        """
        Assigns a certain acquisition to this device.

        Parameters
        ----------
        port: str
            The port the pulse needs to be sent to.
        clock: str
            The clock for modulation of the pulse. Can be a BasebandClock.
        acq_info: OpInfo
            Data structure containing all the information regarding this specific
            acquisition operation.

        Returns
        -------

        """
        self._acquisitions[(port, clock)].append(acq_info)

    @property
    def portclocks_with_data(self) -> Set[Tuple[str, str]]:
        """
        All the port-clock combinations associated with at least one pulse or
        acquisition.

        Returns
        -------
        Set[Tuple[str, str]]
            A set containing all the port-clock combinations
        """
        portclocks_used = set()
        portclocks_used.update(self._pulses.keys())
        portclocks_used.update(self._acquisitions.keys())
        return portclocks_used

    @abstractmethod
    def hardware_compile(self, repetitions: int = 1) -> Any:
        """
        An abstract method that should be overridden by a subclass to implement the
        actual compilation. Method turns the pulses and acquisitions added to the device
        into device specific instructions.

        Parameters
        ----------
        repetitions: int
            Number of times execution the schedule is repeated

        Returns
        -------
            Any
                A data structure representing the compiled program. The type is
                dependent on implementation.
        """
        pass


class LocalOscillator(InstrumentCompiler):
    """
    Implementation of an `InstrumentCompiler` that compiles for a generic LO. The main
    difference between this class and the other compiler classes is that it doesn't take
    pulses and acquisitions.
    """

    def __init__(
        self,
        name: str,
        total_play_time: float,
        lo_freq: Optional[int] = None,
    ):
        """
        Constructor for a local oscillator compiler.

        Parameters
        ----------
        name: str
            QCoDeS name of the device it compiles for.
        total_play_time: float
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        lo_freq: Optional[int]
            LO frequency it needs to be set to. Either this is passed to the constructor
            or set later in the compilation process, in case the LO frequency is not
            initially given and needs to be calculated.
        """
        super().__init__(name, total_play_time)
        self._lo_freq = lo_freq

    def assign_frequency(self, freq: float):
        """
        Sets the lo frequency for this device if no frequency is specified, but raises
        an exception otherwise.

        Parameters
        ----------
        freq: float
            The frequency to set it to.

        Returns
        -------

        Raises
        -------
        ValueError
            Occurs when a frequency has been previously set and attempting to set the
            frequency to a different value than what it is currently set to. This would
            indicate an invalid configuration in the hardware mapping.
        """
        if self._lo_freq is not None:
            if freq != self._lo_freq:
                raise ValueError(
                    f"Attempting to set LO {self.name} to frequency {freq}, "
                    f"while it has previously already been set to {self._lo_freq}!"
                )
        self._lo_freq = freq

    @property
    def frequency(self) -> float:
        """
        Getter for the frequency.

        Returns
        -------
        float
            The current frequency
        """
        return self._lo_freq

    def hardware_compile(self, repetitions: int = 1) -> Dict[str, Any]:
        """
        Compiles the program for the LO control stack component.

        Parameters
        ----------
        repetitions: int
            Number of times execution the schedule is repeated

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all the information the cs component needs to set the
            parameters appropriately.
        """
        return {"lo_freq": self._lo_freq}


# ---------- data structures ----------
@dataclass
class OpInfo(DataClassJsonMixin):
    """
    Data structure containing all the information describing a pulse or acquisition
    needed to play it.

    Attributes
    ----------
    uuid: int
        A unique identifier for this pulse/acquisition.
    data: dict
        The pulse/acquisition info taken from the `data` property of the
        pulse/acquisition in the schedule.
    timing: float
        The start time of this pulse/acquisition.
        Note that this is a combination of the start time "t_abs" of the schedule
        operation, and the t0 of the pulse/acquisition which specifies a time relative
        to "t_abs".
    pulse_settings: Optional[QASMRuntimeSettings]
        Settings that are to be set by the sequencer before playing this
        pulse/acquisition. This is used for parameterized behavior e.g. setting a gain
        parameter to change the pulse amplitude, instead of changing the waveform. This
        allows to reuse the same waveform multiple times despite a difference in
        amplitude.
    """

    uuid: int
    data: dict
    timing: float
    pulse_settings: Optional[QASMRuntimeSettings] = None

    @property
    def duration(self) -> float:
        """
        The duration of the pulse/acquisition.

        Returns
        -------
        float
            The duration of the pulse/acquisition.
        """
        return self.data["duration"]

    @property
    def is_acquisition(self):
        """
        Returns true if this is an acquisition, false if it's a pulse.

        Returns
        -------
        bool
            Is this an acquisition?
        """
        return "acq_index" in self.data

    def __repr__(self):
        s = 'Acquisition "' if self.is_acquisition else 'Pulse "'
        s += str(self.uuid)
        s += f'" (t={self.timing} to {self.timing+self.duration})'
        s += f" data={self.data}"
        return s


@dataclass
class PulsarSettings(DataClassJsonMixin):
    """
    Global settings for the pulsar to be set in the control stack component. This is
    kept separate from the settings that can be set on a per sequencer basis, which are
    specified in `SequencerSettings`.

    Attributes
    ----------
    ref: str
        The reference source. Should either be "internal" or "external", will raise an
        exception in the cs component otherwise.
    """

    ref: str


@dataclass
class SequencerSettings(DataClassJsonMixin):
    """
    Sequencer level settings. In the drivers these settings are typically recognised by
    parameter names of the form "sequencer_{index}_{setting}". These settings are set
    once at the start and will remain unchanged after. Meaning that these correspond to
    the "slow" QCoDeS parameters and not settings that are changed dynamically by the
    sequencer.

    Attributes
    ----------
    nco_en: bool
        Specifies whether the nco will be used or not.
    sync_en: bool
        Enables party-line synchronization.
    modulation_freq: float
        Specifies the frequency of the modulation.
    awg_offset_path_0: float
        Sets the DC offset on path 0. This is used e.g. for calibration of lo leakage
        when using IQ mixers.
    awg_offset_path_1: float
        Sets the DC offset on path 1. This is used e.g. for calibration of lo leakage
        when using IQ mixers.
    """

    nco_en: bool
    sync_en: bool
    modulation_freq: float = None
    awg_offset_path_0: float = 0.0
    awg_offset_path_1: float = 0.0


@dataclass
class MixerCorrections(DataClassJsonMixin):
    """
    Data structure that holds all the mixer correction parameters to compensate for
    skewness/lo feed-through. This class is used to correct the waveforms to compensate
    for skewness and to set the `SequencerSettings`.

    Attributes
    ----------
    amp_ratio: float
        Amplitude ratio between the I and Q paths to correct for the imbalance in the
        two path in the IQ mixer.
    phase_error: float
        Phase shift used to compensate for quadrature errors.
    offset_I: float
        DC offset on the I path used to compensate for lo feed-through.
    offset_Q: float
        DC offset on the Q path used to compensate for lo feed-through.
    """

    amp_ratio: float = 1.0
    phase_error: float = 0.0
    offset_I: float = 0.0
    offset_Q: float = 0.0

    def correct_skewness(self, waveform: np.ndarray) -> np.ndarray:
        """
        Applies the pre-distortion needed to compensate for amplitude and phase errors
        in the IQ mixer. In practice this is simply a wrapper around the
        `apply_mixer_skewness_corrections` function, that uses the attributes specified
        here.

        Parameters
        ----------
        waveform: np.ndarray
            The (complex-valued) waveform before correction.

        Returns
        -------
        np.ndarray
            The complex-valued waveform after correction.
        """
        return apply_mixer_skewness_corrections(
            waveform, self.amp_ratio, self.phase_error
        )


@dataclass
class QASMRuntimeSettings:
    """
    Settings that can be changed dynamically by the sequencer during execution of the
    schedule. This is in contrast to the relatively static `SequencerSettings`.

    Attributes
    ----------
    awg_gain_0: float
        Gain set to the AWG output path 0. Value should be in the range -1.0 < param <
        1.0. Else an exception will be raised during compilation.
    awg_gain_1: float
        Gain set to the AWG output path 1. Value should be in the range -1.0 < param <
        1.0. Else an exception will be raised during compilation.
    awg_offset_0: float
        Offset applied to the AWG output path 0. Value should be in the range -1.0 <
        param < 1.0. Else an exception will be raised during compilation.
    awg_offset_1: float
        Offset applied to the AWG output path 1. Value should be in the range -1.0 <
        param < 1.0. Else an exception will be raised during compilation.
    """

    awg_gain_0: float
    awg_gain_1: float
    awg_offset_0: float = 0.0
    awg_offset_1: float = 0.0


# ---------- utility classes ----------
class PulsarInstructions:
    """
    Class that holds all the valid instructions that can be executed by the sequencer.
    """

    # Control
    ILLEGAL = "illegal"
    STOP = "stop"
    NOP = "nop"
    NEW_LINE = ""
    # Jumps
    JUMP = "jmp"
    LOOP = "loop"
    JUMP_GREATER_EQUALS = "jge"
    JUMP_LESS_EQUALS = "jle"
    # Arithmetic
    MOVE = "move"
    NOT = "not"
    ADD = "add"
    SUB = "sub"
    AND = "and"
    OR = "or"
    XOR = "xor"
    ARITHMETIC_SHIFT_LEFT = "asl"
    ARITHMETIC_SHIFT_RIGHT = "asr"
    # Real-time pipeline instructions
    SET_MARKER = "set_mrk"
    PLAY = "play"
    ACQUIRE = "acquire"
    WAIT = "wait"
    WAIT_SYNC = "wait_sync"
    WAIT_TRIGGER = "wait_trigger"
    UPDATE_PARAMETERS = "upd_param"
    SET_AWG_GAIN = "set_awg_gain"
    SET_ACQ_GAIN = "set_acq_gain"
    SET_AWG_OFFSET = "set_awg_offs"
    SET_ACQ_OFFSET = "set_acq_offs"
    SET_NCO_PHASE = "set_ph"
    SET_NCO_PHASE_OFFSET = "set_ph_delta"


class QASMProgram(list):
    """
    Class that holds the compiled Q1ASM program that is to be executed by the sequencer.

    The object itself is a list which holds the instructions in order of execution. The
    instructions in turn are also lists, which hold the instruction strings themselves
    along with labels, comments and parameters.

    Apart from this the class holds some convenience functions that auto generate
    certain instructions with parameters, as well as update the elapsed time.

    Attributes
    ----------
    elapsed_time: int
        The time elapsed after finishing the program in its current form. This is used
        to keep track of the overall timing and necessary waits.
    """

    elapsed_time: int = 0

    @staticmethod
    def get_instruction_as_list(
        instruction: str,
        *args: Union[int, str],
        label: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> List[Union[str, int], ...]:
        """
        Takes an instruction with arguments, label and comment and turns it into the
        list required by the class.

        Parameters
        ----------
        instruction: str
            The instruction to use. This should be one specified in `PulsarInstructions`
            or the assembler will raise an exception.
        args: Union[int, str]
            Arguments to be passed.
        label: Optional[str]
            Adds a label to the line. Used for jumps and loops.
        comment: Optional[str]
            Optionally add a comment to the instruction.

        Returns
        -------
        List[Union[str, int]]
            List that contains all the passed information in the valid format for the
            program.

        Raises
        -------
        SyntaxError
            More arguments passed than the sequencer allows.
        """
        max_args_amount = 3
        if len(args) > max_args_amount:
            raise SyntaxError(
                f"Too many arguments supplied to `get_instruction_tuple` for "
                f"instruction {instruction}."
            )
        instr_args = ",".join(str(arg) for arg in args)

        label_str = f"{label}:" if label is not None else ""
        comment_str = f"# {comment}" if comment is not None else ""
        return [label_str, instruction, instr_args, comment_str]

    def emit(self, *args, **kwargs):
        """
        Wrapper around the `get_instruction_as_list` which adds it to the program.

        Parameters
        ----------
        args: Any
            All arguments to pass to `get_instruction_as_list`.
        kwargs
            All keyword arguments to pass to `get_instruction_as_list`.

        Returns
        -------

        """
        self.append(self.get_instruction_as_list(*args, **kwargs))

    # --- QOL functions -----

    def auto_wait(self, wait_time: int):
        """
        Automatically emits a correct wait command. If the wait time is longer than
        allowed by the sequencer it correctly breaks it up into multiple wait
        instructions.

        Parameters
        ----------
        wait_time: int
            Time to wait in ns.

        Returns
        -------

        Raises
        ------
        ValueError
            If `wait_time` <= 0
        """
        if wait_time <= 0:
            raise ValueError(
                f"Invalid wait time. Attempting to wait "
                f"for {wait_time} ns at t={self.elapsed_time}"
                f" ns."
            )

        immediate_sz = Pulsar_sequencer_base.IMMEDIATE_SZ
        if wait_time > immediate_sz:
            for _ in range(wait_time // immediate_sz):
                self.emit(
                    PulsarInstructions.WAIT, immediate_sz, comment="auto generated wait"
                )
            time_left = wait_time % immediate_sz
        else:
            time_left = int(wait_time)

        if time_left > 0:
            self.emit(PulsarInstructions.WAIT, time_left)

        self.elapsed_time += wait_time

    def wait_till_start_operation(self, operation: OpInfo):
        """
        Waits until the start of a pulse or acquisition.

        Parameters
        ----------
        operation: OpInfo
            The pulse or acquisition that we want to wait for.

        Returns
        -------

        Raises
        ------
        ValueError
            If wait time < 0
        """
        start_time = self.to_pulsar_time(operation.timing)
        wait_time = start_time - self.elapsed_time
        if wait_time > 0:
            self.auto_wait(wait_time)
        elif wait_time < 0:
            raise ValueError(
                f"Invalid timing. Attempting to wait for {wait_time} "
                f"ns before {repr(operation)}. Please note that a wait time of at least"
                f" {Pulsar_sequencer_base.GRID_TIME_ns} ns is required between "
                f"operations.\nAre multiple operations being started at the same time?"
            )

    def wait_till_start_then_play(self, pulse: OpInfo, idx0: int, idx1: int):
        """
        Waits until the start of the pulse, sets the QASMRuntimeSettings and plays the
        pulse.

        Parameters
        ----------
        pulse: OpInfo
            The pulse to play.
        idx0: int
            Index corresponding to the I channel of the pulse in the awg dict.
        idx1
            Index corresponding to the Q channel of the pulse in the awg dict.

        Returns
        -------

        """
        self.wait_till_start_operation(pulse)
        self.update_runtime_settings(pulse)
        self.emit(
            PulsarInstructions.PLAY, idx0, idx1, Pulsar_sequencer_base.GRID_TIME_ns
        )
        self.elapsed_time += Pulsar_sequencer_base.GRID_TIME_ns

    def wait_till_start_then_acquire(self, acquisition: OpInfo, idx0: int, idx1: int):
        """
        Waits until the start of the acquisition, then starts the acquisition.

        Parameters
        ----------
        acquisition: OpInfo
            The pulse to perform.
        idx0: int
            Index corresponding to the I channel of the acquisition weights in the acq
            dict.
        idx1: int
            Index corresponding to the Q channel of the acquisition weights in the acq
            dict.

        Returns
        -------

        """
        self.wait_till_start_operation(acquisition)
        self.emit(
            PulsarInstructions.ACQUIRE, idx0, idx1, Pulsar_sequencer_base.GRID_TIME_ns
        )
        self.elapsed_time += Pulsar_sequencer_base.GRID_TIME_ns

    def update_runtime_settings(self, operation: OpInfo):
        """
        Adds the commands needed to correctly set the QASMRuntimeSettings.

        Parameters
        ----------
        operation: OpInfo
            The pulse to prepare the settings for.

        Returns
        -------

        Notes
        -----
            Currently only the AWG gain is set correctly, as that is the only one
            actually used currently by the backend. Will be expanded in the future.
        """
        if operation.pulse_settings is None:
            raise RuntimeError(f"No real-time settings found for {repr(operation)}.")

        awg_gain_path0 = self._expand_from_normalised_range(
            operation.pulse_settings.awg_gain_0, "awg_gain_0", operation
        )
        awg_gain_path1 = self._expand_from_normalised_range(
            operation.pulse_settings.awg_gain_1, "awg_gain_1", operation
        )
        self.emit(
            PulsarInstructions.SET_AWG_GAIN,
            awg_gain_path0,
            awg_gain_path1,
            comment=f"setting gain for {operation.uuid}",
        )

    @staticmethod
    def _expand_from_normalised_range(
        val: float, param: Optional[str] = None, operation: Optional[OpInfo] = None
    ):
        """
        Takes a the value of a parameter in normalized form (abs(param) <= 1.0), and
        expands it to an integer in the appropriate range required by the sequencer.

        Parameters
        ----------
        val: float
            The value of the parameter to expand.
        param: Optional[str]
            The name of the parameter, to make a possible exception message more
            descriptive.
        operation: Optional[OpInfo]
            The operation this value is expanded for, to make a possible exception
            message more descriptive.

        Returns
        -------
        int
            The expanded value of the parameter.

        Raises
        ------
        ValueError
            Parameter is not in the normalized range.
        """
        immediate_sz = Pulsar_sequencer_base.IMMEDIATE_SZ
        if np.abs(val) > 1.0:
            raise ValueError(
                f"{param} parameter must be in the range "
                f"-1.0 <= param <= 1.0 for {repr(operation)}."
            )
        return int(val * immediate_sz / 2)

    @staticmethod
    def to_pulsar_time(time: float) -> int:
        """
        Takes a float value representing a time in seconds as used by the schedule, and
        returns the integer valued time in nanoseconds that the sequencer uses.

        Parameters
        ----------
        time: float
            The time to convert

        Returns
        -------
        int
            The integer valued nanosecond time
        """
        time_ns = int(np.round(time * 1e9))
        if time_ns % Pulsar_sequencer_base.GRID_TIME_ns != 0:
            raise ValueError(
                f"Pulsar can only work in a timebase of {Pulsar_sequencer_base.GRID_TIME_ns}"
                f" ns. Attempting to use {time_ns} ns."
            )
        return time_ns

    def __str__(self) -> str:
        """
        Returns a string representation of the program. The pulsar expects the program
        to be such a string.

        The conversion to str is done using `columnar`, which expects a list of lists,
        and turns it into a string with rows and columns corresponding to those lists.

        Returns
        -------
        str
            The string representation of the program.
        """
        try:
            return columnar(list(self), headers=None, no_borders=True)
        # running in a sphinx environment can trigger a TableOverFlowError
        except TableOverflowError:
            return columnar(
                list(self), headers=None, no_borders=True, terminal_width=120
            )

    @contextmanager
    def loop(self, register: str, label: str, repetitions: int = 1):
        """
        Defines a context manager that can be used to generate a loop in the QASM
        program.

        Parameters
        ----------
        register: str
            The register to use for the loop iterator.
        label: str
            The label to use for the jump.
        repetitions: int
            The amount of iterations to perform.

        Returns
        -------

        Examples
        --------
        qasm = QASMProgram()
        with qasm.loop(register='R0', label='repeat', repetitions=10):
             qasm.auto_wait(100)

        This adds a loop to the program that loops 10 times over a wait of 100 ns.
        """
        comment = f"iterator for loop with label {label}"

        def gen_start():
            self.emit(PulsarInstructions.MOVE, repetitions, register, comment=comment)
            self.emit(PulsarInstructions.NEW_LINE, label=label)

        try:
            yield gen_start()
        finally:
            self.emit(PulsarInstructions.LOOP, register, f"@{label}")


# ---------- pulsar sequencer classes ----------
class Pulsar_sequencer_base(metaclass=ABCMeta):
    """
    Abstract base class that specify the compilation steps on the sequencer level. The
    distinction between Pulsar QCM and Pulsar QRM is made by the subclasses.
    """

    IMMEDIATE_SZ = pow(2, 16) - 1
    GRID_TIME_ns = 4
    SAMPLING_RATE = 1_000_000_000  # 1GS/s

    def __init__(
        self,
        parent: Pulsar_base,
        name: str,
        portclock: Tuple[str, str],
        modulation_freq: Optional[float] = None,
    ):
        """
        Constructor for the sequencer compiler.

        Parameters
        ----------
        parent: Pulsar_base
            A reference to the parent instrument this sequencer belongs to.
        name: str
            Name of the sequencer. This is supposed to match "seq{index}".
        portclock: Tuple[str, str]
            Tuple that specifies the unique port and clock combination for this
            sequencer. The first value is the port, second is the clock.
        modulation_freq: Optional[float]
            The frequency used for modulation. This can either be passed in the
            constructor, or assigned in a later stage using `assign_frequency`.
        """
        self.parent = parent
        self._name = name
        self.port = portclock[0]
        self.clock = portclock[1]
        self.pulses: List[OpInfo] = list()
        self.acquisitions: List[OpInfo] = list()
        self._settings = SequencerSettings(
            nco_en=False, sync_en=True, modulation_freq=modulation_freq
        )
        self.mixer_corrections = None

    @property
    def portclock(self) -> Tuple[str, str]:
        """
        A tuple containing the unique port and clock combination for this sequencer.

        Returns
        -------
        Tuple[str, str]
            The portclock
        """
        return self.port, self.clock

    @property
    def modulation_freq(self) -> float:
        """
        The frequency used for modulation of the pulses.

        Returns
        -------
        float
            The frequency
        """
        return self._settings.modulation_freq

    @property
    def settings(self) -> SequencerSettings:
        """
        Gives the current settings.

        Returns
        -------
        SequencerSettings
            The settings set to this sequencer.
        """
        return self._settings

    @property
    def name(self):
        """
        The name assigned to this specific sequencer.

        Returns
        -------
        str
            The name.
        """
        return self._name

    @property
    @abstractmethod
    def AWG_OUTPUT_VOLT(self) -> float:
        """
        The output range in volts. This is to be overridden by the subclass to account
        for the differences between a QCM and a QRM.

        Returns
        -------
        float
            The output range in volts.
        """
        pass

    @property
    def has_data(self):
        """
        Whether or not the sequencer has any data (meaning pulses or acquisitions)
        assigned to it or not.

        Returns
        -------
        bool
            Has data been assigned to this sequencer?
        """
        return len(self.acquisitions) > 0 or len(self.pulses) > 0

    def assign_frequency(self, freq: float):
        """
        Assigns a modulation frequency to the sequencer.

        Parameters
        ----------
        freq: float
            The frequency to be used for modulation.

        Returns
        -------

        Raises
        ------
        ValueError
            Attempting to set the modulation frequency to a new value even though a
            value has been previously assigned.
        """
        if self._settings.modulation_freq != freq:
            if self._settings.modulation_freq is not None:
                raise ValueError(
                    f"Attempting to set the modulation frequency of {self._name} of "
                    f"{self.parent.name} to {freq}, while it has previously been set "
                    f"to {self._settings.modulation_freq}."
                )
        self._settings.modulation_freq = freq

    def _generate_awg_dict(self) -> Dict[str, Any]:
        """
        Generates the dictionary that corresponds that contains the awg waveforms in the
        format accepted by the driver.

        Notes
        -----
        The final dictionary to be included in the json that is uploaded to the pulsar
        is of the form:

            program
            awg
                waveform_name
                    data
                    index
            acq
                waveform_name
                    data
                    index

        This function generates the awg dictionary.

        Returns
        -------
        Dict[str, Any]
            The awg dictionary
        """
        waveforms_complex = dict()
        for pulse in self.pulses:
            if pulse.uuid not in waveforms_complex:
                raw_wf_data = _generate_waveform_data(
                    pulse.data, sampling_rate=self.SAMPLING_RATE
                )
                raw_wf_data = self._apply_corrections_to_waveform(
                    raw_wf_data, pulse.duration, pulse.timing
                )
                raw_wf_data, amp_i, amp_q = self._normalize_waveform_data(raw_wf_data)
                pulse.pulse_settings = QASMRuntimeSettings(
                    awg_gain_0=amp_i, awg_gain_1=amp_q
                )
                waveforms_complex[pulse.uuid] = raw_wf_data
        return _generate_waveform_dict(waveforms_complex)

    def _generate_acq_dict(self) -> Dict[str, Any]:
        """
        Generates the dictionary that corresponds that contains the acq weights
        waveforms in the format accepted by the driver.

        Notes
        -----
        The final dictionary to be included in the json that is uploaded to the pulsar
        is of the form:

            program
            awg
                waveform_name
                    data
                    index
            acq
                waveform_name
                    data
                    index

        This function generates the acq dictionary.

        Returns
        -------
        Dict[str, Any]
            The acq dictionary

        Raises
        ------
        NotImplementedError
            Currently, only two one dimensional waveforms can be used as acquisition
            weights. This exception is raised when either or both waveforms contain
            both a real and imaginary part.
        """
        waveforms_complex = dict()
        for acq in self.acquisitions:
            if acq.uuid not in waveforms_complex:
                raw_wf_data_real = _generate_waveform_data(
                    acq.data["waveforms"][0], sampling_rate=self.SAMPLING_RATE
                )
                raw_wf_data_imag = _generate_waveform_data(
                    acq.data["waveforms"][1], sampling_rate=self.SAMPLING_RATE
                )
                if not (
                    np.all(np.isreal(raw_wf_data_real))
                    and np.all(np.isreal(1.0j * raw_wf_data_imag))
                ):  # since next step will break if either is complex
                    raise NotImplementedError(
                        f"Complex weights not implemented. Please use two 1d "
                        f"real-valued weights. Exception was triggered because of "
                        f"{repr(acq)}."
                    )
                waveforms_complex[acq.uuid] = raw_wf_data_real + raw_wf_data_imag
        return _generate_waveform_dict(waveforms_complex)

    def _apply_corrections_to_waveform(
        self, waveform_data: np.ndarray, time_duration: float, t0: Optional[float] = 0
    ) -> np.ndarray:
        """
        Applies all the needed pre-processing on the waveform data. This includes mixer
        corrections and modulation.

        Parameters
        ----------
        waveform_data: np.ndarray
            The data to correct.
        time_duration: float
            Total time is seconds that the waveform is used.
        t0: Optional[float]
            The start time of the pulse/acquisition. This is used for instance to make
            the make the phase change continuously when the start time is not zero.

        Returns
        -------
        np.ndarray
            The waveform data after applying all the transformations.
        """
        t = np.linspace(t0, time_duration + t0, int(time_duration * self.SAMPLING_RATE))
        corrected_wf = modulate_waveform(t, waveform_data, self.modulation_freq)
        if self.mixer_corrections is not None:
            corrected_wf = self.mixer_corrections.correct_skewness(corrected_wf)
        return corrected_wf

    def _normalize_waveform_data(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Rescales the waveform data so that the maximum amplitude is abs(amp) == 1.

        Parameters
        ----------
        data: np.ndarray
            The waveform data to rescale.

        Returns
        -------
        np.ndarray
            The rescaled data.
        float
            The original amplitude of the real part.
        float
            The original amplitude of the imaginary part.
        """
        amp_real, amp_imag = np.max(np.abs(data.real)), np.max(np.abs(data.imag))
        norm_data_r = data.real / amp_real / self.AWG_OUTPUT_VOLT
        norm_data_i = data.imag / amp_imag / self.AWG_OUTPUT_VOLT
        return norm_data_r + 1.0j * norm_data_i, amp_real, amp_imag

    def update_settings(self):
        """
        Updates the sequencer settings to set all parameters that are determined by the
        compiler. Currently, this only changes the offsets based on the mixer
        calibration parameters.

        Returns
        -------

        """
        if self.mixer_corrections is not None:
            self._settings.awg_offset_path_0 = (
                self.mixer_corrections.offset_I / self.AWG_OUTPUT_VOLT
            )
            self._settings.awg_offset_path_1 = (
                self.mixer_corrections.offset_Q / self.AWG_OUTPUT_VOLT
            )

    @staticmethod
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
        str
            Name for the I waveform.
        str
            Name for the Q waveform.
        """
        return f"{str(uuid)}_I", f"{str(uuid)}_Q"

    @classmethod
    def generate_qasm_program(
        cls,
        total_sequence_time: float,
        pulses: Optional[List[OpInfo]] = None,
        awg_dict: Optional[Dict[str, Any]] = None,
        acquisitions: Optional[List[OpInfo]] = None,
        acq_dict: Optional[Dict[str, Any]] = None,
        repetitions: Optional[int] = 1,
    ) -> str:
        """
        Generates a QASM program for a sequencer. Requires the awg and acq dicts to
        already have been generated.

        Example of a program generated by this function:

                    wait_sync     4
                    set_mrk       1
                    move          10,R0         # iterator for loop with label start
            start:
                    wait          4
                    set_awg_gain  22663,10206  # setting gain for 9056793381316377208
                    play          0,1,4
                    wait          176
                    loop          R0,@start
                    set_mrk       0
                    upd_param     4
                    stop

        Parameters
        ----------
        total_sequence_time: float
            Total time the program needs to play for. If the sequencer would be done
            before this time, a wait is added at the end to ensure synchronization.
        pulses: Optional[List[OpInfo]]
            A list containing all the pulses that are to be played.
        awg_dict: Optional[Dict[str, Any]]
            Dictionary containing the pulse waveform data and the index that is assigned
            to the I and Q waveforms, as generated by the `generate_awg_dict` function.
            This is used to extract the relevant indexes when adding a play instruction.
        acquisitions: Optional[List[OpInfo]]
            A list containing all the acquisitions that are to be performed.
        acq_dict: Optional[Dict[str, Any]]
            Dictionary containing the acquisition waveform data and the index that is
            assigned to the I and Q waveforms, as generated by the `generate_acq_dict`
            function. This is used to extract the relevant indexes when adding an
            acquire instruction.
        repetitions: Optional[int]
            Number of times to repeat execution of the schedule.

        Returns
        -------
        str
            The generated QASM program.
        """
        loop_label = "start"
        loop_register = "R0"

        qasm = QASMProgram()
        # program header
        qasm.emit(PulsarInstructions.WAIT_SYNC, cls.GRID_TIME_ns)
        qasm.emit(PulsarInstructions.SET_MARKER, 1)

        # program body
        pulses = list() if pulses is None else pulses
        acquisitions = list() if acquisitions is None else acquisitions
        op_list = pulses + acquisitions
        op_list = sorted(op_list, key=lambda p: (p.timing, p.is_acquisition))

        with qasm.loop(
            label=loop_label, register=loop_register, repetitions=repetitions
        ):
            op_queue = deque(op_list)
            while len(op_queue) > 0:
                operation = op_queue.popleft()
                if operation.is_acquisition:
                    idx0, idx1 = cls.get_indices_from_wf_dict(operation.uuid, acq_dict)
                    qasm.wait_till_start_then_acquire(operation, idx0, idx1)
                else:
                    idx0, idx1 = cls.get_indices_from_wf_dict(operation.uuid, awg_dict)
                    qasm.wait_till_start_then_play(operation, idx0, idx1)

            end_time = qasm.to_pulsar_time(total_sequence_time)
            wait_time = end_time - qasm.elapsed_time
            qasm.auto_wait(wait_time)

        # program footer
        qasm.emit(PulsarInstructions.SET_MARKER, 0)
        qasm.emit(PulsarInstructions.UPDATE_PARAMETERS, cls.GRID_TIME_ns)
        qasm.emit(PulsarInstructions.STOP)
        return str(qasm)

    @staticmethod
    def get_indices_from_wf_dict(uuid: int, wf_dict: Dict[str, Any]) -> Tuple[int, int]:
        """
        Takes a awg_dict or acq_dict and extracts the waveform indices based off of the
        uuid of the pulse/acquisition.

        Parameters
        ----------
        uuid: int
            The unique identifier of the pulse/acquisition.
        wf_dict: Dict[str, Any]
            The awg or acq dict that holds the waveform data and indices.

        Returns
        -------
        int
            Index of the I waveform.
        int
            Index of the Q waveform.
        """
        name_real, name_imag = Pulsar_sequencer_base.generate_waveform_names_from_uuid(
            uuid
        )
        return wf_dict[name_real]["index"], wf_dict[name_imag]["index"]

    @staticmethod
    def _generate_waveforms_and_program_dict(
        program: str,
        awg_dict: Dict[str, Any],
        acq_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates the full waveforms and program dict that is to be uploaded to the
        sequencer from the program string and the awg and acq dicts, by combining them
        and assigning the appropriate keys.

        Parameters
        ----------
        program: str
            The compiled QASM program as a string.
        awg_dict: Dict[str, Any]
            The dictionary containing all the awg data and indices. This is expected to
            be of the form generated by the `generate_awg_dict` method.
        acq_dict: Optional[Dict[str, Any]]
            The dictionary containing all the acq data and indices. This is expected to
            be of the form generated by the `generate_acq_dict` method.

        Returns
        -------
        Dict[str, Any]
            The combined program.
        """
        compiled_dict = dict()
        compiled_dict["program"] = program
        compiled_dict["waveforms"] = dict()
        compiled_dict["waveforms"]["awg"] = awg_dict
        if acq_dict is not None:
            compiled_dict["waveforms"]["acq"] = acq_dict
        return compiled_dict

    @staticmethod
    def _dump_waveforms_and_program_json(
        wf_and_pr_dict: Dict[str, Any], label: Optional[str] = None
    ) -> str:
        """
        Takes a combined waveforms and program dict and dumps it as a json file.

        Parameters
        ----------
        wf_and_pr_dict: Dict[str, Any]
            The dict to dump as a json file.
        label
            A label that is appended to the filename.

        Returns
        -------
        str
            The full absolute path where the json file is stored.
        """
        data_dir = get_datadir()
        folder = os.path.join(data_dir, "schedules")
        os.makedirs(folder, exist_ok=True)

        filename = (
            f"{gen_tuid()}.json" if label is None else f"{gen_tuid()}_{label}.json"
        )
        filename = _sanitize_file_name(filename)
        file_path = os.path.join(folder, filename)

        with open(file_path, "w") as f:
            json.dump(wf_and_pr_dict, f, cls=NumpyJSONEncoder, indent=4)

        return file_path

    def sequencer_compile(self, repetitions: int = 1) -> Optional[Dict[str, Any]]:
        """
        Performs the full sequencer level compilation based on the assigned data and
        settings. If no data is assigned to this sequencer, the compilation is skipped
        and None is returned instead.

        Parameters
        ----------
        repetitions: int
            Number of times execution the schedule is repeated

        Returns
        -------
        Optional[Dict[str, Any]]
            The compiled program. If no data is assigned to this sequencer, the
            compilation is skipped and None is returned instead.
        """
        if not self.has_data:
            return None

        awg_dict = self._generate_awg_dict()
        acq_dict = self._generate_acq_dict() if len(self.acquisitions) > 0 else None

        qasm_program = self.generate_qasm_program(
            self.parent.total_play_time,
            self.pulses,
            awg_dict,
            self.acquisitions,
            acq_dict,
            repetitions=repetitions,
        )

        wf_and_pr_dict = self._generate_waveforms_and_program_dict(
            qasm_program, awg_dict, acq_dict
        )

        json_filename = self._dump_waveforms_and_program_json(
            wf_and_pr_dict, f"{self.port}_{self.clock}"
        )
        self.update_settings()
        settings_dict = self.settings.to_dict()
        return {"seq_fn": json_filename, "settings": settings_dict}


class QCM_sequencer(Pulsar_sequencer_base):
    """
    Subclass of Pulsar_sequencer_base that is meant to implement all the parts that are
    specific to a Pulsar QCM sequencer.

    Attributes
    ----------
    AWG_OUTPUT_VOLT: float
        Voltage range of the awg output paths.
    """

    AWG_OUTPUT_VOLT = 2.5


class QRM_sequencer(Pulsar_sequencer_base):
    """
    Subclass of Pulsar_sequencer_base that is meant to implement all the parts that are
    specific to a Pulsar QRM sequencer.

    Attributes
    ----------
    AWG_OUTPUT_VOLT: float
        Voltage range of the awg output paths.
    """

    AWG_OUTPUT_VOLT = 0.5


# ---------- pulsar instrument classes ----------
class Pulsar_base(InstrumentCompiler, metaclass=ABCMeta):
    """
    `InstrumentCompiler` level compiler object for a pulsar. The class is defined as an
    abstract base class since the distinction between Pulsar QRM and Pulsar QCM specific
    implementations are defined in subclasses.

    Attributes
    ----------
    OUTPUT_TO_SEQ: Dict[str, int]
        Dictionary that maps output names to specific sequencer indices. This
        implementation is temporary and will change when multiplexing is supported by
        the hardware.
    """

    OUTPUT_TO_SEQ = {"complex_output_0": 0, "complex_output_1": 1}

    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
        """
        Constructor function.

        Parameters
        ----------
        name: str
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time: str
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        hw_mapping: Optional[Dict[str, Any]]
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        """
        super().__init__(name, total_play_time, hw_mapping)

        self.portclock_map = self._generate_portclock_to_seq_map()
        self.sequencers = self._construct_sequencers()
        self._settings = self._extract_settings_from_mapping(hw_mapping)

    @property
    @abstractmethod
    def SEQ_TYPE(self):
        pass

    @property
    @abstractmethod
    def MAX_SEQUENCERS(self):
        pass

    @staticmethod
    def _extract_settings_from_mapping(mapping: Dict[str, Any]) -> PulsarSettings:
        """
        TODO move to PulsarSettings

        Takes all the settings defined in the mapping and generates a `PulsarSettings`
        object from it.

        Parameters
        ----------
        mapping: Dict[str, Any]


        Returns
        -------

        """
        ref: str = mapping["ref"]
        return PulsarSettings(ref=ref)

    def assign_modulation_frequency(self, portclock: Tuple[str, str], freq: float):
        """
        Sets the modulation frequency for a certain portclock belonging to this
        instrument.

        Parameters
        ----------
        portclock: Tuple[str, str]
            A tuple with the port as first element and clock as second.
        freq: float
            The modulation frequency to assign to the portclock.

        Returns
        -------

        """
        seq_name = self.portclock_map[portclock]
        seq = self.sequencers[seq_name]
        seq.assign_frequency(freq)

    def _generate_portclock_to_seq_map(self) -> Dict[Tuple[str, str], str]:
        """
        Generates a mapping from portclock tuples to sequencer names.

        Returns
        -------
        Dict[Tuple[str, str], str]
            A dictionary with as key a portclock tuple and as value the name of a
            sequencer.

        Raises
        ------
        NotImplementedError
            When the hardware mapping contains a dictionary, which is assumed to
            correspond to an output channel, that does not have a name defined in
            self.OUTPUT_TO_SEQ.keys(). Likely this will occur when attempting to use
            real outputs (instead of complex), or when the hardware mapping is invalid.
        """
        output_to_seq = self.OUTPUT_TO_SEQ

        mapping = dict()
        for io, data in self.hw_mapping.items():
            if not isinstance(data, dict):
                continue

            port_clocks = find_all_port_clock_combinations(data)
            if len(port_clocks) > 1:
                # TODO consider raising NotImplementedError?
                warnings.warn(
                    f"Multiple ({len(port_clocks)}) sequencers set per output. "
                    f"Only one is allowed currently.",
                    RuntimeWarning,
                )

            if len(port_clocks) > 0:
                port_clock = port_clocks[0]
                try:
                    mapping[port_clock] = f"seq{output_to_seq[io]}"
                except KeyError as e:
                    raise NotImplementedError(
                        f"Attempting to use non-supported output {io}. "
                        f"Supported output types: "
                        f"{(str(t) for t in output_to_seq.keys())}"
                    ) from e
        return mapping

    def _construct_sequencers(self) -> Dict[str, Pulsar_sequencer_base]:
        """
        Constructs `Pulsar_sequencer_base` objects for each port and clock combination
        belonging to this device.

        Returns
        -------
        Dict[str, Pulsar_sequencer_base]
            A dictionary containing the sequencer objects, the keys correspond to the
            names of the sequencers.
        """
        sequencers = dict()
        for io, io_cfg in self.hw_mapping.items():
            if not isinstance(io_cfg, dict):
                continue

            portclock_dicts = find_inner_dicts_containing_key(io_cfg, "port")
            if len(portclock_dicts) > 1:
                warnings.warn(
                    f"{len(portclock_dicts)} sequencers specified for "
                    f"output {io} in mapping. Only one currently supported."
                )
            portclock_dict = portclock_dicts[0]
            portclock = portclock_dict["port"], portclock_dict["clock"]
            freq = (
                None
                if "interm_freq" in portclock_dict
                else portclock_dict["interm_freq"]
            )

            seq_name = f"seq{self.OUTPUT_TO_SEQ[io]}"
            sequencers[seq_name] = self.SEQ_TYPE(self, seq_name, portclock, freq)
            if "mixer_corrections" in io_cfg:
                sequencers[seq_name].mixer_corrections = MixerCorrections.from_dict(
                    io_cfg["mixer_corrections"]
                )

        if len(sequencers.keys()) > self.MAX_SEQUENCERS:
            raise ValueError(
                f"Attempting to construct too many sequencer compilers. "
                f"Maximum allowed for {self.__class__} is {self.MAX_SEQUENCERS}!"
            )

        return sequencers

    def _distribute_data(self):
        """
        Distributes the pulses and acquisitions assigned to this pulsar over the
        different sequencers based on their portclocks.

        Returns
        -------

        """
        for portclock, pulse_data_list in self._pulses.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.pulses = pulse_data_list

        for portclock, acq_data_list in self._acquisitions.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.acquisitions = acq_data_list

    def hardware_compile(self, repetitions: int = 1) -> Optional[Dict[str, Any]]:
        """
        Performs the actual compilation steps for this pulsar, by calling the sequencer
        level compilation functions and combining them into a single dictionary. The
        compiled program has a settings key, and keys for every sequencer.

        Parameters
        ----------
        repetitions: int
            Number of times execution the schedule is repeated

        Returns
        -------
        Optional[Dict[str, Any]]
            The compiled program corresponding to this pulsar. It contains an entry for
            every sequencer and general "settings". If the device is not actually used,
            and an empty program is compiled, None is returned instead.
        """
        self._distribute_data()
        program = dict()
        for seq_name, seq in self.sequencers.items():
            seq_program = seq.sequencer_compile(repetitions=repetitions)
            if seq_program is not None:
                program[seq_name] = seq_program

        if len(program) == 0:
            return None

        program["settings"] = self._settings.to_dict()
        return program


class Pulsar_QCM(Pulsar_base):
    """
    Pulsar QCM specific implementation of the pulsar compiler.

    Attributes
    ----------
    SEQ_TYPE: Pulsar_sequencer_base
        Defines the type of sequencer that this pulsar uses.
    MAX_SEQUENCERS: int
        Maximum amount of sequencers that this pulsar implements.
    """

    SEQ_TYPE = QCM_sequencer
    MAX_SEQUENCERS = 2

    def _distribute_data(self):
        """
        Distributes the pulses and acquisitions assigned to this pulsar over the
        different sequencers based on their portclocks. Overrides the function of the
        same name in the superclass to raise an exception in case it attempts to
        distribute acquisitions, since this is not supported by the pulsar QCM.

        Returns
        -------

        Raises
        ------
        RuntimeError
            Pulsar_QCM._acquisitions is not empty
        """
        if self._acquisitions:
            raise RuntimeError(
                f"Attempting to add acquisitions to {self.__class__} {self.name}, "
                f"which is not supported by hardware."
            )
        super()._distribute_data()

    def add_acquisition(self, port: str, clock: str, acq_info: OpInfo):
        """
        Raises an exception when called since the pulsar QCM does not support
        acquisitions.

        Parameters
        ----------
        port: str
            The port the pulse needs to be sent to.
        clock: str
            The clock for modulation of the pulse. Can be a BasebandClock.
        acq_info: OpInfo
            Data structure containing all the information regarding this specific
            acquisition operation.

        Returns
        -------

        Raises
        ------
        RuntimeError
            Always
        """
        raise RuntimeError(
            f"Pulsar QCM {self.name} does not support acquisitions. "
            f"Attempting to add acquisition {repr(acq_info)} "
            f"on port {port} with clock {clock}."
        )


class Pulsar_QRM(Pulsar_base):
    """
    Pulsar QRM specific implementation of the pulsar compiler.

    Attributes
    ----------
    SEQ_TYPE: Pulsar_sequencer_base
        Defines the type of sequencer that this pulsar uses.
    MAX_SEQUENCERS: int
        Maximum amount of sequencers that this pulsar implements.
    """

    SEQ_TYPE = QRM_sequencer
    MAX_SEQUENCERS = 1


# ---------- Compilation methods ----------
def _assign_frequencies(
    device_compilers: Dict[str, InstrumentCompiler],
    lo_compilers: Dict[str, LocalOscillator],
    hw_mapping: Dict[str, Any],
    portclock_mapping: Dict[Tuple[str, str], str],
    schedule_resources: Dict[str, Any],
):
    """
    Determines the IF or LO frequency based on the clock frequency and assigns it to
    the `InstrumentCompiler`. If the IF is specified the LO frequency is calculated
    based on the constraint `clock_freq = interm_freq + lo_freq`, and vice versa.

    Parameters
    ----------
    device_compilers: Dict[str, InstrumentCompiler]
        A dictionary containing all the `InstrumentCompiler` objects for which IQ
        modulation is used. The keys correspond to the QCoDeS names of the instruments.
    lo_compilers: Dict[str, LocalOscillator]
        A dictionary containing all the `LocalOscillator` objects that are used. The
        keys correspond to the QCoDeS names of the instruments.
    hw_mapping: Dict[str, Any]
        The hardware mapping dictionary describing the whole setup.
    portclock_mapping: Dict[Tuple[str, str], str]
        A dictionary that maps tuples containing a port and a clock to names of
        instruments. The port and clock combinations are unique, but multiple portclocks
        can point to the same instrument.
    schedule_resources: Dict[str, Any]
        The schedule resources containing all the clocks.

    Returns
    -------

    """
    lo_info_dicts = find_inner_dicts_containing_key(hw_mapping, "lo_name")
    for lo_info_dict in lo_info_dicts:
        lo_obj = lo_compilers[lo_info_dict["lo_name"]]
        associated_portclock_dicts = find_inner_dicts_containing_key(
            lo_info_dict, "port"
        )

        lo_freq = None
        if "lo_freq" in lo_info_dict:
            lo_freq = lo_info_dict["lo_freq"]
        if lo_freq is None:
            for portclock_dict in associated_portclock_dicts:
                port, clock = portclock_dict["port"], portclock_dict["clock"]
                interm_freq = portclock_dict["interm_freq"]
                if clock in schedule_resources:
                    cl_freq = schedule_resources[clock]["freq"]

                    dev_name = portclock_mapping[(port, clock)]
                    assign_frequency = getattr(
                        device_compilers[dev_name], "assign_modulation_frequency"
                    )
                    # FIXME getattr should probably be removed in favour of inheritance.
                    #  Though this would require an additional layer in structure
                    assign_frequency((port, clock), interm_freq)
                    lo_obj.assign_frequency(cl_freq - interm_freq)
        else:  # lo_freq given
            lo_obj.assign_frequency(lo_freq)
            for portclock_dict in associated_portclock_dicts:
                port, clock = portclock_dict["port"], portclock_dict["clock"]
                dev_name = portclock_mapping[(port, clock)]
                assign_frequency = getattr(
                    device_compilers[dev_name], "assign_modulation_frequency"
                )
                if clock in schedule_resources:
                    cl_freq = schedule_resources[clock]["freq"]
                    assign_frequency((port, clock), cl_freq - lo_freq)


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
    schedule: Schedule
        The schedule to extract the pulse and acquisition info from.
    device_compilers: Dict[str, Any]
        Dictionary containing InstrumentCompilers as values and their names as keys.
    portclock_mapping: Dict[Tuple[str, str], str]
        A dictionary that maps tuples containing a port and a clock to names of
        instruments. The port and clock combinations are unique, but multiple portclocks
        can point to the same instrument.

    Returns
    -------

    Raises
    ------
    RuntimeError
        This exception is raised then the function encountered an operation that has no
        pulse or acquisition info assigned to it.
    """
    for op_hash, op_data in schedule.operations.items():
        if not op_data.valid_pulse and not op_data.valid_acquisition:
            raise RuntimeError(
                f"Operation {op_hash} is not a valid pulse or acquisition. Please check"
                f" whether the device compilation been performed successfully. "
                f"Operation data: {repr(op_data)}"
            )

        operation_start_time = find_abs_time_from_operation_hash(schedule, op_hash)
        for pulse_data in op_data.data["pulse_info"]:
            if "t0" in pulse_data:
                pulse_start_time = operation_start_time + pulse_data["t0"]
            else:
                pulse_start_time = operation_start_time

            port = pulse_data["port"]
            clock = pulse_data["clock"]
            if port is None:
                continue  # ignore idle pulse

            uuid = make_hash(without(pulse_data, "t0"))
            combined_data = OpInfo(data=pulse_data, timing=pulse_start_time, uuid=uuid)

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
            uuid = make_hash(hashed_dict)

            combined_data = OpInfo(data=acq_data, timing=acq_start_time, uuid=uuid)
            dev = portclock_mapping[(port, clock)]
            device_compilers[dev].add_acquisition(port, clock, acq_info=combined_data)


def _construct_compiler_objects(
    total_play_time: float,
    mapping: Dict[str, Any],
) -> Dict[str, InstrumentCompiler]:
    """
    Traverses the hardware mapping dictionary and instantiates the appropriate
    instrument compiler objects for all the devices that make up the setup. Local
    oscillators are excluded from this step due to them being defined implicitly in the
    hardware mapping.

    Parameters
    ----------
    total_play_time: float
        Total time that it takes to execute a single repetition of the schedule with the
        current hardware setup as defined in the mapping.
    mapping: Dict[str, Any]
        The hardware mapping dictionary.

    Returns
    -------
    Dict[str, InstrumentCompiler]
        A dictionary with an `InstrumentCompiler` as value and the QCoDeS name of the
        instrument the compiler compiles for as key.
    """
    device_compilers = dict()
    for device, dev_cfg in mapping.items():
        if not isinstance(dev_cfg, dict):
            continue
        device_type = dev_cfg["type"]

        device_compiler: Callable = getattr(sys.modules[__name__], device_type)
        device_compilers[device] = device_compiler(
            device,
            total_play_time,
            mapping[device],
        )
    return device_compilers


def hardware_compile(
    schedule: Schedule, hardware_map: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main function driving the compilation. The principle behind the overall compilation
    works as follows:

    For every instrument in the hardware mapping, we instantiate a compiler object. Then
    we assign all the pulses/acquisitions that need to be played by that instrument to
    the compiler, which then compiles for each instrument individually.

    This function then returns all the compiled programs bundled together in a
    dictionary with the QCoDeS name of the instrument as key.

    Parameters
    ----------
    schedule: Schedule
        The schedule to compile. It is assumed the pulse and acquisition info is
        already added to the operation. Otherwise and exception is raised.
    mapping: Dict[str, Any]
        The hardware mapping of the setup.

    Returns
    -------
    Dict[str, Any]
        The compiled program
    """
    total_play_time = _calculate_total_play_time(schedule)

    portclock_map = generate_port_clock_to_device_map(hardware_map)

    device_compilers = _construct_compiler_objects(
        total_play_time=total_play_time,
        mapping=hardware_map,
    )
    _assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=device_compilers,
        portclock_mapping=portclock_map,
    )

    lo_compilers = generate_ext_local_oscillators(total_play_time, hardware_map)
    _assign_frequencies(
        device_compilers,
        lo_compilers,
        hw_mapping=hardware_map,
        portclock_mapping=portclock_map,
        schedule_resources=schedule.resources,
    )
    device_compilers.update(lo_compilers)

    compiled_schedule = dict()
    for name, compiler in device_compilers.items():
        compiled_dev_program = compiler.hardware_compile(
            repetitions=schedule.repetitions
        )

        if compiled_dev_program is not None:
            compiled_schedule[name] = compiled_dev_program

    return compiled_schedule
