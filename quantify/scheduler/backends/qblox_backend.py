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


# ---------- functions ----------
def _sanitize_file_name(filename: str):
    invalid = ',<>:"/\\|!?* '
    sanitized_fn = filename
    for char in invalid:
        sanitized_fn = sanitized_fn.replace(char, "_")
    return sanitized_fn


def modulate_waveform(t: np.ndarray, envelope: np.ndarray, freq: float, t0: float = 0):
    modulation = np.exp(1.0j * 2 * np.pi * freq * (t + t0))
    return envelope * modulation


def apply_mixer_skewness_corrections(
    wf: np.ndarray, amplitude_ratio: float, phase_shift: float
):
    def calc_corrected_re(wf, amplitude_ratio: float, phase_shift: float):
        original_amp = np.max(np.abs(wf.real))
        wf_re = wf.real + wf.imag * np.tan(phase_shift)
        wf_re = wf_re / np.max(np.abs(wf_re))
        return wf_re * original_amp * np.sqrt(amplitude_ratio)

    def calc_corrected_imag(wf: np.ndarray, amplitude_ratio: float, phase_shift: float):
        original_amp = np.max(np.abs(wf.imag))
        wf_im = wf.imag / np.cos(phase_shift)
        wf_im = wf_im / np.max(np.abs(wf_im))
        return wf_im * original_amp / np.sqrt(amplitude_ratio)

    corrected_re = calc_corrected_re(wf, amplitude_ratio, np.deg2rad(phase_shift))
    corrected_imag = calc_corrected_imag(wf, amplitude_ratio, np.deg2rad(phase_shift))
    return corrected_re + 1.0j * corrected_imag


def _generate_waveform_data(data_dict: dict, sampling_rate: float) -> np.ndarray:
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
    # TODO more generic with get_inner_dicts_containing_key?
    lo_dict = dict()
    for key, device in hardware_cfg.items():
        if not isinstance(device, dict):  # is not a device
            continue

        for io, io_cfg in device.items():
            if not isinstance(io_cfg, dict):  # is not a in/output
                continue

            if "lo_name" in io_cfg.keys():
                lo_name = io_cfg["lo_name"]
                if lo_name not in lo_dict.keys():
                    lo_obj = LocalOscillator(
                        lo_name,
                        total_play_time,
                    )
                    lo_obj.register_portclocks(
                        *find_all_port_clock_combinations(io_cfg)
                    )
                    lo_dict[lo_name] = lo_obj

                if "lo_freq" in io_cfg.keys():
                    lo_dict[lo_name].assign_frequency(io_cfg["lo_freq"])

    return lo_dict


def _calculate_total_play_time(schedule: Schedule) -> float:
    max_found: float = 0.0
    for time_constraint in schedule.timing_constraints:
        pulse_id = time_constraint["operation_hash"]
        operation = schedule.operations[pulse_id]
        # TODO operation.duration already accounts for t0 right?
        end_time = operation.duration + time_constraint["abs_time"]

        if end_time > max_found:
            max_found = end_time

    return max_found


def find_inner_dicts_containing_key(d: Union[Dict, UserDict], key: Any) -> List[dict]:
    dicts_found = list()
    if key in d.keys():
        dicts_found.append(d)
    for val in d.values():
        if isinstance(val, dict) or isinstance(val, UserDict):
            dicts_found.extend(find_inner_dicts_containing_key(val, key))
        elif isinstance(val, Iterable) and not isinstance(val, str):
            for i_item in val:
                dicts_found.extend(find_inner_dicts_containing_key(i_item, key))
        else:
            continue
    return dicts_found


def find_all_port_clock_combinations(d: Union[Dict, UserDict]) -> List[Tuple[str, str]]:
    port_clocks = list()
    dicts_with_port = find_inner_dicts_containing_key(d, "port")
    for d in dicts_with_port:
        if "port" in d.keys():
            port = d["port"]
            if "clock" not in d.keys():
                raise AttributeError(f"Port {d['port']} missing clock")
            clock = d["clock"]
            port_clocks.append((port, clock))
    return port_clocks


def generate_port_clock_to_device_map(
    mapping: Dict[str, Any]
) -> Dict[Tuple[str, str], str]:

    portclock_map = dict()
    for device_name, device_info in mapping.items():
        if not isinstance(device_info, dict):
            continue

        portclocks = find_all_port_clock_combinations(device_info)

        for portclock in portclocks:
            portclock_map[portclock] = device_name

    return portclock_map


def find_abs_time_from_operation_hash(schedule: Schedule, op_hash: int):
    timing_constraints = schedule.timing_constraints
    for tc in timing_constraints:
        if tc["operation_hash"] == op_hash:
            return tc["abs_time"]


def find_devices_needed_in_schedule(
    schedule: Schedule, device_map: Dict[Tuple[str, str], str]
) -> Set[str]:
    portclocks = find_all_port_clock_combinations(schedule.operations)

    devices_found = set()
    for pc in portclocks:
        if pc not in devices_found:
            devices_found.add(device_map[pc])

    return devices_found


# ---------- classes ----------
class InstrumentCompiler(metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.total_play_time = total_play_time
        self.hw_mapping = hw_mapping
        self._pulses = defaultdict(list)
        self._acquisitions = defaultdict(list)

    def add_pulse(self, port: str, clock: str, pulse_info: OpInfo):
        self._pulses[(port, clock)].append(pulse_info)

    def add_acquisition(self, port: str, clock: str, acq_info: OpInfo):
        self._acquisitions[(port, clock)].append(acq_info)

    @abstractmethod
    def hardware_compile(self) -> Any:
        pass


class LocalOscillator(InstrumentCompiler):
    def __init__(
        self,
        name: str,
        total_play_time: float,
        lo_freq: Optional[int] = None,
    ):
        super().__init__(name, total_play_time)
        self._lo_freq = lo_freq
        self.portclocks = set()

    def register_portclocks(self, *to_register: Tuple[str, str]):
        self.portclocks.update(to_register)

    def assign_frequency(self, freq: float):
        if self._lo_freq is not None:
            if freq != self._lo_freq:
                raise ValueError(
                    f"Attempting to set LO {self.name} to frequency {freq}, "
                    f"while it has previously already been set to {self._lo_freq}!"
                )
        self._lo_freq = freq

    @property
    def frequency(self):
        return self._lo_freq

    def get_related_clocks(self) -> Set[str]:
        return {portclock[1] for portclock in self.portclocks}

    def hardware_compile(self) -> Dict[str, Any]:
        return {"lo_freq": self._lo_freq}


# ---------- data structures ----------
@dataclass
class OpInfo(DataClassJsonMixin):
    uuid: int
    data: dict
    timing: float

    @property
    def duration(self):
        return self.data["duration"]

    @property
    def is_acquisition(self):
        return "acq_index" in self.data.keys()


@dataclass
class PulsarSettings(DataClassJsonMixin):
    ref: str


@dataclass
class SequencerSettings(DataClassJsonMixin):
    nco_en: bool
    sync_en: bool
    modulation_freq: float = None


@dataclass
class MixerCorrections(DataClassJsonMixin):
    amp_ratio: float = 1.0
    phase_error: float = 0.0
    offset_I: float = 0.0
    offset_Q: float = 0.0

    def correct_skewness(self, waveform: np.ndarray) -> np.ndarray:
        return apply_mixer_skewness_corrections(
            waveform, self.amp_ratio, self.phase_error
        )


class PulsarInstructions:
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
    elapsed_time: int = 0

    @staticmethod
    def get_instruction_as_list(
        instruction: str,
        *args: Union[int, str],
        label: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> List[Union[str, int], ...]:
        max_args_amount = 3
        if len(args) > max_args_amount:
            raise SyntaxError(
                f"Too many arguments supplied to `get_instruction_tuple` for instruction {instruction}."
            )
        instr_args = ",".join(str(arg) for arg in args)

        label_str = f"{label}:" if label is not None else ""
        comment_str = f"# {comment}" if comment is not None else ""
        return [label_str, instruction, instr_args, comment_str]

    def emit(self, *args, **kwargs):
        self.append(self.get_instruction_as_list(*args, **kwargs))

    # --- QOL functions -----

    def auto_wait(self, wait_time: int):
        if wait_time < 0:
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
        start_time = self.to_pulsar_time(operation.timing)
        wait_time = start_time - self.elapsed_time
        if wait_time > 0:
            self.auto_wait(wait_time)
        elif wait_time < 0:
            raise ValueError(
                f"Invalid timing. Attempting to wait for {wait_time} "
                f"ns before {repr(operation)}.\n"
                f"Are multiple operations being played at the same time?"
            )

    def wait_till_start_then_play(self, pulse: OpInfo, idx0: int, idx1: int):
        self.wait_till_start_operation(pulse)
        self.emit(
            PulsarInstructions.PLAY, idx0, idx1, Pulsar_sequencer_base.GRID_TIME_ns
        )
        self.elapsed_time += Pulsar_sequencer_base.GRID_TIME_ns

    def wait_till_start_then_acquire(self, acquisition: OpInfo, idx0: int, idx1: int):
        self.wait_till_start_operation(acquisition)
        self.emit(
            PulsarInstructions.ACQUIRE, idx0, idx1, Pulsar_sequencer_base.GRID_TIME_ns
        )
        self.elapsed_time += Pulsar_sequencer_base.GRID_TIME_ns

    def reset_timing(self):
        self.elapsed_time = 0

    @staticmethod
    def to_pulsar_time(time: float) -> int:
        time_ns = int(time * 1e9)
        if time_ns % Pulsar_sequencer_base.GRID_TIME_ns != 0:
            raise ValueError(
                f"Pulsar can only work in a timebase of {Pulsar_sequencer_base.GRID_TIME_ns}"
                f" ns. Attempting to use {time_ns} ns."
            )
        return time_ns

    def __str__(self):
        try:
            return columnar(list(self), headers=None, no_borders=True)
        # running in a sphinx environment can trigger a TableOverFlowError
        except TableOverflowError:
            return columnar(
                list(self), headers=None, no_borders=True, terminal_width=120
            )

    @contextmanager
    def loop(self, register: str, label: str, repetitions: int = 1):
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
        return self.port, self.clock

    @property
    def modulation_freq(self) -> float:
        return self._settings.modulation_freq

    @property
    def settings(self) -> SequencerSettings:
        return self._settings

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def AWG_OUTPUT_VOLT(self):
        pass

    @property
    def has_data(self):
        return len(self.acquisitions) > 0 or len(self.pulses) > 0

    def assign_frequency(self, freq: float):
        if self._settings.modulation_freq != freq:
            if self._settings.modulation_freq is not None:
                raise ValueError(
                    f"Attempting to set modulation frequency of {self._name} of {self.parent.name} to {freq}, "
                    f"while it has previously been set to {self._settings.modulation_freq}."
                )
        self._settings.modulation_freq = freq

    def _generate_awg_dict(self) -> Dict[str, Any]:
        waveforms_complex = dict()
        for pulse in self.pulses:
            if pulse.uuid not in waveforms_complex.keys():
                raw_wf_data = _generate_waveform_data(
                    pulse.data, sampling_rate=self.SAMPLING_RATE
                )
                waveforms_complex[pulse.uuid] = self._apply_corrections_to_waveform(
                    raw_wf_data, pulse.duration, pulse.timing
                )
        return self._generate_waveform_dict(waveforms_complex)

    def _generate_acq_dict(self) -> Dict[str, Any]:
        waveforms_complex = dict()
        for acq in self.acquisitions:
            if acq.uuid not in waveforms_complex.keys():
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
                        f"Complex weights not implemented. "
                        f"Please use two 1d real-valued weights. Exception was "
                        f"triggered because of {repr(acq)}."
                    )
                waveforms_complex[acq.uuid] = raw_wf_data_real + raw_wf_data_imag
        return self._generate_waveform_dict(waveforms_complex)

    def _apply_corrections_to_waveform(
        self, waveform_data, time_duration: float, t0: float = 0
    ):
        t = np.arange(t0, time_duration + t0, 1 / self.SAMPLING_RATE)
        corrected_wf = modulate_waveform(t, waveform_data, self.modulation_freq)
        if self.mixer_corrections is not None:
            corrected_wf = self.mixer_corrections.correct_skewness(corrected_wf)
        return corrected_wf

    def _normalize_waveform_data(self, data):
        return data / self.AWG_OUTPUT_VOLT

    @staticmethod
    def _generate_waveform_names_from_uuid(uuid):
        return f"{uuid}_I", f"{uuid}_Q"

    @staticmethod
    def _generate_waveform_dict(waveforms_complex) -> dict:
        wf_dict = dict()
        for idx, (uuid, complex_data) in enumerate(waveforms_complex.items()):
            name_i, name_q = Pulsar_sequencer_base._generate_waveform_names_from_uuid(
                uuid
            )
            to_add = {
                name_i: {"data": complex_data.real, "index": 2 * idx},
                name_q: {"data": complex_data.imag, "index": 2 * idx + 1},
            }
            wf_dict.update(to_add)
        return wf_dict

    @classmethod
    def generate_qasm_program(
        cls,
        total_sequence_time: float,
        pulses: Optional[List[OpInfo]] = None,
        awg_dict: Optional[Dict[str, Any]] = None,
        acquisitions: Optional[List[OpInfo]] = None,
        acq_dict: Optional[Dict[str, Any]] = None,
        repetitions: int = 1,
    ) -> str:
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
    def get_indices_from_wf_dict(uuid: int, wf_dict: Dict[str, Any]):
        name_real, name_imag = Pulsar_sequencer_base._generate_waveform_names_from_uuid(
            uuid
        )
        return wf_dict[name_real]["index"], wf_dict[name_imag]["index"]

    @staticmethod
    def _generate_waveforms_and_program_dict(
        program: str, awg_dict: Dict[str, Any], acq_dict: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        compiled_dict = dict()
        compiled_dict["program"] = program
        compiled_dict["awg"] = awg_dict
        if acq_dict is not None:
            compiled_dict["acq"] = acq_dict
        return compiled_dict

    @staticmethod
    def _dump_waveforms_and_program_json(
        wf_and_pr_dict: Dict[str, Any], label: Optional[str] = None
    ) -> str:
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

    def sequencer_compile(self) -> Optional[Dict[str, Any]]:
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
        )

        wf_and_pr_dict = self._generate_waveforms_and_program_dict(
            qasm_program, awg_dict, acq_dict
        )

        json_filename = self._dump_waveforms_and_program_json(
            wf_and_pr_dict, f"{self.port}_{self.clock}"
        )
        settings_dict = self.settings.to_dict()

        return {"seq_fn": json_filename, "settings": settings_dict}


class QCM_sequencer(Pulsar_sequencer_base):
    AWG_OUTPUT_VOLT = 2.5


class QRM_sequencer(Pulsar_sequencer_base):
    AWG_OUTPUT_VOLT = 0.5


# ---------- pulsar instrument classes ----------
class Pulsar_base(InstrumentCompiler, metaclass=ABCMeta):

    OUTPUT_TO_SEQ = {"complex_output_0": 0, "complex_output_1": 1}

    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
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

    @property
    def portclocks_with_data(self):
        portclocks_used = set()
        portclocks_used.update(self._pulses.keys())
        portclocks_used.update(self._acquisitions.keys())
        return portclocks_used

    @staticmethod
    def _extract_settings_from_mapping(mapping: Dict[str, Any]) -> PulsarSettings:
        ref: str = mapping["ref"]
        return PulsarSettings(ref=ref)

    def assign_modulation_frequency(self, portclock: Tuple[str, str], freq: float):
        seq_name = self.portclock_map[portclock]
        seq = self.sequencers[seq_name]
        seq.assign_frequency(freq)

    def _generate_portclock_to_seq_map(self) -> Dict[Tuple[str, str], str]:
        output_to_seq = self.OUTPUT_TO_SEQ

        mapping = dict()
        for io, data in self.hw_mapping.items():
            if not isinstance(data, dict):
                continue

            port_clocks = find_all_port_clock_combinations(data)
            if len(port_clocks) > 1:
                warnings.warn(
                    f"Multiple ({len(port_clocks)}) sequencers set per output. "
                    f"Only one is allowed.",
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
                if "interm_freq" in portclock_dict.keys()
                else portclock_dict["interm_freq"]
            )

            seq_name = f"seq{self.OUTPUT_TO_SEQ[io]}"
            sequencers[seq_name] = self.SEQ_TYPE(self, seq_name, portclock, freq)
            if "mixer_corrections" in io_cfg.keys():
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
        for portclock, pulse_data_list in self._pulses.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.pulses = pulse_data_list

        for portclock, acq_data_list in self._acquisitions.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.acquisitions = acq_data_list

    def hardware_compile(self) -> Optional[Dict[str, Any]]:
        self._distribute_data()
        program = dict()
        for seq_name, seq in self.sequencers.items():
            seq_program = seq.sequencer_compile()
            if seq_program is not None:
                program[seq_name] = seq_program

        if len(program) == 0:
            return None

        program["settings"] = self._settings.to_dict()
        return program


class Pulsar_QCM(Pulsar_base):
    SEQ_TYPE = QCM_sequencer
    MAX_SEQUENCERS = 2

    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
        super().__init__(name, total_play_time, hw_mapping)

    def _distribute_data(self):
        if self._acquisitions:
            raise ValueError(
                f"Attempting to add acquisitions to {self.__class__} {self.name}, "
                f"which is not supported by hardware."
            )
        super()._distribute_data()

    def add_acquisition(self, port: str, clock: str, acq_info: OpInfo):
        raise RuntimeError(
            f"Pulsar QCM {self.name} does not support acquisitions. "
            f"Attempting to add acquisition {repr(acq_info)} "
            f"on port {port} with clock {clock}."
        )


class Pulsar_QRM(Pulsar_base):
    SEQ_TYPE = QRM_sequencer
    MAX_SEQUENCERS = 1

    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
        super().__init__(name, total_play_time, hw_mapping)


# ---------- Compilation methods ----------
def _assign_frequencies(
    device_compilers: Dict[str, InstrumentCompiler],
    lo_compilers: Dict[str, LocalOscillator],
    hw_mapping: Dict[str, Any],
    portclock_mapping: Dict[Tuple[str, str], str],
    schedule_resources: Dict[str, Any],
):
    lo_info_dicts = find_inner_dicts_containing_key(hw_mapping, "lo_name")
    for lo_info_dict in lo_info_dicts:
        lo_obj = lo_compilers[lo_info_dict["lo_name"]]
        associated_portclock_dicts = find_inner_dicts_containing_key(
            lo_info_dict, "port"
        )

        lo_freq = None
        if "lo_freq" in lo_info_dict.keys():
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
                    )  # FIXME getattr should probably be removed in favour of inheritance.
                    # This would require an additional layer in structure
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
                if clock in schedule_resources.keys():
                    cl_freq = schedule_resources[clock]["freq"]
                    assign_frequency((port, clock), cl_freq - lo_freq)


def _assign_pulse_and_acq_info_to_devices(
    schedule: Schedule,
    device_compilers: Dict[str, Any],
    portclock_mapping: Dict[Tuple[str, str], str],
):
    for op_hash, op_data in schedule.operations.items():
        operation_start_time = find_abs_time_from_operation_hash(schedule, op_hash)
        for pulse_data in op_data.data["pulse_info"]:
            if "t0" in pulse_data:
                pulse_start_time = operation_start_time + pulse_data["t0"]
            else:
                pulse_start_time = operation_start_time

            port = pulse_data["port"]
            clock = pulse_data["clock"]
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

            hashed_dict = without(acq_data, ["t0", "waveforms"])
            hashed_dict["waveforms"] = list()
            for acq in acq_data["waveforms"]:
                hashed_dict["waveforms"].append(without(acq, ["t0"]))
            uuid = make_hash(hashed_dict)

            combined_data = OpInfo(data=acq_data, timing=acq_start_time, uuid=uuid)
            dev = portclock_mapping[(port, clock)]
            device_compilers[dev].add_acquisition(port, clock, acq_info=combined_data)


def _construct_compiler_objects(
    total_play_time,
    mapping: Dict[str, Any],
) -> Dict[str, InstrumentCompiler]:
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


def hardware_compile(schedule: Schedule, mapping: Dict[str, Any]) -> Dict[str, Any]:
    total_play_time = _calculate_total_play_time(schedule)

    portclock_map = generate_port_clock_to_device_map(mapping)

    device_compilers = _construct_compiler_objects(
        total_play_time=total_play_time,
        mapping=mapping,
    )
    _assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=device_compilers,
        portclock_mapping=portclock_map,
    )

    lo_compilers = generate_ext_local_oscillators(total_play_time, mapping)
    _assign_frequencies(
        device_compilers,
        lo_compilers,
        hw_mapping=mapping,
        portclock_mapping=portclock_map,
        schedule_resources=schedule.resources,
    )
    device_compilers.update(lo_compilers)

    compiled_schedule = dict()
    for name, compiler in device_compilers.items():
        compiled_dev_program = compiler.hardware_compile()

        if compiled_dev_program is not None:
            compiled_schedule[name] = compiled_dev_program

    return compiled_schedule
