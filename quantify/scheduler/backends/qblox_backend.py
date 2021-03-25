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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple, Union, Type

import numpy as np
from dataclasses_json import DataClassJsonMixin
from quantify.data.handling import get_datadir, gen_tuid
from quantify.utilities.general import make_hash, without, import_func_from_string

if TYPE_CHECKING:
    from quantify.scheduler.types import Schedule


# ---------- functions ----------
def _sanitize_file_name(filename: str):
    invalid = '<>:"/\\|?* '

    sanitized_fn = filename
    for char in invalid:
        sanitized_fn = sanitized_fn.replace(char, "_")

    return sanitized_fn


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

    lo_dict = dict()
    for key, device in hardware_cfg.items():
        if not isinstance(device, dict):  # is not a device
            continue

        for io, io_cfg in device.items():
            if not isinstance(io_cfg, dict):  # is not a in/output
                continue

            if "lo_name" in io_cfg.keys():
                if io_cfg["lo_name"] not in lo_dict.keys():
                    lo_dict[io_cfg["lo_name"]] = LocalOscillator(
                        io_cfg["lo_name"], total_play_time
                    )

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


def find_all_port_clock_combinations(d: Union[Dict, UserDict]) -> List[Tuple[str, str]]:
    port_clocks = list()
    if "port" in d.keys():
        if "clock" not in d.keys():
            raise AttributeError(f"Port {d['port']} missing clock")

        port, clock = d["port"], d["clock"]
        port_clocks.append((port, clock))

    for val in d.values():
        if isinstance(val, dict) or isinstance(val, UserDict):
            nested_port_clocks = find_all_port_clock_combinations(val)

            for item in nested_port_clocks:
                if item not in port_clocks:
                    port_clocks.append(item)
        elif isinstance(val, list):
            for l_item in val:
                nested_port_clocks = find_all_port_clock_combinations(l_item)

                for item in nested_port_clocks:
                    if item not in port_clocks:
                        port_clocks.append(item)

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
) -> List[str]:
    portclocks = find_all_port_clock_combinations(schedule.operations)

    devices_found = list()
    for pc in portclocks:
        if pc not in devices_found:
            devices_found.append(device_map[pc])

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

    @abstractmethod
    def hardware_compile(self) -> Any:
        pass


class LocalOscillator(InstrumentCompiler):
    def __init__(
        self, name: str, total_play_time: float, lo_freq: Optional[int] = None
    ):
        super().__init__(name, total_play_time)
        if lo_freq:
            self.lo_freq = lo_freq

    def hardware_compile(self) -> Dict[str, Any]:
        return {"interm_freq": self.lo_freq}


# ---------- data structures ----------
@dataclass
class OpInfo(DataClassJsonMixin):
    uuid: int
    data: dict
    timing: float

    @property
    def duration(self):
        return self.data["duration"]


@dataclass
class SequencerSettings(DataClassJsonMixin):
    nco_en: bool
    sync_en: bool


class PulsarInstructions:
    # Control
    ILLEGAL = "illegal"
    STOP = "stop"
    NOP = "nop"
    NEW_LINE = ""
    # Jumps
    JUMP = "jmp"
    LOOP = "loop"
    # Arithmetic
    MOVE = "move"
    # Real-time pipeline instructions
    SET_MARKER = "set_mrk"
    PLAY = "play"
    ACQUIRE = "acquire"
    WAIT = "wait"
    WAIT_SYNC = "wait_sync"


class QASMProgram(list):
    @staticmethod
    def get_instruction_as_list(
        instruction: str,
        *args: int,
        label: Optional[str] = "",
        comment: Optional[str] = None,
    ) -> List[Union[str, int], ...]:
        max_args_amount = 3
        if len(args) > max_args_amount:
            raise SyntaxError(
                f"Too many arguments supplied to `get_instruction_tuple` for instruction {instruction}"
            )
        instr_args = [""] * max_args_amount
        instr_args[: len(args)] = args

        comment_str = f"#{comment}" if comment is not None else ""
        return [label, instruction, *instr_args, comment_str]

    def emit(self, *args, **kwargs):
        self.append(self.get_instruction_as_list(*args, **kwargs))

    def auto_wait(self, wait_time: int):
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

    @staticmethod
    def to_pulsar_time(time: float) -> int:
        time_ns = int(time * 1e9)
        if time_ns % Pulsar_sequencer_base.GRID_TIME_ns != 0:
            raise ValueError(
                f"Pulsar can only work in a timebase of {Pulsar_sequencer_base.GRID_TIME_ns}."
            )
        return time_ns

    def __str__(self):
        try:
            return columnar(list(self), no_borders=True)
        # running in a sphinx environment can trigger a TableOverFlowError
        except TableOverflowError:
            return columnar(list(self), no_borders=True, terminal_width=120)


# ---------- pulsar sequencer classes ----------
class Pulsar_sequencer_base(metaclass=ABCMeta):

    IMMEDIATE_SZ = pow(2, 16) - 1
    GRID_TIME_ns = 4
    SAMPLING_RATE = 1_000_000_000  # 1GS/s

    def __init__(self, parent: Pulsar_base, name: str, portclock: Tuple[str, str]):
        self.parent = parent
        self.name = name
        self.port = portclock[0]
        self.clock = portclock[1]
        self.pulses: List[OpInfo] = list()
        self.acquisitions: List[OpInfo] = list()
        self.settings = SequencerSettings(nco_en=False, sync_en=True)

    @property
    def portclock(self) -> Tuple[str, str]:
        return self.port, self.clock

    @property
    @abstractmethod
    def AWG_OUTPUT_VOLT(self):
        pass

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
            nested_dict = {
                name_i: {"data": complex_data.real, "index": 2 * idx},
                name_q: {"data": complex_data.imag, "index": 2 * idx + 1},
            }
            wf_dict.update(nested_dict)
        return wf_dict

    def _generate_awg_dict(self) -> Dict[str, Any]:
        waveforms_complex = dict()
        for pulse in self.pulses:
            if pulse.uuid not in waveforms_complex.keys():
                raw_wf_data = _generate_waveform_data(
                    pulse.data, sampling_rate=self.SAMPLING_RATE
                )
                waveforms_complex[pulse.uuid] = self._apply_corrections_to_waveform(
                    raw_wf_data
                )
        return self._generate_waveform_dict(waveforms_complex)

    @classmethod
    def generate_qasm_program(
        cls,
        total_sequence_time: float,
        pulses: List[OpInfo],
        awg_dict: Dict[str, Any],
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
        qasm.emit(PulsarInstructions.MOVE, loop_register, repetitions)
        qasm.emit(PulsarInstructions.NEW_LINE, label=loop_label)

        # program body
        pulse_queue = deque(sorted(pulses, key=lambda p: p.timing))
        current_time: int = 0
        while len(pulse_queue) > 0:
            p = pulse_queue.popleft()
            pulse_time = qasm.to_pulsar_time(p.timing)
            if pulse_time > current_time:
                wait_time = pulse_time - current_time
                qasm.auto_wait(wait_time)
                current_time += wait_time
            idx0, idx1 = cls.get_indices_from_wf_dict(p.uuid, awg_dict)
            qasm.emit(PulsarInstructions.PLAY, idx0, idx1)
            current_time += qasm.to_pulsar_time(p.duration)

        # program footer
        wait_time = qasm.to_pulsar_time(total_sequence_time) - current_time
        if wait_time < 0:
            raise ValueError(f"Invalid timing. Attempting to wait for {wait_time} ns.")
        qasm.auto_wait(wait_time)
        qasm.emit(PulsarInstructions.LOOP, loop_register, loop_label)
        qasm.emit(PulsarInstructions.STOP)
        return str(qasm)

    @staticmethod
    def get_indices_from_wf_dict(uuid: int, wf_dict: Dict[str, Any]):
        name_real, name_imag = Pulsar_sequencer_base._generate_waveform_names_from_uuid(
            uuid
        )
        return wf_dict[name_real]["index"], wf_dict[name_imag]["index"]

    def _apply_corrections_to_waveform(
        self, waveform_data
    ):  # mixer phase and modulation mainly
        # TODO
        return waveform_data

    def _normalize_waveform_data(self, data):
        return data / self.AWG_OUTPUT_VOLT

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

    def sequencer_compile(self) -> Dict[str, Any]:
        awg_dict = self._generate_awg_dict()
        qasm_program = self.generate_qasm_program(
            self.parent.total_play_time, self.pulses, awg_dict
        )

        wf_and_pr_dict = self._generate_waveforms_and_program_dict(
            qasm_program, awg_dict, {}
        )

        json_filename = self._dump_waveforms_and_program_json(
            wf_and_pr_dict, f"{self.port}_{self.clock}"
        )
        settings_dict = self.settings.to_dict()

        return {
            "seq_fn": json_filename,
            "settings": settings_dict,
            "debug": f"Compilation for {self.__class__} named {self.name} successful. "
            f"Port, clock: {self.port, self.clock}",
        }


class QCM_sequencer(Pulsar_sequencer_base):
    AWG_OUTPUT_VOLT = 2.5


class QRM_sequencer(Pulsar_sequencer_base):
    AWG_OUTPUT_VOLT = 0.5


# ---------- pulsar instrument classes ----------
class Pulsar_base(InstrumentCompiler, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
        max_sequencers: int,
        seq_type: Type[Pulsar_sequencer_base],
    ):
        super().__init__(name, total_play_time, hw_mapping)
        self.max_sequencers = max_sequencers
        self.seq_type = seq_type

        self.port_clock_map = self._generate_portclock_to_seq_map()

    def _generate_portclock_to_seq_map(self) -> Dict[Tuple[str, str], str]:
        output_to_seq = {"complex_output_0": 0, "complex_output_1": 1}

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
                    raise KeyError(
                        f"Attempting to use non-supported output {io}."
                    ) from e
        return mapping

    def _construct_sequencers(self):
        portclocks_used = set(self._pulses.keys())
        portclocks_used = portclocks_used.union(set(self._acquisitions.keys()))

        sequencers = list()
        for i, portclock in enumerate(portclocks_used):
            if i >= self.max_sequencers:
                raise ValueError(
                    f"Attempting to construct too many sequencer compilers. "
                    f"Maximum allowed for {self.__class__} is {self.max_sequencers}!"
                )
            seq = self.seq_type(self, self.port_clock_map[portclock], portclock)
            sequencers.append(seq)
        return sequencers

    @abstractmethod
    def _distribute_data(self, sequencers: List[Pulsar_sequencer_base]):
        pass

    def hardware_compile(self) -> Dict[str, Any]:
        sequencers = self._construct_sequencers()
        self._distribute_data(sequencers)

        program = dict()
        for seq in sequencers:
            program[seq.name] = seq.sequencer_compile()
        return program


class Pulsar_QCM(Pulsar_base):
    SEQ_TYPE = QCM_sequencer

    def __init__(self, name: str, total_play_time: float, hw_mapping: Dict[str, Any]):
        super().__init__(
            name, total_play_time, hw_mapping, max_sequencers=2, seq_type=self.SEQ_TYPE
        )

    def _distribute_data(self, sequencers: List[QCM_sequencer]):
        if self._acquisitions:
            raise ValueError(
                f"Attempting to add acquisitions to {self.__class__} {self.name}, "
                f"which is not supported by hardware."
            )

        for portclock, pulse_data_list in self._pulses.items():
            for seq in sequencers:
                if seq.portclock == portclock:
                    seq.pulses = pulse_data_list


class Pulsar_QRM(Pulsar_base):
    SEQ_TYPE = QRM_sequencer

    def __init__(self, name: str, total_play_time: float, hw_mapping: Dict[str, Any]):
        super().__init__(
            name, total_play_time, hw_mapping, max_sequencers=1, seq_type=self.SEQ_TYPE
        )

    def _distribute_data(self, sequencers: List[QRM_sequencer]):
        pass


# ---------- Compilation methods ----------
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


def _assign_frequencies(
    device_compilers: Dict[str, Any],
    lo_compilers: Dict[str, Any],
    mapping: Dict[str, Any],
    schedule_resources: UserDict,
):
    for lo in lo_compilers.values():
        lo.lo_freq = 0


def _construct_compiler_objects(
    device_names: List[str], total_play_time, mapping: Dict[str, Any]
) -> Dict[str, InstrumentCompiler]:
    device_compilers = dict()
    for device in device_names:
        device_type = mapping[device]["type"]

        device_compiler = getattr(sys.modules[__name__], device_type)
        device_compilers[device] = device_compiler(
            device, total_play_time, mapping[device]
        )
    return device_compilers


def hardware_compile(schedule: Schedule, mapping: Dict[str, Any]) -> Dict[str, Any]:
    total_play_time = _calculate_total_play_time(schedule)

    portclock_map = generate_port_clock_to_device_map(mapping)
    devices_used = find_devices_needed_in_schedule(schedule, portclock_map)

    device_compilers = _construct_compiler_objects(
        device_names=devices_used, total_play_time=total_play_time, mapping=mapping
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
        mapping=mapping,
        schedule_resources=schedule.resources,
    )
    device_compilers.update(lo_compilers)

    compiled_schedule = dict()
    for name, compiler in device_compilers.items():
        compiled_schedule[name] = compiler.hardware_compile()

    return compiled_schedule
