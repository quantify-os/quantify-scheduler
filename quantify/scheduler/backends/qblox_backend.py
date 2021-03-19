# -----------------------------------------------------------------------------
# Description:    Compiler backend for Qblox hardware.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from collections import UserDict

if TYPE_CHECKING:
    from quantify.scheduler.types import Schedule


# ---------- functions ----------


def generate_ext_local_oscillators(
    hardware_cfg: Dict[str, Any]
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
                    lo_dict[io_cfg["lo_name"]] = LocalOscillator(io_cfg["lo_name"])

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


def find_devices_needed_in_schedule(
    schedule: Schedule, device_map: Dict[Tuple[str, str], str]
) -> List[str]:
    portclocks = find_all_port_clock_combinations(schedule.operations)

    devices_found = list()
    for pc in portclocks:
        if pc not in devices_found:
            devices_found.append(device_map[pc])

    return devices_found


# ---------- structures ----------


@dataclass
class LocalOscillator:
    name: str
    lo_freq: Optional[int] = None


class Pulsar_QCM:
    pass


class Pulsar_QRM:
    pass


# ---------- Compilation ----------


def _assign_frequencies(
    device_compilers: Dict[str, Any],
    lo_compilers: Dict[str, Any],
    mapping: Dict[str, Any],
):
    pass


def _construct_compiler_objects(device_names: List[str], mapping: Dict[str, Any]):
    device_compilers = dict()
    for device in device_names:
        device_type = mapping[device]["type"]

        device_compiler = getattr(sys.modules[__name__], device_type)
        device_compilers[device] = device_compiler()
    return device_compilers


def hardware_compile(schedule: Schedule, mapping: Dict[str, Any]) -> Dict[str, Any]:
    total_play_time = _calculate_total_play_time(schedule)

    device_map = generate_port_clock_to_device_map(mapping)
    devices_used = find_devices_needed_in_schedule(schedule, device_map)

    device_compilers = _construct_compiler_objects(
        device_names=devices_used, mapping=mapping
    )

    lo_compilers = generate_ext_local_oscillators(mapping)
    _assign_frequencies(device_compilers, lo_compilers, mapping=mapping)
    device_compilers.update(lo_compilers)

    compiled_schedule = dict()
    for name, compiler in device_compilers.items():
        compiled_schedule[name] = compiler.hardware_compile()

    return compiled_schedule
