# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends.qblox import compiler_container, helpers
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.helpers.collections import without
from quantify_scheduler.operations.pulse_library import WindowOperation


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

        portclocks = helpers.find_all_port_clock_combinations(device_info)

        for portclock in portclocks:
            portclock_map[portclock] = device_name

    return portclock_map


def hardware_compile(
    schedule: Schedule, hardware_cfg: Dict[str, Any]
) -> CompiledSchedule:
    """
    Main function driving the compilation. The principle behind the overall compilation
    works as follows:

    For every instrument in the hardware configuration, we instantiate a compiler
    object. Then we assign all the pulses/acquisitions that need to be played by that
    instrument to the compiler, which then compiles for each instrument individually.

    This function then returns all the compiled programs bundled together in a
    dictionary with the QCoDeS name of the instrument as key.

    Parameters
    ----------
    schedule
        The schedule to compile. It is assumed the pulse and acquisition info is
        already added to the operation. Otherwise and exception is raised.
    hardware_cfg
        The hardware configuration of the setup.

    Returns
    -------
    :
        The compiled schedule.
    """

    container = compiler_container.CompilerContainer.from_mapping(
        schedule, hardware_cfg
    )

    helpers.assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        mapping=hardware_cfg,
        device_compilers=container.instrument_compilers,
    )

    container.prepare()
    compiled_instructions = container.compile(repetitions=schedule.repetitions)
    # add the compiled instructions to the schedule data structure
    schedule["compiled_instructions"] = compiled_instructions
    # Mark the schedule as a compiled schedule
    return CompiledSchedule(schedule)
