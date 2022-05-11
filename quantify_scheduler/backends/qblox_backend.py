# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

from typing import Any, Dict

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends.qblox import compiler_container, helpers


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

    converted_hw_config = helpers.convert_hw_config_to_portclock_configs_spec(
        hardware_cfg
    )
    old_hw_config_spec = hardware_cfg != converted_hw_config

    if old_hw_config_spec:
        ValueError(helpers._pre_portclock_configs_err())

    container = compiler_container.CompilerContainer.from_mapping(
        schedule, hardware_cfg
    )

    helpers._assign_pulse_and_acq_info_to_devices(
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
