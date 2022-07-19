# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

import warnings
from typing import Any, Dict

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends.corrections import apply_distortion_corrections
from quantify_scheduler.backends.qblox import compiler_container, helpers

from quantify_scheduler.backends.graph_compilation import (
    CompilationNode,
)

from quantify_scheduler.backends.device_compile import DeviceCompile


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
        already added to the operation. Otherwise an exception is raised.
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

    # Directly comparing dictionaries that contain numpy arrays raises a
    # ValueError. It is however sufficient to compare all the keys of nested
    # dictionaries.
    def _get_flattened_keys_from_dictionary(
        dictionary, parent_key: str = "", sep: str = "."
    ):
        flattened_keys = set()
        for key, value in dictionary.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                flattened_keys = flattened_keys.union(
                    _get_flattened_keys_from_dictionary(value, new_key, sep=sep)
                )
            else:
                flattened_keys = flattened_keys.union({new_key})
        return flattened_keys

    hw_config_keys = _get_flattened_keys_from_dictionary(hardware_cfg)
    converted_hw_config_keys = _get_flattened_keys_from_dictionary(converted_hw_config)

    if hw_config_keys != converted_hw_config_keys:
        warnings.warn(
            "The provided hardware config adheres to a specification "
            "that is now deprecated. Please learn about the new "
            "Qblox hardware config specification at:\n"
            "https://gitlab.com/quantify-os/quantify-scheduler/-/wikis/"
            "Qblox-backend:-Dynamic-Sequencer-Allocation \n"
            "You may upgrade an old config to the new specification using the "
            "'quantify_scheduler.backends.qblox.helpers."
            "convert_hw_config_to_portclock_configs_spec' function.",
            DeprecationWarning,
        )
        hardware_cfg = converted_hw_config

    schedule = apply_distortion_corrections(schedule, hardware_cfg)

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        schedule, hardware_cfg
    )

    helpers.assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        hardware_cfg=hardware_cfg,
        device_compilers=container.instrument_compilers,
    )

    container.prepare()
    compiled_instructions = container.compile(repetitions=schedule.repetitions)
    # add the compiled instructions to the schedule data structure
    schedule["compiled_instructions"] = compiled_instructions
    # Mark the schedule as a compiled schedule
    return CompiledSchedule(schedule)


qblox_hardware_compile = CompilationNode(
    name="qblox_hardware_compile",
    compilation_func=hardware_compile,
    config_key="hardware_cfg",
    config_validator=None,
)


class QbloxBackend(DeviceCompile):
    """
    Backend for compiling a schedule from the Quantum-circuit layer to
    instructions suitable for Qblox hardware.

    This backend extends the :class:`.backends.DeviceCompile` backend.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

        # find the last node of the graph we are extending and remove the output node
        old_output_nodes = list(self.predecessors("output"))
        self.remove_node("output")

        # add the new node and connect it to all previous output nodes.
        self.add_node(qblox_hardware_compile)
        for old_output_node in old_output_nodes:
            self.add_edge(old_output_node, qblox_hardware_compile)
        self.add_edge(qblox_hardware_compile, "output")
