# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends.corrections import (
    apply_distortion_corrections,
    determine_relative_latency_corrections,
)
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    LatencyCorrections,
)
from quantify_scheduler.backends.qblox import compiler_container, helpers


def hardware_compile(
    schedule: Schedule,
    config: CompilationConfig | Dict[str, Any] | None = None,
    # config can be Dict to support (deprecated) calling with hardware config
    # as positional argument.
    *,  # Support for (deprecated) calling with hardware_cfg as keyword argument:
    hardware_cfg: Optional[Dict[str, Any]] = None,
) -> CompiledSchedule:
    """
    Generate qblox hardware instructions for executing the schedule.

    The principle behind the overall compilation is as follows:

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
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`, of
        which only the :attr:`.CompilationConfig.connectivity`
        is currently extracted in this compilation step.
    hardware_cfg
        (deprecated) The hardware configuration of the setup. Pass a full compilation
        config instead using `config` argument.

    Returns
    -------
    :
        The compiled schedule.

    Raises
    ------
    ValueError
        When both `config` and `hardware_cfg` are supplied.
    """
    if not ((config is not None) ^ (hardware_cfg is not None)):
        raise ValueError(
            f"Qblox `{hardware_compile.__name__}` was called with {config=} and "
            f"{hardware_cfg=}. Please make sure this function is called with "
            f"one of the two (CompilationConfig recommended)."
        )
    if not isinstance(config, CompilationConfig):
        warnings.warn(
            f"Qblox `{hardware_compile.__name__}` will require a full "
            f"CompilationConfig as input as of quantify-scheduler >= 0.15.0",
            FutureWarning,
        )
    if isinstance(config, CompilationConfig):
        # Extract the hardware config from the CompilationConfig
        hardware_cfg = config.connectivity
        if config.hardware_options.latency_corrections is not None:
            hardware_cfg[
                "latency_corrections"
            ] = config.hardware_options.latency_corrections.corrections
    elif config is not None:
        # Support for (deprecated) calling with hardware_cfg as positional argument.
        hardware_cfg = config

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
            "The provided hardware config adheres to a specification that is deprecated"
            ". See https://quantify-quantify-scheduler.readthedocs-hosted.com/en/0.8.0/"
            "tutorials/qblox/recent.html",
            FutureWarning,
        )
        hardware_cfg = converted_hw_config

    if "latency_corrections" in hardware_cfg.keys():
        # Important: currently only used to validate the input, should also be
        # used for storing the latency corrections
        # (see also https://gitlab.com/groups/quantify-os/-/epics/1)
        LatencyCorrections(corrections=converted_hw_config["latency_corrections"])

        # Subtract minimum latency to allow for negative latency corrections
        hardware_cfg["latency_corrections"] = determine_relative_latency_corrections(
            hardware_cfg
        )

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
