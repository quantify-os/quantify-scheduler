# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import Field, model_validator

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends.corrections import (
    apply_distortion_corrections,
    determine_relative_latency_corrections,
)
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.qblox import compiler_container, constants, helpers
from quantify_scheduler.backends.qblox.operations import long_square_pulse
from quantify_scheduler.backends.qblox.operations.stitched_pulse import StitchedPulse
from quantify_scheduler.backends.types.common import (
    HardwareCompilationConfig,
    HardwareDescription,
    HardwareOptions,
)
from quantify_scheduler.backends.types.qblox import (
    QbloxHardwareDescription,
    QbloxHardwareOptions,
)
from quantify_scheduler.helpers.collections import find_inner_dicts_containing_key


def _get_square_pulses_to_replace(schedule: Schedule) -> Dict[str, List[int]]:
    """
    Generate a dict referring to long square pulses to replace in the schedule.

    This function generates a mapping (dict) from the keys in the
    :meth:`~quantify_scheduler.schedules.schedule.ScheduleBase.operations` dict to a
    list of indices, which refer to entries in the `"pulse_info"` list that describe a
    square pulse.

    Parameters
    ----------
    schedule : Schedule
        A :class:`~quantify_scheduler.schedules.schedule.Schedule`, possibly containing
        long square pulses.

    Returns
    -------
    square_pulse_idx_map : Dict[str, List[int]]
        The mapping from ``operation_id`` to ``"pulse_info"`` indices to be replaced.
    """
    square_pulse_idx_map: Dict[str, List[int]] = {}
    for ref, operation in schedule.operations.items():
        square_pulse_idx_to_replace: List[int] = []
        for i, pulse_info in enumerate(operation.data["pulse_info"]):
            if (
                pulse_info.get("wf_func", "") == "quantify_scheduler.waveforms.square"
                and pulse_info["duration"] >= constants.PULSE_STITCHING_DURATION
            ):
                square_pulse_idx_to_replace.append(i)
        if square_pulse_idx_to_replace:
            square_pulse_idx_map[ref] = square_pulse_idx_to_replace
    return square_pulse_idx_map


def _replace_long_square_pulses(
    schedule: Schedule, pulse_idx_map: Dict[str, List[int]]
) -> Schedule:
    """
    Replace any square pulses indicated by pulse_idx_map by a ``long_square_pulse``.

    Parameters
    ----------
    schedule : Schedule
        A :class:`~quantify_scheduler.schedules.schedule.Schedule`, possibly containing
        long square pulses.
    pulse_idx_map : Dict[str, List[int]]
        A mapping from the keys in the
        :meth:`~quantify_scheduler.schedules.schedule.ScheduleBase.operations` dict to
        a list of indices, which refer to entries in the `"pulse_info"` list that
        describe a square pulse.

    Returns
    -------
    Schedule
        The schedule with square pulses longer than
        :class:`~quantify_scheduler.backends.qblox.constants.PULSE_STITCHING_DURATION`
        replaced by
        :func:`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_square_pulse`. If no
        replacements were done, this is the original unmodified schedule.
    """
    for ref, square_pulse_idx_to_replace in pulse_idx_map.items():
        # Below, we replace entries in-place in a list that we loop over. The
        # indices here are the entries to be replaced. We sort such that popping
        # from the end returns indices in descending order.
        square_pulse_idx_to_replace.sort()

        operation = schedule.operations[ref]

        while square_pulse_idx_to_replace:
            idx = square_pulse_idx_to_replace.pop()
            pulse_info = operation.data["pulse_info"].pop(idx)
            new_square_pulse = long_square_pulse(
                amp=pulse_info["amp"],
                duration=pulse_info["duration"],
                port=pulse_info["port"],
                clock=pulse_info["clock"],
                t0=pulse_info["t0"],
                reference_magnitude=pulse_info["reference_magnitude"],
            )
            operation.add_pulse(new_square_pulse)

            # To not break __str__ in some cases, the operation type must be a
            # StitchedPulse.
            if idx == 0 and not (operation.valid_acquisition or operation.valid_gate):
                schedule.operations[ref] = StitchedPulse(operation.data["pulse_info"])
    return schedule


def compile_long_square_pulses_to_awg_offsets(schedule: Schedule, **_: Any) -> Schedule:
    """
    Replace square pulses in the schedule with long square pulses.

    Introspects operations in the schedule to find square pulses with a duration
    longer than
    :class:`~quantify_scheduler.backends.qblox.constants.PULSE_STITCHING_DURATION`. Any
    of these square pulses are converted to
    :func:`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_square_pulse`, which
    consist of AWG voltage offsets.

    If any operations are to be replaced, a deepcopy will be made of the schedule, which
    is returned by this function. Otherwise the original unmodified schedule will be
    returned.

    Parameters
    ----------
    schedule : Schedule
        A :class:`~quantify_scheduler.schedules.schedule.Schedule`, possibly containing
        long square pulses.

    Returns
    -------
    schedule : Schedule
        The schedule with square pulses longer than
        :class:`~quantify_scheduler.backends.qblox.constants.PULSE_STITCHING_DURATION`
        replaced by
        :func:`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_square_pulse`. If no
        replacements were done, this is the original unmodified schedule.
    """
    pulse_idx_map = _get_square_pulses_to_replace(schedule)
    if pulse_idx_map:
        schedule = _replace_long_square_pulses(schedule, pulse_idx_map)
    return schedule


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
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`.
    hardware_cfg
        (deprecated) The hardware configuration of the setup. Pass a full compilation
        config instead using ``config`` argument.

    Returns
    -------
    :
        The compiled schedule.

    Raises
    ------
    ValueError
        When both ``config`` and ``hardware_cfg`` are supplied.
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
            f"CompilationConfig as input as of quantify-scheduler >= 0.19.0",
            FutureWarning,
        )
        debug_mode = False
    else:
        debug_mode = config.debug_mode

    if isinstance(config, CompilationConfig):
        # Extract the hardware config from the CompilationConfig
        hardware_cfg = helpers.generate_hardware_config(
            schedule=schedule, compilation_config=config
        )
    elif config is not None:
        # Support for (deprecated) calling with hardware_cfg as positional argument.
        hardware_cfg = helpers._preprocess_legacy_hardware_config(config)

    # To be removed when hardware config validation is implemented. See
    # https://gitlab.com/groups/quantify-os/-/epics/1
    for faulty_key in ["thresholded_acq_rotation", "thresholded_acq_threshold"]:
        if find_inner_dicts_containing_key(hardware_cfg, faulty_key):
            raise KeyError(
                f"'{faulty_key}' found in hardware configuration. Please configure "
                "thresholded acquisition via the device elements. See documentation "
                "for `ThresholdedAcquisition` for more information."
            )

    if "latency_corrections" in hardware_cfg.keys():
        # Important: currently only used to validate the input, should also be
        # used for storing the latency corrections
        # (see also https://gitlab.com/groups/quantify-os/-/epics/1)
        HardwareOptions(latency_corrections=hardware_cfg["latency_corrections"])

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

    compiled_instructions = container.compile(
        debug_mode=debug_mode, repetitions=schedule.repetitions
    )
    # Create compiled instructions key if not already present. This can happen if this
    # compilation function is called directly instead of through a `QuantifyCompiler`.
    if "compiled_instructions" not in schedule:
        schedule["compiled_instructions"] = {}
    # add the compiled instructions to the schedule data structure
    schedule["compiled_instructions"].update(compiled_instructions)
    # Mark the schedule as a compiled schedule
    return CompiledSchedule(schedule)


class QbloxHardwareCompilationConfig(HardwareCompilationConfig):
    """
    Datastructure containing the information needed to compile to the Qblox backend.

    This information is structured in the same way as in the generic
    :class:`~quantify_scheduler.backends.types.common.HardwareCompilationConfig`, but
    contains fields for hardware-specific settings.
    """

    config_type: Type[QbloxHardwareCompilationConfig] = Field(
        default="quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        validate_default=True,
    )
    """
    A reference to the
    :class:`~quantify_scheduler.backends.types.common.HardwareCompilationConfig`
    DataStructure for the Qblox backend.
    """
    hardware_description: Dict[
        str, Union[QbloxHardwareDescription, HardwareDescription]
    ]
    """Description of the instruments in the physical setup."""
    hardware_options: QbloxHardwareOptions
    """
    Options that are used in compiling the instructions for the hardware, such as
    :class:`~quantify_scheduler.backends.types.common.LatencyCorrection` or
    :class:`~quantify_scheduler.backends.types.qblox.SequencerOptions`.
    """
    compilation_passes: List[SimpleNodeConfig] = [
        SimpleNodeConfig(
            name="compile_long_square_pulses_to_awg_offsets",
            compilation_func=compile_long_square_pulses_to_awg_offsets,
        ),
        SimpleNodeConfig(
            name="qblox_hardware_compile", compilation_func=hardware_compile
        ),
    ]
    """
    The list of compilation nodes that should be called in succession to compile a 
    schedule to instructions for the Qblox hardware.
    """

    @model_validator(mode="before")
    @classmethod
    def from_old_style_hardware_config(
        cls: type[QbloxHardwareCompilationConfig], data: Any
    ) -> Any:
        """Convert old style hardware config dict to new style before validation."""
        if (
            isinstance(data, dict)
            and data.get("backend")
            == "quantify_scheduler.backends.qblox_backend.hardware_compile"
        ):
            # Input is an old style Qblox hardware config dict
            data = helpers._generate_new_style_hardware_compilation_config(data)

        return data
