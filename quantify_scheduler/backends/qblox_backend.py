# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import Field, model_validator

from quantify_scheduler.backends.corrections import (
    apply_distortion_corrections,
    determine_relative_latency_corrections,
)
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.qblox import compiler_container, constants
from quantify_scheduler.backends.qblox.helpers import (
    _generate_legacy_hardware_config,
    _generate_new_style_hardware_compilation_config,
    _preprocess_legacy_hardware_config,
    assign_pulse_and_acq_info_to_devices,
    find_channel_names,
)
from quantify_scheduler.backends.qblox.operations import long_square_pulse
from quantify_scheduler.backends.qblox.operations.stitched_pulse import StitchedPulse
from quantify_scheduler.backends.types.common import (
    Connectivity,
    HardwareCompilationConfig,
    HardwareDescription,
    HardwareOptions,
)
from quantify_scheduler.backends.types.qblox import (
    PulsarQCMDescription,
    PulsarQRMDescription,
    QbloxHardwareDescription,
    QbloxHardwareOptions,
    QCMDescription,
    QCMRFDescription,
    QRMDescription,
    QRMRFDescription,
)
from quantify_scheduler.helpers.collections import find_inner_dicts_containing_key
from quantify_scheduler.schedules.schedule import (
    CompiledSchedule,
    Schedulable,
    Schedule,
)


def _get_square_pulses_to_replace(schedule: Schedule) -> dict[str, list[int]]:
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
    square_pulse_idx_map : dict[str, list[int]]
        The mapping from ``operation_id`` to ``"pulse_info"`` indices to be replaced.
    """
    square_pulse_idx_map: dict[str, list[int]] = {}
    for ref, operation in schedule.operations.items():
        square_pulse_idx_to_replace: list[int] = []
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
    schedule: Schedule, pulse_idx_map: dict[str, list[int]]
) -> Schedule:
    """
    Replace any square pulses indicated by pulse_idx_map by a ``long_square_pulse``.

    Parameters
    ----------
    schedule : Schedule
        A :class:`~quantify_scheduler.schedules.schedule.Schedule`, possibly containing
        long square pulses.
    pulse_idx_map : dict[str, list[int]]
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
                schedule.operations[ref] = StitchedPulse(
                    name=operation.data["name"], pulse_info=operation.data["pulse_info"]
                )
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
    config: CompilationConfig | dict[str, Any] | None = None,
    # config can be dict to support (deprecated) calling with hardware config
    # as positional argument.
    *,  # Support for (deprecated) calling with hardware_cfg as keyword argument:
    hardware_cfg: Optional[dict[str, Any]] = None,
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
        hardware_cfg = _generate_legacy_hardware_config(
            schedule=schedule, compilation_config=config
        )
    elif config is not None:
        # Support for (deprecated) calling with hardware_cfg as positional argument.
        hardware_cfg = _preprocess_legacy_hardware_config(config)

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

    validate_non_overlapping_stitched_pulse(schedule)

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        schedule, hardware_cfg
    )

    assign_pulse_and_acq_info_to_devices(
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


def find_qblox_instruments(
    hardware_config: Dict[str, Any], instrument_type: str
) -> Dict[str, Any]:
    """Find all inner dictionaries representing a qblox instrument of the given type."""
    instruments = {}
    for key, value in hardware_config.items():
        try:
            if value["instrument_type"] == instrument_type:
                instruments[key] = value
        except (KeyError, TypeError):
            pass

    return instruments


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

    @model_validator(mode="after")
    def _validate_connectivity_channel_names(  # noqa: PLR0912
        self,
    ) -> QbloxHardwareCompilationConfig:
        all_channel_names = []

        # Fetch channel_names from connectivity datastructure
        if isinstance(self.connectivity, Connectivity):
            for edge in self.connectivity.graph.edges:
                for node in edge:
                    try:
                        cluster_name, module_name, channel_name = node.split(".")
                        if (
                            self.hardware_description[cluster_name].instrument_type
                            == "Cluster"
                        ):
                            slot_idx = int(re.search(r"\d+$", module_name).group())
                            module_type = (
                                self.hardware_description[cluster_name]
                                .modules[slot_idx]
                                .instrument_type
                            )
                            all_channel_names.append(
                                (
                                    f"{cluster_name}.{module_name}",
                                    module_type,
                                    channel_name,
                                )
                            )
                    except ValueError:
                        pass

        # Fetch channel_names from legacy hardware config
        else:
            for cluster_name, cluster_config in find_qblox_instruments(
                hardware_config=self.connectivity, instrument_type="Cluster"
            ).items():
                for module_type in ["QCM", "QRM", "QCM_RF", "QRM_RF"]:
                    for module_name, module in find_qblox_instruments(
                        hardware_config=cluster_config, instrument_type=module_type
                    ).items():
                        for channel_name in find_channel_names(module):
                            all_channel_names.append(
                                (cluster_name, module_type, channel_name)
                            )

            for pulsar_type in ["Pulsar_QRM", "Pulsar_QCM"]:
                for pulsar_name, pulsar_config in find_qblox_instruments(
                    hardware_config=self.connectivity, instrument_type=pulsar_type
                ).items():
                    for channel_name in find_channel_names(pulsar_config):
                        all_channel_names.append(
                            (pulsar_name, pulsar_type, channel_name)
                        )

        # Validate channel_names
        instrument_type_to_description = {
            description.get_instrument_type(): description
            for description in [
                QCMDescription,
                QRMDescription,
                QCMRFDescription,
                QRMRFDescription,
                PulsarQCMDescription,
                PulsarQRMDescription,
            ]
        }

        for instrument_name, instrument_type, channel_name in all_channel_names:
            valid_channel_names = instrument_type_to_description[
                instrument_type
            ].get_valid_channels()
            if channel_name not in valid_channel_names:
                raise ValueError(
                    f"Invalid connectivity: '{channel_name}' of "
                    f"{instrument_name} ({instrument_type}) "
                    f"is not a valid name of an input/output."
                    f"\n\nSupported names for {instrument_type}:\n"
                    f"{valid_channel_names}"
                )

        return self

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
            data = _generate_new_style_hardware_compilation_config(data)

        return data


def validate_non_overlapping_stitched_pulse(schedule: Schedule, **_: Any) -> None:
    """
    Raise an error when pulses overlap, if at least one contains a voltage offset.

    Since voltage offsets are sometimes used to construct pulses (see e.g.
    :func:`.long_square_pulse`), overlapping these with regular pulses in time on the
    same port-clock can lead to undefined behaviour.

    Note that for each schedulable, all pulse info entries with the same port and clock
    count as one pulse for that port and clock. This is because schedulables, starting
    before another schedulable has finished, could affect the waveforms or offsets in
    the remaining time of that other schedulable.

    Parameters
    ----------
    schedule : Schedule
        A :class:`~quantify_scheduler.schedules.schedule.Schedule`, possibly containing
        long square pulses.

    Returns
    -------
    schedule : Schedule
        A :class:`~quantify_scheduler.schedules.schedule.Schedule`, possibly containing
        long square pulses.

    Raises
    ------
    RuntimeError
        If the schedule contains overlapping pulses (containing voltage offsets) on the
        same port and clock.
    """
    schedulables = sorted(schedule.schedulables.values(), key=lambda x: x["abs_time"])
    # Iterate through the schedulables in chronological order, and keep track of the
    # latest end time of schedulables containing pulses.
    # When a schedulable contains a voltage offset, check if it starts before the end of
    # a previously found pulse, else look ahead for any schedulables with pulses
    # starting before this schedulable ends.
    last_pulse_end = -np.inf
    last_pulse_sched = None
    for i, schedulable in enumerate(schedulables):
        if _has_voltage_offset(schedulable, schedule):
            # Add 1e-10 for possible floating point errors (strictly >).
            if last_pulse_end > schedulable["abs_time"] + 1e-10:
                _raise_if_pulses_overlap_on_same_port_clock(
                    schedulable, last_pulse_sched, schedule  # type: ignore
                )
            elif other := _exists_pulse_starting_before_current_end(
                schedulables, i, schedule
            ):
                _raise_if_pulses_overlap_on_same_port_clock(
                    schedulable, other, schedule
                )
        if _has_pulse(schedulable, schedule):
            last_pulse_end = _operation_end(schedulable, schedule)
            last_pulse_sched = schedulable


def _exists_pulse_starting_before_current_end(
    sorted_schedulables: list[Schedulable], current_idx: int, schedule: Schedule
) -> Schedulable | Literal[False]:
    current_end = _operation_end(sorted_schedulables[current_idx], schedule)
    for schedulable in sorted_schedulables[current_idx + 1 :]:
        # Schedulable starting at the exact time a previous one ends does not count as
        # overlapping. Subtract 1e-10 for possible floating point errors.
        if schedulable["abs_time"] >= current_end - 1e-10:
            return False
        if _has_pulse(schedulable, schedule) or _has_voltage_offset(
            schedulable, schedule
        ):
            return schedulable
    return False


def _raise_if_pulses_overlap_on_same_port_clock(
    schble_a: Schedulable, schble_b: Schedulable, schedule: Schedule
) -> None:
    """
    Raise an error if any pulse operations overlap on the same port-clock.

    A pulse here means a waveform or a voltage offset.
    """
    pulse_start_ends_per_port_a = _get_pulse_start_ends(schble_a, schedule)
    pulse_start_ends_per_port_b = _get_pulse_start_ends(schble_b, schedule)
    common_ports = set(pulse_start_ends_per_port_a.keys()) & set(
        pulse_start_ends_per_port_b.keys()
    )
    for port_clock in common_ports:
        start_a, end_a = pulse_start_ends_per_port_a[port_clock]
        start_b, end_b = pulse_start_ends_per_port_b[port_clock]
        if (
            start_b < start_a < end_b
            or start_b < end_a < end_b
            or start_a < start_b < end_a
            or start_a < end_b < end_a
        ):
            op_a = schedule.operations[schble_a["operation_id"]]
            op_b = schedule.operations[schble_b["operation_id"]]
            raise RuntimeError(
                f"{op_a} at t={schble_a['abs_time']} and {op_b} at t="
                f"{schble_b['abs_time']} contain pulses with voltage offsets that "
                "overlap in time on the same port and clock. This leads to undefined "
                "behaviour."
            )


def _get_pulse_start_ends(
    schedulable: Schedulable, schedule: Schedule
) -> dict[str, tuple[float, float]]:
    pulse_start_ends_per_port: dict[str, tuple[float, float]] = defaultdict(
        lambda: (np.inf, -np.inf)
    )
    for pulse_info in schedule.operations[schedulable["operation_id"]]["pulse_info"]:
        if (
            pulse_info.get("wf_func") is None
            and pulse_info.get("offset_path_I") is None
        ):
            continue
        prev_start, prev_end = pulse_start_ends_per_port[
            f"{pulse_info['port']}_{pulse_info['clock']}"
        ]
        new_start = schedulable["abs_time"] + pulse_info["t0"]
        new_end = new_start + pulse_info["duration"]
        if new_start < prev_start:
            prev_start = new_start
        if new_end > prev_end:
            prev_end = new_end
        pulse_start_ends_per_port[f"{pulse_info['port']}_{pulse_info['clock']}"] = (
            prev_start,
            prev_end,
        )
    return pulse_start_ends_per_port


def _has_voltage_offset(schedulable: Schedulable, schedule: Schedule) -> bool:
    return schedule.operations[schedulable["operation_id"]].has_voltage_offset


def _has_pulse(schedulable: Schedulable, schedule: Schedule) -> bool:
    return schedule.operations[schedulable["operation_id"]].valid_pulse


def _operation_end(schedulable: Schedulable, schedule: Schedule) -> float:
    return (
        schedulable["abs_time"]
        + schedule.operations[schedulable["operation_id"]].duration
    )
