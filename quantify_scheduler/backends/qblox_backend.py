# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

import itertools
import re
import warnings
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
from pydantic import Field, model_validator

from quantify_scheduler.backends.corrections import (
    apply_software_distortion_corrections,
    determine_relative_latency_corrections,
)
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.qblox import compiler_container, constants
from quantify_scheduler.backends.qblox.crosstalk_compensation import (
    crosstalk_compensation,
)
from quantify_scheduler.backends.qblox.enums import ChannelMode
from quantify_scheduler.backends.qblox.exceptions import NcoOperationTimingError
from quantify_scheduler.backends.qblox.helpers import (
    _generate_new_style_hardware_compilation_config,
    assign_pulse_and_acq_info_to_devices,
    to_grid_time,
)
from quantify_scheduler.backends.qblox.operations import long_square_pulse
from quantify_scheduler.backends.qblox.operations.pulse_library import (
    LatchReset,
)
from quantify_scheduler.backends.qblox.stack_pulses import stack_pulses
from quantify_scheduler.backends.types.common import (
    Connectivity,
    HardwareCompilationConfig,
    HardwareDescription,
    LatencyCorrection,
    LocalOscillatorDescription,
    ModulationFrequencies,
    SoftwareDistortionCorrection,
)
from quantify_scheduler.backends.types.qblox import (
    ClusterDescription,
    ClusterModuleDescription,
    ComplexChannelDescription,
    ComplexInputGain,
    DigitalChannelDescription,
    DigitizationThresholds,
    QbloxHardwareDescription,
    QbloxHardwareDistortionCorrection,
    QbloxHardwareOptions,
    QbloxMixerCorrections,
    RealChannelDescription,
    RealInputGain,
    SequencerOptions,
)
from quantify_scheduler.helpers.schedule import _extract_port_clocks_used
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    ControlFlowOperation,
    LoopOperation,
)
from quantify_scheduler.operations.pulse_library import (
    ResetClockPhase,
    SetClockFrequency,
    ShiftClockPhase,
)
from quantify_scheduler.schedules.schedule import (
    CompiledSchedule,
    Schedulable,
    Schedule,
    ScheduleBase,
)
from quantify_scheduler.structure.model import DataStructure

if TYPE_CHECKING:
    from quantify_scheduler.operations.operation import Operation


def _replace_long_square_pulses_recursively(
    operation: Operation | Schedule,
) -> Operation | None:
    """
    Generate a dict referring to long square pulses to replace in the schedule.

    This function generates a mapping (dict) from the keys in the
    :meth:`~quantify_scheduler.schedules.schedule.ScheduleBase.operations` dict to a
    list of indices, which refer to entries in the `"pulse_info"` list that describe a
    square pulse.

    Parameters
    ----------
    operation
        An operation, possibly containing long square pulses.

    """
    if isinstance(operation, ScheduleBase):
        for inner_operation_id, inner_operation in operation.operations.items():
            replacing_operation = _replace_long_square_pulses_recursively(inner_operation)
            if replacing_operation:
                operation.operations[inner_operation_id] = replacing_operation
        return None
    elif isinstance(operation, ControlFlowOperation):
        replacing_operation = _replace_long_square_pulses_recursively(operation.body)
        if replacing_operation:
            operation.body = replacing_operation
        return None
    else:
        square_pulse_idx_to_replace: list[int] = []
        for i, pulse_info in enumerate(operation.data["pulse_info"]):
            if (
                pulse_info.get("wf_func", "") == "quantify_scheduler.waveforms.square"
                and pulse_info["duration"] >= constants.PULSE_STITCHING_DURATION
            ):
                square_pulse_idx_to_replace.append(i)
        replacing_operation = _replace_long_square_pulses(operation, square_pulse_idx_to_replace)
        if replacing_operation:
            return replacing_operation
        return None


def _replace_long_square_pulses(
    operation: Operation,
    square_pulse_idx_to_replace: list[int],
) -> Operation | None:
    """
    Replace any square pulses indicated by pulse_idx_map by a ``long_square_pulse``.

    Parameters
    ----------
    operation
        Operation to be replaced.
    square_pulse_idx_to_replace
        A list of indices in the pulse info to be replaced.

    Returns
    -------
    operation
        The operation to be replaced. If returns ``None``, the operation does
        not need to be replaced in the schedule or control flow.
    square_pulse_idx_to_replace
        The pulse indices that need to be replaced in the operation.

    """
    square_pulse_idx_to_replace.sort()

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
    return None


def _all_conditional_acqs_and_control_flows_and_latch_reset(
    operation: Operation | Schedule,
    time_offset: float,
    accumulator: list[tuple[float, Operation]],
) -> None:
    if isinstance(operation, ScheduleBase):
        for schedulable in operation.schedulables.values():
            abs_time = schedulable.data["abs_time"]
            inner_operation = operation.operations[schedulable["operation_id"]]
            _all_conditional_acqs_and_control_flows_and_latch_reset(
                inner_operation,
                time_offset + abs_time,
                accumulator,
            )
    elif isinstance(operation, LoopOperation):
        assert operation.body.duration is not None
        for i in range(operation.data["control_flow_info"]["repetitions"]):
            _all_conditional_acqs_and_control_flows_and_latch_reset(
                operation.body,
                time_offset + i * operation.body.duration,
                accumulator,
            )
    elif isinstance(operation, (ConditionalOperation, LatchReset)) or (
        operation.valid_acquisition and operation.is_conditional_acquisition
    ):
        accumulator.append((time_offset, operation))


@dataclass
class OperationTimingInfo:
    """Timing information for an Operation."""

    start: float
    """start time of the operation."""
    end: float
    """end time of the operation."""

    @classmethod
    def from_operation_and_schedulable(
        cls, operation: Operation, schedulable: Schedulable  # noqa: ANN102
    ) -> OperationTimingInfo:
        """Create an ``OperationTimingInfo`` from an operation and a schedulable."""
        start: float = schedulable.data["abs_time"]
        duration: float = operation.duration
        end: float = start + duration

        return cls(
            start=start,
            end=end,
        )

    def overlaps_with(self, operation_timing_info: OperationTimingInfo) -> bool:
        """Check if this operation timing info overlaps with another."""
        return self.start <= operation_timing_info.end and operation_timing_info.start <= self.end


@dataclass
class ConditionalAddress:
    """Container for conditional address data."""

    portclocks: set[tuple[str, str]]
    address: int


def _set_conditional_address_map(
    operation: Operation | Schedule,
    conditional_address_map: defaultdict[str, ConditionalAddress],
) -> None:
    if isinstance(operation, ScheduleBase):
        schedulables = list(operation.schedulables.values())
        for schedulable in schedulables:
            inner_operation = operation.operations[schedulable["operation_id"]]
            _set_conditional_address_map(
                operation=inner_operation,
                conditional_address_map=conditional_address_map,
            )
    elif isinstance(operation, ConditionalOperation):
        # Store `feedback_trigger_address` in the pulse that corresponds
        # to a conditional control flow.
        # Note, we do not allow recursive conditional calls, so no need to
        # go recursively into conditionals.
        control_flow_info: dict = operation.data["control_flow_info"]
        feedback_trigger_label: str = control_flow_info["feedback_trigger_label"]
        control_flow_info["feedback_trigger_address"] = conditional_address_map[
            feedback_trigger_label
        ].address
        conditional_address_map[feedback_trigger_label].portclocks |= _extract_port_clocks_used(
            operation.body
        )
    elif isinstance(operation, ControlFlowOperation):
        _set_conditional_address_map(
            operation=operation.body,
            conditional_address_map=conditional_address_map,
        )
    elif operation.valid_acquisition and operation.is_conditional_acquisition:
        # Store `feedback_trigger_address` in the correct acquisition, so that it can
        # be passed to the correct Sequencer via ``SequencerSettings``.
        acq_info = operation["acquisition_info"]
        for info in acq_info:
            if (feedback_trigger_label := info.get("feedback_trigger_label")) is not None:
                info["feedback_trigger_address"] = conditional_address_map[
                    feedback_trigger_label
                ].address


def _insert_latch_reset(
    operation: Operation | Schedule,
    abs_time_relative_to_schedule: float,
    schedule: Schedule,
    conditional_address_map: defaultdict[str, ConditionalAddress],
) -> None:
    if isinstance(operation, ScheduleBase):
        schedulables = list(operation.schedulables.values())
        for schedulable in schedulables:
            abs_time = schedulable.data["abs_time"]
            inner_operation = operation.operations[schedulable["operation_id"]]
            _insert_latch_reset(
                operation=inner_operation,
                abs_time_relative_to_schedule=abs_time,
                schedule=operation,
                conditional_address_map=conditional_address_map,
            )
    elif isinstance(operation, LoopOperation):
        _insert_latch_reset(
            operation=operation.body,
            abs_time_relative_to_schedule=abs_time_relative_to_schedule,
            schedule=schedule,
            conditional_address_map=conditional_address_map,
        )
    elif operation.valid_acquisition and operation.is_conditional_acquisition:
        acq_info = operation["acquisition_info"]
        for info in acq_info:
            if (feedback_trigger_label := info.get("feedback_trigger_label")) is not None:
                at = abs_time_relative_to_schedule + info["t0"] + constants.MAX_MIN_INSTRUCTION_WAIT
                for portclock in conditional_address_map[feedback_trigger_label].portclocks:
                    schedulable = schedule.add(LatchReset(portclock=portclock))
                    schedulable.data["abs_time"] = at


def compile_conditional_playback(  # noqa: D417
    schedule: Schedule, config: DataStructure | dict  # noqa: ARG001
) -> Schedule:
    """
    Compiles conditional playback.

    This compiler pass will determine the mapping between trigger labels and
    trigger addresses that the hardware will use. The feedback trigger address
    is stored under the key ``feedback_trigger_address`` in ``pulse_info`` and
    in ``acquisition_info`` of the corresponding operation.

    A valid conditional playback consists of two parts: (1) a conditional
    acquisition or measure, and (2) a conditional control flow. The first should
    always be followed by the second, else an error is raised. A conditional
    acquisition sends a trigger after the acquisition ends and if the
    acquisition crosses a certain threshold. Each sequencer that is subscribed
    to this trigger will increase their *latch* counters by one. To ensure the
    latch counters contain either 0 or 1 trigger counts, a
    :class:`~quantify_scheduler.backends.qblox.operations.pulse_library.LatchReset`
    operation is inserted right after the start of a conditional acquisition, on
    all sequencers. If this is not possible (e.g. due to concurring operations),
    a :class:`RuntimeError` is raised.

    Parameters
    ----------
    schedule :
        The schedule to compile.

    Returns
    -------
    Schedule
        The returned schedule is a reference to the original ``schedule``, but
        updated.

    Raises
    ------
    RuntimeError
        - If a conditional acquisitions/measures is not followed by a
          conditional control flow.
        - If a conditional control flow is not preceded by a conditional
          acquisition/measure.
        - If the compilation pass is unable to insert
          :class:`~quantify_scheduler.backends.qblox.operations.pulse_library.LatchReset`
          on all sequencers.

    """
    # TODO: this logic needs to be moved to a cluster compiler container. With
    # this implementation the `address_map` is shared among multiple
    # clusters, but each cluster should have its own map. (SE-332)
    address_counter = itertools.count(1)
    conditional_address_map = defaultdict(
        lambda: ConditionalAddress(portclocks=set(), address=address_counter.__next__())
    )
    _set_conditional_address_map(schedule, conditional_address_map)
    _insert_latch_reset(schedule, 0, schedule, conditional_address_map)
    address_map_addresses = [a.address for a in conditional_address_map.values()]
    if max(address_map_addresses, default=0) > constants.MAX_FEEDBACK_TRIGGER_ADDRESS:
        raise ValueError(
            "Maximum number of feedback trigger addresses received. "
            "Currently a Qblox cluster can store a maximum of "
            f"{constants.MAX_FEEDBACK_TRIGGER_ADDRESS} addresses."
        )

    all_conditional_acqs_and_control_flows: list[tuple[float, Operation]] = list()
    _all_conditional_acqs_and_control_flows_and_latch_reset(
        schedule, 0, all_conditional_acqs_and_control_flows
    )
    all_conditional_acqs_and_control_flows.sort(key=lambda time_op_sched: time_op_sched[0])

    current_ongoing_conditional_acquire = None
    for (
        time,
        operation,
    ) in all_conditional_acqs_and_control_flows:
        if isinstance(operation, ConditionalOperation):
            if current_ongoing_conditional_acquire is None:
                raise RuntimeError(
                    f"Conditional control flow, ``{operation}``,  found without a preceding "
                    "Conditional acquisition. "
                    "Please ensure that the preceding acquisition or Measure is conditional, "
                    "by passing `feedback_trigger_label=qubit_name` "
                    "to the corresponding operation, "
                    "e.g.\n\n"
                    "> schedule.add(Measure(qubit_name, ..., feedback_trigger_label=qubit_name))\n"
                )
            else:
                current_ongoing_conditional_acquire = None
        elif operation.valid_acquisition and operation.is_conditional_acquisition:
            if current_ongoing_conditional_acquire is None:
                current_ongoing_conditional_acquire = operation
            else:
                raise RuntimeError(
                    "Two subsequent conditional acquisitions found, without a "
                    "conditional control flow operation in between. Conditional "
                    "playback will only work if a conditional measure or acquisition "
                    "is followed by a conditional control flow operation.\n"
                    "The following two operations caused this problem: \n"
                    f"{current_ongoing_conditional_acquire}\nand\n"
                    f"{operation}\n"
                )

    return schedule


def compile_long_square_pulses_to_awg_offsets(  # noqa: D417
    schedule: Schedule, config: DataStructure | dict  # noqa: ARG001
) -> Schedule:
    """
    Replace square pulses in the schedule with long square pulses.

    Introspects operations in the schedule to find square pulses with a duration
    longer than
    :class:`~quantify_scheduler.backends.qblox.constants.PULSE_STITCHING_DURATION`. Any
    of these square pulses are converted to
    :func:`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_square_pulse`,
    which consist of AWG voltage offsets.

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
        :func:`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_square_pulse`.
        If no replacements were done, this is the original unmodified schedule.

    """
    _replace_long_square_pulses_recursively(schedule)
    return schedule


def hardware_compile(
    schedule: Schedule,
    config: CompilationConfig,
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

    Returns
    -------
    :
        The compiled schedule.

    """
    hardware_cfg = deepcopy(config.hardware_compilation_config)

    assert isinstance(hardware_cfg, QbloxHardwareCompilationConfig)

    if hardware_cfg.hardware_options.latency_corrections is not None:
        # Subtract minimum latency to allow for negative latency corrections
        hardware_cfg.hardware_options.latency_corrections = determine_relative_latency_corrections(
            schedule=schedule,
            hardware_cfg=hardware_cfg,
        )

    # Apply software distortion corrections. Hardware distortion corrections are
    # compiled into the compiler container that follows.
    if hardware_cfg.hardware_options.distortion_corrections is not None:
        replacing_schedule = apply_software_distortion_corrections(
            schedule, hardware_cfg.hardware_options.distortion_corrections
        )
        if replacing_schedule is not None:
            schedule = replacing_schedule

    _add_clock_freqs_to_set_clock_frequency(schedule)

    validate_non_overlapping_stitched_pulse(schedule)

    if not hardware_cfg.allow_off_grid_nco_ops:
        _check_nco_operations_on_nco_time_grid(schedule)

    container = compiler_container.CompilerContainer.from_hardware_cfg(schedule, hardware_cfg)

    assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=container.clusters,
    )

    container.prepare()

    compiled_instructions = container.compile(
        debug_mode=config.debug_mode, repetitions=schedule.repetitions
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

    config_type: type[QbloxHardwareCompilationConfig] = Field(  # type: ignore
        default="quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        validate_default=True,
    )
    """
    A reference to the
    :class:`~quantify_scheduler.backends.types.common.HardwareCompilationConfig`
    DataStructure for the Qblox backend.
    """
    version: str = Field(default="0.1")
    """
    Version of the specific hardware compilation config used.
    """
    hardware_description: dict[str, QbloxHardwareDescription | HardwareDescription]  # type: ignore
    """Description of the instruments in the physical setup."""
    hardware_options: QbloxHardwareOptions  # type: ignore
    """
    Options that are used in compiling the instructions for the hardware, such as
    :class:`~quantify_scheduler.backends.types.common.LatencyCorrection` or
    :class:`~quantify_scheduler.backends.types.qblox.SequencerOptions`.
    """
    allow_off_grid_nco_ops: bool | None = None
    """
    Flag to allow NCO operations to play at times that are not aligned with the NCO
    grid.
    """
    compilation_passes: list[SimpleNodeConfig] = [
        SimpleNodeConfig(
            name="crosstalk_compensation",
            compilation_func=crosstalk_compensation,  # type: ignore
        ),
        SimpleNodeConfig(
            name="stack_pulses",
            compilation_func=stack_pulses,
        ),
        SimpleNodeConfig(
            name="compile_long_square_pulses_to_awg_offsets",
            compilation_func=compile_long_square_pulses_to_awg_offsets,
        ),
        SimpleNodeConfig(
            name="qblox_compile_conditional_playback",
            compilation_func=compile_conditional_playback,
        ),
        SimpleNodeConfig(
            name="qblox_hardware_compile", compilation_func=hardware_compile  # type: ignore
        ),
    ]
    """
    The list of compilation nodes that should be called in succession to compile a
    schedule to instructions for the Qblox hardware.
    """

    @model_validator(mode="after")
    def _validate_connectivity_channel_names(self) -> QbloxHardwareCompilationConfig:
        module_name_to_channel_names_map: dict[tuple[str, str], set[str]] = defaultdict(set)
        assert isinstance(self.connectivity, Connectivity)
        for node in self.connectivity.graph.nodes:
            try:
                cluster_name, module_name, channel_name = node.split(".")
            except ValueError:
                continue

            if isinstance(self.hardware_description.get(cluster_name), ClusterDescription):
                module_name_to_channel_names_map[cluster_name, module_name].add(channel_name)

        for (
            cluster_name,
            module_name,
        ), channel_names in module_name_to_channel_names_map.items():
            module_idx_str = re.search(r"module(\d+)", module_name)
            assert module_idx_str is not None
            module_idx = int(module_idx_str.group(1))

            # Create new variable to help pyright.
            cluster_descr = self.hardware_description[cluster_name]
            assert isinstance(cluster_descr, ClusterDescription)
            try:
                cluster_descr.modules[module_idx].validate_channel_names(channel_names)
            except ValueError as exc:
                instrument_type = cluster_descr.modules[module_idx].instrument_type
                valid_channels = cluster_descr.modules[module_idx].get_valid_channels()
                # Add some information to the raised exception. The original exception
                # is included in the message because pydantic suppresses the traceback.
                raise ValueError(
                    f"Error validating channel names for {cluster_name}.{module_name} "
                    f"({instrument_type}). Full error message:\n{exc}\n\nSupported "
                    f"names for {instrument_type}:\n{valid_channels}."
                ) from exc
            except KeyError:
                raise KeyError(
                    f"""Module '{module_idx}' of cluster
                        '{cluster_name}' not found in the hardware description. Please
                        ensure all modules mentioned in the connectivity are
                        present in the hardware description. """
                )
        return self

    @model_validator(mode="after")
    def _warn_mix_lo_false(
        self,
    ) -> QbloxHardwareCompilationConfig:
        channels_with_laser = []

        if isinstance(self.connectivity, Connectivity):
            # Find channels coupled to lasers
            for edge in self.connectivity.graph.edges:
                source, target = edge
                if len(source.split(".")) == 3 and "laser" in target:
                    channels_with_laser.append(ChannelPath.from_path(source))

                # Sometimes source and target appear swapped. This block can be removed
                # after making graph directed. (SE-477)
                elif len(target.split(".")) == 3 and "laser" in source:
                    channels_with_laser.append(ChannelPath.from_path(target))

            # Find mix_lo value in hardware description
            for channel_path in channels_with_laser:
                # New variable to help pyright.
                cluster_descr = self.hardware_description[channel_path.cluster_name]
                assert isinstance(cluster_descr, ClusterDescription)
                module_description = cluster_descr.modules[channel_path.module_idx].model_dump(
                    exclude_unset=True
                )
                channel_description = module_description.get(channel_path.channel_name, None)
                if channel_description is not None:
                    mix_lo = channel_description.get("mix_lo", None)
                    # FIXME: https://qblox.atlassian.net/browse/SE-490
                    if mix_lo is not None and mix_lo is False:
                        warnings.warn(
                            "Using `mix_lo=False` in channels coupled to lasers might "
                            "cause undefined behavior.",
                            UserWarning,
                        )

        return self

    # TODO Remove together with deprecated  _generate_new_style_hardware_compilation_config
    @model_validator(mode="before")
    @classmethod
    def from_old_style_hardware_config(
        cls, data: Any  # noqa: ANN401 deprecated
    ) -> Any:  # noqa: ANN401 deprecated
        """Convert old style hardware config dict to new style before validation."""
        if (
            isinstance(data, dict)
            and data.get("backend") == "quantify_scheduler.backends.qblox_backend.hardware_compile"
        ):
            # Input is an old style Qblox hardware config dict
            data = _generate_new_style_hardware_compilation_config(data)

        return data

    @model_validator(mode="before")
    def _validate_versioning(cls, config: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        if "version" in config:  # noqa: SIM102
            if config["version"] not in ["0.1", "0.2"]:
                raise ValueError("Unknown hardware config version.")

        return config

    def _extract_instrument_compilation_configs(
        self, portclocks_used: set[tuple]
    ) -> dict[str, Any]:
        """
        Extract an instrument compiler config
        for each instrument mentioned in ``hardware_description``.
        Each instrument config has a similar structure as ``QbloxHardwareCompilationConfig``
        , but contains only the settings related to their related instrument.
        Each config must contain at least one ortclock referenced in ``portclocks_used``,
        otherwise the config is deleted.
        """
        cluster_configs: dict[str, _ClusterCompilationConfig] = {}
        lo_configs: dict[str, _LocalOscillatorCompilationConfig] = {}

        # Extract instrument hardware descriptions
        for (
            instrument_name,
            instrument_description,
        ) in self.hardware_description.items():
            if isinstance(instrument_description, ClusterDescription):
                cluster_configs[instrument_name] = _ClusterCompilationConfig(
                    hardware_description=instrument_description,
                    hardware_options=QbloxHardwareOptions(),
                    parent_config_version=self.version,
                )

            elif isinstance(instrument_description, LocalOscillatorDescription):
                lo_configs[instrument_name] = _LocalOscillatorCompilationConfig(
                    hardware_description=instrument_description,
                    frequency=None,
                )

        self._get_all_portclock_to_path_and_lo_name_to_path(
            portclocks_used, cluster_configs, lo_configs
        )

        # Add name to LO hardware description and add frequency to LO compiler config
        if self.hardware_options.modulation_frequencies is not None:
            for (
                portclock,
                frequencies,
            ) in self.hardware_options.modulation_frequencies.items():
                for instr_name, cfg in cluster_configs.items():
                    if portclock not in cfg.portclock_to_path:
                        continue

                    for lo_name, lo_path in cfg.lo_to_path.items():
                        if cfg.portclock_to_path[portclock] == lo_path:
                            lo_config = lo_configs[lo_name]
                            if lo_config.hardware_description.instrument_name is None:
                                lo_config.hardware_description.instrument_name = lo_name
                                lo_config.frequency = frequencies.lo_freq

        # Extract hardware options
        clusters_hardware_options: dict[str, dict[str, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        options_by_portclock: dict[str, dict[str, Any]] = {}
        for option, pc_to_value_map in self.hardware_options.model_dump(exclude_unset=True).items():
            for pc, value in pc_to_value_map.items():
                options_by_portclock.setdefault(pc, {})[option] = value

        for instr_name, cfg in cluster_configs.items():
            for pc in cfg.portclock_to_path:
                if pc not in options_by_portclock:
                    continue
                for option, value in options_by_portclock[pc].items():
                    clusters_hardware_options[instr_name][option][pc] = value

        for cluster_name, options in clusters_hardware_options.items():
            cluster_configs[cluster_name].hardware_options = QbloxHardwareOptions.model_validate(
                options
            )

        for instr_cfg in cluster_configs.values():
            instr_cfg.allow_off_grid_nco_ops = self.allow_off_grid_nco_ops

        # Delete hardware descriptions of unused modules
        unused_modules = defaultdict(list)
        for instrument_name, cfg in cluster_configs.items():
            used_modules_idx = [path.module_idx for path in cfg.portclock_to_path.values()]
            for module_idx in cfg.hardware_description.modules:
                if module_idx not in used_modules_idx:
                    unused_modules[instrument_name].append(module_idx)

        for cluster_name, indices in unused_modules.items():
            for module_idx in indices:
                del cluster_configs[cluster_name].hardware_description.modules[module_idx]

        # Delete empty configs of unused clusters
        unused_clusters = []
        for instrument_name, cfg in cluster_configs.items():
            if not cfg.portclock_to_path:
                unused_clusters.append(instrument_name)

        for cluster_name in unused_clusters:
            del cluster_configs[cluster_name]

        # Delete empty configs of unused local oscillators
        unused_los = []
        for instrument_name, cfg in lo_configs.items():
            if cfg.hardware_description.instrument_name is None:
                unused_los.append(instrument_name)

        for lo_name in unused_los:
            del lo_configs[lo_name]

        return {**cluster_configs, **lo_configs}

    def _get_all_portclock_to_path_and_lo_name_to_path(  # noqa: PLR0915
        self,
        portclocks_used: set[tuple[str, str]],
        cluster_configs: dict[str, _ClusterCompilationConfig],
        lo_configs: dict[str, _LocalOscillatorCompilationConfig],
    ) -> None:
        assert isinstance(self.connectivity, Connectivity)
        cluster_pc_to_path: dict[str, dict[str, ChannelPath]] = {}
        cluster_lo_to_path: dict[str, dict[str, ChannelPath]] = {}
        for instr in cluster_configs:
            cluster_pc_to_path[instr] = {}
            cluster_lo_to_path[instr] = {}

        def is_path(value: str) -> bool:
            possible_cluster_name = value.split(".")[0]
            # Important: must be exact match.
            return any(cluster_name == possible_cluster_name for cluster_name in cluster_pc_to_path)

        def is_lo_port(value: str) -> bool:
            possible_lo_name = value.split(".")[0]
            # Important: must be exact match.
            return any(lo_name == possible_lo_name for lo_name in lo_configs)

        def get_all_mixer_nodes(value: str) -> Iterable[str]:
            return (
                n
                for n in self.connectivity.graph  # type: ignore
                if "." in n and n[: n.rindex(".")] == value
            )

        port_to_clocks: dict[str, list[str]] = {}
        for port, clock in portclocks_used:
            port_to_clocks.setdefault(port, []).append(clock)

        # NV center hack
        mixer_part_to_clock_part = {
            "spinpump_laser": "ge1",
            "green_laser": "ionization",
            "red_laser": "ge0",
        }

        def get_optical_clock(mixer: str, clocks: list[str]) -> str | None:
            for mixer_part, clock_part in mixer_part_to_clock_part.items():
                if mixer_part in mixer:
                    for clock in clocks:
                        if clock_part in clock:
                            return clock
            return None

        def get_module_and_lo_for_mixer(mixer: str) -> tuple[str, str]:
            mixer_nodes = get_all_mixer_nodes(mixer)
            mixer_path: str = None  # type: ignore
            mixer_lo: str = None  # type: ignore
            unidentified: list[str] = []
            for mixer_node in mixer_nodes:
                for mixer_nbr in self.connectivity.graph.neighbors(mixer_node):  # type: ignore
                    if is_path(mixer_nbr):
                        mixer_path = mixer_nbr
                    elif is_lo_port(mixer_nbr):
                        mixer_lo = mixer_nbr[: mixer_nbr.rindex(".")]
                    elif mixer_nbr in port_to_clocks:
                        pass
                    else:
                        unidentified.append(mixer_nbr)
            err_add = (
                f" Did find unidentified nodes {unidentified}. "
                f"Make sure these are specified in the hardware description."
                if unidentified
                else ""
            )
            if mixer_path is None and mixer_lo is not None:
                raise RuntimeError(
                    f"Could not find a cluster module port for '{mixer}', which is "
                    f"connected to local oscillator '{mixer_lo}' and port '{port}' "
                    f"in the connectivity.{err_add}"
                )
            if mixer_path is not None and mixer_lo is None:
                raise RuntimeError(
                    f"Could not find local oscillator device for '{mixer}', which is "
                    f"connected to cluster module port '{mixer_path}' and port "
                    f"'{port}' in the connectivity.{err_add}"
                )
            if mixer_path is None or mixer_lo is None:
                raise RuntimeError(
                    f"Could not find cluster module port or local oscillator in the "
                    f"connectivity for '{mixer}', which is connected to port '{port}'."
                    f"{err_add}"
                )
            return mixer_path, mixer_lo

        for port in port_to_clocks:
            if port not in self.connectivity.graph:
                raise KeyError(f"{port} was not found in the connectivity.")
            for port_nbr in self.connectivity.graph.neighbors(port):
                if is_path(port_nbr):
                    path = ChannelPath.from_path(port_nbr)
                    for clock in port_to_clocks[port]:
                        repeated_pc = False
                        pc = f"{port}-{clock}"
                        # Add extra channel name for `Measure` operation. This takes place after
                        # the first channel name is added ("if not repeated_pc..." block below)
                        for cluster_name, pc_to_path in cluster_pc_to_path.items():
                            if pc in pc_to_path:
                                if (
                                    path.cluster_name != pc_to_path[pc].cluster_name
                                    or path.module_name != pc_to_path[pc].module_name
                                ):
                                    raise ValueError(
                                        f"Provided channel names for port-clock {pc} are defined "
                                        f"for diferent modules, but they must be defined in the "
                                        f"same module."
                                    )
                                pc_to_path[pc].add_channel_name_measure(path.channel_name)
                                repeated_pc = True

                        if not repeated_pc:
                            cluster_pc_to_path[path.cluster_name][pc] = path

                elif "." in port_nbr:
                    mixer = port_nbr[: port_nbr.rindex(".")]
                    mixer_path, mixer_lo = get_module_and_lo_for_mixer(mixer)
                    path = ChannelPath.from_path(mixer_path)
                    cluster_lo_to_path[path.cluster_name][mixer_lo] = path
                    for clock in port_to_clocks[port]:
                        # NV center hack:
                        fixed_clock = get_optical_clock(mixer, port_to_clocks[port]) or clock
                        cluster_pc_to_path[path.cluster_name][f"{port}-{fixed_clock}"] = path

        for cluster_name, pc_to_path in cluster_pc_to_path.items():
            cluster_configs[cluster_name].portclock_to_path = pc_to_path
        for cluster_name, lo_to_path in cluster_lo_to_path.items():
            cluster_configs[cluster_name].lo_to_path = lo_to_path


class _LocalOscillatorCompilationConfig(DataStructure):
    """
    Configuration values for a
    :class:`quantify_scheduler.backends.qblox.instrument_compilers.LocalOscillatorCompiler`.
    """

    hardware_description: LocalOscillatorDescription
    """Description of the physical setup of this local oscillator."""
    frequency: float | None = None
    """The frequency of this local oscillator."""


class _ClusterModuleCompilationConfig(ABC, DataStructure):
    """Configuration values for a :class:`~.ClusterModuleCompiler`."""

    hardware_description: ClusterModuleDescription
    """Description of the physical setup of this module."""
    hardware_options: QbloxHardwareOptions
    """Options that are used in compiling the instructions for the hardware."""
    portclock_to_path: dict[str, ChannelPath] = {}
    """Mapping between portclocks and their associated channel name paths
    (e.g. cluster0.module1.complex_output_0)."""
    lo_to_path: dict[str, ChannelPath] = {}
    """Mapping between lo names and their associated channel name paths
    (e.g. cluster0.module1.complex_output_0)."""
    allow_off_grid_nco_ops: bool | None = None
    """
    Flag to allow NCO operations to play at times that are not aligned with the NCO
    grid.
    """
    parent_config_version: str
    """
    Version of the parent hardware compilation config used.
    """

    def _extract_sequencer_compilation_configs(
        self,
    ) -> dict[int, _SequencerCompilationConfig]:

        sequencer_configs = {}
        channel_to_lo = {path.channel_name: lo_name for lo_name, path in self.lo_to_path.items()}

        # Sort to ensure deterministic order in sequencer instantiation
        for seq_idx, portclock in enumerate(sorted(self.portclock_to_path)):
            path = self.portclock_to_path[portclock]
            sequencer_options = (
                self.hardware_options.sequencer_options.get(portclock, {})
                if self.hardware_options.sequencer_options is not None
                else {}
            )
            hardware_description = {}
            for description in self.hardware_description.model_fields_set:
                if description == path.channel_name:
                    hardware_description = getattr(
                        self.hardware_description,
                        description,
                    )
            latency_correction = (
                self.hardware_options.latency_corrections.get(portclock, 0)
                if self.hardware_options.latency_corrections is not None
                else 0
            )
            distortion_correction = (
                self.hardware_options.distortion_corrections.get(portclock, None)
                if self.hardware_options.distortion_corrections is not None
                else None
            )
            if isinstance(
                distortion_correction,
                (list, QbloxHardwareDistortionCorrection),
            ):
                distortion_correction = None
            modulation_frequencies = (
                self.hardware_options.modulation_frequencies.get(portclock, {})
                if self.hardware_options.modulation_frequencies is not None
                else {}
            )
            mixer_corrections = (
                self.hardware_options.mixer_corrections.get(portclock, None)
                if self.hardware_options.mixer_corrections is not None
                else None
            )
            digitization_thresholds = (
                self.hardware_options.digitization_thresholds.get(portclock, None)
                if self.hardware_options.digitization_thresholds
                else None
            )
            # Types: arguments are allowed to be dict. Pydantic takes care of that.
            sequencer_configs[seq_idx] = _SequencerCompilationConfig(
                sequencer_options=sequencer_options,  # type: ignore
                hardware_description=hardware_description,  # type: ignore
                portclock=portclock,
                channel_name=path.channel_name,
                channel_name_measure=path.channel_name_measure,
                latency_correction=latency_correction,
                distortion_correction=distortion_correction,
                lo_name=channel_to_lo.get(path.channel_name),
                modulation_frequencies=modulation_frequencies,  # type: ignore
                mixer_corrections=mixer_corrections,
                allow_off_grid_nco_ops=self.allow_off_grid_nco_ops,
                digitization_thresholds=digitization_thresholds,
            )

        return sequencer_configs

    # This is a temporary solution until distortion corrections are
    # properly defined (SE-544)
    def _validate_hardware_distortion_corrections_mode(
        self,
    ) -> _ClusterModuleCompilationConfig:

        distortion_corrections = (
            self.hardware_options.distortion_corrections
            if self.hardware_options is not None
            else None
        )

        if distortion_corrections is not None:
            for portclock, corrections in distortion_corrections.items():
                channel_name = self.portclock_to_path[portclock].channel_name
                if ChannelMode.REAL in channel_name and isinstance(corrections, list):
                    raise ValueError(
                        f"Several distortion corrections were assigned to portclock '{portclock}' "
                        f"which is a real channel, but only one correction is required."
                    )
                elif ChannelMode.COMPLEX in channel_name and isinstance(
                    corrections, QbloxHardwareDistortionCorrection
                ):
                    raise ValueError(
                        f"One distortion correction was assigned to portclock '{portclock}' "
                        f"which is a complex channel, but two corrections are required."
                    )

        return self

    def _validate_input_gain_mode(
        self,
    ) -> _ClusterModuleCompilationConfig:

        input_gain = self.hardware_options.input_gain if self.hardware_options is not None else None

        if input_gain is not None:
            for portclock, gain in input_gain.items():
                channel_name = self.portclock_to_path[portclock].channel_name

                if ChannelMode.REAL in channel_name and isinstance(gain, ComplexInputGain):
                    raise ValueError(
                        f"A complex input gain was assigned to portclock '{portclock}', "
                        f"which is a real channel."
                    )
                elif ChannelMode.COMPLEX in channel_name and isinstance(gain, RealInputGain):
                    raise ValueError(
                        f"A real input gain was assigned to portclock '{portclock}', "
                        f"which is a complex channel."
                    )

        return self

    def _validate_channel_name_measure(self) -> None:
        pass


class _QCMCompilationConfig(_ClusterModuleCompilationConfig):
    """QCM-specific configuration values for a :class:`~.ClusterModuleCompiler`."""

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:
                raise ValueError(
                    f"Found two channel names {path.channel_name} and {path.channel_name_measure} "
                    f"for portclock {pc}. Repeated portclocks are forbidden for QCM modules."
                )

        return super()._validate_channel_name_measure()


class _QRMCompilationConfig(_ClusterModuleCompilationConfig):
    """QRM-specific configuration values for a :class:`~.ClusterModuleCompiler`."""

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:  # noqa: SIM102
                if len(path.channel_name_measure) == 1:
                    if not (
                        "complex_output" in path.channel_name or "real_output" in path.channel_name
                    ) or not (
                        "complex_input" in path.channel_name_measure[0]
                        or "real_input" in path.channel_name_measure[0]
                    ):
                        raise ValueError(
                            f"Found two channel names {path.channel_name} and "
                            f"{path.channel_name_measure[0]} that are not of the same mode for "
                            f"portclock {pc}. Only channel names of the same mode (e.g. "
                            f"`complex_output_0` and `complex_input_0`) are allowed when they "
                            f"share a portclock in QRM modules."
                        )
                elif len(path.channel_name_measure) == 2:
                    if not (
                        "complex_output" in path.channel_name or "real_output" in path.channel_name
                    ) or not (
                        "real_input_0" in path.channel_name_measure
                        and "real_input_1" in path.channel_name_measure
                    ):
                        # This is not true, other combinations are possible but are hidden from
                        # the user interface. Those are allowed only to keep compatibility
                        # with hardware config version 0.1 (SE-427)
                        raise ValueError(
                            f"Found an incorrect combination of three channel names for portclock "
                            f"{pc}. Please try to use two channel names instead."
                        )
                else:
                    # Extreme edge case
                    raise ValueError(
                        f"Found four channel names for portclock {pc}. "
                        "Please try to use two channel names instead."
                    )

        return super()._validate_channel_name_measure()


class _QCMRFCompilationConfig(_ClusterModuleCompilationConfig):
    """QCM_RF-specific configuration values for a :class:`~.ClusterModuleCompiler`."""

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:
                raise ValueError(
                    f"Found two channel names {path.channel_name} and {path.channel_name_measure} "
                    f"for portclock {pc}. Repeated portclocks are forbidden for QCM_RF modules."
                )
        return super()._validate_channel_name_measure()


class _QRMRFCompilationConfig(_ClusterModuleCompilationConfig):
    """QRMRF-specific configuration values for a :class:`~.ClusterModuleCompiler`."""

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:  # noqa: SIM102
                if not (
                    "complex_output" in path.channel_name
                    and "complex_input" in path.channel_name_measure[0]
                ):
                    raise ValueError(
                        f"Found two channel names {path.channel_name} and "
                        f"{path.channel_name_measure[0]} that are not of the same mode for "
                        f"portclock {pc}. Only channel names of the same mode (e.g. "
                        f"`complex_output_0` and `complex_input_0`) are allowed when they share a "
                        f"portclock in QRM_RF modules."
                    )

        return super()._validate_channel_name_measure()


class _QTMCompilationConfig(_ClusterModuleCompilationConfig):
    """QTM-specific configuration values for a :class:`~.ClusterModuleCompiler`."""

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:
                raise NotImplementedError(
                    f"Found two channel names {path.channel_name} and {path.channel_name_measure} "
                    f"for portclock {pc}. Circuit-level operations involving several channels "
                    f"(e.g. `Measure`) are not implemented for QTM modules."
                )
        return super()._validate_channel_name_measure()


class _SequencerCompilationConfig(DataStructure):
    """Configuration values for a :class:`~.SequencerCompiler`."""

    hardware_description: (
        ComplexChannelDescription | RealChannelDescription | DigitalChannelDescription
    )
    """Information needed to specify a complex/real/digital input/output."""
    sequencer_options: SequencerOptions
    """Configuration options for this sequencer."""
    portclock: str
    """Portclock associated to this sequencer."""
    channel_name: str
    """Channel name associated to this sequencer."""
    channel_name_measure: None | list[str]
    """Extra channel name necessary to define a `Measure` operation."""
    latency_correction: LatencyCorrection
    """Latency correction that should be applied to operations on this sequencer."""
    distortion_correction: SoftwareDistortionCorrection | None
    """Distortion corrections that should be applied to waveforms on this sequencer."""
    lo_name: str | None
    """Local oscilator associated to this sequencer."""
    modulation_frequencies: ModulationFrequencies
    """Modulation frequencies associated to this sequencer."""
    mixer_corrections: QbloxMixerCorrections | None
    """Mixer correction settings."""
    allow_off_grid_nco_ops: bool | None = None
    """
    Flag to allow NCO operations to play at times that are not aligned with the NCO
    grid.
    """
    digitization_thresholds: DigitizationThresholds | None = None
    """The settings that determine when an analog voltage is counted as a pulse."""


class _ClusterCompilationConfig(DataStructure):
    """Configuration values for a :class:`~.ClusterCompiler`."""

    hardware_description: ClusterDescription
    """Description of the physical setup of this cluster."""
    hardware_options: QbloxHardwareOptions
    """Options that are used in compiling the instructions for the hardware."""
    portclock_to_path: dict[str, ChannelPath] = {}
    """Mapping between portclocks and their associated channel name paths
    (e.g. cluster0.module1.complex_output_0)."""
    lo_to_path: dict[str, ChannelPath] = {}
    """Mapping between lo names and their associated channel name paths
    (e.g. cluster0.module1.complex_output_0)."""
    allow_off_grid_nco_ops: bool | None = None
    """
    Flag to allow NCO operations to play at times that are not aligned with the NCO
    grid.
    """
    parent_config_version: str
    """
    Version of the parent hardware compilation config used.
    """

    module_config_classes: dict = {
        "QCM": _QCMCompilationConfig,
        "QRM": _QRMCompilationConfig,
        "QCM_RF": _QCMRFCompilationConfig,
        "QRM_RF": _QRMRFCompilationConfig,
        "QTM": _QTMCompilationConfig,
    }

    def _extract_module_compilation_configs(
        self,
    ) -> dict[int, _ClusterModuleCompilationConfig]:

        module_configs: dict[int, _ClusterModuleCompilationConfig] = {}

        # Create configs and distribute `hardware_description`
        for module_idx, module_description in self.hardware_description.modules.items():
            module_configs[module_idx] = self.module_config_classes[
                module_description.instrument_type
            ](
                hardware_description=module_description,
                hardware_options=QbloxHardwareOptions(),
                parent_config_version=self.parent_config_version,
            )

        # Distribute module `hardware_options`
        modules_hardware_options: dict[int, dict[str, dict[str, Any]]] = {
            module_idx: {} for module_idx in module_configs
        }

        for option, values in self.hardware_options.model_dump(exclude_unset=True).items():
            for pc, option_value in values.items():
                module_idx = self.portclock_to_path[pc].module_idx

                if not modules_hardware_options[module_idx].get(option):
                    modules_hardware_options[module_idx][option] = {}
                modules_hardware_options[module_idx][option][pc] = option_value

        for module_idx, options in modules_hardware_options.items():
            module_configs[module_idx].hardware_options = QbloxHardwareOptions.model_validate(
                options
            )

        # Distribute `portclock_to_path`
        for portclock, path in self.portclock_to_path.items():
            module_configs[path.module_idx].portclock_to_path[portclock] = path

        # Distribute `lo_to_path`
        for lo_name, path in self.lo_to_path.items():
            module_configs[path.module_idx].lo_to_path[lo_name] = path

        # Distribute `allow_off_grid_nco_ops`
        for module_cfg in module_configs.values():
            module_cfg.allow_off_grid_nco_ops = self.allow_off_grid_nco_ops

        # Add support for old versions of hardware config
        if self.parent_config_version == "0.1":
            for cfg in module_configs.values():
                _add_support_input_channel_names(cfg)

        # Validate module configs
        for cfg in module_configs.values():
            cfg._validate_input_gain_mode()
            cfg._validate_hardware_distortion_corrections_mode()
            cfg._validate_channel_name_measure()

        return module_configs


def _add_support_input_channel_names(
    module_config: _ClusterModuleCompilationConfig,
) -> None:
    # Update cluster module config with extra channel names (`channel_name_measure`) in order
    # to keep backwards compatibility after enforcing the use of two channel names for `Measure`
    # operations (see also SE-427). Sorry for the hacky style, couldn't find a better solution
    complex_output_pcs = {}
    real_output_pcs = {}
    complex_inputs = []
    real_inputs = []

    for pc, path in module_config.portclock_to_path.items():
        if ChannelMode.COMPLEX in path.channel_name:
            if "output" in path.channel_name:
                complex_output_pcs[path.channel_name] = pc
            else:
                complex_inputs.append(path.channel_name)
        elif ChannelMode.REAL in path.channel_name:
            if "output" in path.channel_name:
                real_output_pcs[path.channel_name] = pc
            else:
                real_inputs.append(path.channel_name)

    if module_config.hardware_description.instrument_type in ["QRM", "QRM_RF"]:
        # The specific value of the portclock is not important, as long as it has been
        # defined before for an output in the same module
        if not complex_inputs and not real_inputs:
            if complex_output_pcs:
                pc = list(complex_output_pcs.values())[0]
                module_config.portclock_to_path[pc].add_channel_name_measure("complex_input_0")
            elif real_output_pcs:
                pc = list(real_output_pcs.values())[0]
                module_config.portclock_to_path[pc].add_channel_name_measure("real_input_0")
                # removing the deepcopy is breaking.
                module_config.portclock_to_path[deepcopy(pc)].add_channel_name_measure(
                    "real_input_1"
                )
        elif (
            not complex_inputs
            and len(real_inputs) == 1
            and not complex_output_pcs
            and real_output_pcs
        ):
            pc = list(real_output_pcs.values())[0]
            if int(real_inputs[0][-1]) == 0:
                module_config.portclock_to_path[pc].add_channel_name_measure("real_input_1")
            else:
                module_config.portclock_to_path[pc].add_channel_name_measure("real_input_0")


@dataclass
class ChannelPath:
    """Path of a sequencer channel."""

    cluster_name: str
    module_name: str
    channel_name: str
    module_idx: int
    channel_name_measure: None | list[str] = field(init=False, default=None)

    @classmethod
    def from_path(cls: type[ChannelPath], path: str) -> ChannelPath:
        """Instantiate a `ChannelPath` object from a path string."""
        cluster_name, module_name, channel_name = path.split(".")
        module_idx = int(module_name.replace("module", ""))
        return cls(
            cluster_name=cluster_name,
            module_name=module_name,
            channel_name=channel_name,
            module_idx=module_idx,
        )

    def add_channel_name_measure(self, channel_name_measure: str) -> None:
        """Add an extra input channel name for measure operation."""
        if self.channel_name_measure is None:
            channel_name = deepcopy(self.channel_name)
            # By convention, the "output" channel name is the main channel name in
            # measure operations
            if "input" in channel_name_measure:
                self.channel_name_measure = [channel_name_measure]
            else:
                self.channel_name = channel_name_measure
                self.channel_name_measure = [channel_name]
        else:
            self.channel_name_measure.append(channel_name_measure)


def _all_abs_times_ops_with_voltage_offsets_pulses(
    operation: Operation | Schedule,
    time_offset: float,
    accumulator: list[tuple[float, Operation]],
) -> None:
    if isinstance(operation, ScheduleBase):
        for schedulable in operation.schedulables.values():
            abs_time = schedulable["abs_time"]
            inner_operation = operation.operations[schedulable["operation_id"]]
            _all_abs_times_ops_with_voltage_offsets_pulses(
                inner_operation, time_offset + abs_time, accumulator
            )
    elif isinstance(operation, (ConditionalOperation, LoopOperation)):
        # Note: we don't need to check every cycle of the loop,
        # only the first one, because if it fails for any cycle,
        # it should also fail for the first cyce (we check each sequencer separately).
        _all_abs_times_ops_with_voltage_offsets_pulses(
            operation.body,
            time_offset,
            accumulator,
        )
    elif operation.valid_pulse or operation.has_voltage_offset:
        accumulator.append((time_offset, operation))


def _add_clock_freqs_to_set_clock_frequency(
    schedule: Schedule, operation: Operation | Schedule | None = None
) -> None:
    if operation is None:
        operation = schedule

    if isinstance(operation, ScheduleBase):
        for schedulable in operation.schedulables.values():
            inner_operation = operation.operations[schedulable["operation_id"]]
            _add_clock_freqs_to_set_clock_frequency(operation=inner_operation, schedule=operation)
    elif isinstance(operation, ControlFlowOperation):
        _add_clock_freqs_to_set_clock_frequency(operation=operation.body, schedule=schedule)
    else:
        for pulse_info in operation["pulse_info"]:
            clock_freq = schedule.resources.get(pulse_info["clock"], {}).get("freq", None)

            if "clock_freq_new" in pulse_info:
                pulse_info.update(
                    {
                        "clock_freq_old": clock_freq,
                    }
                )


def validate_non_overlapping_stitched_pulse(schedule: Schedule) -> None:
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
    abs_times_and_operations: list[tuple[float, Operation]] = list()
    _all_abs_times_ops_with_voltage_offsets_pulses(schedule, 0, abs_times_and_operations)
    abs_times_and_operations.sort(key=lambda abs_time_and_op: abs_time_and_op[0])

    # Iterate through all relevant operations in chronological order, and keep track of the
    # latest end time of schedulables containing pulses.
    # When a schedulable contains a voltage offset, check if it starts before the end of
    # a previously found pulse, else look ahead for any schedulables with pulses
    # starting before this schedulable ends.
    last_pulse_end = -np.inf
    last_pulse_abs_time = None
    last_pulse_op = None
    for i, (abs_time_op, op) in enumerate(abs_times_and_operations):
        if op.has_voltage_offset:
            # Add 1e-10 for possible floating point errors (strictly >).
            if last_pulse_end > abs_time_op + 1e-10:
                if last_pulse_abs_time is None or last_pulse_op is None:
                    # This error should be unreachable.
                    raise RuntimeError(
                        f"{last_pulse_abs_time=} and {last_pulse_op} may not be None "
                        "at this point."
                    )
                _raise_if_pulses_overlap_on_same_port_clock(
                    abs_time_op,
                    op,
                    last_pulse_abs_time,
                    last_pulse_op,
                )
            elif other := _exists_pulse_starting_before_current_end(
                abs_times_and_operations,
                i,
            ):
                abs_time_other = other[0]
                op_other = other[1]
                _raise_if_pulses_overlap_on_same_port_clock(
                    abs_time_op,
                    op,
                    abs_time_other,
                    op_other,
                )
        if op.valid_pulse:
            last_pulse_end = _operation_end((abs_time_op, op))
            last_pulse_abs_time = abs_time_op
            last_pulse_op = op


def _exists_pulse_starting_before_current_end(
    abs_times_and_operations: list[tuple[float, Operation]], current_idx: int
) -> tuple[float, Operation] | Literal[False]:
    current_end = _operation_end(abs_times_and_operations[current_idx])
    for i in range(current_idx + 1, len(abs_times_and_operations)):
        abs_time = abs_times_and_operations[i][0]
        operation = abs_times_and_operations[i][1]
        # Schedulable starting at the exact time a previous one ends does not count as
        # overlapping. Subtract 1e-10 for possible floating point errors.
        if abs_time >= current_end - 1e-10:
            return False
        if operation.valid_pulse or operation.has_voltage_offset:
            return abs_time, operation
    return False


def _raise_if_pulses_overlap_on_same_port_clock(
    abs_time_a: float,
    op_a: Operation,
    abs_time_b: float,
    op_b: Operation,
) -> None:
    """
    Raise an error if any pulse operations overlap on the same port-clock.

    A pulse here means a waveform or a voltage offset.
    """
    pulse_start_ends_per_port_a = _get_pulse_start_ends(abs_time_a, op_a)
    pulse_start_ends_per_port_b = _get_pulse_start_ends(abs_time_b, op_b)
    common_ports = set(pulse_start_ends_per_port_a.keys()) & set(pulse_start_ends_per_port_b.keys())
    for port_clock in common_ports:
        start_a, end_a = pulse_start_ends_per_port_a[port_clock]
        start_b, end_b = pulse_start_ends_per_port_b[port_clock]
        if (
            start_b < start_a < end_b
            or start_b < end_a < end_b
            or start_a < start_b < end_a
            or start_a < end_b < end_a
        ):
            raise RuntimeError(
                f"{op_a} at t={abs_time_a} and {op_b} at t="
                f"{abs_time_b} contain pulses with voltage offsets that "
                "overlap in time on the same port and clock. This leads to undefined "
                "behaviour."
            )


def _get_pulse_start_ends(abs_time: float, operation: Operation) -> dict[str, tuple[float, float]]:
    pulse_start_ends_per_port: dict[str, tuple[float, float]] = defaultdict(
        lambda: (np.inf, -np.inf)
    )
    for pulse_info in operation["pulse_info"]:
        if pulse_info.get("wf_func") is None and pulse_info.get("offset_path_I") is None:
            continue
        prev_start, prev_end = pulse_start_ends_per_port[
            f"{pulse_info['port']}_{pulse_info['clock']}"
        ]
        new_start = abs_time + pulse_info["t0"]
        new_end = new_start + pulse_info["duration"]
        prev_start = min(new_start, prev_start)
        prev_end = max(new_end, prev_end)
        pulse_start_ends_per_port[f"{pulse_info['port']}_{pulse_info['clock']}"] = (
            prev_start,
            prev_end,
        )
    return pulse_start_ends_per_port


def _operation_end(abs_time_and_operation: tuple[float, Operation]) -> float:
    abs_time = abs_time_and_operation[0]
    operation = abs_time_and_operation[1]
    return abs_time + operation.duration


def _check_nco_operations_on_nco_time_grid(schedule: Schedule) -> Schedule:
    """
    Check whether NCO operations are on the 4ns time grid _and_ sub-schedules (including
    control-flow) containing NCO operations start/end on the 4 ns time grid.
    """
    _check_nco_operations_on_nco_time_grid_recursively(schedule)
    return schedule


def _check_nco_operations_on_nco_time_grid_recursively(
    operation: Operation | Schedule,
    schedulable: Schedulable | None = None,
    parent_control_flow_op: ControlFlowOperation | None = None,
) -> bool:
    """
    Check whether NCO operations, or Schedules/ControlFlowOperations containing NCO
    operations, align with the NCO grid.

    Parameters
    ----------
    operation : Operation | Schedule
        The Operation or Schedule to be checked.
    schedulable : Schedulable | None, optional
        The Schedulable the operation is a part of. None if it is the top-level
        Schedule.
    parent_control_flow_op : ControlFlowOperation | None, optional
        The ControlFlowOperation that the operation is part of, if any. This is used to
        create the correct error message.

    Returns
    -------
    bool
        True if the operation is a, or contains NCO operation(s), else False.

    """
    contains_nco_op = False
    if isinstance(operation, Schedule):
        for sub_schedulable in operation.schedulables.values():
            sub_operation = operation.operations[sub_schedulable["operation_id"]]
            contains_nco_op = contains_nco_op or _check_nco_operations_on_nco_time_grid_recursively(
                operation=sub_operation,
                schedulable=sub_schedulable,
                parent_control_flow_op=None,
            )
        if contains_nco_op:
            _check_nco_grid_timing(
                operation=operation,
                schedulable=schedulable,
                parent_control_flow_op=parent_control_flow_op,
            )
    elif isinstance(operation, ControlFlowOperation):
        contains_nco_op = _check_nco_operations_on_nco_time_grid_recursively(
            operation=operation.body,
            schedulable=schedulable,
            parent_control_flow_op=operation,
        )
        if contains_nco_op:
            _check_nco_grid_timing(
                operation=operation,
                schedulable=schedulable,
                parent_control_flow_op=parent_control_flow_op,
            )
    elif _is_nco_operation(operation):
        _check_nco_grid_timing(
            operation=operation,
            schedulable=schedulable,
            parent_control_flow_op=parent_control_flow_op,
        )
        contains_nco_op = True
    return contains_nco_op


def _check_nco_grid_timing(
    operation: Operation | Schedule,
    schedulable: Schedulable | None,
    parent_control_flow_op: ControlFlowOperation | None = None,
) -> None:
    """
    Assumes `operation` is a, or contains NCO operation(s), and checks the alignment
    of the `operation` with the NCO grid.
    """
    abs_time = 0 if schedulable is None else schedulable["abs_time"]
    start_time = abs_time + operation.get("t0", 0)
    if isinstance(operation, Schedule) and parent_control_flow_op is None:
        assert operation.duration is not None
        try:
            to_grid_time(start_time, constants.NCO_TIME_GRID)
            to_grid_time(operation.duration, constants.NCO_TIME_GRID)
        except ValueError as e:
            raise NcoOperationTimingError(
                f"Schedule {operation.name}, which contains NCO related operations, "
                f"cannot start at t={round(start_time*1e9)} ns and end at "
                f"t={round((start_time+operation.duration)*1e9)} ns. This schedule "
                f"must start and end on the {constants.NCO_TIME_GRID} ns time grid."
            ) from e

    elif isinstance(operation, ControlFlowOperation) or parent_control_flow_op is not None:
        assert operation.duration is not None
        try:
            to_grid_time(start_time, constants.NCO_TIME_GRID)
            to_grid_time(operation.duration, constants.NCO_TIME_GRID)
        except ValueError as e:
            if parent_control_flow_op is None:
                raise NcoOperationTimingError(
                    f"ControlFlow operation {operation.name}, which contains NCO "
                    f"related operations, cannot start at t="
                    f"{round(start_time*1e9)} ns and end at t="
                    f"{round((start_time+operation.duration)*1e9)} ns. This operation "
                    f"must start and end on the {constants.NCO_TIME_GRID} ns time grid."
                ) from e
            else:
                raise NcoOperationTimingError(
                    f"ControlFlow operation {parent_control_flow_op.name}, starting at "
                    f"t={round(start_time*1e9)} ns and ending at t="
                    f"{round((start_time+parent_control_flow_op.duration)*1e9)} ns, "
                    "contains NCO related operations that may not be aligned with the "
                    f"{constants.NCO_TIME_GRID} ns time grid. Please make sure all "
                    "iterations and/or branches start and end on the "
                    f"{constants.NCO_TIME_GRID} ns time grid."
                ) from e

    # Type: guaranteed to be Operation at this point.
    elif _is_nco_operation(operation):  # type: ignore
        try:
            to_grid_time(start_time, constants.NCO_TIME_GRID)
        except ValueError as e:
            raise NcoOperationTimingError(
                f"NCO related operation {operation} cannot start at "
                f"t={round(start_time*1e9)} ns. This operation must be on the "
                f"{constants.NCO_TIME_GRID} ns time grid."
            ) from e


def _is_nco_operation(operation: Operation) -> bool:
    return isinstance(operation, (ShiftClockPhase, ResetClockPhase, SetClockFrequency))
