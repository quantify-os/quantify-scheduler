# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

import itertools
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple, Type, Union

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
from quantify_scheduler.backends.qblox.exceptions import NcoOperationTimingError
from quantify_scheduler.backends.qblox.helpers import (
    _generate_legacy_hardware_config,
    _generate_new_style_hardware_compilation_config,
    assign_pulse_and_acq_info_to_devices,
    find_channel_names,
    to_grid_time,
)
from quantify_scheduler.backends.qblox.operations import long_square_pulse
from quantify_scheduler.backends.qblox.operations.pulse_library import LatchReset
from quantify_scheduler.backends.types.common import (
    Connectivity,
    HardwareCompilationConfig,
    HardwareDescription,
    HardwareOptions,
)
from quantify_scheduler.backends.types.qblox import (
    ClusterDescription,
    QbloxHardwareDescription,
    QbloxHardwareOptions,
    QCMDescription,
    QCMRFDescription,
    QRMDescription,
    QRMRFDescription,
    QTMDescription,
    _ClusterCompilerConfig,
    _ClusterModuleCompilerConfig,
    _LocalOscillatorCompilerConfig,
)
from quantify_scheduler.helpers.schedule import _extract_port_clocks_used
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    ControlFlowOperation,
    LoopOperation,
)
from quantify_scheduler.operations.operation import Operation
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
            replacing_operation = _replace_long_square_pulses_recursively(
                inner_operation
            )
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
        replacing_operation = _replace_long_square_pulses(
            operation, square_pulse_idx_to_replace
        )
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
    accumulator: List[Tuple[float, Operation]],
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
        return (
            self.start <= operation_timing_info.end
            and operation_timing_info.start <= self.end
        )


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
        conditional_address_map[
            feedback_trigger_label
        ].portclocks |= _extract_port_clocks_used(operation.body)
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
            if (
                feedback_trigger_label := info.get("feedback_trigger_label")
            ) is not None:
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
            if (
                feedback_trigger_label := info.get("feedback_trigger_label")
            ) is not None:
                at = (
                    abs_time_relative_to_schedule
                    + info["t0"]
                    + constants.MAX_MIN_INSTRUCTION_WAIT
                )
                for portclock in conditional_address_map[
                    feedback_trigger_label
                ].portclocks:
                    schedulable = schedule.add(LatchReset(portclock=portclock))
                    schedulable.data["abs_time"] = at


def compile_conditional_playback(schedule: Schedule, **_: Any) -> Schedule:
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

    all_conditional_acqs_and_control_flows: List[Tuple[float, Operation]] = list()
    _all_conditional_acqs_and_control_flows_and_latch_reset(
        schedule, 0, all_conditional_acqs_and_control_flows
    )
    all_conditional_acqs_and_control_flows.sort(
        key=lambda time_op_sched: time_op_sched[0]
    )

    current_ongoing_conditional_acquire = None
    for (
        time,
        operation,
    ) in all_conditional_acqs_and_control_flows:
        if isinstance(operation, ConditionalOperation):
            if current_ongoing_conditional_acquire is None:
                raise RuntimeError(
                    f"Conditional control flow, ``{operation}``,  found without a preceding "
                    "Conditional acquisition. Please ensure that the preceding acquisition or Measure "
                    "is conditional, by passing `feedback_trigger_label=qubit_name` to the "
                    "corresponding operation, e.g.\n\n"
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
    # Extract the old-style hardware config from the CompilationConfig
    hardware_cfg = _generate_legacy_hardware_config(
        schedule=schedule, compilation_config=config
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

    # Apply software distortion corrections. Hardware distortion corrections are
    # compiled into the compiler container that follows.
    if (
        distortion_corrections := hardware_cfg.get("distortion_corrections")
    ) is not None:
        replacing_schedule = apply_software_distortion_corrections(
            schedule, distortion_corrections
        )
        if replacing_schedule is not None:
            schedule = replacing_schedule

    _add_clock_freqs_to_set_clock_frequency(schedule)

    validate_non_overlapping_stitched_pulse(schedule)

    _check_nco_operations_on_nco_time_grid(schedule)

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        schedule, hardware_cfg
    )

    assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        hardware_cfg=hardware_cfg,
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
            name="qblox_compile_conditional_playback",
            compilation_func=compile_conditional_playback,
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
    def _validate_connectivity_channel_names(self) -> QbloxHardwareCompilationConfig:
        if isinstance(self.connectivity, Connectivity):
            self._validate_channel_names_new_config()

        else:
            self._validate_channel_names_old_config()

        return self

    def _validate_channel_names_old_config(self) -> None:
        instrument_type_to_description = {
            description.get_instrument_type(): description
            for description in [
                QCMDescription,
                QRMDescription,
                QCMRFDescription,
                QRMRFDescription,
                QTMDescription,
            ]
        }

        for cluster_name, cluster_config in find_qblox_instruments(
            hardware_config=self.connectivity, instrument_type="Cluster"
        ).items():
            for instrument_type, class_ in instrument_type_to_description.items():
                for module_name, module in find_qblox_instruments(
                    hardware_config=cluster_config, instrument_type=instrument_type
                ).items():
                    try:
                        class_.validate_channel_names(find_channel_names(module))
                    except ValueError as exc:
                        # Add some information to the raised exception. The original exception
                        # is included in the message because pydantic suppresses the traceback.
                        raise ValueError(
                            "Error validating channel names for "
                            f"{cluster_name}.{module_name} ({instrument_type}). Full "
                            f"error message:\n{exc}\n\nSupported names for "
                            f"{instrument_type}:\n{class_.get_valid_channels()}"
                        ) from exc

    def _validate_channel_names_new_config(self) -> None:
        module_name_to_channel_names_map: dict[tuple[str, str], set[str]] = defaultdict(
            set
        )
        for node in self.connectivity.graph.nodes:
            try:
                cluster_name, module_name, channel_name = node.split(".")
            except ValueError:
                continue

            if isinstance(
                self.hardware_description.get(cluster_name), ClusterDescription
            ):
                module_name_to_channel_names_map[cluster_name, module_name].add(
                    channel_name
                )

        for (
            cluster_name,
            module_name,
        ), channel_names in module_name_to_channel_names_map.items():
            module_idx = int(re.search(r"module(\d+)", module_name).group(1))
            try:
                self.hardware_description[cluster_name].modules[
                    module_idx
                ].validate_channel_names(channel_names)
            except ValueError as exc:
                instrument_type = (
                    self.hardware_description[cluster_name]
                    .modules[module_idx]
                    .instrument_type
                )
                valid_channels = (
                    self.hardware_description[cluster_name]
                    .modules[module_idx]
                    .get_valid_channels()
                )
                # Add some information to the raised exception. The original exception
                # is included in the message because pydantic suppresses the traceback.
                raise ValueError(
                    f"Error validating channel names for {cluster_name}.{module_name} "
                    f"({instrument_type}). Full error message:\n{exc}\n\nSupported "
                    f"names for {instrument_type}:\n{valid_channels}."
                ) from exc

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
                    channels_with_laser.append(source)

                # Sometimes source and target appear swapped. This block can be removed
                # after making graph directed. (SE-477)
                elif len(target.split(".")) == 3 and "laser" in source:
                    channels_with_laser.append(target)

            # Find mix_lo value in hardware description
            for channel_path in channels_with_laser:
                cluster, module, channel = channel_path.split(".")
                module_idx = int(module.replace("module", ""))
                module_description = (
                    self.hardware_description[cluster]
                    .modules[module_idx]
                    .model_dump(exclude_unset=True)
                )
                channel_description = module_description.get(channel, None)
                if channel_description is not None:
                    mix_lo = channel_description.get("mix_lo", None)
                    # FIXME: https://qblox.atlassian.net/browse/SE-490
                    if mix_lo is not None and mix_lo is False:
                        warnings.warn(
                            "Using `mix_lo=False` in channels coupled to lasers might cause ill-behavior. "
                            "Please use quantify_scheduler=0.20.1.",
                            FutureWarning,
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

    # TODO: Remove this validator (and test) when substituting `networkx.Graph` with `networkx.DiGraph` (SE-477)
    #       (introduced to find errors during conversion of hardware config versions)
    @model_validator(mode="after")
    def _validate_connectivity_graph_structure(self) -> QbloxHardwareCompilationConfig:
        """Validate connectivity graph structure."""
        EXC_MESSAGE = (
            "Channels, rf-signals of iq mixers and lo outputs must be source nodes "
            "(left), and ports and if/lo-signals of iq mixers must be target nodes (right)."
        )

        def _is_channel(node: str) -> bool:
            return bool(re.match(r"^\w+\.module\d+\.\w+$", node))

        def _is_iq_mixer(node: str, signal_type: str) -> bool:
            return bool(re.match(rf"^iq_mixer_lo\d+\.{signal_type}$", node))

        def _is_lo(node: str) -> bool:
            return bool(re.match(r"^lo\d+\.output$", node))

        def _is_port(node: str) -> bool:
            # Exclude special case of `optical_control` ports in nv centers. This will be fixed
            # when making connectivity graph directed. (SE-477)
            return bool(len(node.split(":")) == 2 and "optical_control" not in node)

        if isinstance(self.connectivity, Connectivity):
            for edge in self.connectivity.graph.edges:
                source, target = edge

                if (
                    _is_port(source)
                    or _is_iq_mixer(source, "if")
                    or _is_iq_mixer(source, "lo")
                ):
                    raise ValueError(
                        f"Node {source} in connectivity graph is a source. {EXC_MESSAGE}"
                    )

                if _is_channel(target) or _is_lo(target) or _is_iq_mixer(target, "rf"):
                    raise ValueError(
                        f"Node {target} in connectivity graph is a target. {EXC_MESSAGE}"
                    )

        return self

    def _extract_instrument_compiler_configs(  # noqa: PLR0912, PLR0915
        self, portclocks_used: set[tuple]
    ) -> Dict[str, DataStructure]:
        """
        Extract an instrument compiler config for each instrument mentioned in ``hardware_description``.
        Each instrument config has a similar structure than ``QbloxHardwareCompilationConfig``, but
        contains only the settings related to their related instrument. Each config must contain at least one
        portclock referenced in ``portclocks_used``, otherwise the config is deleted.
        """
        compiler_configs = {}

        # Extract instrument hardware descriptions
        for (
            instrument_name,
            instrument_description,
        ) in self.hardware_description.items():
            if instrument_description.instrument_type == "Cluster":
                cluster_config = _ClusterCompilerConfig(
                    instrument_type="Cluster",
                    ref=instrument_description.ref,
                    sequence_to_file=instrument_description.sequence_to_file,
                )
                for (
                    module_idx,
                    module_description,
                ) in instrument_description.modules.items():
                    cluster_config.modules[module_idx] = _ClusterModuleCompilerConfig(
                        instrument_type=instrument_description.modules[
                            module_idx
                        ].instrument_type,
                        hardware_description=module_description,
                        hardware_options={},
                        connectivity={"graph": []},
                    )

                compiler_configs[instrument_name] = cluster_config

            elif instrument_description.instrument_type == "LocalOscillator":
                compiler_configs[instrument_name] = _LocalOscillatorCompilerConfig(
                    instrument_type="LocalOscillator",
                    hardware_description=instrument_description,
                    frequency=None,
                )

        # Distribute connectivity edges to the different cluster modules
        # and get portclock to module / lo_name mappings
        portclock_to_path = {}

        mixers_and_ports = []
        for edge in self.connectivity.graph.edges:
            instr_source = edge[0].split(".")[0]
            target_0 = edge[1].split(".")[0]

            edge_portclocks = [pc for pc in portclocks_used if target_0 in pc[0]]
            if len(edge_portclocks) > 0:
                if self.hardware_description[instr_source].instrument_type == "Cluster":
                    module_idx = int(edge[0].split(".")[1].replace("module", ""))
                    compiler_configs[instr_source].modules[
                        module_idx
                    ].connectivity.graph.add_edge(*edge)
                    for pc in edge_portclocks:
                        portclock_to_path[f"{pc[0]}-{pc[1]}"] = edge[0]
                else:
                    mixers_and_ports.append(edge)

        for edge_with_mixer in mixers_and_ports:
            edge_with_lo, edge_with_channel = None, None

            for edge in self.connectivity.graph.edges:
                instr_source = edge[0].split(".")[0]
                target_0 = edge[1].split(".")[0]

                if target_0 in edge_with_mixer[0]:
                    if (
                        self.hardware_description[instr_source].instrument_type
                        == "Cluster"
                    ):
                        cluster_name = instr_source
                        module_idx = int(edge[0].split(".")[1].replace("module", ""))
                        edge_with_channel = edge
                    else:
                        edge_with_lo = edge

                if edge_with_channel and edge_with_lo:
                    compiler_configs[cluster_name].modules[
                        module_idx
                    ].connectivity.graph.add_edges_from(
                        [
                            edge_with_channel,
                            edge_with_lo,
                            edge_with_mixer,
                        ]
                    )

                    channel_name = edge_with_channel[0].split(".")[-1]
                    lo_name = edge_with_lo[0].split(".")[0]
                    compiler_configs[cluster_name].modules[module_idx].channel_to_lo[
                        channel_name
                    ] = lo_name

                    edge_portclocks = [
                        pc for pc in portclocks_used if edge_with_mixer[1] in pc[0]
                    ]
                    for pc in edge_portclocks:
                        portclock = f"{pc[0]}-{pc[1]}"
                        portclock_to_path[portclock] = edge_with_channel[0]
                    break

        # Extract hardware options
        hardware_options = self.hardware_options.model_dump()
        modules_hardware_options = {}

        for (
            instrument_name,
            instrument_description,
        ) in self.hardware_description.items():
            if instrument_description.instrument_type == "Cluster":
                modules_hardware_options[instrument_name] = {}
                for module_idx in instrument_description.modules.keys():
                    modules_hardware_options[instrument_name][module_idx] = {}

        for option, values in hardware_options.items():
            if values is not None:
                for portclock, option_value in values.items():
                    if portclock in portclock_to_path:
                        path = portclock_to_path[portclock]
                        cluster_name, module_name, channel_name = path.split(".")
                        module_idx = int(module_name.replace("module", ""))
                        if not modules_hardware_options[cluster_name][module_idx].get(
                            option
                        ):
                            modules_hardware_options[cluster_name][module_idx][
                                option
                            ] = {}
                        modules_hardware_options[cluster_name][module_idx][option][
                            portclock
                        ] = option_value

                        # Populate LO hardware description
                        channel_to_lo = (
                            compiler_configs[cluster_name]
                            .modules[module_idx]
                            .channel_to_lo
                        )
                        if (
                            option == "modulation_frequencies"
                            and channel_name in channel_to_lo.keys()
                        ):
                            lo_name = channel_to_lo[channel_name]
                            lo_config = compiler_configs[lo_name]
                            if lo_config.hardware_description.instrument_name is None:
                                lo_config.hardware_description.instrument_name = lo_name
                                lo_config.frequency = option_value.get("lo_freq")

        for cluster_name, value in modules_hardware_options.items():
            for module_idx, options in value.items():
                compiler_configs[cluster_name].modules[module_idx].hardware_options = (
                    QbloxHardwareOptions.model_validate(options)
                )

        # Distribute portclock paths and channel LOs between the different clusters and modules
        for portclock, path in portclock_to_path.items():
            cluster_name = path.split(".")[0]
            module_idx = int(path.split(".")[1].replace("module", ""))
            portclock_tuple = tuple(portclock.split("-"))
            compiler_configs[cluster_name].portclock_to_path[portclock_tuple] = path
            compiler_configs[cluster_name].modules[module_idx].portclock_to_path[
                portclock_tuple
            ] = path

        # Delete empty configs of unused modules
        cluster_names = []
        unused_modules = defaultdict(list)
        for instrument_name, cfg in compiler_configs.items():
            if cfg.instrument_type == "Cluster":
                cluster_names.append(instrument_name)
                for module_idx in cfg.modules.keys():
                    if len(cfg.modules[module_idx].portclock_to_path) == 0:
                        unused_modules[instrument_name].append(module_idx)

        for cluster_name, indices in unused_modules.items():
            for module_idx in indices:
                del compiler_configs[cluster_name].modules[module_idx]

        # Delete empty configs of unused clusters
        unused_clusters = []
        for cluster_name in cluster_names:
            if len(compiler_configs[cluster_name].modules) == 0:
                unused_clusters.append(cluster_name)

        for cluster_name in unused_clusters:
            del compiler_configs[cluster_name]

        return compiler_configs


def _all_abs_times_ops_with_voltage_offsets_pulses(
    operation: Operation | Schedule,
    time_offset: float,
    accumulator: List[Tuple[float, Operation]],
) -> None:
    if isinstance(operation, ScheduleBase):
        for schedulable in operation.schedulables.values():
            abs_time = schedulable["abs_time"]
            inner_operation = operation.operations[schedulable["operation_id"]]
            _all_abs_times_ops_with_voltage_offsets_pulses(
                inner_operation, time_offset + abs_time, accumulator
            )
    elif isinstance(operation, LoopOperation):
        for i in range(operation.data["control_flow_info"]["repetitions"]):
            _all_abs_times_ops_with_voltage_offsets_pulses(
                operation.body,
                time_offset + i * operation.body.duration,
                accumulator,
            )
    elif isinstance(operation, ConditionalOperation):
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
            _add_clock_freqs_to_set_clock_frequency(
                operation=inner_operation, schedule=operation
            )
    elif isinstance(operation, ControlFlowOperation):
        _add_clock_freqs_to_set_clock_frequency(
            operation=operation.body, schedule=schedule
        )
    else:
        for pulse_info in operation["pulse_info"]:
            clock_freq = schedule.resources.get(pulse_info["clock"], {}).get(
                "freq", None
            )

            if "clock_freq_new" in pulse_info:
                pulse_info.update(
                    {
                        "clock_freq_old": clock_freq,
                    }
                )


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
    abs_times_and_operations: list[Tuple[float, Operation]] = list()
    _all_abs_times_ops_with_voltage_offsets_pulses(
        schedule, 0, abs_times_and_operations
    )
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
    abs_times_and_operations: list[Tuple[float, Operation]], current_idx: int
) -> Tuple[float, Operation] | Literal[False]:
    current_end = _operation_end(abs_times_and_operations[current_idx])
    for abs_time, operation in abs_times_and_operations[current_idx + 1 :]:
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
            raise RuntimeError(
                f"{op_a} at t={abs_time_a} and {op_b} at t="
                f"{abs_time_b} contain pulses with voltage offsets that "
                "overlap in time on the same port and clock. This leads to undefined "
                "behaviour."
            )


def _get_pulse_start_ends(
    abs_time: float, operation: Operation
) -> dict[str, tuple[float, float]]:
    pulse_start_ends_per_port: dict[str, tuple[float, float]] = defaultdict(
        lambda: (np.inf, -np.inf)
    )
    for pulse_info in operation["pulse_info"]:
        if (
            pulse_info.get("wf_func") is None
            and pulse_info.get("offset_path_I") is None
        ):
            continue
        prev_start, prev_end = pulse_start_ends_per_port[
            f"{pulse_info['port']}_{pulse_info['clock']}"
        ]
        new_start = abs_time + pulse_info["t0"]
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


def _operation_end(abs_time_and_operation: Tuple[float, Operation]) -> float:
    abs_time = abs_time_and_operation[0]
    operation = abs_time_and_operation[1]
    return abs_time + operation.duration


def _check_nco_operations_on_nco_time_grid(schedule: Schedule, **_: Any) -> Schedule:
    """
    Check whether NCO operations are on the 4ns time grid _and_ sub-schedules (including
    control-flow) containing NCO operations start/end on the 4 ns time grid.
    """
    _check_nco_operations_on_nco_time_grid_recursively(schedule)
    return schedule


def _check_nco_operations_on_nco_time_grid_recursively(
    operation: Operation | Schedule, schedulable: Schedulable | None = None
) -> bool:
    contains_nco_op = False
    if isinstance(operation, Schedule):
        for schedulable in operation.schedulables.values():
            sub_operation = operation.operations[schedulable["operation_id"]]
            contains_nco_op = (
                contains_nco_op
                or _check_nco_operations_on_nco_time_grid_recursively(
                    sub_operation, schedulable
                )
            )
        if contains_nco_op:
            _check_nco_grid_timing(schedulable, operation)
    elif isinstance(operation, ControlFlowOperation):
        contains_nco_op = _check_nco_operations_on_nco_time_grid_recursively(
            operation.body
        )
        if contains_nco_op:
            _check_nco_grid_timing(schedulable, operation)
    elif _is_nco_operation(operation):
        _check_nco_grid_timing(schedulable, operation)
        contains_nco_op = True
    return contains_nco_op


def _check_nco_grid_timing(
    schedulable: Schedulable | None, operation: Operation | Schedule
) -> None:
    abs_time = 0 if schedulable is None else schedulable["abs_time"]
    start_time = abs_time + operation.get("t0", 0)
    if isinstance(operation, Schedule):
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

    elif isinstance(operation, ControlFlowOperation):
        try:
            to_grid_time(start_time, constants.NCO_TIME_GRID)
            to_grid_time(operation.duration, constants.NCO_TIME_GRID)
        except ValueError as e:
            raise NcoOperationTimingError(
                f"ControlFlow operation {operation.name}, which contains NCO related "
                f"operations, cannot start at t={round(start_time*1e9)} ns and end at "
                f"t={round((start_time+operation.duration)*1e9)} ns. This operation "
                f"must start and end on the {constants.NCO_TIME_GRID} ns time grid."
            ) from e

    elif _is_nco_operation(operation):
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
