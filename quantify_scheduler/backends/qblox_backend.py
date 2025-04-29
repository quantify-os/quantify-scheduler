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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, model_validator
from qblox_instruments import InstrumentType

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
from quantify_scheduler.backends.qblox.helpers import (
    _generate_new_style_hardware_compilation_config,
    assign_pulse_and_acq_info_to_devices,
)
from quantify_scheduler.backends.qblox.operations import long_square_pulse
from quantify_scheduler.backends.qblox.operations.pulse_library import (
    LatchReset,
)
from quantify_scheduler.backends.qblox.schedule import CompiledInstructions
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
from quantify_scheduler.enums import DualThresholdedTriggerCountLabels, TriggerCondition
from quantify_scheduler.helpers.generate_acq_channels_data import (
    generate_acq_channels_data,
)
from quantify_scheduler.operations import (
    DualThresholdedTriggerCount,
    ThresholdedAcquisition,
    ThresholdedTriggerCount,
    WeightedThresholdedAcquisition,
)
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    ControlFlowOperation,
    LoopOperation,
)
from quantify_scheduler.schedules.schedule import (
    CompiledSchedule,
    Schedulable,
    Schedule,
    ScheduleBase,
)
from quantify_scheduler.structure.model import DataStructure

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        cls,
        operation: Operation,
        schedulable: Schedulable,
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
class ConditionalInfo:
    """Container for conditional address data."""

    portclocks: set[tuple[str, str]]
    """Port-clocks reading from the trigger address."""
    address: int
    """Trigger address."""
    _trigger_invert: bool | None = None
    _trigger_count: int | None = None

    @property
    def trigger_invert(self) -> bool | None:
        """
        If True, inverts the threshold comparison result when reading from the trigger address
        counter.

        If a ThresholdedTriggerCount acquisition is done with a QRM, this must be set according to
        the condition you are trying to measure (greater than /equal to the threshold, or less than
        the threshold). If it is done with a QTM, this is set to False.
        """
        return self._trigger_invert

    @trigger_invert.setter
    def trigger_invert(self, value: bool) -> None:
        if self._trigger_invert is not None and self._trigger_invert != value:
            raise ValueError(
                f"Trying to set conflicting settings for feedback trigger inversion. Setting "
                f"{value} while the previous value was {self._trigger_invert}. This may happen "
                "because multiple ThresholdedTriggerCount acquisitions with conflicting threshold "
                "settings are scheduled, or ThresholdedTriggerCount acquisitions with the same "
                "feedback trigger label are scheduled on different modules."
            )
        self._trigger_invert = value

    @property
    def trigger_count(self) -> int | None:
        """
        The sequencer trigger address counter threshold.

        If a ThresholdedTriggerCount acquisition is done with a QRM, this must be set to the counts
        threshold. If it is done with a QTM, this is set to 1.
        """
        return self._trigger_count

    @trigger_count.setter
    def trigger_count(self, value: int) -> None:
        if self._trigger_count is not None and self._trigger_count != value:
            raise ValueError(
                f"Trying to set conflicting settings for feedback trigger address count threshold. "
                f"Setting {value} while the previous value was {self._trigger_count}. This may "
                "happen because multiple ThresholdedTriggerCount acquisitions with conflicting "
                "threshold settings are scheduled, or ThresholdedTriggerCount acquisitions with "
                "the same feedback trigger label are scheduled on different modules."
            )
        self._trigger_count = value


def _get_module_type(
    port: str, clock: str, compilation_config: CompilationConfig
) -> InstrumentType:
    config = deepcopy(compilation_config)  # _extract_... call modifies the config
    assert isinstance(config.hardware_compilation_config, QbloxHardwareCompilationConfig)
    device_cfgs = config.hardware_compilation_config._extract_instrument_compilation_configs(
        {(port, clock)}
    )
    port_clock_str = f"{port}-{clock}"
    for device_cfg in device_cfgs.values():
        if (
            isinstance(device_cfg, _ClusterCompilationConfig)
            and port_clock_str in device_cfg.portclock_to_path
        ):
            module_idx = device_cfg.portclock_to_path[port_clock_str].module_idx
            module_type = device_cfg.hardware_description.modules[module_idx].instrument_type
            break
    else:
        raise KeyError(
            f"Could not determine module type associated with port-clock {port_clock_str}"
        )
    return InstrumentType(module_type)


def _update_conditional_info_from_acquisition(
    acq_info: dict[str, Any],
    cond_info: defaultdict[str, ConditionalInfo],
    compilation_config: CompilationConfig,
) -> None:
    if acq_info["protocol"] in [
        ThresholdedAcquisition.__name__,
        WeightedThresholdedAcquisition.__name__,
    ]:
        label = acq_info["feedback_trigger_label"]
        acq_info["feedback_trigger_address"] = cond_info[label].address
    elif acq_info["protocol"] == ThresholdedTriggerCount.__name__:
        try:
            TriggerCondition(acq_info["thresholded_trigger_count"]["condition"])
        except (ValueError, TypeError):
            raise ValueError(
                f"Trigger condition {acq_info['thresholded_trigger_count']['condition']} is not "
                "supported."
            ) from None

        label = acq_info["feedback_trigger_label"]
        acq_info["feedback_trigger_address"] = cond_info[label].address
        module_type = _get_module_type(acq_info["port"], acq_info["clock"], compilation_config)
        if module_type == InstrumentType.QRM:
            # The QRM sends a trigger on every count. Therefore, any sequencers doing conditional
            # playback based on thresholded trigger count need to respond only when the count
            # threshold is breached.
            cond_info[label].trigger_invert = (
                acq_info["thresholded_trigger_count"]["condition"] == TriggerCondition.LESS_THAN
            )
            cond_info[label].trigger_count = acq_info["thresholded_trigger_count"]["threshold"]
        elif module_type == InstrumentType.QTM:
            # The QTM sends a trigger only after the acquisition is done, to a trigger address based
            # on the "trigger condition" (this is handled in acquisition info). Therefore, if a
            # QTM acquires, we check for only 0 or 1 triggers.
            cond_info[label].trigger_invert = False
            cond_info[label].trigger_count = 1
    elif acq_info["protocol"] == DualThresholdedTriggerCount.__name__:
        if "feedback_trigger_addresses" not in acq_info:
            acq_info["feedback_trigger_addresses"] = {}
        for kind in DualThresholdedTriggerCountLabels:  # type: ignore
            if (label := acq_info["feedback_trigger_labels"].get(kind)) is None:
                continue
            module_type = _get_module_type(acq_info["port"], acq_info["clock"], compilation_config)
            if module_type != InstrumentType.QTM:
                raise RuntimeError(
                    f"{DualThresholdedTriggerCount.__name__} cannot be scheduled on a module of "
                    f"type {module_type}."
                )

            acq_info["feedback_trigger_addresses"][kind] = cond_info[label].address

            cond_info[label].trigger_invert = False
            cond_info[label].trigger_count = 1
    else:
        raise ValueError(
            f"Error evaluating unknown thresholded acquisition type {acq_info['protocol']}"
        )


def _set_conditional_info_map(
    operation: Operation | Schedule,
    conditional_info_map: defaultdict[str, ConditionalInfo],
    compilation_config: CompilationConfig,
) -> None:
    if isinstance(operation, ScheduleBase):
        schedulables = list(operation.schedulables.values())
        for schedulable in schedulables:
            inner_operation = operation.operations[schedulable["operation_id"]]
            _set_conditional_info_map(
                operation=inner_operation,
                conditional_info_map=conditional_info_map,
                compilation_config=compilation_config,
            )
    elif isinstance(operation, ConditionalOperation):
        # Store `feedback_trigger_address` in the pulse that corresponds
        # to a conditional control flow.
        # Note, we do not allow recursive conditional calls, so no need to
        # go recursively into conditionals.
        control_flow_info: dict = operation.data["control_flow_info"]
        feedback_trigger_label: str = control_flow_info["feedback_trigger_label"]
        control_flow_info["feedback_trigger_address"] = conditional_info_map[
            feedback_trigger_label
        ].address
        control_flow_info["feedback_trigger_invert"] = conditional_info_map[
            feedback_trigger_label
        ].trigger_invert
        control_flow_info["feedback_trigger_count"] = conditional_info_map[
            feedback_trigger_label
        ].trigger_count

        conditional_info_map[
            feedback_trigger_label
        ].portclocks |= operation.body.get_used_port_clocks()

    elif isinstance(operation, ControlFlowOperation):
        _set_conditional_info_map(
            operation=operation.body,
            conditional_info_map=conditional_info_map,
            compilation_config=compilation_config,
        )
    elif operation.valid_acquisition and operation.is_conditional_acquisition:
        # Store `feedback_trigger_address` in the correct acquisition, so that it can
        # be passed to the correct Sequencer via ``SequencerSettings``.
        acq_info = operation["acquisition_info"]
        for info in acq_info:
            if info["protocol"] in (
                ThresholdedAcquisition.__name__,
                ThresholdedTriggerCount.__name__,
                DualThresholdedTriggerCount.__name__,
            ):
                _update_conditional_info_from_acquisition(
                    acq_info=info,
                    cond_info=conditional_info_map,
                    compilation_config=compilation_config,
                )


def _insert_latch_reset(
    operation: Operation | Schedule,
    abs_time_relative_to_schedule: float,
    schedule: Schedule,
    conditional_info_map: defaultdict[str, ConditionalInfo],
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
                conditional_info_map=conditional_info_map,
            )
    elif isinstance(operation, LoopOperation):
        _insert_latch_reset(
            operation=operation.body,
            abs_time_relative_to_schedule=abs_time_relative_to_schedule,
            schedule=schedule,
            conditional_info_map=conditional_info_map,
        )
    elif operation.valid_acquisition and operation.is_conditional_acquisition:
        acq_info = operation["acquisition_info"]
        for info in acq_info:
            if info["protocol"] in (
                ThresholdedAcquisition.__name__,
                ThresholdedTriggerCount.__name__,
            ):
                if (feedback_trigger_label := info.get("feedback_trigger_label")) is not None:
                    at = (
                        abs_time_relative_to_schedule
                        + info["t0"]
                        + constants.MAX_MIN_INSTRUCTION_WAIT
                    )
                    for portclock in conditional_info_map[feedback_trigger_label].portclocks:
                        schedulable = schedule.add(LatchReset(portclock=portclock))
                        schedulable.data["abs_time"] = at
            elif info["protocol"] == DualThresholdedTriggerCount.__name__:
                # With this protocol, multiple labels can be used simultaneously, but for each
                # sequencer doing conditional playback, only one LatchReset is needed each time an
                # acquisition is done, because LatchReset resets _all_ address counters.
                unique_portclocks_and_times = set()
                for kind in DualThresholdedTriggerCountLabels:  # type: ignore
                    if (label := info["feedback_trigger_labels"].get(kind)) is None:
                        continue
                    at_ns = round(
                        (
                            abs_time_relative_to_schedule
                            + info["t0"]
                            + constants.MAX_MIN_INSTRUCTION_WAIT
                        )
                        * 1e9
                    )
                    for portclock in conditional_info_map[label].portclocks:
                        unique_portclocks_and_times.add((portclock, at_ns))
                for portclock, at_ns in unique_portclocks_and_times:
                    schedulable = schedule.add(LatchReset(portclock=portclock))
                    schedulable.data["abs_time"] = at_ns * 1e-9


def compile_conditional_playback(  # noqa: D417
    schedule: Schedule,
    config: CompilationConfig,
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
    conditional_info_map = defaultdict(
        lambda: ConditionalInfo(portclocks=set(), address=address_counter.__next__())
    )
    _set_conditional_info_map(schedule, conditional_info_map, config)
    _insert_latch_reset(schedule, 0, schedule, conditional_info_map)
    address_map_addresses = [a.address for a in conditional_info_map.values()]
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

    currently_active_trigger_labels: set[str] = set()
    for (
        _time,
        operation,
    ) in all_conditional_acqs_and_control_flows:
        if isinstance(operation, ConditionalOperation):
            if (
                operation.data["control_flow_info"]["feedback_trigger_label"]
                not in currently_active_trigger_labels
            ):
                raise RuntimeError(
                    f"Conditional control flow, ``{operation}``,  found without a preceding "
                    "Conditional acquisition. "
                    "Please ensure that the preceding acquisition or Measure is conditional, "
                    "by passing `feedback_trigger_label=dev_element` "
                    "to the corresponding operation, "
                    "e.g.\n\n"
                    "schedule.add(Measure(dev_element, ..., feedback_trigger_label=dev_element))\n"
                )
            else:
                currently_active_trigger_labels.remove(
                    operation.data["control_flow_info"]["feedback_trigger_label"]
                )
        elif operation.valid_acquisition and operation.is_conditional_acquisition:
            acq_info = operation.data["acquisition_info"][0]
            if acq_info["protocol"] == DualThresholdedTriggerCount.__name__:
                # These labels must _all_ be added to `currently_active_trigger_labels` if they are
                # not None and were not present before. This is because, for
                # DualThresholdedTriggerCount, multiple subsequent conditional operations are
                # allowed if they use different labels.
                labels = set(
                    acq_info["feedback_trigger_labels"][key]
                    for key in DualThresholdedTriggerCountLabels  # type: ignore
                    if acq_info["feedback_trigger_labels"][key] is not None
                )
            else:
                labels = {acq_info["feedback_trigger_label"]}
            if labels:
                if currently_active_trigger_labels:
                    raise RuntimeError(
                        "A conditional acquisition was scheduled while some labels from previous "
                        "conditional acquisition(s) were not used yet. To avoid ambiguity, "
                        "conditional playback will only work if all conditional acquisition labels "
                        "are used in a conditional control flow operation before a next "
                        "conditional acquisition is scheduled.\nA conditional acquisition with "
                        f"feedback labels {labels} was scheduled while the labels "
                        f"{currently_active_trigger_labels} were not used yet:\n{operation}"
                    )
                else:
                    currently_active_trigger_labels = labels

    return schedule


def compile_long_square_pulses_to_awg_offsets(  # noqa: D417
    schedule: Schedule,
    config: DataStructure | dict,  # noqa: ARG001
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

    acq_channels_data, schedulable_label_to_acq_index = generate_acq_channels_data(schedule)

    container = compiler_container.CompilerContainer.from_hardware_cfg(schedule, hardware_cfg)

    assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=container.clusters,
        schedulable_label_to_acq_index=schedulable_label_to_acq_index,
    )

    container.prepare()

    compiled_instructions = container.compile(
        debug_mode=config.debug_mode, repetitions=schedule.repetitions
    )
    # Create compiled instructions key if not already present. This can happen if this
    # compilation function is called directly instead of through a `QuantifyCompiler`.
    if "compiled_instructions" not in schedule:
        schedule["compiled_instructions"] = {}

    schedule["compiled_instructions"].update(compiled_instructions)
    schedule["compiled_instructions"] = CompiledInstructions(schedule["compiled_instructions"])

    # Add the acquisition channel data to the schedule data structure.
    schedule["acq_channels_data"] = acq_channels_data

    # Mark the schedule as a compiled schedule.
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
    allow_off_grid_nco_ops: bool | None = Field(
        default=None,
        deprecated="`allow_off_grid_nco_ops` is deprecated as NCO operations can be executed on a "
        "1 ns time grid.",
    )
    """
    Flag to allow NCO operations to play at times that are not aligned with the NCO
    grid.
    """
    compilation_passes: list[SimpleNodeConfig] = [
        SimpleNodeConfig(
            name="crosstalk_compensation",
            compilation_func=crosstalk_compensation,
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
        SimpleNodeConfig(name="qblox_hardware_compile", compilation_func=hardware_compile),
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
                ) from None
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
        cls,
        data: Any,  # noqa: ANN401 deprecated
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
    @classmethod
    def _validate_versioning(cls, config: dict[str, Any]) -> dict[str, Any]:
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
                for cfg in cluster_configs.values():
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

        # For each port-clock combination, we look up the port in the connectivity graph
        # and check what cluster module channel it is connected to.
        # Each port-clock combination can be connected to at most one output channel and
        # either one complex input channel or two real input channels.
        # FIXME (SE-672) the logic below is very difficult to understand and not likely
        # to be correct.
        for port, clocks in port_to_clocks.items():
            if port not in self.connectivity.graph:
                raise KeyError(f"{port} was not found in the connectivity.")
            for port_nbr in self.connectivity.graph.neighbors(port):
                if is_path(port_nbr):
                    path = ChannelPath.from_path(port_nbr)
                    for clock in clocks:
                        repeated_pc = False
                        pc = f"{port}-{clock}"
                        # Add extra channel name for `Measure` operation. This takes place after
                        # the first channel name is added ("if not repeated_pc..." block below)
                        for pc_to_path in cluster_pc_to_path.values():
                            if pc in pc_to_path:
                                if (
                                    path.cluster_name != pc_to_path[pc].cluster_name
                                    or path.module_name != pc_to_path[pc].module_name
                                ):
                                    raise ValueError(
                                        f"Provided channel names for port-clock {pc} are defined "
                                        f"for different modules, but they must be defined in the "
                                        f"same module."
                                    )
                                pc_to_path[pc].add_channel_name_measure(path.channel_name)
                                repeated_pc = True

                        if not repeated_pc:
                            cluster_pc_to_path[path.cluster_name][pc] = path

                elif "." in port_nbr:
                    # In this case, there is an element (e.g. I/Q mixer) between the
                    # port and the channel.
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


SequencerIndex = int
"""
Index of a sequencer.
"""


class AllowedChannels(DataStructure):
    """Allowed channels for a specific sequencer."""

    output: set[str]
    """
    Allowed outputs.

    For example `{"complex_output_0", "real_output_0", `digital_output_0"}`.
    """
    input: set[str]
    """
    Allowed inputs.

    For example `{"complex_input_1", "real_input_1"}`.
    """


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
    parent_config_version: str
    """
    Version of the parent hardware compilation config used.
    """
    sequencer_allowed_channels: dict[SequencerIndex, AllowedChannels]
    """Allowed channels for each sequencer."""

    def _sequencer_to_portclock(self) -> dict[SequencerIndex, str]:
        # This logic assumes that each sequencer uses at most
        # only one output, and one input. This is only a current
        # restriction on quantify, but not in the hardware.
        #
        # See SE-672, with that change, this data and logic might need to also change.
        sequencer_to_portclock: dict[SequencerIndex, str] = {}

        def find_sequencer_index(portclock: str) -> SequencerIndex:
            path: ChannelPath = self.portclock_to_path[portclock]
            for sequencer_index, allowed_channels in self.sequencer_allowed_channels.items():
                if sequencer_index not in sequencer_to_portclock:
                    if path.channel_name_measure is None:
                        # In this case path.channel_name can also be the input channel name.
                        if (
                            path.channel_name in allowed_channels.output
                            or path.channel_name in allowed_channels.input
                        ):
                            return sequencer_index
                    else:
                        # In this case path.channel_name_measure can only be the input channel name.
                        channel_name_measure = path.channel_name_measure or {}
                        if path.channel_name in allowed_channels.output and all(
                            (ch in allowed_channels.input) for ch in channel_name_measure
                        ):
                            return sequencer_index
            raise ValueError(
                f"Cannot reserve a Qblox sequencer for the module "
                f"{self.hardware_description.instrument_type}, portclock {portclock}. "
                f"There are not enough appropriate sequencers for "
                f"the given portclocks and paths. "
                f"Already reserved sequencer indices: {list(sequencer_to_portclock.keys())}. "
                f"Output channel: {path.channel_name}, input channel: {path.channel_name_measure}."
            )

        # To make sure we efficiently reserve the sequencers,
        # we first reserve the ones that use both output and input.
        # Sort to ensure deterministic order sequencer allocation.
        reserved_portclocks: set[str] = set()
        for portclock in sorted(self.portclock_to_path):
            path: ChannelPath = self.portclock_to_path[portclock]
            if path.channel_name_measure is not None:
                sequencer_to_portclock[find_sequencer_index(portclock)] = portclock
                reserved_portclocks.add(portclock)
        for portclock in sorted(self.portclock_to_path):
            if portclock not in reserved_portclocks:
                sequencer_to_portclock[find_sequencer_index(portclock)] = portclock

        return sequencer_to_portclock

    def _extract_sequencer_compilation_configs(
        self,
    ) -> dict[int, _SequencerCompilationConfig]:
        sequencer_configs = {}
        channel_to_lo = {path.channel_name: lo_name for lo_name, path in self.lo_to_path.items()}

        for seq_idx, portclock in self._sequencer_to_portclock().items():
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
                    break
            else:
                # No channel descriptions were set, make a default one based on the channel name.
                if path.channel_name.startswith("real"):
                    hardware_description = RealChannelDescription()
                elif path.channel_name.startswith("complex"):
                    hardware_description = ComplexChannelDescription()
                elif path.channel_name.startswith("digital"):
                    hardware_description = DigitalChannelDescription()
                else:
                    raise ValueError(f"Cannot parse channel name {path.channel_name}")
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
                channel_name_measure=(
                    list(path.channel_name_measure)
                    if path.channel_name_measure is not None
                    else None
                ),
                latency_correction=latency_correction,
                distortion_correction=distortion_correction,
                lo_name=channel_to_lo.get(path.channel_name),
                modulation_frequencies=modulation_frequencies,  # type: ignore
                mixer_corrections=mixer_corrections,
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

    sequencer_allowed_channels: dict[SequencerIndex, AllowedChannels] = {
        sequencer: AllowedChannels(
            output={
                "complex_output_0",
                "complex_output_1",
                "real_output_0",
                "real_output_1",
                "real_output_2",
                "real_output_3",
                "digital_output_0",
                "digital_output_1",
                "digital_output_2",
                "digital_output_3",
            },
            input=set(),
        )
        for sequencer in (0, 1, 2, 3, 4, 5)
    }

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

    sequencer_allowed_channels: dict[SequencerIndex, AllowedChannels] = {
        sequencer: AllowedChannels(
            output={
                "complex_output_0",
                "real_output_0",
                "real_output_1",
                "digital_output_0",
                "digital_output_1",
                "digital_output_2",
                "digital_output_3",
            },
            input={
                "complex_input_0",
                "real_input_0",
                "real_input_1",
            },
        )
        for sequencer in (0, 1, 2, 3, 4, 5)
    }

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:
                if len(path.channel_name_measure) == 1:
                    channel_name_measure = next(iter(path.channel_name_measure))
                    if not (
                        "complex_output" in path.channel_name or "real_output" in path.channel_name
                    ) or not (
                        "complex_input" in channel_name_measure
                        or "real_input" in channel_name_measure
                    ):
                        raise ValueError(
                            f"Found two channel names {path.channel_name} and "
                            f"{channel_name_measure} that are not of the same mode for "
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

    sequencer_allowed_channels: dict[SequencerIndex, AllowedChannels] = {
        sequencer: AllowedChannels(
            output={
                "complex_output_0",
                "complex_output_1",
                "digital_output_0",
                "digital_output_1",
            },
            input=set(),
        )
        for sequencer in (0, 1, 2, 3, 4, 5)
    }

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

    sequencer_allowed_channels: dict[SequencerIndex, AllowedChannels] = {
        sequencer: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "digital_output_0", "digital_output_1"},
            input={"complex_input_0"},
        )
        for sequencer in (0, 1, 2, 3, 4, 5)
    }

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:  # noqa: SIM102
                if not (
                    "complex_output" in path.channel_name
                    and all("complex_input" in ch_name for ch_name in path.channel_name_measure)
                ):
                    raise ValueError(
                        f"Found channel names {path.channel_name} and "
                        f"{path.channel_name_measure} that are not of the same mode for "
                        f"portclock {pc}. Only channel names of the same mode (e.g. "
                        f"`complex_output_0` and `complex_input_0`) are allowed when they share a "
                        f"portclock in QRM_RF modules."
                    )

        return super()._validate_channel_name_measure()


class _QRCCompilationConfig(_ClusterModuleCompilationConfig):
    """QRC-specific configuration values for a :class:`~.ClusterModuleCompiler`."""

    sequencer_allowed_channels: dict[SequencerIndex, AllowedChannels] = {
        0: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_2", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
        1: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_3", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
        2: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_4", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
        3: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_5", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
        4: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_2", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
        5: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_3", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
        6: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_4", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
        7: AllowedChannels(
            output={"complex_output_0", "complex_output_1", "complex_output_5", "digital_output_0"},
            input={"complex_input_0", "complex_input_1"},
        ),
    } | {
        sequencer: AllowedChannels(
            output={
                "complex_output_2",
                "complex_output_3",
                "complex_output_4",
                "complex_output_5",
                "digital_output_0",
            },
            input=set(),
        )
        for sequencer in (8, 9, 10, 11)
    }

    def _validate_channel_name_measure(self) -> None:
        for pc, path in self.portclock_to_path.items():
            if path.channel_name_measure is not None:  # noqa: SIM102
                # TODO: check for whenever there is a complex_input_0 in channel_name_measure,
                # then the output is complex_output_0,
                # and similarly for complex_input_1 and complex_output_1.
                if not (
                    "complex" in path.channel_name
                    and all("complex_input" in ch_name for ch_name in path.channel_name_measure)
                ):
                    raise ValueError(
                        f"Found channel names {path.channel_name} and "
                        f"{path.channel_name_measure} that are not of the same mode for "
                        f"portclock {pc}. Only channel names of the same mode (e.g. "
                        f"`complex_output_0` and `complex_input_0`) are allowed when they share a "
                        f"portclock in QRC modules."
                    )

        return super()._validate_channel_name_measure()


class _QTMCompilationConfig(_ClusterModuleCompilationConfig):
    """QTM-specific configuration values for a :class:`~.ClusterModuleCompiler`."""

    sequencer_allowed_channels: dict[SequencerIndex, AllowedChannels] = {
        sequencer: AllowedChannels(
            output={
                "digital_output_0",
                "digital_output_1",
                "digital_output_2",
                "digital_output_3",
                "digital_output_4",
                "digital_output_5",
                "digital_output_6",
                "digital_output_7",
            },
            input={
                "digital_input_0",
                "digital_input_1",
                "digital_input_2",
                "digital_input_3",
                "digital_input_4",
                "digital_input_5",
                "digital_input_6",
                "digital_input_7",
            },
        )
        for sequencer in (0, 1, 2, 3, 4, 5, 6, 7)
    }

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
    """Local oscillator associated to this sequencer."""
    modulation_frequencies: ModulationFrequencies
    """Modulation frequencies associated to this sequencer."""
    mixer_corrections: QbloxMixerCorrections | None
    """Mixer correction settings."""
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
    parent_config_version: str
    """
    Version of the parent hardware compilation config used.
    """

    module_config_classes: dict = {
        "QCM": _QCMCompilationConfig,
        "QRM": _QRMCompilationConfig,
        "QCM_RF": _QCMRFCompilationConfig,
        "QRM_RF": _QRMRFCompilationConfig,
        "QRC": _QRCCompilationConfig,
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
    complex_output_pcs: dict[str, list[str]] = defaultdict(list)
    real_output_pcs: dict[str, list[str]] = defaultdict(list)
    complex_inputs: list[str] = []
    real_inputs: list[str] = []

    for pc, path in module_config.portclock_to_path.items():
        if ChannelMode.COMPLEX in path.channel_name:
            if "output" in path.channel_name:
                complex_output_pcs[path.channel_name].append(pc)
            else:
                complex_inputs.append(path.channel_name)
        elif ChannelMode.REAL in path.channel_name:
            if "output" in path.channel_name:
                real_output_pcs[path.channel_name].append(pc)
            else:
                real_inputs.append(path.channel_name)

    if module_config.hardware_description.instrument_type in ["QRM", "QRM_RF"]:
        # The specific value of the portclock is not important, as long as it has been
        # defined before for an output in the same module
        if not complex_inputs and not real_inputs:
            for pc_list in complex_output_pcs.values():
                for pc in pc_list:
                    module_config.portclock_to_path[pc].add_channel_name_measure("complex_input_0")
            for pc_list in real_output_pcs.values():
                for pc in pc_list:
                    module_config.portclock_to_path[pc].add_channel_name_measure("real_input_0")
                    module_config.portclock_to_path[pc].add_channel_name_measure("real_input_1")
        elif (
            not complex_inputs
            and len(real_inputs) == 1
            and not complex_output_pcs
            and real_output_pcs
        ):
            for pc_list in real_output_pcs.values():
                for pc in pc_list:
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
    channel_name_measure: None | set[str] = field(init=False, default=None)

    def __hash__(self) -> int:
        return hash(tuple(self.__dataclass_fields__.values()))

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

    def __str__(self) -> str:
        return f"{self.cluster_name}.{self.module_name}.{self.channel_name}"

    @property
    def channel_idx(self) -> int:
        """
        The channel index in the channel name.

        A channel name is always formatted as "type_direction_#" where # is the channel index. This
        property extracts the channel index.
        """
        return int(self.channel_name[self.channel_name.rfind("_") + 1 :])

    def add_channel_name_measure(self, channel_name_measure: str) -> None:
        """Add an extra input channel name for measure operation."""
        # FIXME (SE-672) this method has turned this class from a description of a
        # single input/output on a module ('channel'), into a description of the I/O
        # connections of a sequencer. The fields _except_ 'channel_name_measure'
        # represent a single output channel (which in itself is incorrect, since you can
        # have _two_ "real" output channels), and the `channel_name_measure` are the
        # input channels.
        # We need to decide what the actual purpose of this class is and clean up the
        # usage.

        if self.channel_name_measure is None:
            channel_name = deepcopy(self.channel_name)
            # By convention, the "output" channel name is the main channel name in
            # measure operations
            if "input" in channel_name_measure:
                self.channel_name_measure = {channel_name_measure}
            else:
                self.channel_name = channel_name_measure
                self.channel_name_measure = {channel_name}
        else:
            self.channel_name_measure.add(channel_name_measure)


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
                        f"{last_pulse_abs_time=} and {last_pulse_op} may not be None at this point."
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
