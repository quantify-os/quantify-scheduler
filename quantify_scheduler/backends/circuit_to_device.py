# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compilation backend for quantum-circuit to quantum-device layer."""
from __future__ import annotations

import warnings
from copy import deepcopy
from itertools import permutations
from typing import TYPE_CHECKING, Sequence, overload

import numpy as np

from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    DeviceCompilationConfig,
    OperationCompilationConfig,
)
from quantify_scheduler.operations.control_flow_library import ControlFlowOperation
from quantify_scheduler.operations.pulse_compensation_library import (
    Port,
    PulseCompensation,
)
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import Schedulable, Schedule, ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler.operations.operation import Operation


def compile_circuit_to_device_with_config_validation(
    schedule: Schedule,
    config: CompilationConfig,
) -> Schedule:
    """
    Add pulse information to all gates in the schedule.

    Before calling this function, the schedule can contain abstract operations (gates or
    measurements). This function adds pulse and acquisition information with respect to
    ``config`` as they are expected to arrive to device (latency or distortion corrections
    are not taken into account).

    From a point of view of :ref:`sec-compilation`, this function converts a schedule
    defined on a quantum-circuit layer to a schedule defined on a quantum-device layer.

    Parameters
    ----------
    schedule
        The schedule to be compiled.
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`, of
        which only the :attr:`.CompilationConfig.device_compilation_config`
        is used in this compilation step.

    Returns
    -------
    :
        The modified ``schedule`` with pulse information added to all gates, or the unmodified
        schedule if circuit to device compilation is not necessary.

    """
    device_cfg = DeviceCompilationConfig.model_validate(config.device_compilation_config)

    return _compile_circuit_to_device(
        operation=schedule, device_cfg=device_cfg, device_overrides={}
    )


# It is important that if the operation is a Schedule type, we always return a Schedule.
# Otherwise, we can return an Operation or a Schedule.
@overload
def _compile_circuit_to_device(
    operation: Schedule,
    device_cfg: DeviceCompilationConfig,
    device_overrides: dict,
) -> Schedule: ...
@overload
def _compile_circuit_to_device(
    operation: Operation | Schedule,
    device_cfg: DeviceCompilationConfig,
    device_overrides: dict,
) -> Operation | Schedule: ...
def _compile_circuit_to_device(  # noqa: PLR0911
    operation,
    device_cfg,
    device_overrides,
):
    device_overrides = {
        **operation.data.get("gate_info", {}).get("device_overrides", {}),
        **device_overrides,
    }
    if isinstance(operation, ScheduleBase):
        for inner_op_key in operation.operations:
            operation.operations[inner_op_key] = _compile_circuit_to_device(
                operation=operation.operations[inner_op_key],
                device_cfg=device_cfg,
                device_overrides=device_overrides,
            )
        return operation
    elif isinstance(operation, ControlFlowOperation):
        operation.body = _compile_circuit_to_device(
            operation=operation.body,
            device_cfg=device_cfg,
            device_overrides=device_overrides,
        )
        return operation
    elif isinstance(operation, PulseCompensation):
        return _compile_circuit_to_device_pulse_compensation(operation, device_cfg)
    elif not (operation.valid_pulse or operation.valid_acquisition):
        # If operation is a valid pulse or acquisition it will not attempt to
        # add pulse/acquisition info in the lines below (if operation.valid_gate
        # will not work here for e.g. Measure, which is also a valid
        # acquisition)
        qubits: Sequence[str] = operation.data["gate_info"]["qubits"]
        operation_type: str = operation.data["gate_info"]["operation_type"]

        # single qubit operations
        if len(qubits) == 1:
            return _compile_single_qubit(
                operation=operation,
                qubit=qubits[0],
                operation_type=operation_type,
                device_cfg=device_cfg,
                device_overrides=device_overrides,
            )

        # it is a two-qubit operation if the operation not in the qubit config
        elif len(qubits) == 2 and operation_type not in device_cfg.elements[qubits[0]]:
            return _compile_two_qubits(
                operation=operation,
                qubits=qubits,
                operation_type=operation_type,
                device_cfg=device_cfg,
                device_overrides=device_overrides,
            )
        # we only support 2-qubit operations and single-qubit operations.
        # some single-qubit operations (reset, measure) can be expressed as acting
        # on multiple qubits simultaneously. That is covered through this for-loop.
        else:
            return _compile_multiplexed(
                operation=operation,
                qubits=qubits,
                operation_type=operation_type,
                device_cfg=device_cfg,
                device_overrides=device_overrides,
            )
    else:
        return operation


def set_pulse_and_acquisition_clock(
    schedule: Schedule,
    config: CompilationConfig,
) -> Schedule:
    """
    Ensures that each pulse/acquisition-level clock resource is added to the schedule,
    and validates the given configuration.

    If a pulse/acquisition-level clock resource has not been added
    to the schedule and is present in device_cfg, it is added to the schedule.

    A warning is given when a clock resource has conflicting frequency
    definitions, and an error is raised if the clock resource is unknown.

    Parameters
    ----------
    schedule
        The schedule to be compiled.
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`, of
        which only the :attr:`.CompilationConfig.device_compilation_config`
        is used in this compilation step.

    Returns
    -------
    :
        The modified ``schedule`` with all clock resources added, or the unmodified
        schedule if circuit to device compilation is not necessary.

    Warns
    -----
    RuntimeWarning
        When clock has conflicting frequency definitions.

    Raises
    ------
    RuntimeError
        When operation is not at pulse/acquisition-level.
    ValueError
        When clock frequency is unknown.
    ValueError
        When clock frequency is NaN.

    """
    device_cfg = DeviceCompilationConfig.model_validate(config.device_compilation_config)

    all_clock_freqs: dict[str, float] = {}
    _extract_clock_freqs(schedule, all_clock_freqs)
    for clock, freq in device_cfg.clocks.items():
        if clock in all_clock_freqs:
            _clocks_compatible(clock, device_cfg, all_clock_freqs)
        else:
            all_clock_freqs[clock] = freq

    return _set_pulse_and_acquisition_clock(
        schedule=schedule,
        operation=schedule,
        all_clock_freqs=all_clock_freqs,
        verified_clocks=[],
    )


def _extract_clock_freqs(
    operation: Operation | Schedule, all_clock_freqs: dict[str, float]
) -> None:
    if isinstance(operation, ScheduleBase):
        for inner_operation in operation.operations.values():
            _extract_clock_freqs(operation=inner_operation, all_clock_freqs=all_clock_freqs)
        for clock, clock_data in operation.resources.items():
            if "freq" in clock_data:
                freq = clock_data["freq"]
                if clock in all_clock_freqs and freq != all_clock_freqs[clock]:
                    raise ValueError(
                        f"Inconsistent clock frequencies in the schedule. "
                        f"Clock '{clock}' is defined with frequencies "
                        f"{freq} Hz and {all_clock_freqs[clock]} Hz."
                    )
                all_clock_freqs[clock] = freq
    elif isinstance(operation, ControlFlowOperation):
        _extract_clock_freqs(operation=operation.body, all_clock_freqs=all_clock_freqs)


# It is important that if the operation is a Schedule type, we always return a Schedule.
# Otherwise, we can return an Operation or a Schedule.
@overload
def _set_pulse_and_acquisition_clock(
    schedule: Schedule,
    operation: Schedule,
    all_clock_freqs: dict[str, float],
    verified_clocks: list,
) -> Schedule: ...
@overload
def _set_pulse_and_acquisition_clock(
    schedule: Schedule,
    operation: Operation | Schedule,
    all_clock_freqs: dict[str, float],
    verified_clocks: list,
) -> Operation | Schedule: ...
def _set_pulse_and_acquisition_clock(
    schedule: Schedule,
    operation: Operation | Schedule,
    all_clock_freqs: dict[str, float],
    verified_clocks: list,
) -> Operation | Schedule:
    """
    Ensures that each pulse/acquisition-level clock resource is added to the schedule.

    Parameters
    ----------
    schedule
        The resources from ``operation`` are added to ``schedule``
        if ``operation`` is not a ``Schedule``.
    operation
        The ``operation`` to collect resources from.
    all_clock_freqs
        All clock frequencies.
    verified_clocks
        Already verified clocks.

    Returns
    -------
    :
        The modified ``operation`` with all clock resources added.

    """
    if isinstance(operation, ScheduleBase):
        # verify that required clocks are present; print warning if they are inconsistent
        verified_clocks = []
        for inner_op_key in operation.operations:
            # Only if we have a valid device-level operation, we can assign clocks
            operation.operations[inner_op_key] = _set_pulse_and_acquisition_clock(
                schedule=operation,
                all_clock_freqs=all_clock_freqs,
                operation=operation.operations[inner_op_key],
                verified_clocks=verified_clocks,
            )
    elif isinstance(operation, (ControlFlowOperation, PulseCompensation)):
        operation.body = _set_pulse_and_acquisition_clock(
            schedule=schedule,
            operation=operation.body,
            all_clock_freqs=all_clock_freqs,
            verified_clocks=verified_clocks,
        )
    else:
        _assert_operation_valid_device_level(operation)

        operation_info = operation["pulse_info"] + operation["acquisition_info"]
        clocks_used = set([info["clock"] for info in operation_info])
        for clock in clocks_used:
            if clock in verified_clocks:
                continue
            # raises ValueError if no clock found;
            # enters if condition if clock only in device config
            if not _valid_clock_in_schedule(clock, all_clock_freqs, schedule, operation):
                clock_resource = ClockResource(name=clock, freq=all_clock_freqs[clock])
                schedule.add_resource(clock_resource)
            verified_clocks.append(clock)

    return operation


def _valid_clock_in_schedule(
    clock: str,
    all_clock_freqs: dict[str, float],
    schedule: Schedule,
    operation: Operation,
) -> bool:
    """
    Asserts that valid clock is present. Returns whether clock is already in schedule.

    Parameters
    ----------
    clock
        Name of the clock
    all_clock_freqs
        All clock frequencies
    schedule
        Schedule that potentially has the clock in its resources
    operation
        Quantify operation, to which the clock belongs. Only used for error message.

    Raises
    ------
    ValueError
        Returns ValueError if (i) the device config is the only defined clock and
        contains nan values or (ii) no clock is defined.

    """
    if clock in schedule.resources:
        return True
    else:
        if clock in all_clock_freqs:
            # Clock only in device config
            if np.isnan(all_clock_freqs[clock]).any():
                raise ValueError(
                    f"Operation '{operation}' contains clock '{clock}' with an "
                    f"undefined (initial) frequency; ensure this resource has been "
                    f"added to the schedule or to the device config."
                )
            return False

        # Clock neither in device config nor schedule.
        raise ValueError(
            f"Operation '{operation}' contains an unknown clock '{clock}'; "
            f"ensure this resource has been added to the schedule "
            f"or to the device config."
        )


def _clocks_compatible(
    clock: str,
    device_cfg: DeviceCompilationConfig,
    schedule_clock_resources: dict[str, float],
) -> bool:
    """
    Compare device config and schedule resources for compatibility of their clocks.

    Clocks can be defined in the device_cfg and in the schedule. They are consistent if

    - they have the same value
    - if the clock in the device config is nan (not the other way around)

    These conditions are also generalized to numpy arrays. Arrays of different length
    are only equal if all frequencies in the device config are nan.

    If the clocks are inconsistent, a warning message is emitted.

    Parameters
    ----------
    clock
        Name of the clock found in the device config and schedule
    device_cfg
        Device config containing the ``clock``
    schedule_clock_resources
        All clock resources in the schedule

    Returns
    -------
        True if the clock frequencies are consistent.

    """
    clock_freq_device_cfg = np.asarray(device_cfg.clocks[clock])
    clock_freq_schedule = np.asarray(schedule_clock_resources[clock])

    is_nan = np.isnan(clock_freq_device_cfg)
    if is_nan.all():
        return True

    try:
        is_equal = clock_freq_device_cfg == clock_freq_schedule
    except ValueError:
        return False

    if (np.logical_or(is_nan, is_equal)).all():
        return True

    warnings.warn(
        f"Clock '{clock}' has conflicting frequency definitions: "
        f"{clock_freq_schedule} Hz in the schedule and "
        f"{clock_freq_device_cfg} Hz in the device config. "
        f"The clock is set to '{clock_freq_schedule}'. "
        f"Ensure the schedule clock resource matches the "
        f"device config clock frequency or set the "
        f"clock frequency in the device config to np.NaN "
        f"to omit this warning.",
        RuntimeWarning,
    )
    return False


def _assert_operation_valid_device_level(operation: Operation) -> None:
    """
    Verifies that the operation has been compiled to device level.

    Parameters
    ----------
    operation
        Quantify operation

    """
    if not (operation.valid_pulse or operation.valid_acquisition or operation.has_voltage_offset):
        raise RuntimeError(
            f"Operation '{operation}' is a gate-level operation and must be "
            f"compiled from circuit to device; ensure compilation "
            f"is made in the correct order."
        )


def _compile_multiplexed(
    operation: Operation,
    qubits: Sequence[str],
    operation_type: str,
    device_cfg: DeviceCompilationConfig,
    device_overrides: dict,
) -> Operation | Schedule:
    """
    Compiles gate with multiple qubits.

    Note: it updates the `operation`, if it can directly add pulse representation.
    """
    inner_subschedules: list = []
    operation_has_device_representation: bool = False

    for mux_idx, qubit in enumerate(qubits):
        if qubit not in device_cfg.elements:
            raise ConfigKeyError(
                kind="element",
                missing=qubit,
                allowed=list(device_cfg.elements.keys()),
            )

        element_cfg = device_cfg.elements[qubit]

        if operation_type not in element_cfg:
            raise ConfigKeyError(
                kind="operation",
                missing=operation_type,
                allowed=list(element_cfg.keys()),
            )

        device_op: Operation | Schedule = _get_device_repr_from_cfg_multiplexed(
            operation,
            element_cfg[operation_type],
            mux_idx=mux_idx,
            device_overrides=device_overrides,
        )

        device_op_device_overrides = device_op.data.get("gate_info", {}).get("device_overrides", {})
        new_device_overrides = {**device_op_device_overrides, **device_overrides}

        if isinstance(device_op, ScheduleBase):
            inner_subschedules.append(
                _compile_circuit_to_device(
                    operation=device_op,
                    device_cfg=device_cfg,
                    device_overrides=new_device_overrides,
                )
            )
        else:
            operation.add_device_representation(device_op)
            operation_has_device_representation = True

    if len(inner_subschedules) != 0:
        inner_schedule: Schedule = Schedule(f"Inner schedule for {str(operation)}")
        # All operations in the inner schedule
        # should happen at the same time;
        # this reference time is the start time
        # of the `ref_schedulable` operation / schedule
        ref_schedulable: Schedulable | None = None
        if operation_has_device_representation:
            ref_schedulable = inner_schedule.add(operation)

        for inner_subschedule in inner_subschedules:
            if ref_schedulable is not None:
                inner_schedule.add(
                    operation=inner_subschedule,
                    rel_time=0,
                    ref_op=ref_schedulable,
                    ref_pt="start",
                )
            else:
                ref_schedulable = inner_schedule.add(operation=inner_subschedule)
        return inner_schedule
    else:
        return operation


def _compile_single_qubit(
    operation: Operation,
    qubit: str,
    operation_type: str,
    device_cfg: DeviceCompilationConfig,
    device_overrides: dict,
) -> Operation | Schedule:
    """
    Compiles gate with multiple qubits.

    Note: it updates the `operation`, if it can directly add pulse representation.
    """
    if qubit not in device_cfg.elements:
        raise ConfigKeyError(
            kind="element",
            missing=qubit,
            allowed=list(device_cfg.elements.keys()),
        )

    element_cfg = device_cfg.elements[qubit]
    if operation_type not in element_cfg:
        raise ConfigKeyError(
            kind="operation",
            missing=operation_type,
            allowed=list(element_cfg.keys()),
        )

    device_op: Operation | Schedule = _get_device_repr_from_cfg(
        operation=operation,
        operation_cfg=element_cfg[operation_type],
        device_overrides=device_overrides,
    )

    device_op_device_overrides = device_op.data.get("gate_info", {}).get("device_overrides", {})
    new_device_overrides = {**device_op_device_overrides, **device_overrides}

    if isinstance(device_op, ScheduleBase):
        return _compile_circuit_to_device(
            operation=device_op,
            device_cfg=device_cfg,
            device_overrides=new_device_overrides,
        )
    else:
        operation.add_device_representation(device_op)
        return operation


def _compile_two_qubits(
    operation: Operation,
    qubits: Sequence[str],
    operation_type: str,
    device_cfg: DeviceCompilationConfig,
    device_overrides: dict,
) -> Operation | Schedule:
    """
    Compiles gate with multiple qubits.

    Note: it updates the `operation`, if it can directly add pulse representation.
    """
    parent_qubit, child_qubit = qubits
    edge = f"{parent_qubit}_{child_qubit}"

    if edge not in device_cfg.edges:
        symmetric_operation = operation.get("gate_info", {}).get("symmetric", False)

        if symmetric_operation:
            possible_permutations = permutations(qubits, 2)
            operable_edges = {
                f"{permutation[0]}_{permutation[1]}" for permutation in possible_permutations
            }
            valid_edge_list = list(operable_edges.intersection(device_cfg.edges))
            if len(valid_edge_list) == 1:
                edge = valid_edge_list[0]
            elif len(valid_edge_list) < 1:
                raise ConfigKeyError(
                    kind="edge", missing=edge, allowed=list(device_cfg.edges.keys())
                )
            elif len(valid_edge_list) > 1:
                raise MultipleKeysError(operation=operation_type, matches=valid_edge_list)
        else:
            raise ConfigKeyError(kind="edge", missing=edge, allowed=list(device_cfg.edges.keys()))

    edge_config = device_cfg.edges[edge]
    if operation_type not in edge_config:
        # only raise exception if it is also not a single-qubit operation
        raise ConfigKeyError(
            kind="operation",
            missing=operation_type,
            allowed=list(edge_config.keys()),
        )

    device_op: Operation | Schedule = _get_device_repr_from_cfg(
        operation, edge_config[operation_type], device_overrides
    )

    device_op_device_overrides = device_op.data.get("gate_info", {}).get("device_overrides", {})
    new_device_overrides = {**device_op_device_overrides, **device_overrides}

    if isinstance(device_op, ScheduleBase):
        return _compile_circuit_to_device(
            operation=device_op,
            device_cfg=device_cfg,
            device_overrides=new_device_overrides,
        )
    else:
        operation.add_device_representation(device_op)
        return operation


def _compile_circuit_to_device_pulse_compensation(
    operation: PulseCompensation, device_cfg: DeviceCompilationConfig
) -> PulseCompensation:
    """Compiles circuit-level pulse compensation operation to device-level."""
    if (qubits := operation.data.get("pulse_compensation_info", {}).get("qubits")) is not None:

        max_compensation_amp: dict[Port, float] = {}
        time_grid: float | None = None
        sampling_rate: float | None = None

        for qubit in qubits:
            if (
                pulse_compensation_element := device_cfg.elements.get(qubit, {}).get(
                    "pulse_compensation"
                )
            ) is not None:
                if pulse_compensation_element.factory_func is not None:
                    raise ValueError(
                        f"'factory_func' in the device configuration for pulse compensation "
                        f"for device element '{qubit}' is not 'None'. "
                        f"Only 'None' is allowed for 'factory_func' for pulse compensation."
                    )
                current_time_grid = pulse_compensation_element.factory_kwargs["time_grid"]
                if (time_grid != current_time_grid) and (time_grid is not None):
                    raise ValueError(
                        f"'time_grid' must be the same for every device element "
                        f"for pulse compensation. 'time_grid' for "
                        f"device element '{qubit}' is '{current_time_grid}', "
                        f"for others it is '{time_grid}'."
                    )
                time_grid = current_time_grid

                current_sampling_rate = pulse_compensation_element.factory_kwargs["sampling_rate"]
                if (sampling_rate != current_sampling_rate) and (sampling_rate is not None):
                    raise ValueError(
                        f"'sampling_rate' must be the same for "
                        f"every device element for pulse compensation. "
                        f"'sampling_rate' for device element '{qubit}' is "
                        f"'{current_sampling_rate}', for others it is '{sampling_rate}'."
                    )
                sampling_rate = current_sampling_rate

                port = pulse_compensation_element.factory_kwargs["port"]
                max_compensation_amp[port] = pulse_compensation_element.factory_kwargs[
                    "max_compensation_amp"
                ]

        return PulseCompensation(
            body=operation.body,
            max_compensation_amp=max_compensation_amp,
            time_grid=time_grid,
            sampling_rate=sampling_rate,
        )
    else:
        return operation


def _get_device_repr_from_cfg(
    operation: Operation,
    operation_cfg: OperationCompilationConfig,
    device_overrides: dict,
) -> Operation | Schedule:
    # deepcopy because operation_type can occur multiple times
    # (e.g., parametrized operations).
    operation_cfg = deepcopy(operation_cfg)
    factory_func = operation_cfg.factory_func

    factory_kwargs: dict = operation_cfg.factory_kwargs

    # retrieve keyword args for parametrized operations from the gate info
    if operation_cfg.gate_info_factory_kwargs is not None:
        for key in operation_cfg.gate_info_factory_kwargs:
            factory_kwargs[key] = operation.data["gate_info"][key]

    # Add operation defined custom device overrides.
    for key, value in device_overrides.items():
        if key in factory_kwargs:
            factory_kwargs[key] = value

    assert factory_func is not None
    return factory_func(**factory_kwargs)


def _get_device_repr_from_cfg_multiplexed(
    operation: Operation,
    operation_cfg: OperationCompilationConfig,
    mux_idx: int,
    device_overrides: dict,
) -> Operation | Schedule:
    operation_cfg = deepcopy(operation_cfg)
    factory_func = operation_cfg.factory_func

    factory_kwargs: dict = operation_cfg.factory_kwargs

    # retrieve keyword args for parametrized operations from the gate info
    if operation_cfg.gate_info_factory_kwargs is not None:
        for key in operation_cfg.gate_info_factory_kwargs:
            gate_info = operation.data["gate_info"][key]
            # Hack alert: not all parameters in multiplexed operation are
            # necessary passed for each element separately. We assume that if they do
            # (say, acquisition index and channel for measurement), they are passed as
            # a list or tuple. If they don't (say, it is hard to imagine different
            # acquisition protocols for qubits during multiplexed readout), they are
            # assumed to NOT be a list or tuple. If this spoils the correct behaviour of
            # your program in future: sorry :(
            if isinstance(gate_info, (tuple, list)):
                factory_kwargs[key] = gate_info[mux_idx]
            else:
                factory_kwargs[key] = gate_info

    # Add operation defined custom device overrides.
    for key, value in device_overrides.items():
        if key in factory_kwargs:
            factory_kwargs[key] = value

    assert factory_func is not None
    return factory_func(**factory_kwargs)


class ConfigKeyError(KeyError):
    """Custom exception for when a key is missing in a configuration file."""

    def __init__(self, kind: str, missing: str, allowed: list[str]) -> None:
        self.value = (
            f'{kind} "{missing}" is not present in the configuration file;'
            + f" {kind} must be one of the following: {allowed}"
        )

    def __str__(self) -> str:
        return repr(self.value)


class MultipleKeysError(KeyError):
    """Custom exception for when symmetric keys are found in a configuration file."""

    def __init__(self, operation: str, matches: list[str]) -> None:
        self.value = (
            f"Symmetric Operation {operation} matches the following edges {matches}"
            f" in the QuantumDevice. You can only specify a single edge for a symmetric"
            " operation."
        )

    def __str__(self) -> str:
        return repr(self.value)
