# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Compilation backend for quantum-circuit to quantum-device layer.
"""
import warnings
from copy import deepcopy
from itertools import permutations
from typing import Dict, Optional, Union

import numpy as np
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    DeviceCompilationConfig,
    OperationCompilationConfig,
)
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import Schedule


def compile_circuit_to_device(
    schedule: Schedule,
    config: Optional[CompilationConfig] = None,
    # device_cfg for backwards compatibility:
    device_cfg: Optional[Union[DeviceCompilationConfig, dict]] = None,
) -> Schedule:
    """
    Adds the information required to represent operations on the quantum-device
    abstraction layer to operations that contain information on how to be represented
    on the quantum-circuit layer.

    Parameters
    ----------
    schedule
        The schedule to be compiled.
    config
        CompilationConfig used in the :class:`~QuantifyCompiler`, from which only
        the :class:`~DeviceCompilationConfig` is used in this compilation step.
    device_cfg
        Device specific configuration, defines the compilation step from
        the quantum-circuit layer to the quantum-device layer description.
        Note, if a dictionary is passed, it will be parsed to a
        :class:`~DeviceCompilationConfig`.

    """
    if not (bool(config) ^ bool(device_cfg)):
        raise ValueError(
            f"compile_circuit_to_device was called with config={config} and device_cfg={device_cfg}. "
            "Please make sure this function is called with either of the two (CompilationConfig recommended)."
        )
    # In the graph-based compilation, CompilationNodes should accept the full
    # CompilationConfig as input (#405, !615, &1)
    if config:
        device_cfg = config.device_compilation_config

    if not isinstance(device_cfg, DeviceCompilationConfig):
        # this is a special case to be supported to enable compilation for schedules
        # that are defined completely at the quantum-device layer and require no
        # circuit to device compilation.
        # A better solution would be to omit skip this compile call in a backend,
        # but this is supported for backwards compatibility reasons.
        if device_cfg is None:
            return schedule

        device_cfg = DeviceCompilationConfig.parse_obj(device_cfg)

    # to prevent the original input schedule from being modified.
    schedule = deepcopy(schedule)

    for operation in schedule.operations.values():
        # if operation is a valid pulse or acquisition it will not attempt to add
        # pulse/acquisition info in the lines below.
        if operation.valid_pulse or operation.valid_acquisition:
            continue

        qubits = operation.data["gate_info"]["qubits"]
        operation_type = operation.data["gate_info"]["operation_type"]

        # single qubit operations
        if len(qubits) == 1:
            _compile_single_qubit(
                operation=operation,
                qubit=qubits[0],
                operation_type=operation_type,
                device_cfg=device_cfg,
            )

        # it is a two-qubit operation if the operation not in the qubit config
        elif len(qubits) == 2 and operation_type not in device_cfg.elements[qubits[0]]:
            _compile_two_qubits(
                operation=operation,
                qubits=qubits,
                operation_type=operation_type,
                device_cfg=device_cfg,
            )
        # we only support 2-qubit operations and single-qubit operations.
        # some single-qubit operations (reset, measure) can be expressed as acting
        # on multiple qubits simultaneously. That is covered through this for-loop.
        else:
            _compile_multiplexed(
                operation=operation,
                qubits=qubits,
                operation_type=operation_type,
                device_cfg=device_cfg,
            )

    return schedule


def set_pulse_and_acquisition_clock(
    schedule: Schedule,
    config: Optional[CompilationConfig] = None,
    # device_cfg for backwards compatibility:
    device_cfg: Optional[Union[DeviceCompilationConfig, dict]] = None,
) -> Schedule:
    """
    Ensures that each pulse/acquisition-level clock resource has either been added
    to the schedule or, if present in device_cfg, adds it to the schedule.
    A warning is given when a clock resource has conflicting frequency
    definitions, and an error is raised if the clock resource is unknown.

    Parameters
    ----------
    schedule
        The schedule to be compiled.
    config
        CompilationConfig used in the :class:`~QuantifyCompiler`, from which only
        the :class:`~DeviceCompilationConfig` is used in this compilation step.
    device_cfg
        Device specific configuration, defines the compilation step from
        the quantum-circuit layer to the quantum-device layer description.
        Note, if a dictionary is passed, it will be parsed to a
        :class:`~DeviceCompilationConfig`.

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
    if not (bool(config) ^ bool(device_cfg)):
        raise ValueError(
            f"set_pulse_and_acquisition_clock was called with config={config} and device_cfg={device_cfg}. "
            "Please make sure this function is called with either of the two (CompilationConfig recommended)."
        )
    # In the graph-based compilation, CompilationNodes should accept the full
    # CompilationConfig as input (#405, !615, &1)
    if config:
        device_cfg = config.device_compilation_config

    if not isinstance(device_cfg, DeviceCompilationConfig):
        # this is a special case to be supported to enable compilation for schedules
        # that are defined completely at the quantum-device layer and require no
        # circuit to device compilation.
        # A better solution would be to omit skip this compile call in a backend,
        # but this is supported for backwards compatibility reasons.
        if device_cfg is not None:
            device_cfg = DeviceCompilationConfig.parse_obj(device_cfg)

    # to prevent the original input schedule from being modified.
    schedule = deepcopy(schedule)

    for operation in schedule.operations.values():
        # if the operation is at gate-level it must be compiled from
        # circuit to device first.
        if not operation.valid_pulse and not operation.valid_acquisition:
            raise RuntimeError(
                f"Operation '{operation}' is a gate-level operation and must be "
                f"compiled from circuit to device; ensure compilation "
                f"is made in the correct order."
            )

        clocks_used = []
        for info in operation["pulse_info"] + operation["acquisition_info"]:
            clocks_used.append(info["clock"])
        for clock in set(clocks_used):
            # if clock is defined both in the schedule and device_cfg,
            # ensures the frequency is the same in both
            if clock in schedule.resources:
                if device_cfg is not None and clock in device_cfg.clocks:
                    clock_freq_device_cfg = device_cfg.clocks[clock]
                    clock_freq_schedule = schedule.resources[clock]["freq"]
                    if (
                        not np.isnan(clock_freq_device_cfg)
                        and clock_freq_device_cfg != clock_freq_schedule
                    ):
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
            else:
                if device_cfg is None or clock not in device_cfg.clocks:
                    raise ValueError(
                        f"Operation '{operation}' contains an unknown clock '{clock}'; "
                        f"ensure this resource has been added to the schedule "
                        f"or to the device config."
                    )

                if np.isnan(clock_freq_device_cfg := device_cfg.clocks[clock]):
                    raise ValueError(
                        f"Operation '{operation}' contains clock '{clock}' with an "
                        f"undefined (initial) frequency; ensure this resource has been "
                        f"added to the schedule or to the device config."
                    )

                clock_resource = ClockResource(name=clock, freq=clock_freq_device_cfg)
                schedule.add_resource(clock_resource)

    return schedule


def _compile_multiplexed(operation, qubits, operation_type, device_cfg):
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

        _add_device_repr_from_cfg_multiplexed(
            operation, element_cfg[operation_type], mux_idx=mux_idx
        )


def _compile_single_qubit(operation, qubit, operation_type, device_cfg):
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

    _add_device_repr_from_cfg(
        operation=operation,
        operation_cfg=element_cfg[operation_type],
    )


def _compile_two_qubits(operation, qubits, operation_type, device_cfg):
    parent_qubit, child_qubit = qubits
    edge = f"{parent_qubit}_{child_qubit}"

    symmetric_operation = operation.get("gate_info", {}).get("symmetric", False)

    if symmetric_operation:
        possible_permutations = permutations(qubits, 2)
        operable_edges = {
            f"{permutation[0]}_{permutation[1]}"
            for permutation in possible_permutations
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

    if edge not in device_cfg.edges:
        raise ConfigKeyError(
            kind="edge", missing=edge, allowed=list(device_cfg.edges.keys())
        )

    edge_config = device_cfg.edges[edge]
    if operation_type not in edge_config:
        # only raise exception if it is also not a single-qubit operation
        raise ConfigKeyError(
            kind="operation",
            missing=operation_type,
            allowed=list(edge_config.keys()),
        )

    _add_device_repr_from_cfg(operation, edge_config[operation_type])


def _add_device_repr_from_cfg(
    operation: Operation, operation_cfg: OperationCompilationConfig
):
    # deepcopy because operation_type can occur multiple times
    # (e.g., parametrized operations).
    operation_cfg = deepcopy(operation_cfg)
    factory_func = operation_cfg.factory_func

    factory_kwargs: Dict = operation_cfg.factory_kwargs

    # retrieve keyword args for parametrized operations from the gate info
    if operation_cfg.gate_info_factory_kwargs is not None:
        for key in operation_cfg.gate_info_factory_kwargs:
            factory_kwargs[key] = operation.data["gate_info"][key]

    device_op = factory_func(**factory_kwargs)
    operation.add_device_representation(device_op)


def _add_device_repr_from_cfg_multiplexed(
    operation: Operation, operation_cfg: OperationCompilationConfig, mux_idx: int
):
    operation_cfg = deepcopy(operation_cfg)
    factory_func = operation_cfg.factory_func

    factory_kwargs: Dict = operation_cfg.factory_kwargs

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

    device_op = factory_func(**factory_kwargs)
    operation.add_device_representation(device_op)


# pylint: disable=super-init-not-called
class ConfigKeyError(KeyError):
    """
    Custom exception for when a key is missing in a configuration file.
    """

    def __init__(self, kind, missing, allowed):
        self.value = (
            f'{kind} "{missing}" is not present in the configuration file;'
            + f" {kind} must be one of the following: {allowed}"
        )

    def __str__(self):
        return repr(self.value)


# pylint: disable=super-init-not-called
class MultipleKeysError(KeyError):
    """
    Custom exception for when symmetric keys are found in a configuration file.
    """

    def __init__(self, operation, matches):
        self.value = (
            f"Symmetric Operation {operation} matches the following edges {matches}"
            f" in the QuantumDevice. You can only specify a single edge for a symmetric"
            " operation."
        )

    def __str__(self):
        return repr(self.value)
