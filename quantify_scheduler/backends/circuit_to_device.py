# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Compilation backend for quantum-circuit to quantum-device layer.
"""
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.structure import DataStructure


class OperationCompilationConfig(DataStructure):
    """
    A datastructure containing the information required to compile an individual
    operation to the representation at the device level.

    Parameters
    ----------
    factory_func:
        A callable designating a factory function used to create the representation
        of the operation at the quantum-device level.
    factory_kwargs:
        a dictionary containing the keyword arguments and corresponding values to use
        when creating the operation by evaluating the factory function.
    gate_info_factory_kwargs:
        A list of keyword arguments of the factory function for which the value must
        be retrieved from the `gate_info` of the operation.
    """

    factory_func: Union[Callable, str]
    factory_kwargs: Dict[str, Any]
    gate_info_factory_kwargs: Optional[List[str]]


# pylint: disable=line-too-long
class DeviceCompilationConfig(DataStructure):
    """
    A datastructure containing the information required to compile a
    schedule to the representation at the quantum-device layer.

    Parameters
    ----------
    backend:
        a . separated string specifying the location of the compilation backend this
        configuration is intended for e.g.,
        :func:`~.backends.circuit_to_device.compile_circuit_to_device`.
    clocks:
        a dictionary specifying the clock frequencies available on the device e.g.,
        :code:`{"q0.01": 6.123e9}`.
    elements:
        a dictionary specifying the elements on the device, what operations can be
        applied to them and how to compile them.
    edges:
        a dictionary specifying the edges, links between elements on the device to which
        operations can be applied, the operations tha can be  applied to them and how
        to compile them.



    .. admonition:: Examples
        :class: dropdown

        The DeviceCompilationConfig is structured such that it should allow the
        specification of the circuit-to-device compilation for many different qubit
        platforms.
        Here we show a basic configuration for a two-transmon quantum device.
        In this example, the DeviceCompilationConfig is created by parsing a dictionary
        containing the relevant information.

        .. important::

            Although it is possible to manually create a configuration using
            dictionaries, this is not recommended. The
            :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice`
            is responsible for managing and generating configuration files.

        .. jupyter-execute::

            from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
            import pprint
            from quantify_scheduler.schemas.examples.circuit_to_device_example_cfgs import (
                example_transmon_cfg,
            )

            pprint.pprint(example_transmon_cfg)


        The dictionary can be parsed using the :code:`parse_obj` method.

        .. jupyter-execute::

            device_cfg = DeviceCompilationConfig.parse_obj(example_transmon_cfg)
            device_cfg


    """

    backend: str
    clocks: Dict[str, float]
    elements: Dict[str, Dict[str, OperationCompilationConfig]]
    edges: Dict[str, Dict[str, OperationCompilationConfig]]


# pylint: disable=too-many-branches
def compile_circuit_to_device(
    schedule: Schedule, device_cfg: Union[DeviceCompilationConfig, dict]
) -> Schedule:
    """
    Adds the information required to represent operations on the quantum-device
    abstraction layer to operations that contain information on how to be represented
    on the quantum-circuit layer.

    Parameters
    ----------
    schedule
        The schedule to be compiled.
    device_cfg
        Device specific configuration, defines the compilation step from
        the quantum-circuit layer to the quantum-device layer description.
        Note, if a dictionary is passed, it will be parsed to a
        :class:`~DeviceCompilationConfig`.

    """
    if not isinstance(device_cfg, DeviceCompilationConfig):
        device_cfg = DeviceCompilationConfig.parse_obj(device_cfg)

    # to prevent the original input schedule from being modified.
    schedule = deepcopy(schedule)

    for name, frequency in device_cfg.clocks.items():
        if name not in schedule.resources:
            # some schedules manually add a clock and set it's frequency
            # (e.g., spectroscopy schedules. In that case, this should not overwrite
            # the information that was explicitly set in the schedule.
            schedule.add_resources([ClockResource(name=name, freq=frequency)])

    for operation in schedule.operations.values():
        # if operation is a valid pulse or acquisition it will not attempt to add
        # pulse/acquisition info in the lines below.
        if operation.valid_pulse:
            _verify_pulse_clock_present(operation=operation, schedule=schedule)
            continue
        if operation.valid_acquisition:
            # no verification at this point.
            continue

        qubits = operation.data["gate_info"]["qubits"]
        operation_type = operation.data["gate_info"]["operation_type"]

        # single qubit operations
        if len(qubits) == 1:
            qubit = qubits[0]
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

        # it is a two-qubit operation if the operation not in the qubit config
        elif len(qubits) == 2 and operation_type not in device_cfg.elements[qubits[0]]:
            edge = f"{qubits[0]}-{qubits[1]}"
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

        # we only support 2-qubit operations and single-qubit operations.
        # some single-qubit operations (reset, measure) can be expressed as acting
        # on multiple qubits simultaneously. That is covered through this for-loop.
        else:
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

    return schedule


def _add_device_repr_from_cfg(
    operation: Operation, operation_cfg: OperationCompilationConfig
):

    # deepcopy because operation_type can occur multiple times
    # (e.g., parametrized operations).
    operation_cfg = deepcopy(operation_cfg)
    factory_func = operation_cfg.factory_func

    # if specified as an importable string, import the function.
    if isinstance(factory_func, str):
        factory_func = import_python_object_from_string(factory_func)

    factory_kwargs: Dict = operation_cfg.factory_kwargs

    # retrieve keyword args for parametrized operations from the gate info
    if operation_cfg.gate_info_factory_kwargs is not None:
        for key in operation_cfg.gate_info_factory_kwargs:
            factory_kwargs[key] = operation.data["gate_info"][key]

    device_op = factory_func(**factory_kwargs)
    operation.add_device_representation(device_op)


def _verify_pulse_clock_present(operation, schedule):
    for pulse in operation["pulse_info"]:
        if "clock" in pulse:
            if pulse["clock"] not in schedule.resources:
                raise ValueError(
                    "Operation '{}' contains an unknown clock '{}'; ensure "
                    "this resource has been added to the schedule.".format(
                        str(operation), pulse["clock"]
                    )
                )


def _add_device_repr_from_cfg_multiplexed(
    operation: Operation, operation_cfg: OperationCompilationConfig, mux_idx: int
):
    operation_cfg = deepcopy(operation_cfg)
    factory_func = operation_cfg.factory_func

    # if specified as an importable string, import the function.
    if isinstance(factory_func, str):
        factory_func = import_python_object_from_string(factory_func)

    factory_kwargs: Dict = operation_cfg.factory_kwargs

    # retrieve keyword args for parametrized operations from the gate info
    if operation_cfg.gate_info_factory_kwargs is not None:
        for key in operation_cfg.gate_info_factory_kwargs:
            factory_kwargs[key] = operation.data["gate_info"][key][mux_idx]

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
