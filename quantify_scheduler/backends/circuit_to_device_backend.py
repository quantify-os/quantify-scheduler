# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Compilation backend for quantum-circuit to quantum-device layer.
"""
from copy import deepcopy
from typing import List
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.schedules.schedule import Schedule

from quantify_scheduler.operations.acquisition_library import AcquisitionOperation
from quantify_core.utilities.general import import_python_object_from_string
from quantify_scheduler.resources import ClockResource, BasebandClockResource


def compile_circuit_to_device(schedule: Schedule, device_cfg: dict) -> Schedule:
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

    """
    # to prevent the original input schedule from being modified.
    schedule = deepcopy(schedule)
    # the input schedule is currently not being validated.
    # validate_config(device_cfg, scheme_fn=new_config_format)

    for name, frequency in device_cfg["clocks"].items():
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

        # assume it is a two-qubit operation if the operation not in the qubit config
        if len(qubits) == 2 and operation_type not in device_cfg["qubits"][qubits[0]]:
            edge = f"{qubits[0]}-{qubits[1]}"
            if edge not in device_cfg["edges"]:
                raise EdgeKeyError(
                    missing=edge, allowed=list(device_cfg["edges"].keys())
                )
            edge_config = device_cfg["edges"][edge]
            if operation_type not in edge_config:
                # only raise exception if it is also not a single-qubit operation
                raise OperationKeyError(
                    missing=operation_type,
                    allowed=list(edge_config.keys()),
                    acting_on=edge,
                )
            _add_device_repr_from_cfg(operation, edge_config[operation_type])

        else:
            # we only support 2-qubit operations and single-qubit operations.
            # some single-qubit operations (reset, measure) can be expressed as acting
            # on multiple qubit simultaneously. That is covered through this for-loop.
            for qubit in qubits:
                if qubit not in device_cfg["qubits"].keys():
                    raise QubitKeyError(
                        missing=qubit, allowed=list(device_cfg["qubits"].keys())
                    )
                qubit_cfg = device_cfg["qubits"][qubit]

                if operation_type not in qubit_cfg:
                    raise OperationKeyError(
                        missing=operation_type,
                        allowed=list(qubit_cfg.keys()),
                        acting_on=qubit,
                    )
                _add_device_repr_from_cfg(operation, qubit_cfg[operation_type])

    return schedule


def _add_device_repr_from_cfg(operation: Operation, operation_cfg: dict):

    # deepcopy because operation_type can occur multiple times
    # (e.g., parametrized operations).
    operation_cfg = deepcopy(operation_cfg)
    generator_func = operation_cfg.pop("generator_func")
    # if specified as an importable string, import the function.
    if isinstance(generator_func, str):
        generator_func = import_python_object_from_string(generator_func)

    generator_kwargs = {}
    # retrieve keyword args for parametrized operations from the gate info
    if "gate_info_generator_kwargs" in operation_cfg:
        for key in operation_cfg.pop("gate_info_generator_kwargs"):
            generator_kwargs[key] = operation.data["gate_info"][key]

    # add all other keyword args from the device configuration file.
    # the pop of generator_func and _kwargs should ensure the arguments match.
    generator_kwargs.update(operation_cfg)
    device_op = generator_func(**generator_kwargs)
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


class QubitKeyError(KeyError):
    """
    Custom exception for when a qubit is missing in a configuration file.
    """

    def __init__(self, missing, allowed):
        self.value = (
            f'Qubit "{missing}" is not present in the configuration file;'
            + f" qubit must be one of the following: {allowed}"
        )

    def __str__(self):
        return repr(self.value)


class EdgeKeyError(KeyError):
    """
    Custom exception for when an edge is missing in a configuration file.
    """

    def __init__(self, missing, allowed):
        self.value = (
            f'Edge "{missing}" is not present in the configuration file;'
            + f" edge must be one of the following: {allowed}"
        )

    def __str__(self):
        return repr(self.value)


class OperationKeyError(KeyError):
    """
    Custom exception for when a specific operation is missing in a configuration file.
    """

    def __init__(self, missing: str, allowed: List[str], acting_on: str):
        self.value = (
            f'Operation "{missing}" for "{acting_on}" is not present in the '
            + f"configuration file; operation must be one of the following: {allowed}"
        )

    def __str__(self):
        return repr(self.value)
