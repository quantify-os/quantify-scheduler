"""
Compilation backend for quantum-circuit to quantum-device layer.
"""
from copy import deepcopy
from typing import List
from quantify_scheduler.schedules.schedule import Schedule

from quantify_scheduler.operations.acquisition_library import AcquisitionOperation
from quantify_core.utilities.general import import_python_object_from_string


def compile_circuit_to_device(schedule: Schedule, device_cfg: dict) -> Schedule:
    """
    Adds pulse_info and acquisition_info to all operations that have gate_info
    specified
    """
    # to prevent the original input schedule from being modified.
    schedule = deepcopy(schedule)

    # validate_config(device_cfg, scheme_fn=new_config_format)

    for operation in schedule.operations.values():
        if operation.valid_pulse:
            _verify_pulse_clock_present(operation=operation, schedule=schedule)
            continue
        if operation.valid_acquisition:
            # no verification at this point.
            continue

        # if operation is a valid pulse or acquisition it will not attempt to add
        # pulse/acquisition info in the lines below.

        qubits = operation.data["gate_info"]["qubits"]
        operation_type = operation.data["gate_info"]["operation_type"]

        if len(qubits) == 1:
            qubit = qubits[0]

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

            # deepcopy because operation_type can occur multiple times
            # (e.g., parametrized operations)
            operation_cfg = deepcopy(qubit_cfg[operation_type])

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

            # FIXME: check if this is how the operations should be added.
            if isinstance(device_op, AcquisitionOperation):
                operation.add_acquisition(device_op)
            else:
                operation.add_pulse(device_op)

        elif len(qubits) == 2:
            edge = f"{qubits[0]}-{qubits[1]}"
            operation_cfg = device_cfg["edges"][edge][operation_type]
        else:
            raise ValueError("Operations on more than 2 qubits are not supported")
    return schedule


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


def add_clock_resource(schedule: Schedule, clock_name: str, frequency: float):
    schedule.add_resources(
        [
            ClockResource(
                clock_name,
                freq=frequency,
            )
        ]
    )
