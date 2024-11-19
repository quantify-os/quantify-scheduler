# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing logic to handle crosstalk compensation."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import CompilationConfig
    from quantify_scheduler.operations import Operation

import numpy as np

from quantify_scheduler.schedules import Schedulable, Schedule


def crosstalk_compensation(
    schedule: Schedule, config: CompilationConfig  # noqa: D417
) -> Schedule:  # noqa: D103
    """
    Apply crosstalk compensation to the given schedule based on the provided configuration.
    It adds compensation operations to port clocks affected by crosstalk.
    It also adjusts the amplitude of the original operation.

    Parameters
    ----------
    schedule
        The schedule to which cross-talk compensation will be applied.
    config
        The configuration containing hardware options.

    Returns
    -------
    Schedule:
        The schedule with crosstalk compensation applied.

    """
    compilation_config = getattr(config, "hardware_compilation_config", None)
    hardware_options = getattr(compilation_config, "hardware_options", None)

    if not hardware_options or not hardware_options.crosstalk:
        return schedule

    port_clock_list = _get_unique_port_clocks(hardware_options.crosstalk)
    compensation_matrix = _calculate_compensation_matrix(
        hardware_options.crosstalk, port_clock_list
    )

    for schedulable in list(schedule.schedulables.values()):
        operation = schedule.operations[schedulable["operation_id"]]

        if isinstance(operation, Schedule):
            # Recursively apply compensation to nested schedules
            schedule.operations[schedulable["operation_id"]] = crosstalk_compensation(
                operation, config
            )
        elif is_pulse(operation):
            _apply_compensation_to_operation(
                schedule, operation, schedulable, port_clock_list, compensation_matrix
            )

    return schedule


def _get_unique_port_clocks(crosstalk: dict[str, dict[str, float]]) -> list[str]:
    port_clocks = set(crosstalk.keys())
    for connections in crosstalk.values():
        port_clocks.update(connections.keys())
    return sorted(port_clocks)


def _calculate_compensation_matrix(
    crosstalk: dict[str, dict[str, complex]], port_clock_list: list[str]
) -> np.ndarray:
    return np.linalg.inv(_construct_crosstalk_matrix(crosstalk, port_clock_list))


def _construct_crosstalk_matrix(
    crosstalk: dict[str, dict[str, complex]], port_clock_list: list[str]
) -> np.ndarray:
    size = len(port_clock_list)
    matrix = np.eye(size, dtype=complex)

    for src_port_clock, connections in crosstalk.items():
        for dest_port_clock, value in connections.items():
            src_index = port_clock_list.index(src_port_clock)
            dest_index = port_clock_list.index(dest_port_clock)
            matrix[src_index, dest_index] = value

    return matrix


def is_pulse(operation: Operation) -> bool:
    """
    Check if the operation is a pulse.

    Parameters
    ----------
    operation:
        The operation to check.

    Returns
    -------
    :
        True if the operation is a pulse, False otherwise.

    """
    return (
        len(operation.data["pulse_info"]) > 0
        and operation.data["pulse_info"][0]["wf_func"] is not None
    )


def _apply_compensation_to_operation(
    schedule: Schedule,
    operation: Operation,
    schedulable: Schedulable,
    port_clock_list: list[str],
    compensation_matrix: np.ndarray,
) -> None:
    port_clock = (
        f"{operation.data['pulse_info'][0]['port']}-" f"{operation.data['pulse_info'][0]['clock']}"
    )
    port_clock_index = port_clock_list.index(port_clock)
    compensation_row = compensation_matrix[port_clock_index]

    # Add compensation pulses to port clocks that are affected by crosstalk
    for i, compensation_value in enumerate(compensation_row):
        if compensation_value != 0 and i != port_clock_index:
            _add_compensation_operation(
                schedule,
                operation,
                schedulable,
                port_clock_list[i],
                compensation_value,
                i,
            )

    # Adjust all params that have amp in them
    for key in operation.data["pulse_info"][0]:
        if "amp" in key:
            operation.data["pulse_info"][0][key] *= compensation_row[port_clock_index]


def _add_compensation_operation(
    schedule: Schedule,
    original_operation: Operation,
    original_schedulable: Schedulable,
    target_port_clock: str,
    compensation_value: float,
    index: int,
) -> None:
    compensation_op = deepcopy(original_operation)
    target_port, target_clock = target_port_clock.split("-")

    # amp of compensation operation is product of the original amplitude and compensation value.
    compensation_op.data["pulse_info"][0].update(
        {
            "port": target_port,
            "clock": target_clock,
        }
    )
    for key in compensation_op.data["pulse_info"][0]:
        if "amp" in key:
            compensation_op.data["pulse_info"][0][key] = (
                original_operation.data["pulse_info"][0][key] * compensation_value
            )

    schedule.operations[compensation_op.hash] = compensation_op

    compensation_schedulable = deepcopy(original_schedulable)
    compensation_schedulable["operation_id"] = compensation_op.hash
    schedule.schedulables[f"{original_operation.hash}_compensation_{index}"] = (
        compensation_schedulable
    )
