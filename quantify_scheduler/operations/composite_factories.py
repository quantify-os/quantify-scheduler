# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""A module containing factory functions for composite gates, which are replaced by schedules."""

from quantify_scheduler.operations.gate_library import CZ, Y90, H, Z
from quantify_scheduler.schedules.schedule import Schedule


def hadamard_as_y90z(
    qubit: str,
) -> Schedule:
    """
    Generate a :class:`~.schedules.schedule.Schedule` Y90 * Z (equivalent to a Hadamard gate).

    Parameters
    ----------
    qubit
        Qubit to which the Hadamard gate is applied.

    Returns
    -------
    :
        Schedule.

    """
    schedule = Schedule("Hadamard")
    schedule.add(Z(qubit))
    schedule.add(Y90(qubit))
    return schedule


def cnot_as_h_cz_h(
    control_qubit: str,
    target_qubit: str,
) -> Schedule:
    """
    Generate a :class:`~.schedules.schedule.Schedule` for a CNOT gate using a CZ gate
    interleaved with Hadamard gates on the target qubit.

    Parameters
    ----------
    control_qubit
        Qubit acting as the control qubit.
    target_qubit
        Qubit acting as the target qubit.

    Returns
    -------
    Schedule
        Schedule for the CNOT gate.

    """
    schedule = Schedule("CNOT")
    schedule.add(H(target_qubit))
    schedule.add(CZ(control_qubit, target_qubit))
    schedule.add(H(target_qubit))
    return schedule
