# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""A module containing factory functions for composite gates, which are replaced by schedules."""

from quantify_scheduler.operations.gate_library import Y90, Z
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
