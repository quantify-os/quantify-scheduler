# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utility class for dynamically allocating registers for Qblox sequencers."""
from typing import Set

from quantify_scheduler.backends.qblox import constants


class RegisterManager:
    """
    Utility class that keeps track of all the registers that are still available.
    """

    def __init__(self) -> None:
        """
        Instantiates the `RegisterManager`.
        """
        self._available_registers: Set[str] = {
            f"R{idx}" for idx in range(constants.NUMBER_OF_REGISTERS)
        }

    def allocate_register(self) -> str:
        """
        Allocates a register to be used within the q1asm program.

        Returns
        -------
        :
            A register that can be used.

        Raises
        ------
        IndexError
            When the RegisterManager runs out of registers to allocate.
        """
        if len(self.available_registers) < 1:
            raise IndexError(
                "Out of registers. Attempting to use more registers than "
                "available in the Q1 sequence processor. This can be "
                "caused, e.g., by attempting to use too many acquisition "
                "channels."
            )
        # to ensure deterministic behavior as sets are unsorted
        first_element = sorted(self._available_registers)[0]
        self._available_registers.remove(first_element)
        return first_element

    def free_register(self, register: str) -> None:
        """
        Frees up a register to be reused.

        Parameters
        ----------
        register
            The register to free up.

        Raises
        ------
        ValueError
            The value provided is not a valid register.
        RuntimeError
            Attempting to free a register that is already free.
        """
        _verify_valid_register(register)
        if register in self.available_registers:
            raise RuntimeError(
                f"Attempting to free register '{register}', but this register is not in"
                f"use."
            )
        self._available_registers.add(register)

    @property
    def available_registers(self) -> Set[str]:
        """
        Getter for the available registers.

        Returns
        -------
        :
            A set containing all the available registers.
        """
        return self._available_registers


def _verify_valid_register(register_name: str) -> None:
    """
    Verifies whether the passed name is a valid register name. Raises on any of the
    conditions:

    1. `register_name` does not start with "R" or
    2. `register_name` does not have an integer next
    3. the integer is higher than the number of registers in the sequence processor
    4. the integer is negative valued

    Parameters
    ----------
    register_name
        The register to verify.

    Raises
    -------
    ValueError
        Invalid register name passed.
    """

    def raise_error() -> None:
        raise ValueError(
            f"Invalid register '{register_name}'! The correct format is 'R' followed by"
            f" an integer between 0 and {constants.NUMBER_OF_REGISTERS}."
        )

    prefix = register_name[0]
    if prefix != "R":
        raise_error()

    register_idx: int = 0
    try:
        register_idx = int(register_name[1:])
    except ValueError:
        raise_error()

    if register_idx < 0 or register_idx > constants.NUMBER_OF_REGISTERS:
        raise_error()
