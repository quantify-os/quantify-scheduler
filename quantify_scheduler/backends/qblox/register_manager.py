# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utility class for dynamically allocating registers for Qblox sequencers."""

from __future__ import annotations

from quantify_scheduler.backends.qblox import constants


class RegisterManager:
    """Utility class that keeps track of all the registers that are still available."""

    def __init__(self) -> None:
        self._available_registers: list[str] = [
            f"R{idx}" for idx in range(constants.NUMBER_OF_REGISTERS)
        ]

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
        return self._available_registers.pop(0)

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
                f"Attempting to free register '{register}', but this register is not inuse."
            )
        self._available_registers.insert(0, register)  # LIFO

    @property
    def available_registers(self) -> list[str]:
        """
        Getter for the available registers.

        Returns
        -------
        :
            A copy of the list containing all the available registers.

        """
        return self._available_registers[:]


def _verify_valid_register(register_name: str) -> None:
    """
    Verifies whether the passed name is a valid register name.

    Raises on any of the conditions:

    1. ``register_name`` does not start with "R" or
    2. ``register_name`` does not have an integer next
    3. the integer is higher than the number of registers in the sequence processor
    4. the integer is negative valued

    Parameters
    ----------
    register_name
        The register to verify.

    Raises
    ------
    ValueError
        Invalid register name passed.

    """
    if not register_name:
        raise ValueError("Register can not be None or empty string")

    if register_name[0] != "R":
        raise ValueError(
            f"Invalid register '{register_name}'. The register should start with a capital 'R'"
        )

    if not register_name[1:].isdigit():
        raise ValueError(
            f"Invalid register '{register_name}'. The correct format is 'R' followed by digits"
        )

    register_idx: int = int(register_name[1:])

    if register_idx < 0 or register_idx > constants.NUMBER_OF_REGISTERS:
        raise ValueError(
            f"The register index '{register_name}' should be between "
            f"0 and {constants.NUMBER_OF_REGISTERS}"
        )
