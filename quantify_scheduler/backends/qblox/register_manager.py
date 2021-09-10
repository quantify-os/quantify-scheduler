# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Utility class for dynamically allocating registers.
"""
from typing import Set
from quantify_scheduler.backends.qblox import constants


class RegisterManager:
    def __init__(self):
        self._available_registers: Set[str] = {
            f"R{idx}" for idx in range(constants.NUMBER_OF_REGISTERS)
        }

    def allocate_register(self) -> str:
        if len(self.available_registers) < 1:
            raise IndexError(
                "Out of registers. Attempting to use more registers than "
                "available in the Q1 sequence processor. This can be "
                "caused e.g. by attempting to use too many acquisition "
                "channels."
            )
        return self._available_registers.pop()

    def free_register(self, register: str):
        _verify_valid_register(register)
        self._available_registers.add(register)

    @property
    def available_registers(self) -> Set[str]:
        return self._available_registers


def _verify_valid_register(register_name: str):
    def raise_error():
        raise ValueError(
            f'Invalid register "{register_name}"! The correct format is "R" followed by'
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
