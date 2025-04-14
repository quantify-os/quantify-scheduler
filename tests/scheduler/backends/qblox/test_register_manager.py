# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox backend register manager."""

import pytest

from quantify_scheduler.backends.qblox import constants, register_manager
from quantify_scheduler.backends.qblox.register_manager import RegisterManager


class TestAvailableRegisters:
    """Test suite for verifying the available registers functionality in RegisterManager.

    Tests include:
    - Checking that the available registers are correctly reported.
    - Ensuring that the `available_registers` property returns a copy, not the original list.
    """

    def test_available_registers(
        self,
    ) -> None:
        manager = RegisterManager()
        assert manager.available_registers == manager._available_registers
        # Test that it's a copy
        assert manager.available_registers is not manager._available_registers

    def test_available_registers_content(self) -> None:
        manager = RegisterManager()
        registers = manager.available_registers
        assert all(reg.startswith("R") for reg in registers)
        assert all(0 <= int(reg[1:]) < constants.NUMBER_OF_REGISTERS for reg in registers)
        assert len(registers) == constants.NUMBER_OF_REGISTERS


class TestAllocateRegister:
    """Test suite for register allocation functionality.

    Verifies the behavior of register allocation including:
    - Basic allocation
    - Sequential allocation
    - Exhaustion handling
    """

    def test_allocate_register(
        self,
    ) -> None:
        manager = RegisterManager()
        initial_amount_of_registers = len(manager.available_registers)
        manager.allocate_register()

        # since we remove one from the pool when we allocate
        assert len(manager.available_registers) == initial_amount_of_registers - 1

    def test_no_more_free_registers(self):
        manager = RegisterManager()
        for register_index in range(constants.NUMBER_OF_REGISTERS):
            register = manager.allocate_register()
            assert register == f"R{register_index}"
        with pytest.raises(IndexError):
            manager.allocate_register()

    def test_sequential_allocation(self) -> None:
        manager = RegisterManager()
        assert manager.allocate_register() == "R0"
        assert manager.allocate_register() == "R1"
        assert manager.allocate_register() == "R2"


class TestFreeRegister:
    """Test suite for freeing registers in RegisterManager.

    Ensures:
    - Registers can be freed and made available for re-allocation.
    - Appropriate exceptions are raised when freeing an already free register.
    - Invalid registers cannot be freed and raise an exception.
    """

    def test_free_register(self) -> None:
        manager = RegisterManager()
        reg = manager.allocate_register()
        assert reg not in manager.available_registers
        manager.free_register(reg)
        assert reg in manager.available_registers

    def test_was_already_free(self):
        manager = RegisterManager()
        with pytest.raises(RuntimeError):
            manager.free_register("R1")

    def test_invalid_register_cant_be_freed(self):
        manager = RegisterManager()
        with pytest.raises(ValueError):
            manager.free_register("Invalid")

    def test_reallocate_freed_register(self) -> None:
        manager = RegisterManager()
        reg = manager.allocate_register()
        manager.free_register(reg)
        new_reg = manager.allocate_register()
        assert new_reg == reg  # Should reuse the freed register


class TestVerifyValidRegister:
    """Test suite for validating register names with `_verify_valid_register`.

    Validates:
    - Invalid register names correctly raise a `ValueError`.
    - Valid register names pass validation without error.
    """

    @pytest.mark.parametrize(
        "register",
        ["R-1", "2", "S2", "R65", "R2a", "hello@qblox.com", "", None, "R 0", "r0"],
    )
    def test__verify_invalid_register(self, register: str) -> None:
        with pytest.raises(ValueError):
            register_manager._verify_valid_register(register)

    @pytest.mark.parametrize(
        "register",
        ["R0", "R1", "R64"],
    )
    def test__verify_valid_register(self, register: str) -> None:
        register_manager._verify_valid_register(register)
        assert True  # Didn't raise an error
