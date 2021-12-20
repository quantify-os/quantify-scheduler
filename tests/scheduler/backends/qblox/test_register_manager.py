# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring
# pylint: disable=no-self-use

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Tests for Qblox backend register manager."""
import pytest

from quantify_scheduler.backends.qblox import register_manager


class TestRegisterManager:
    @pytest.fixture(name="make_rm")
    def fixture_make_rm(self) -> register_manager.RegisterManager:
        return register_manager.RegisterManager()

    def test_available_registers(
        self, make_rm: register_manager.RegisterManager
    ) -> None:
        pool = make_rm
        assert pool.available_registers == pool._available_registers

    def test_allocate_register(self, make_rm: register_manager.RegisterManager) -> None:
        pool = make_rm
        initial_amount_of_registers = len(pool.available_registers)
        pool.allocate_register()

        # since we remove one from the pool when we allocate
        assert len(pool.available_registers) == initial_amount_of_registers - 1

    def test_free_register(self, make_rm: register_manager.RegisterManager) -> None:
        pool = make_rm
        reg = pool.allocate_register()
        assert reg not in pool.available_registers
        pool.free_register(reg)
        assert reg in pool.available_registers


@pytest.mark.parametrize(
    "register, is_valid",
    [
        ("R0", True),
        ("R1", True),
        ("R64", True),
        ("R-1", False),
        ("S2", False),
        ("R65", False),
        ("R2a", False),
        ("hello@qblox.com", False),
    ],
)
def test__verify_invalid_register(register: str, is_valid: bool) -> None:
    if is_valid:
        register_manager._verify_valid_register(register)
    else:
        with pytest.raises(ValueError):
            register_manager._verify_valid_register(register)
