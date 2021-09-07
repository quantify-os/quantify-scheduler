# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring

from quantify_scheduler.backends.qblox import reserved_registers


def test_registers_unique():
    regs = [
        getattr(reserved_registers, r)
        for r in dir(reserved_registers)
        if r.startswith("REG")
    ]

    # assert that all register assignments are unique
    assert len(regs) == len(set(regs))


def test_registers_valid_range():
    # registers in qblox hardware are allowed to be (0,63)

    regs = [
        getattr(reserved_registers, r)
        for r in dir(reserved_registers)
        if r.startswith("REG")
    ]
    for reg in regs:
        assert reg.startswith("R")
        assert len(reg) < 4
        assert int(reg[1:]) < 64
        assert int(reg[1:]) >= 0
