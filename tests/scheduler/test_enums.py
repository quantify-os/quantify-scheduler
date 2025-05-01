from qcodes import validators
from qcodes.parameters.parameter import ManualParameter

from quantify_scheduler.enums import BinMode


def test_str_enum():
    assert BinMode.APPEND == "append"
    assert BinMode.AVERAGE == "average"
    assert BinMode.APPEND < BinMode.DISTRIBUTION


def test_str_enum_qcodes_cache():
    param = ManualParameter(
        name="param",
        vals=validators.Strings(),
    )
    param(BinMode.APPEND)
    assert param.cache.get() == "append"
    assert param.cache.get() == BinMode.APPEND
