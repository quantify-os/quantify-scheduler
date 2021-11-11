# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=too-few-public-methods
# pylint: disable=no-self-use
from __future__ import annotations

import pytest
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.instrument_coordinator import utility


@pytest.fixture(name="mock_instrument")
def fixture_mock_instrument() -> Instrument:
    instr = Instrument("ParameterHolder")

    instr.add_parameter(
        "test_param",
        initial_value=1,
        unit="foo",
        label="Test Parameter",
        parameter_class=ManualParameter,
    )
    yield instr

    instr.close()


class CallCounter:
    def __init__(self, wrap_func):
        self.count: int = 0
        self.wrap_func = wrap_func

    def __call__(self, *args, **kwargs):
        self.count += 1
        self.wrap_func(*args, **kwargs)


class TestLazySet:
    def test_equal(self, mocker, mock_instrument):
        # arrange
        instr = mock_instrument
        counter = CallCounter(instr.test_param.set)
        mocker.patch.object(instr.test_param, "set", counter)

        # act
        utility.lazy_set(instr, "test_param", 10)
        utility.lazy_set(instr, "test_param", 10)

        # assert
        assert counter.count == 1

    def test_different(self, mocker, mock_instrument):
        # arrange
        instr = mock_instrument
        counter = CallCounter(instr.test_param.set)
        mocker.patch.object(instr.test_param, "set", counter)

        # act
        utility.lazy_set(instr, "test_param", 10)
        utility.lazy_set(instr, "test_param", 11)

        # assert
        assert counter.count == 2

    def test_none(self, mocker, mock_instrument):
        # arrange
        instr = mock_instrument
        counter = CallCounter(instr.test_param.set)
        mocker.patch.object(instr.test_param, "set", counter)

        # act
        utility.lazy_set(instr, "test_param", None)
        utility.lazy_set(instr, "test_param", None)

        # assert
        assert counter.count == 2
