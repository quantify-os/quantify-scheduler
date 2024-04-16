# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch


from __future__ import annotations

import pytest
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
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

    class DummyInstrumentChannel(InstrumentChannel):
        def __init__(self, parent: Instrument, name: str) -> None:
            super().__init__(parent, name)
            self.bar = ManualParameter(
                "bar", label="Test Child Parameter", instrument=self
            )

    channel_name = "ch_foo"
    channel = DummyInstrumentChannel(parent=instr, name=channel_name)
    instr.add_submodule(channel_name, channel)

    yield instr


@pytest.mark.parametrize(
    "channel_name, parameter_name",
    [("", "test_param"), ("", "ch_foo.bar"), ("ch_foo", "bar")],
)
def test_search_settable_param_success(mock_instrument, channel_name, parameter_name):
    instrument = (
        mock_instrument.submodules[channel_name] if channel_name else mock_instrument
    )

    settable_param = utility.search_settable_param(
        instrument=instrument, nested_parameter_name=parameter_name
    )
    assert isinstance(settable_param, ManualParameter)


@pytest.mark.parametrize("parameter_name", ["ch_foo2.bar", "ch_foo.bar2", "foo"])
def test_search_settable_param_fail(mock_instrument, parameter_name):
    with pytest.raises(ValueError):
        utility.search_settable_param(
            instrument=mock_instrument, nested_parameter_name=parameter_name
        )


@pytest.mark.parametrize("parameter_name", ["missing.ch_foo.bar"])
def test_search_settable_param_fail_wrong_nesting(mock_instrument, parameter_name):
    with pytest.raises(ValueError):
        utility.search_settable_param(
            instrument=mock_instrument, nested_parameter_name=parameter_name
        )


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
        assert counter.count == 1

    def test_invalid(self, mocker, mock_instrument):
        # arrange
        instr = mock_instrument
        counter = CallCounter(instr.test_param.set)
        mocker.patch.object(instr.test_param, "set", counter)

        # act
        utility.lazy_set(instr, "test_param", 10)
        instr.test_param.cache.invalidate()
        utility.lazy_set(instr, "test_param", 10)

        # assert
        assert counter.count == 2
