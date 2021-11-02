# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=unused-argument
# pylint: disable=too-many-arguments
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List
from unittest.mock import call

import pytest
from qcodes import Instrument

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.instrument_coordinator import (
    InstrumentCoordinator,
    ZIInstrumentCoordinator,
)
from quantify_scheduler.instrument_coordinator.components import base as base_component


class MyICC(base_component.InstrumentCoordinatorComponentBase):
    @property
    def is_running(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def prepare(self, options):
        pass

    def retrieve_acquisition(self):
        pass

    def wait_done(self, timeout_sec: int = 10):
        pass


# creates a few dummy components avialable to be used in each test
@pytest.fixture(scope="function", name="dummy_components")
def fixture_dummy_components(
    mocker, request
) -> base_component.InstrumentCoordinatorComponentBase:

    # Create a QCoDeS intrument for realistic emulation
    instruments = [Instrument(f"dev{i}") for i in range(3)]
    components = []

    for instrument in instruments:
        comp = MyICC(instrument)
        for func in ("prepare", "start", "stop", "wait_done", "retrieve_acquisition"):
            mocker.patch.object(
                comp,
                func,
                wraps=getattr(comp, func),
            )
        components.append(comp)

    def cleanup_tmp():
        # This should prevent the garbage collector from colleting the qcodes instrument
        for instrument in instruments:
            instrument.close()

    request.addfinalizer(cleanup_tmp)

    return components


@pytest.fixture(scope="function", name="instrument_coordinator")
def fixture_instrument_coordinator(request) -> InstrumentCoordinator:
    instrument_coordinator = InstrumentCoordinator("ic_0000")

    def cleanup_tmp():
        # This should prevent the garbage collector from colleting the qcodes instrument
        instrument_coordinator.close()

    request.addfinalizer(cleanup_tmp)

    return instrument_coordinator


@pytest.fixture(scope="function", name="zi_instrument_coordinator")
def fixture_zi_instrument_coordinator(request) -> ZIInstrumentCoordinator:
    zi_instrument_coordinator = ZIInstrumentCoordinator("ic_zi_0000")

    def cleanup_tmp():
        # This should prevent the garbage collector from colleting the qcodes instrument
        zi_instrument_coordinator.close()

    request.addfinalizer(cleanup_tmp)

    return zi_instrument_coordinator


def test_constructor(close_all_instruments, instrument_coordinator):

    # Assert
    assert len(instrument_coordinator.components()) == 0


@pytest.mark.parametrize(
    "states,expected",
    [
        ([True, True], True),
        ([False, True], True),
        ([False, False], False),
    ],
)
def test_is_running(
    close_all_instruments,
    instrument_coordinator,
    dummy_components,
    states: List[bool],
    expected: bool,
    mocker,
):
    # Arrange
    mocker.patch.object(MyICC, "is_running")  # necessary for overriding `is_running`

    for state in states:
        # popping ensures the reference to the object is released after this for loop
        component = dummy_components.pop(0)
        instrument_coordinator.add_component(component)
        component.is_running = state

    # force garbage collection to emulate qcodes correcly
    gc.collect()

    # Act
    is_running = instrument_coordinator.is_running

    # Assert
    assert is_running == expected


def test_get_component(close_all_instruments, instrument_coordinator, dummy_components):
    for i in range(len(dummy_components)):
        # Arrange
        component_ = dummy_components.pop(0)
        instrument_coordinator.add_component(component_)

        # Act
        component = instrument_coordinator.get_component(f"ic_dev{i}")

        # Assert
        assert component_ == component


def test_get_component_failed(close_all_instruments, instrument_coordinator):

    # Act
    with pytest.raises(KeyError) as execinfo:
        instrument_coordinator.get_component("ic_dev1234")

    # Assert
    assert execinfo.value.args[0] == "'ic_dev1234' is not a component of ic_0000!"


def test_add_component_failed_duplicate(
    close_all_instruments, instrument_coordinator, dummy_components
):
    # Arrange
    component1 = dummy_components.pop(0)
    instrument_coordinator.add_component(component1)

    # Act
    with pytest.raises(ValueError) as execinfo:
        instrument_coordinator.add_component(component1)

    # Assert
    assert execinfo.value.args[0] == "'ic_dev0' has already been added!"


def test_add_component_failed_type_validation(
    close_all_instruments, instrument_coordinator
):
    @dataclass
    class DummyComponent:
        name: str

        def __repr__(self) -> str:
            return "<DummyComponent>"

    component = DummyComponent("abc")

    # Act
    with pytest.raises(TypeError) as execinfo:
        instrument_coordinator.add_component(component)

    # Assert
    assert execinfo.value.args[0] == (
        "<DummyComponent> is not quantify_scheduler.instrument_coordinator."
        "components.base.InstrumentCoordinatorComponentBase."
    )


def test_remove_component(
    close_all_instruments, instrument_coordinator, dummy_components
):
    # Arrange
    component1, component2 = dummy_components.pop(0), dummy_components.pop(0)
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)

    # Act
    assert instrument_coordinator.components() == ["ic_dev0", "ic_dev1"]
    instrument_coordinator.remove_component("ic_dev0")
    assert instrument_coordinator.components() == ["ic_dev1"]


def test_prepare(
    close_all_instruments, instrument_coordinator, dummy_components, mocker
):  # NB order of fixtures matters for teardown, keep mocker as last!
    # Arrange
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)

    get_component_spy = mocker.patch.object(
        instrument_coordinator,
        "get_component",
        wraps=instrument_coordinator.get_component,
    )

    # Act
    test_sched = Schedule(name="test_schedule")
    args = {"ic_dev0": {"foo": 0}, "ic_dev1": {"foo": 1}}
    test_sched["compiled_instructions"] = args
    compiled_sched = CompiledSchedule(test_sched)

    instrument_coordinator.prepare(compiled_sched)

    # Assert
    assert get_component_spy.call_args_list == [call("ic_dev0"), call("ic_dev1")]

    component1.prepare.assert_called_with(args["ic_dev0"])
    component2.prepare.assert_called_with(args["ic_dev1"])


def test_start(close_all_instruments, instrument_coordinator, dummy_components):
    # Arrange
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)

    # Act
    instrument_coordinator.start()

    # Assert
    component1.start.assert_called()
    component2.start.assert_called()


def test_stop(close_all_instruments, instrument_coordinator, dummy_components):
    # Arrange
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)

    # Act
    instrument_coordinator.stop()

    # Assert
    component1.stop.assert_called()
    component2.stop.assert_called()


def test_retrieve_acquisition(
    close_all_instruments, instrument_coordinator, dummy_components
):
    # Arrange
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)

    component1.retrieve_acquisition.return_value = {(0, 0): [1, 2, 3, 4]}
    component2.retrieve_acquisition.return_value = None

    # Act
    data = instrument_coordinator.retrieve_acquisition()

    # Assert
    component1.retrieve_acquisition.assert_called()
    component2.retrieve_acquisition.assert_called()
    assert {(0, 0): [1, 2, 3, 4]} == data


def test_reacquire_acquisition_successful(
    close_all_instruments, zi_instrument_coordinator, dummy_components
):
    # Arrange
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    zi_instrument_coordinator.add_component(component1)
    zi_instrument_coordinator.add_component(component2)

    component1.retrieve_acquisition.return_value = {(0, 0): [1, 2, 3, 4]}
    component2.retrieve_acquisition.return_value = None

    # Set the last cache of the ZIInstrumentCoordinator to something different
    zi_instrument_coordinator._last_acquisition = {(0, 0): [5, 6, 7, 8]}

    # Act
    data = zi_instrument_coordinator.retrieve_acquisition()

    # Assert
    component1.retrieve_acquisition.assert_called()
    component2.retrieve_acquisition.assert_called()
    assert zi_instrument_coordinator._last_acquisition is not None
    assert {(0, 0): [1, 2, 3, 4]} == data


def test_reacquire_acquisition_failed(
    close_all_instruments, zi_instrument_coordinator, dummy_components
):
    # Arrange
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    zi_instrument_coordinator.add_component(component1)
    zi_instrument_coordinator.add_component(component2)

    component1.retrieve_acquisition.return_value = {(0, 0): [1, 2, 3, 4]}
    component2.retrieve_acquisition.return_value = None

    # Assert
    with pytest.raises(RuntimeError):
        # Act
        zi_instrument_coordinator.retrieve_acquisition()


def test_wait_done(close_all_instruments, instrument_coordinator, dummy_components):
    # Arrange
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)

    timeout: int = 1

    # Act
    instrument_coordinator.wait_done(timeout)

    # Assert
    component1.wait_done.assert_called_with(timeout)
    component2.wait_done.assert_called_with(timeout)


def test_last_schedule(close_all_instruments, instrument_coordinator, dummy_components):
    component1 = dummy_components.pop(0)
    component2 = dummy_components.pop(0)
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)

    # assert that first there is no schedule prepared yet
    with pytest.raises(ValueError):
        instrument_coordinator.last_schedule

    test_sched = Schedule(name="test_schedule")
    compiled_sched = CompiledSchedule(test_sched)

    # assert that the uploaded schedule is retrieved
    instrument_coordinator.prepare(compiled_sched)
    last_sched = instrument_coordinator.last_schedule

    assert last_sched == compiled_sched
