# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from __future__ import annotations
from typing import List
import gc

from dataclasses import dataclass
from unittest.mock import call

import pytest
from qcodes import Instrument
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
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


@pytest.fixture(autouse=True, name="close_all_instruments")
def fixture_close_all_instruments():
    """Makes sure that after startup and teardown all instruments are closed"""
    Instrument.close_all()
    yield
    Instrument.close_all()

# cretes a few dummy compoents avialable to be used in each test
@pytest.fixture(scope="function", name="dummy_components")
def fixture_dummy_components(request, mocker) -> base_component.InstrumentCoordinatorComponentBase:

    mocker.patch.object(MyICC, "is_running")
    mocker.patch.object(MyICC, "start")
    mocker.patch.object(MyICC, "stop")
    mocker.patch.object(MyICC, "prepare")
    mocker.patch.object(MyICC, "retrieve_acquisition")
    mocker.patch.object(MyICC, "wait_done")

    # Crete a QCoDeS intrument for realistic emulation
    instruments = [Instrument(f"dev{i}") for i in range(3)]
    components = [MyICC(instrument) for instrument in instruments]

    def cleanup_tmp():
        # This also prevent the garbage collector from colleting the qcodes instrument
        for instrument in instruments:
            instrument.close()

    request.addfinalizer(cleanup_tmp)

    return components

@pytest.fixture(scope="function", name="instrument_coordinator")
def fixture_instrument_coordinator(request) -> InstrumentCoordinator:
    instrument_coordinator = InstrumentCoordinator("ic_0000")

    def cleanup_tmp():
        # This also prevent the garbage collector from colleting the qcodes instrument
        instrument_coordinator.close()

    request.addfinalizer(cleanup_tmp)

    return instrument_coordinator


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
    expected: bool
):
    # Arrange

    for state in states:
        # popping ensures the reference to the object is released after this for loop
        component = dummy_components.pop(0)
        component.is_running = state
        instrument_coordinator.add_component(component)

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


# def test_get_component_failed(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")

#     # Act
#     with pytest.raises(KeyError) as execinfo:
#         instrument_coordinator.get_component("dev1234")

#     # Assert
#     assert execinfo.value.args[0] == "'dev1234' is not a component of ic0!"


# def test_add_component_failed_duplicate(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
#     component1 = make_component(mocker, "dev1234")

#     # Act
#     with pytest.raises(ValueError) as execinfo:
#         instrument_coordinator.add_component(component1)
#         instrument_coordinator.add_component(component1)

#     # Assert
#     assert execinfo.value.args[0] == "'dev1234' has already been added!"


# def test_add_component_failed_type_validation(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")

#     @dataclass
#     class DummyComponent:
#         name: str

#         def __repr__(self) -> str:
#             return "<DummyComponent>"

#     component = DummyComponent("abc")

#     # Act
#     with pytest.raises(TypeError) as execinfo:
#         instrument_coordinator.add_component(component)

#     # Assert
#     assert execinfo.value.args[0] == (
#         "<DummyComponent> is not quantify_scheduler.instrument_coordinator."
#         "components.base.InstrumentCoordinatorComponentBase."
#     )


# def test_remove_component(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
#     component1 = make_component(mocker, "dev1234")
#     component2 = make_component(mocker, "dev5678")
#     instrument_coordinator.add_component(component1)
#     instrument_coordinator.add_component(component2)
#     # required for the mock find_instrument to work
#     instrument_coordinator._mock_instr_dict[component1.name] = component1
#     instrument_coordinator._mock_instr_dict[component2.name] = component2

#     # Act
#     assert instrument_coordinator.components() == ["dev1234", "dev5678"]
#     instrument_coordinator.remove_component("dev1234")
#     assert instrument_coordinator.components() == ["dev5678"]


# def test_prepare(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
#     component1 = make_component(mocker, "dev1234")
#     component2 = make_component(mocker, "dev5678")

#     instrument_coordinator.add_component(component1)
#     instrument_coordinator.add_component(component2)
#     # required for the mock find_instrument to work
#     instrument_coordinator._mock_instr_dict[component1.name] = component1
#     instrument_coordinator._mock_instr_dict[component2.name] = component2

#     get_component_spy = mocker.patch.object(
#         instrument_coordinator,
#         "get_component",
#         wraps=instrument_coordinator.get_component,
#     )

#     # Act
#     args = {"dev1234": {"foo": 1}, "dev5678": {"foo": 2}}
#     instrument_coordinator.prepare(args)

#     # Assert
#     assert get_component_spy.call_args_list == [
#         call("dev1234"),
#         call("dev5678"),
#     ]
#     component1.prepare.assert_called_with(args["dev1234"])
#     component2.prepare.assert_called_with(args["dev5678"])


# def test_start(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
#     component1 = make_component(mocker, "dev1234")
#     component2 = make_component(mocker, "dev5678")
#     instrument_coordinator.add_component(component1)
#     instrument_coordinator.add_component(component2)
#     # required for the mock find_instrument to work
#     instrument_coordinator._mock_instr_dict[component1.name] = component1
#     instrument_coordinator._mock_instr_dict[component2.name] = component2

#     # Act
#     instrument_coordinator.start()

#     # Assert
#     component1.start.assert_called()
#     component2.start.assert_called()


# def test_stop(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
#     component1 = make_component(mocker, "dev1234")
#     component2 = make_component(mocker, "dev5678")
#     instrument_coordinator.add_component(component1)
#     instrument_coordinator.add_component(component2)
#     # required for the mock find_instrument to work
#     instrument_coordinator._mock_instr_dict[component1.name] = component1
#     instrument_coordinator._mock_instr_dict[component2.name] = component2

#     # Act
#     instrument_coordinator.stop()

#     # Assert
#     component1.stop.assert_called()
#     component2.stop.assert_called()


# def test_retrieve_acquisition(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
#     component1 = make_component(mocker, "dev1234")
#     component2 = make_component(mocker, "dev5678")
#     instrument_coordinator.add_component(component1)
#     instrument_coordinator.add_component(component2)
#     # required for the mock find_instrument to work
#     instrument_coordinator._mock_instr_dict[component1.name] = component1
#     instrument_coordinator._mock_instr_dict[component2.name] = component2

#     component1.retrieve_acquisition.return_value = {0: [1, 2, 3, 4]}
#     component2.retrieve_acquisition.return_value = None

#     # Act
#     data = instrument_coordinator.retrieve_acquisition()

#     # Assert
#     component1.retrieve_acquisition.assert_called()
#     component2.retrieve_acquisition.assert_called()
#     assert {"dev1234": {0: [1, 2, 3, 4]}} == data


# def test_wait_done(mocker):
#     # Arrange
#     instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
#     component1 = make_component(mocker, "dev1234")
#     component2 = make_component(mocker, "dev5678")
#     instrument_coordinator.add_component(component1)
#     instrument_coordinator.add_component(component2)
#     # required for the mock find_instrument to work
#     instrument_coordinator._mock_instr_dict[component1.name] = component1
#     instrument_coordinator._mock_instr_dict[component2.name] = component2

#     timeout: int = 1

#     # Act
#     instrument_coordinator.wait_done(timeout)

#     # Assert
#     component1.wait_done.assert_called_with(timeout)
#     component2.wait_done.assert_called_with(timeout)
