# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from __future__ import annotations
from typing import List

from dataclasses import dataclass
from unittest.mock import call

import pytest
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components import base as base_component


def make_component(
    mocker, name: str
) -> base_component.InstrumentCoordinatorComponentBase:
    mocker.patch("qcodes.instrument.Instrument.record_instance")
    component = mocker.create_autospec(
        base_component.InstrumentCoordinatorComponentBase, instance=True
    )
    component.name = name
    return component


def make_instrument_coordinator(mocker, name: str) -> InstrumentCoordinator:
    mocker.patch("qcodes.instrument.Instrument.record_instance")
    instrument_coordinator = InstrumentCoordinator(name)

    instrument_coordinator._mock_instr_dict = {}

    # add a mock find instrument
    def mock_find_instrument(instr_name: str):
        return instrument_coordinator._mock_instr_dict[instr_name]

    instrument_coordinator.find_instrument = mock_find_instrument

    return instrument_coordinator


def test_constructor(mocker):
    # Act
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")

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
def test_is_running(mocker, states: List[bool], expected: bool):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")

    for i, state in enumerate(states):
        component = make_component(mocker, f"dev{i}")
        component.is_running = state
        instrument_coordinator.add_component(component)

        instrument_coordinator._mock_instr_dict[component.name] = component

    # Act
    is_running = instrument_coordinator.is_running

    # Assert
    assert is_running == expected


def test_get_component(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")
    instrument_coordinator.add_component(component1)

    # required for the mock find_instrument to work
    instrument_coordinator._mock_instr_dict[component1.name] = component1

    # Act
    component = instrument_coordinator.get_component("dev1234")

    # Assert
    assert component1 == component


def test_get_component_failed(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")

    # Act
    with pytest.raises(KeyError) as execinfo:
        instrument_coordinator.get_component("dev1234")

    # Assert
    assert execinfo.value.args[0] == "'dev1234' is not a component of ic0!"


def test_add_component_failed_duplicate(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")

    # Act
    with pytest.raises(ValueError) as execinfo:
        instrument_coordinator.add_component(component1)
        instrument_coordinator.add_component(component1)

    # Assert
    assert execinfo.value.args[0] == "'dev1234' has already been added!"


def test_add_component_failed_type_validation(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")

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


def test_remove_component(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)
    # required for the mock find_instrument to work
    instrument_coordinator._mock_instr_dict[component1.name] = component1
    instrument_coordinator._mock_instr_dict[component2.name] = component2

    # Act
    assert instrument_coordinator.components() == ["dev1234", "dev5678"]
    instrument_coordinator.remove_component("dev1234")
    assert instrument_coordinator.components() == ["dev5678"]


def test_prepare(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")

    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)
    # required for the mock find_instrument to work
    instrument_coordinator._mock_instr_dict[component1.name] = component1
    instrument_coordinator._mock_instr_dict[component2.name] = component2

    get_component_spy = mocker.patch.object(
        instrument_coordinator,
        "get_component",
        wraps=instrument_coordinator.get_component,
    )

    # Act
    args = {"dev1234": {"foo": 1}, "dev5678": {"foo": 2}}
    instrument_coordinator.prepare(args)

    # Assert
    assert get_component_spy.call_args_list == [
        call("dev1234"),
        call("dev5678"),
    ]
    component1.prepare.assert_called_with(args["dev1234"])
    component2.prepare.assert_called_with(args["dev5678"])


def test_start(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)
    # required for the mock find_instrument to work
    instrument_coordinator._mock_instr_dict[component1.name] = component1
    instrument_coordinator._mock_instr_dict[component2.name] = component2

    # Act
    instrument_coordinator.start()

    # Assert
    component1.start.assert_called()
    component2.start.assert_called()


def test_stop(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)
    # required for the mock find_instrument to work
    instrument_coordinator._mock_instr_dict[component1.name] = component1
    instrument_coordinator._mock_instr_dict[component2.name] = component2

    # Act
    instrument_coordinator.stop()

    # Assert
    component1.stop.assert_called()
    component2.stop.assert_called()


def test_retrieve_acquisition(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)
    # required for the mock find_instrument to work
    instrument_coordinator._mock_instr_dict[component1.name] = component1
    instrument_coordinator._mock_instr_dict[component2.name] = component2

    component1.retrieve_acquisition.return_value = {0: [1, 2, 3, 4]}
    component2.retrieve_acquisition.return_value = None

    # Act
    data = instrument_coordinator.retrieve_acquisition()

    # Assert
    component1.retrieve_acquisition.assert_called()
    component2.retrieve_acquisition.assert_called()
    assert {"dev1234": {0: [1, 2, 3, 4]}} == data


def test_wait_done(mocker):
    # Arrange
    instrument_coordinator = make_instrument_coordinator(mocker, "ic0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    instrument_coordinator.add_component(component1)
    instrument_coordinator.add_component(component2)
    # required for the mock find_instrument to work
    instrument_coordinator._mock_instr_dict[component1.name] = component1
    instrument_coordinator._mock_instr_dict[component2.name] = component2

    timeout: int = 1

    # Act
    instrument_coordinator.wait_done(timeout)

    # Assert
    component1.wait_done.assert_called_with(timeout)
    component2.wait_done.assert_called_with(timeout)
