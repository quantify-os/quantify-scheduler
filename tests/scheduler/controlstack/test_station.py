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
from quantify_scheduler.controlstack import station
from quantify_scheduler.controlstack.components import base as base_component


def make_component(mocker, name: str) -> base_component.ControlStackComponentBase:
    mocker.patch("qcodes.instrument.Instrument.record_instance")
    component = mocker.create_autospec(
        base_component.ControlStackComponentBase, instance=True
    )
    component.name = name
    return component


def make_controlstack(mocker, name: str) -> station.ControlStack:
    mocker.patch("qcodes.instrument.Instrument.record_instance")
    controlstack = station.ControlStack(name)

    controlstack._mock_instr_dict = {}

    # add a mock find instrument
    def mock_find_instrument(instr_name: str):
        return controlstack._mock_instr_dict[instr_name]

    controlstack.find_instrument = mock_find_instrument

    return controlstack


def test_constructor(mocker):
    # Act
    controlstack = make_controlstack(mocker, "cs0")

    # Assert
    assert len(controlstack.components()) == 0


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
    controlstack = make_controlstack(mocker, "cs0")

    for i, state in enumerate(states):
        component = make_component(mocker, f"dev{i}")
        component.is_running = state
        controlstack.add_component(component)

        controlstack._mock_instr_dict[component.name] = component

    # Act
    is_running = controlstack.is_running

    # Assert
    assert is_running == expected


def test_get_component(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")
    controlstack.add_component(component1)

    # required for the mock find_instrument to work
    controlstack._mock_instr_dict[component1.name] = component1

    # Act
    component = controlstack.get_component("dev1234")

    # Assert
    assert component1 == component


def test_get_component_failed(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")

    # Act
    with pytest.raises(KeyError) as execinfo:
        controlstack.get_component("dev1234")

    # Assert
    assert execinfo.value.args[0] == "'dev1234' is not a component of cs0!"


def test_add_component_failed_duplicate(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")

    # Act
    with pytest.raises(ValueError) as execinfo:
        controlstack.add_component(component1)
        controlstack.add_component(component1)

    # Assert
    assert execinfo.value.args[0] == "'dev1234' has already been added!"


def test_add_component_failed_type_validation(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")

    @dataclass
    class DummyComponent:
        name: str

        def __repr__(self) -> str:
            return "<DummyComponent>"

    component = DummyComponent("abc")

    # Act
    with pytest.raises(TypeError) as execinfo:
        controlstack.add_component(component)

    # Assert
    assert execinfo.value.args[0] == (
        "<DummyComponent> is not quantify_scheduler.controlstack."
        "components.base.ControlStackComponentBase."
    )


def test_remove_component(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)
    # required for the mock find_instrument to work
    controlstack._mock_instr_dict[component1.name] = component1
    controlstack._mock_instr_dict[component2.name] = component2

    # Act
    assert controlstack.components() == ["dev1234", "dev5678"]
    controlstack.remove_component("dev1234")
    assert controlstack.components() == ["dev5678"]


def test_prepare(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")

    controlstack.add_component(component1)
    controlstack.add_component(component2)
    # required for the mock find_instrument to work
    controlstack._mock_instr_dict[component1.name] = component1
    controlstack._mock_instr_dict[component2.name] = component2

    get_component_spy = mocker.patch.object(
        controlstack, "get_component", wraps=controlstack.get_component
    )

    # Act
    args = {"dev1234": {"foo": 1}, "dev5678": {"foo": 2}}
    controlstack.prepare(args)

    # Assert
    assert get_component_spy.call_args_list == [
        call("dev1234"),
        call("dev5678"),
    ]
    component1.prepare.assert_called_with(args["dev1234"])
    component2.prepare.assert_called_with(args["dev5678"])


def test_start(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)
    # required for the mock find_instrument to work
    controlstack._mock_instr_dict[component1.name] = component1
    controlstack._mock_instr_dict[component2.name] = component2

    # Act
    controlstack.start()

    # Assert
    component1.start.assert_called()
    component2.start.assert_called()


def test_stop(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)
    # required for the mock find_instrument to work
    controlstack._mock_instr_dict[component1.name] = component1
    controlstack._mock_instr_dict[component2.name] = component2

    # Act
    controlstack.stop()

    # Assert
    component1.stop.assert_called()
    component2.stop.assert_called()


def test_retrieve_acquisition(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)
    # required for the mock find_instrument to work
    controlstack._mock_instr_dict[component1.name] = component1
    controlstack._mock_instr_dict[component2.name] = component2

    component1.retrieve_acquisition.return_value = {0: [1, 2, 3, 4]}
    component2.retrieve_acquisition.return_value = None

    # Act
    data = controlstack.retrieve_acquisition()

    # Assert
    component1.retrieve_acquisition.assert_called()
    component2.retrieve_acquisition.assert_called()
    assert {"dev1234": {0: [1, 2, 3, 4]}} == data


def test_wait_done(mocker):
    # Arrange
    controlstack = make_controlstack(mocker, "cs0")
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)
    # required for the mock find_instrument to work
    controlstack._mock_instr_dict[component1.name] = component1
    controlstack._mock_instr_dict[component2.name] = component2

    timeout: int = 1

    # Act
    controlstack.wait_done(timeout)

    # Assert
    component1.wait_done.assert_called_with(timeout)
    component2.wait_done.assert_called_with(timeout)
