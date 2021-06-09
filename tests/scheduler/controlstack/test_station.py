# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from __future__ import annotations

from unittest.mock import call

import pytest
from quantify.scheduler.controlstack import station
from quantify.scheduler.controlstack.components import base


def make_component(mocker, name: str) -> base.AbstractControlStackComponent:
    component = mocker.create_autospec(
        base.AbstractControlStackComponent, instance=True
    )
    component.name = name
    return component


def test_constructor():
    # Act
    controlstack = station.ControlStack()

    # Assert
    assert len(controlstack.components) == 0


def test_get_component(mocker):
    # Arrange
    controlstack = station.ControlStack()
    component1 = make_component(mocker, "dev1234")
    controlstack.add_component(component1)

    # Act
    component = controlstack.get_component("dev1234")

    # Assert
    assert component1 == component


def test_get_component_failed():
    # Arrange
    controlstack = station.ControlStack()

    # Act
    with pytest.raises(KeyError) as execinfo:
        controlstack.get_component("dev1234")

    # Assert
    assert execinfo.value.args[0] == "Device 'dev1234' not added to 'ControlStack'"


def test_add_component_failed():
    # Arrange
    controlstack = station.ControlStack()

    # Act
    with pytest.raises(TypeError) as execinfo:
        controlstack.add_component(object())

    # Assert
    assert execinfo.value.args[0] == (
        "Expected AbstractControlStackComponent and Instrument for component"
        " argument, instead got <class 'object'>"
    )


def test_prepare(mocker):
    # Arrange
    controlstack = station.ControlStack()
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)

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
    controlstack = station.ControlStack()
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)

    get_component_spy = mocker.patch.object(
        controlstack, "get_component", wraps=controlstack.get_component
    )

    # Act
    controlstack.start()

    # Assert
    assert get_component_spy.call_args_list == [
        call("dev1234"),
        call("dev5678"),
    ]
    component1.start.assert_called()
    component2.start.assert_called()


def test_stop(mocker):
    # Arrange
    controlstack = station.ControlStack()
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)

    get_component_spy = mocker.patch.object(
        controlstack, "get_component", wraps=controlstack.get_component
    )

    # Act
    controlstack.stop()

    # Assert
    assert get_component_spy.call_args_list == [
        call("dev1234"),
        call("dev5678"),
    ]
    component1.stop.assert_called()
    component2.stop.assert_called()


def test_retrieve_acquisition(mocker):
    # Arrange
    controlstack = station.ControlStack()
    component1 = make_component(mocker, "dev1234")
    component2 = make_component(mocker, "dev5678")
    controlstack.add_component(component1)
    controlstack.add_component(component2)

    component1.retrieve_acquisition.return_value = {0: [1, 2, 3, 4]}
    component2.retrieve_acquisition.return_value = None

    # Act
    data = controlstack.retrieve_acquisition()

    # Assert
    component1.retrieve_acquisition.assert_called()
    component2.retrieve_acquisition.assert_called()
    assert {"dev1234": {0: [1, 2, 3, 4]}} == data
