# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Unit tests for the DataStructure, and custom json (de)serialization."""
import pytest

from typing import Any, Callable, Type, Union

from pydantic import validator
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.structure.model import (
    DataStructure,
    deserialize_class,
    deserialize_function,
)


class DummyStructure(DataStructure):
    name: str
    func: Callable[[int], int]

    @validator("func", pre=True)
    @classmethod
    def import_func_if_str(
        cls, fun: Union[str, Callable[[int], int]]
    ) -> Callable[[int], int]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore


class DummyStructure2(DataStructure):
    name: str
    func: Callable[[str], Any]

    @validator("func", pre=True)
    @classmethod
    def import_func_if_str(
        cls, fun: Union[str, Callable[[str], Any]]
    ) -> Callable[[str], Any]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore


class DummyStructure3(DataStructure):
    name: str
    cls: Type[DataStructure]

    @validator("cls", pre=True)
    @classmethod
    def import_class_if_str(
        cls, class_: Union[str, Type[DataStructure]]
    ) -> Type[DataStructure]:
        if isinstance(class_, str):
            return deserialize_class(class_)
        return class_  # type: ignore


def foo(bar: int) -> int:
    return bar * 2


class Random:
    pass


class TestDataStructure:
    def test_constructor(self):
        _ = DummyStructure(name="foobar", func=foo)

    def test_json_loads1(self):
        dummy1 = DummyStructure.parse_raw(
            '{"name": "foobar", "func": "tests.scheduler.structure.test_model.foo"}'
        )
        dummy2 = DummyStructure(name="foobar", func=foo)
        assert dummy1 == dummy2

    def test_json_loads2(self):
        dummy1 = DummyStructure2.parse_raw(
            '{"name": "foobar", "func": "quantify_scheduler.helpers.importers.import_python_object_from_string"}'
        )
        dummy2 = DummyStructure2(name="foobar", func=import_python_object_from_string)
        assert dummy1 == dummy2

    def test_json_loads3(self):
        dummy1 = DummyStructure3.parse_raw(
            '{"name": "foobar", "cls": "quantify_scheduler.structure.model.DataStructure"}'
        )
        dummy2 = DummyStructure3(name="foobar", cls=DataStructure)
        assert dummy1 == dummy2

    def test_bad_function(self):
        with pytest.raises(ValueError):
            _ = DummyStructure(name="", func=3)
        with pytest.raises(ValueError):
            _ = DummyStructure(name="", func="quantify_scheduler.does.not.exist")

    def test_bad_class(self):
        with pytest.raises(ValueError):
            _ = DummyStructure3(name="", cls=Random)
        with pytest.raises(ValueError):
            _ = DummyStructure3(name="", func="quantify_scheduler.does.not.exist")

    def test_json_dumps1(self):
        dummy = DummyStructure(name="foobar", func=foo)
        json_str = dummy.json()
        assert (
            json_str
            == '{"name":"foobar","func":"tests.scheduler.structure.test_model.foo"}'
        )

    def test_json_dumps2(self):
        dummy = DummyStructure2(name="foobar", func=import_python_object_from_string)
        json_str = dummy.json()
        assert (
            json_str
            == '{"name":"foobar","func":"quantify_scheduler.helpers.importers.import_python_object_from_string"}'
        )

    def test_json_dumps3(self):
        dummy = DummyStructure3(name="foobar", cls=DataStructure)
        json_str = dummy.json()
        assert (
            json_str
            == '{"name":"foobar","cls":"quantify_scheduler.structure.model.DataStructure"}'
        )
