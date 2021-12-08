# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing quantify JSON utilities."""
from __future__ import annotations

import ast
import functools
import json
import pathlib
import re
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List, Type, Union

import fastjsonschema
from quantify_core.utilities.general import load_json_schema

from quantify_scheduler.helpers import inspect as inspect_helpers

current_python_version = sys.version_info

lru_cache = functools.lru_cache(maxsize=200)


def validate_json(data, schema):
    """Validate schema using jsonschema-rs"""
    return fastjsonschema.validate(schema, data)


@lru_cache
def load_json_validator(
    relative_to: Union[str, pathlib.Path], filename: str
) -> Callable:
    """
    Load a JSON validator from file. Expects a 'schemas' directory in the same directory
    as `relative_to`.


    Parameters
    ----------
    relative_to
        the file to begin searching from
    filename
        the JSON file to load
    Returns
    -------
    Callable
        The validator
    """
    definition = load_json_schema(relative_to, filename)
    validator = fastjsonschema.compile(definition, handlers={}, formats={})
    return validator


class JSONSchemaValMixin:  # pylint: disable=too-few-public-methods
    """
    A mixin that adds validation utilities to classes that have
    a data attribute like a :class:`UserDict` based on JSONSchema.

    This requires the class to have a class variable "schema_filename"
    """

    @classmethod
    def is_valid(cls, object_to_be_validated) -> bool:
        """Checks if the object is valid according to its schema

        Raises
        ------
        fastjsonschema.JsonSchemaException
            if the data is invalid

        Returns
        -------
        :

        """

        validator_method = load_json_validator(__file__, cls.schema_filename)
        validator_method(object_to_be_validated.data)
        return True  # if no exception was raised during validation


class ScheduleJSONDecoder(json.JSONDecoder):
    """
    The Quantify Schedule JSONDecoder.

    The ScheduleJSONDecoder is used to convert a string with JSON content into a
    :class:`quantify_scheduler.schedules.schedule.Schedule`.

    To avoid the execution of malicious code ScheduleJSONDecoder uses
    :func:`ast.literal_eval` instead of :func:`eval` to convert the data to an instance
    of Schedule.
    """

    classes: Dict[str, Type[Any]]

    def __init__(self, *args, **kwargs) -> None:
        """
        Create new instance of ScheduleJSONDecoder to decode a string into a Schedule.

        The list of serializable classes can be extended with custom classes by
        providing the `modules` keyword argument. These classes have to implement
        :class:`quantify_scheduler.operations.operation.Operation` and overload the
        :code:`__str__` and :code:`__repr__` methods in order to serialize and
        deserialize domain objects into a valid JSON-format.

        Keyword Arguments
        -----------------
        modules : List[ModuleType], *optional*
            A list of custom modules containing serializable classes, by default []
        """
        extended_modules: List[ModuleType] = kwargs.pop("modules", list())
        assert all(isinstance(o, ModuleType) for o in extended_modules)

        super().__init__(
            object_hook=self.custom_object_hook,
            *args,
            **kwargs,
        )

        # Use local import to void Error('Operation' from partially initialized module
        # 'quantify_scheduler')
        # pylint: disable=import-outside-toplevel
        from quantify_scheduler import resources
        from quantify_scheduler.operations import (  # pylint: disable=import-outside-toplevel
            acquisition_library,
            gate_library,
            pulse_library,
        )

        self._modules: List[ModuleType] = [
            gate_library,
            pulse_library,
            acquisition_library,
            resources,
        ] + extended_modules
        self.classes = inspect_helpers.get_classes(*self._modules)

    def decode_dict(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the deserialized JSON dictionary.

        Parameters
        ----------
        obj
            The dictionary to deserialize.

        Returns
        -------
        :
            The deserialized result.
        """
        for key in obj:
            value = obj[key]
            if isinstance(value, str):
                # Check if the string has a signature of a constructor
                # example: Reset('q0', data={...})
                if not re.match(r"^(\w+)\(.*\)$", value):
                    return obj
                obj[key] = self.decode_quantify_type(value)
        return obj

    def decode_quantify_type(self, obj: str) -> object:
        """
        Returns the deserialized result of a possible known type stored in the
        :class:`~.ScheduleJSONDecoder` .classes property.

        For better security the usage of `eval` has been replaced in favour of
        :func:`ast.literal_eval`.

        Parameters
        ----------
        obj
            The value of dictionary pair to deserialize.

        Returns
        -------
        :
            The decoded result.
        """
        kwargs = dict()
        args = list()
        ast_tree = ast.parse(obj)
        class_name: str = ""
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Load):
                break
            if isinstance(node, ast.Call):
                class_name = node.func.id
            elif isinstance(node, ast.Constant):
                args.append(node.value)
            elif isinstance(node, ast.keyword):
                kwargs[node.arg] = ast.literal_eval(node.value)

        if class_name not in self.classes:
            return obj

        class_type: type = self.classes[class_name]
        return class_type(*args, **kwargs)

    def custom_object_hook(self, obj: object) -> object:
        """
        The `object_hook` hook will be called with the result of every JSON object
        decoded and its return value will be used in place of the given ``dict``.

        Parameters
        ----------
        obj
            A pair of JSON objects.

        Returns
        -------
        :
            The deserialized result.
        """
        if isinstance(obj, dict):
            return self.decode_dict(obj)
        return obj


class ScheduleJSONEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder which encodes the quantify Schedule into a JSON file format
    string.
    """

    def default(self, o):
        """
        Overloads the json.JSONEncoder default method that returns a serializable
        object.
        """
        # Use local import to void Error('Operation' from partially initialized module
        # 'quantify_scheduler')
        from quantify_scheduler import (  # pylint: disable=import-outside-toplevel
            Operation,
            resources,
        )

        if isinstance(o, (Operation, resources.Resource)):
            return repr(o)
        if hasattr(o, "__dict__"):
            return o.__dict__

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)
