# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing quantify JSON utilities."""
from __future__ import annotations

import functools
import json
import pathlib
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List, Type, Union

import fastjsonschema
import numpy as np

from qcodes import Instrument

from quantify_scheduler.helpers import inspect as inspect_helpers
from quantify_scheduler import enums

current_python_version = sys.version_info

lru_cache = functools.lru_cache(maxsize=200)


def validate_json(data, schema):
    """Validate schema using jsonschema-rs"""
    return fastjsonschema.validate(schema, data)


def load_json_schema(relative_to: Union[str, pathlib.Path], filename: str):
    """
    Load a JSON schema from file. Expects a 'schemas' directory in the same directory
    as `relative_to`.

    .. tip::

        Typical usage of the form
        `schema = load_json_schema(__file__, 'definition.json')`

    Parameters
    ----------
    relative_to
        the file to begin searching from
    filename
        the JSON file to load
    Returns
    -------
    dict
        the schema
    """
    path = pathlib.Path(relative_to).resolve().parent.joinpath("schemas", filename)
    with path.open(mode="r", encoding="utf-8") as file:
        return json.load(file)


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
        extended_modules: List[ModuleType] = kwargs.pop("modules", [])
        invalid_modules = list(
            filter(lambda o: not isinstance(o, ModuleType), extended_modules)
        )
        if invalid_modules:
            raise ValueError(
                f"Attempting to create a Schedule decoder class ScheduleJSONDecoder. "
                f"The following modules provided are not an instance of the ModuleType:"
                f" {invalid_modules} ."
            )

        super().__init__(
            object_hook=self.custom_object_hook,
            *args,
            **kwargs,
        )

        # Use local import to void Error('Operation' from partially initialized module
        # 'quantify_scheduler')
        # pylint: disable=import-outside-toplevel
        from quantify_scheduler import resources
        from quantify_scheduler.schedules.schedule import (
            AcquisitionMetadata,
            Schedulable,
        )
        from quantify_scheduler.operations import (
            acquisition_library,
            gate_library,
            operation,
            nv_native_library,
            pulse_library,
            shared_native_library,
        )
        from quantify_scheduler.device_under_test import transmon_element

        self._modules: List[ModuleType] = [
            enums,
            operation,
            transmon_element,
            acquisition_library,
            gate_library,
            pulse_library,
            nv_native_library,
            shared_native_library,
            resources,
        ] + extended_modules
        self.classes = inspect_helpers.get_classes(*self._modules)
        self.classes.update({c.__name__: c for c in [AcquisitionMetadata, Schedulable]})
        self.classes.update(
            {
                t.__name__: t
                for t in [
                    complex,
                    float,
                    int,
                    bool,
                    str,
                    np.ndarray,
                    np.complex128,
                    np.int32,
                    np.int64,
                ]
            }
        )

    def decode_dict(
        self, obj: Dict[str, Any]
    ) -> Union[Dict[str, Any], np.ndarray, type]:
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
        # If "deserialization_type" is present in `obj` it means the object was
        # serialized using `__getstate__` and should be deserialized using
        # `__setstate__`.
        if "deserialization_type" in obj:
            class_type: Type = self.classes[obj["deserialization_type"]]

            if "mode" in obj and obj["mode"] == "__init__":
                if class_type == np.ndarray:
                    return np.array(obj["data"])
                if issubclass(class_type, Instrument):
                    return class_type(**obj["data"])
                return class_type(obj["data"])

            if "mode" in obj and obj["mode"] == "type":
                return class_type

            new_obj = class_type.__new__(class_type)
            new_obj.__setstate__(obj)
            return new_obj

        return obj

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
        object. It will try 3 different serialization methods which are, in order,
        check if the object is to be serialized to a string using repr. If not, try
        to use `__getstate__`. Finally, try to serialize the `__dict__` property.
        """
        if hasattr(o, "__getstate__"):
            return o.__getstate__()
        if isinstance(o, (complex, np.int32, np.complex128, np.int64, enums.BinMode)):
            return {
                "deserialization_type": type(o).__name__,
                "mode": "__init__",
                "data": str(o),
            }
        if isinstance(o, (np.ndarray,)):
            return {
                "deserialization_type": type(o).__name__,
                "mode": "__init__",
                "data": list(o),
            }
        if o in [
            complex,
            float,
            int,
            bool,
            str,
            np.ndarray,
            np.complex128,
            np.int32,
            np.int64,
        ]:
            return {"deserialization_type": o.__name__, "mode": "type"}
        if hasattr(o, "__dict__"):
            return o.__dict__

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)
