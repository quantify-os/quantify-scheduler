# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the core concepts of the scheduler."""
from __future__ import annotations

import inspect
import json
import ast
import re
from collections import UserDict
from copy import deepcopy
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List, Type
from uuid import uuid4

import jsonschema
import numpy as np
from typing_extensions import Literal
from quantify.utilities import general
from quantify.scheduler.helpers import inspect as inspect_helpers
from quantify.scheduler import resources
from quantify.scheduler import enums
from quantify.scheduler import acquisition_library
from quantify.scheduler import gate_library
from quantify.scheduler import pulse_library


class Operation(UserDict):  # pylint: disable=too-many-ancestors
    """
    A JSON compatible data structure that contains information on
    how to represent the operation on the Gate, Pulse and/or Logical level.
    It also contains information on the
    :class:`~quantify.scheduler.resources.Resource` s used.

    An operation always has the following attributes:

    - duration (float): duration of the operation in seconds (can be 0).
    - hash (str): an auto generated unique identifier.
    - name (str): a readable identifier, does not have to be unique.

    An Operation can contain information  on several levels of abstraction.
    This information is used when different representations. Note that when
    initializing an operation  not all of this information needs to be available
    as operations are typically modified during the compilation steps.

    .. jsonschema:: schemas/operation.json


    .. note::

        Two different Operations containing the same information generate the
        same hash and are considered identical.
    """

    def __init__(self, name: str, data: dict = None):
        super().__init__()

        # ensure keys exist
        self.data["name"] = name
        self.data["gate_info"] = {}
        self.data["pulse_info"] = []
        self.data["acquisition_info"] = []
        self.data["logic_info"] = {}
        self._duration: float = 0

        if data is not None:
            self.data.update(data)
            self._deserialize()
            self._update()

    def __eq__(self, other) -> bool:
        """
        Returns the equality of two instances based on its content :code:`self.data`.

        Parameters
        ----------
        other :

        Returns
        -------
        :
        """
        return repr(self) == repr(other)

    def __str__(self) -> str:
        """
        Returns a concise string represenation which can be evaluated into a new
        instance using `eval(str(operation))` only when the data dictionary has
        not been modified.

        This representation is guaranteed to be unique.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """
        Returns the string representation  of this instance.

        This represenation can always be evalued to create a new instance.

        .. code-block::

            eval(repr(operation))

        Returns
        -------
        :
        """
        _data = self._serialize()
        data_str = f"{str(self)[:-1]}, data={_data})"
        return data_str

    def _update(self) -> None:
        """Update the Operation's internals."""

        def _get_operation_end(info) -> float:
            """Return the operation end in seconds."""
            return info["t0"] + info["duration"]

        # Iterate over the data and take longest duration
        self._duration = max(
            map(
                _get_operation_end,
                self.data["pulse_info"] + self.data["acquisition_info"],
            ),
            default=0,
        )

    @property
    def name(self) -> str:
        """Return the name of the operation."""
        return self.data["name"]

    @property
    def duration(self) -> float:
        """
        Determine the duration of the operation based on the pulses described in
        pulse_info.

        If the operation contains no pulse info, it is assumed to be ideal and
        have zero duration.
        """
        return self._duration

    @property
    def hash(self) -> int:
        """
        A hash based on the contents of the Operation.
        """
        return general.make_hash(self.data)

    @classmethod
    def _get_signature(cls, parameters: dict) -> str:
        """
        Returns the constructor call signature of this instance for serialization.

        The string constructor representation can be used to recreate the object
        using eval(signature).

        Parameters
        ----------
        parameters : dict
            The current data dictionary.

        Returns
        -------
        :
        """
        signature = inspect.signature(cls)

        def to_kwarg(key) -> str:
            """
            Returns a key-value pair in string format of a keyword argument.

            Parameters
            ----------
            key :

            Returns
            -------
            :
            """
            value = parameters[key]
            if isinstance(value, Enum):
                enum_value = value.value
                value = enum_value
            value = f"'{value}'" if isinstance(value, str) else value
            return f"{key}={value}"

        required_params = list(signature.parameters.keys())[:-1]
        kwargs_list = map(to_kwarg, required_params)

        return f'{cls.__name__}({",".join(kwargs_list)})'

    def add_gate_info(self, gate_operation: Operation) -> None:
        """
        Updates self.data['gate_info'] with contents of gate_operation.

        Parameters
        ----------
        gate_operation :
            an operation containing gate_info.
        """
        self.data["gate_info"].update(gate_operation.data["gate_info"])

    def add_pulse(self, pulse_operation: Operation) -> None:
        """
        Adds pulse_info of pulse_operation Operation to this Operation.

        Parameters
        ----------
        pulse_operation :
            an operation containing pulse_info.
        """
        self.data["pulse_info"] += pulse_operation.data["pulse_info"]
        self._update()

    def add_acquisition(self, acquisition_operation: Operation) -> None:
        """
        Adds acquisition_info of acquisition_operation Operation to this Operation.

        Parameters
        ----------
        acquisition_operation :
            an operation containing acquisition_info.
        """
        self.data["acquisition_info"] += acquisition_operation.data["acquisition_info"]
        self._update()

    def _serialize(self) -> Dict[str, Any]:
        """
        Serializes the data dictionary.

        Returns
        -------
        :
        """
        _data = deepcopy(self.data)
        if "unitary" in _data["gate_info"] and isinstance(
            _data["gate_info"]["unitary"], (np.generic, np.ndarray)
        ):
            _data["gate_info"]["unitary"] = np.array2string(
                _data["gate_info"]["unitary"], separator=", ", precision=9
            )

        for acq_info in _data["acquisition_info"]:
            if "bin_mode" in acq_info and isinstance(
                acq_info["bin_mode"], enums.BinMode
            ):
                acq_info["bin_mode"] = acq_info["bin_mode"].value

            for waveform in acq_info["waveforms"]:
                if "t" in waveform:
                    waveform["t"] = np.array2string(
                        waveform["t"], separator=", ", precision=9
                    )
                if "weights" in waveform:
                    waveform["weights"] = np.array2string(
                        waveform["weights"], separator=", ", precision=9
                    )

        return _data

    def _deserialize(self) -> None:
        """Deserializes the data dictionary."""
        if "unitary" in self.data["gate_info"] and isinstance(
            self.data["gate_info"]["unitary"], str
        ):
            self.data["gate_info"]["unitary"] = np.array(
                ast.literal_eval(self.data["gate_info"]["unitary"])
            )

        for acq_info in self.data["acquisition_info"]:
            if "bin_mode" in acq_info and isinstance(acq_info["bin_mode"], str):
                acq_info["bin_mode"] = enums.BinMode(acq_info["bin_mode"])

            for waveform in acq_info["waveforms"]:
                if "t" in waveform and isinstance(waveform["t"], str):
                    waveform["t"] = np.array(ast.literal_eval(waveform["t"]))
                if "weights" in waveform and isinstance(waveform["weights"], str):
                    waveform["weights"] = np.array(
                        ast.literal_eval(waveform["weights"])
                    )

    @classmethod
    def is_valid(cls, operation) -> bool:
        """Checks if the operation is valid according to its schema."""
        scheme = general.load_json_schema(__file__, "operation.json")
        jsonschema.validate(operation.data, scheme)
        _ = operation.hash  # test that the hash property evaluates
        return True  # if not exception was raised during validation

    @property
    def valid_gate(self) -> bool:
        """
        An operation is a valid gate if it contains information on how to represent
        the operation on the gate level.
        """
        if self.data["gate_info"]:
            return True
        return False

    @property
    def valid_pulse(self) -> bool:
        """
        An operation is a valid pulse if it contains information on how to represent
        the operation on the pulse level.
        """
        if self.data["pulse_info"]:
            return True
        return False

    @property
    def valid_acquisition(self) -> bool:
        """
        An operation is a valid acquisition if it contains information on how to
        represent the operation as a acquisition on the pulse level.
        """
        if len(self.data["acquisition_info"]) > 0:
            return True
        return False


class Schedule(UserDict):  # pylint: disable=too-many-ancestors
    """
    A collection of :class:`~Operation` objects and timing constraints
    that define relations between the operations.

    The Schedule data structure is based on a dictionary.
    This dictionary contains:

        - `operation_dict`     : a hash table containing the unique
            :class:`~Operation` s added to the schedule.
        - `timing_constraints` : a list of all timing constraints added between
            operations.

    .. jsonschema:: schemas/schedule.json

    """

    def __init__(self, name: str, repetitions: int = 1, data: dict = None):
        """
        Initialize a new instance of Schedule.

        Parameters
        ----------
        name :
            The name of the schedule
        repetitions :
            The amount of times the schedule will be repeated, by default 1
        data :
            A dictionary containing a pre-existing schedule., by default None

        Raises
        ------
        NotImplementedError
        """

        # validate the input data to ensure it is valid schedule data
        super().__init__()

        # ensure keys exist
        self.data["operation_dict"] = {}
        self.data["timing_constraints"] = []
        self.data["resource_dict"] = {}
        self.data["name"] = "nameless"
        self.data["repetitions"] = repetitions

        # This is used to define baseband pulses and is expected to always be present
        # in any schedule.
        self.add_resource(
            resources.BasebandClockResource(resources.BasebandClockResource.IDENTITY)
        )

        if name is not None:
            self.data["name"] = name

        if data is not None:
            self.data.update(data)

    @property
    def name(self) -> str:
        """Returns the name of the schedule."""
        return self.data["name"]

    @property
    def repetitions(self) -> int:
        """
        Returns the amount of times this Schedule will be repeated.

        Returns
        -------
        :
            The repetitions count.
        """
        return self.data["repetitions"]

    @repetitions.setter
    def repetitions(self, value: int):
        assert value > 0
        self.data["repetitions"] = int(value)

    @property
    def operations(self):
        """
        A dictionary of all unique operations used in the schedule.
        This specifies information on *what* operation to apply *where*.

        The keys correspond to the :attr:`~Operation.hash` and values are instances
        of :class:`~Operation`.
        """
        return self.data["operation_dict"]

    @property
    def timing_constraints(self):
        # pylint: disable=line-too-long
        """
        A list of dictionaries describing timing constraints between operations.

        Each item in the list is a dictionary with the following keys:

        :code:`['label', 'rel_time', 'ref_op', 'ref_pt_new', 'ref_pt', 'operation_hash']`

        The `label` is used as a unique identifier that can be used as a reference for
        other operations, the `operation_hash` refers to the hash of a unique operation
        in :attr:`~Schedule.operations`.
        """
        return self.data["timing_constraints"]

    @property
    def resources(self):
        """
        A dictionary containing resources. Keys are names (str),
        values are instances of :class:`~quantify.scheduler.resources.Resource` .
        """
        return self.data["resource_dict"]

    def to_json(self) -> str:
        """
        Converts the Schedule data structure to a JSON string.

        Returns
        -------
        :
            The json string result.
        """
        return json.dumps(self.data, cls=ScheduleJSONEncoder)

    @classmethod
    def from_json(cls, data: str) -> Schedule:
        """
        Converts the JSON data to a Schedule.

        Parameters
        ----------
        data :
            The JSON data.

        Returns
        -------
        :
            The Schedule object.
        """
        schedule_data = ScheduleJSONDecoder().decode(data)
        name = schedule_data["name"]

        return Schedule(name, data=schedule_data)

    def add_resources(self, resources_list: list):
        """Add wrapper for adding multiple resources"""
        for resource in resources_list:
            self.add_resource(resource)

    def add_resource(self, resource):
        """
        Add a resource such as a channel or qubit to the schedule.
        """
        assert resources.Resource.is_valid(resource)
        if resource.name in self.data["resource_dict"]:
            raise ValueError("Key {} is already present".format(resource.name))

        self.data["resource_dict"][resource.name] = resource

    def __repr__(self):
        return 'Schedule "{}" containing ({}) {}  (unique) operations.'.format(
            self.data["name"],
            len(self.data["operation_dict"]),
            len(self.data["timing_constraints"]),
        )

    @classmethod
    def is_valid(cls, schedule):
        """
        Checks the schedule validity according to its schema.
        """
        scheme = general.load_json_schema(__file__, "schedule.json")
        jsonschema.validate(schedule.data, scheme)
        return True  # if not exception was raised during validation

    # pylint: disable=too-many-arguments
    def add(
        self,
        operation: Operation,
        rel_time: float = 0,
        ref_op: str = None,
        ref_pt: Literal["start", "center", "end"] = "end",
        ref_pt_new: Literal["start", "center", "end"] = "start",
        label: str = None,
    ) -> str:
        """
        Add an :class:`~Operation` to the schedule and specify timing constraints.

        A timing constraint constrains the operation in time by specifying the time
        (:code:`"rel_time"`) between a reference operation and the added operation.
        The time can be specified with respect to the "start", "center", or "end" of
        the operations.
        The reference operation (:code:`"ref_op"`) is specified using its label
        property.

        Parameters
        ----------
        operation :
            The operation to add to the schedule
        rel_time :
            relative time between the reference operation and the added operation.
            the time is the time between the "ref_pt" in the reference operation and
            "ref_pt_new" of the operation that is added.
        ref_op :
            label of the reference operation. If set to :code:`None`, will default
            to the last added operation.
        ref_pt :
            reference point in reference operation must be one of
            ('start', 'center', 'end').
        ref_pt_new :
            reference point in added operation must be one of
            ('start', 'center', 'end').
        label :
            a unique string that can be used as an identifier when adding operations.
            if set to None, a random hash will be generated instead.
        Returns
        -------
        :
            returns the (unique) label of the last added operation.
        """
        assert isinstance(operation, Operation)

        if label is None:
            label = str(uuid4())

        # assert that the label of the operation does not exists in the
        # timing constraints.
        label_is_unique = (
            len(
                [
                    item
                    for item in self.data["timing_constraints"]
                    if item["label"] == label
                ]
            )
            == 0
        )
        if not label_is_unique:
            raise ValueError('label "{}" must be unique'.format(label))

        # assert that the reference operation exists
        if ref_op is not None:
            ref_exists = (
                len(
                    [
                        item
                        for item in self.data["timing_constraints"]
                        if item["label"] == ref_op
                    ]
                )
                == 1
            )
            if not ref_exists:
                raise ValueError(
                    'Reference "{}" does not exist in schedule.'.format(ref_op)
                )

        operation_id = str(operation)
        self.data["operation_dict"][operation_id] = operation
        timing_constr = {
            "label": label,
            "rel_time": rel_time,
            "ref_op": ref_op,
            "ref_pt_new": ref_pt_new,
            "ref_pt": ref_pt,
            "operation_hash": operation_id,
        }
        self.data["timing_constraints"].append(timing_constr)

        return label


class ScheduleJSONDecoder(json.JSONDecoder):
    """
    The Quantify Schedule JSONDecoder.

    The ScheduleJSONDecoder is used to convert a string with JSON content into a
    :class:`~quantify.scheduler.types.Schedule`.

    To avoid the execution of malicious code ScheduleJSONDecoder uses
    :func:`ast.literal_eval` instead of :func:`eval` to convert the data to an instance
    of Schedule.
    """

    classes: Dict[str, Type[Any]]

    def __init__(self, *args, **kwargs) -> None:
        """
        Create new instance of ScheduleJSONDecoder to decode a string into a Schedule.

        The list of serializable classes can be extended with custom classes by
        providing the `modules` keyword argument. These classes have to overload the
        :func:`__str__` and :func:`__repr__` functions in order to serialize and
        deserialize domain objects into a valid JSON-format.

        Keyword Arguments
        -----------------
        modules : List[ModuleType], optional
            A list of custom modules containing serializable classes, by default []
        """
        extended_modules: List[ModuleType] = kwargs.pop("modules", list())
        assert all(isinstance(o, ModuleType) for o in extended_modules)

        super().__init__(
            object_hook=self.custom_object_hook,
            *args,
            **kwargs,
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
        obj :
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
        :attr:`~.ScheduleJSONDecoder.classes` property.

        For better security the usage of `eval` has been replaced in favour of
        :func:`ast.literal_eval`.

        Parameters
        ----------
        obj : str
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
            elif isinstance(node, ast.Call):
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
        obj :
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
        if isinstance(o, (Operation, resources.Resource)):
            return repr(o)
        if hasattr(o, "__dict__"):
            return o.__dict__

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)
