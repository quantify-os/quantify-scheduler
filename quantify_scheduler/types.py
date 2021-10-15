# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the core concepts of the scheduler."""
from __future__ import annotations

import logging
import inspect
import json
import ast
from abc import ABC
from collections import UserDict
from dataclasses import dataclass
from copy import deepcopy
from pydoc import locate
from enum import Enum
from typing import Any, Dict, List, Type, TYPE_CHECKING
from uuid import uuid4

import numpy as np
from typing_extensions import Literal
from quantify_core.utilities import general
from quantify_scheduler import json_utils
from quantify_scheduler.json_utils import JSONSchemaValMixin
from quantify_scheduler import resources
from quantify_scheduler import enums


if TYPE_CHECKING:
    from quantify_scheduler.resources import Resource


class Operation(JSONSchemaValMixin, UserDict):  # pylint: disable=too-many-ancestors
    """
    A JSON compatible data structure that contains information on
    how to represent the operation on the quantum-circuit and/or the quantum-device
    layer.
    It also contains information on where the operation should be applied: the
    :class:`~quantify_scheduler.resources.Resource` s used.

    An operation always has the following attributes:

    - duration (float): duration of the operation in seconds (can be 0).
    - hash (str): an auto generated unique identifier.
    - name (str): a readable identifier, does not have to be unique.



    An Operation can contain information  on several levels of abstraction.
    This information is used when different representations are required. Note that when
    initializing an operation  not all of this information needs to be available
    as operations are typically modified during the compilation steps.

    .. tip::

        :mod:`quantify_scheduler` comes with a :mod:`~quantify_scheduler.gate_library`
        and a :mod:`~quantify_scheduler.pulse_library` , both containing
        common operations.


    **JSON schema of a valid Operation**

    .. jsonschema:: schemas/operation.json


    .. note::

        Two different Operations containing the same information generate the
        same hash and are considered identical.
    """

    schema_filename = "operation.json"
    _class_signature = None

    def __init__(self, name: str, data: dict = None) -> None:
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
        Returns a concise string representation which can be evaluated into a new
        instance using `eval(str(operation))` only when the data dictionary has
        not been modified.

        This representation is guaranteed to be unique.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """
        Returns the string representation  of this instance.

        This representation can always be evaluated to create a new instance.

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
        if cls._class_signature is None:
            logging.info(f"Caching signature for class {cls.__name__}")
            cls._class_signature = inspect.signature(cls)
        signature = cls._class_signature

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

            # types lead to problems when serialized without casting to string first
            if "<class " in str(acq_info["acq_return_type"]):
                acq_info["acq_return_type"] = str(acq_info["acq_return_type"])

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

            # FIXME # pylint: disable=fixme
            # this workaround is required because we cannot easily specify types and
            # serialize easy. We should change the implementation to dataclasses #159
            if "<class " in str(acq_info["acq_return_type"]):
                # first remove the class prefix
                return_type_str = str(acq_info["acq_return_type"])[7:].strip("'>")
                # and then use locate to retrieve the type class
                acq_info["acq_return_type"] = locate(return_type_str)

            for waveform in acq_info["waveforms"]:
                if "t" in waveform and isinstance(waveform["t"], str):
                    waveform["t"] = np.array(ast.literal_eval(waveform["t"]))
                if "weights" in waveform and isinstance(waveform["weights"], str):
                    waveform["weights"] = np.array(
                        ast.literal_eval(waveform["weights"])
                    )

    @classmethod
    def is_valid(cls, object_to_be_validated) -> bool:
        """
        Checks if the contents of the object_to_be_validated are valid
        according to the schema.
        Additionally checks if the hash property of the object evaluates correctly.
        """
        valid_operation = super().is_valid(object_to_be_validated)
        if valid_operation:
            _ = object_to_be_validated.hash  # test that the hash property evaluates
            return True

        return False

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


# pylint: disable=too-many-ancestors
class ScheduleBase(JSONSchemaValMixin, UserDict, ABC):
    """
    The :class:`~quantify_scheduler.types.ScheduleBase` is a data structure that is at
    the core of the Quantify-scheduler and describes when what operations are applied
    where.

    The :class:`~quantify_scheduler.types.ScheduleBase` is a collection of
    :class:`~Operation` objects and timing constraints that define relations between
    the operations.

    The schedule data structure is based on a dictionary.
    This dictionary contains:

    - operation_dict - a hash table containing the unique :class:`~Operation` s added
        to the schedule.
    - timing_constraints - a list of all timing constraints added between operations.

    The :class:`~quantify_scheduler.types.Schedule` provides an API to create schedules.
    The :class:`~quantify_scheduler.types.CompiledSchedule` represents a schedule after
    it has been compiled for execution on a backend.


    The :class:`~quantify_scheduler.types.Schedule` contains information on the
    :attr:`~quantify_scheduler.types.ScheduleBase.operations` and
    :attr:`~quantify_scheduler.types.ScheduleBase.timing_constraints`.
    The :attr:`~quantify_scheduler.types.ScheduleBase.operations` is a dictionary of all
    unique operations used in the schedule and contain the information on *what*
    operation to apply *where*.
    The :attr:`~quantify_scheduler.types.ScheduleBase.timing_constraints` is a list of
    dictionaries describing timing constraints between operations, i.e. when to apply
    an operation.


    **JSON schema of a valid Schedule**

    .. jsonschema:: schemas/schedule.json

    """

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
    def operations(self) -> Dict[str, Operation]:
        """
        A dictionary of all unique operations used in the schedule.
        This specifies information on *what* operation to apply *where*.

        The keys correspond to the :attr:`~Operation.hash` and values are instances
        of :class:`~Operation`.
        """
        return self.data["operation_dict"]

    @property
    def timing_constraints(self) -> List[Dict[str, Any]]:
        """
        A list of dictionaries describing timing constraints between operations.

        A timing constraint constrains the operation in time by specifying the time
        (:code:`"rel_time"`) between a reference operation and the added operation.
        The time can be specified with respect to a reference point (:code:`"ref_pt"')
        on the reference operation (:code:`"ref_op"`) and a reference point on the next
        added operation (:code:`"ref_pt_new"').
        A reference point can be either the "start", "center", or "end" of an
        operation. The reference operation (:code:`"ref_op"`) is specified using its
        label property.

        Each item in the list represents a timing constraint and is a dictionary with
        the following keys:

        .. code-block::

            ['label', 'rel_time', 'ref_op', 'ref_pt_new', 'ref_pt', 'operation_repr']

        The label is used as a unique identifier that can be used as a reference for
        other operations, the operation_repr refers to the string representation of a
        operation in :attr:`~.ScheduleBase.operations`.

        .. note::

            timing constraints are not intended to be modified directly.
            Instead use the :meth:`~quantify_scheduler.types.Schedule.add`

        """
        return self.data["timing_constraints"]

    @property
    def resources(self) -> Dict[str, Resource]:
        """
        A dictionary containing resources. Keys are names (str),
        values are instances of :class:`~quantify_scheduler.resources.Resource`.
        """
        return self.data["resource_dict"]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} "{self.data["name"]}" containing '
            f'({len(self.data["operation_dict"])}) '
            f'{len(self.data["timing_constraints"])}  (unique) operations.'
        )

    def to_json(self) -> str:
        """
        Converts the Schedule data structure to a JSON string.

        Returns
        -------
        :
            The json string result.
        """
        return json.dumps(self.data, cls=json_utils.ScheduleJSONEncoder)

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
        schedule_data = json_utils.ScheduleJSONDecoder().decode(data)
        name = schedule_data["name"]

        return Schedule(name, data=schedule_data)


class Schedule(ScheduleBase):  # pylint: disable=too-many-ancestors
    """
    A modifiable schedule.

    Operations :class:`~quantify_scheduler.types.Operation` can be added using the
    :meth:`~quantify_scheduler.types.Schedule.add` method, allowing precise
    specification *when* to perform an operation using timing constraints.

    When adding an operation, it is not required to specify how to represent this
    :class:`~quantify_scheduler.types.Operation` on all layers.
    Instead, this information can be added later during
    :ref:`compilation <sec-compilation>`.
    This allows the user to effortlessly mix the gate- and pulse-level descriptions as
    required for many (calibration) experiments.

    """  # pylint: disable=line-too-long

    schema_filename = "schedule.json"

    def __init__(self, name: str, repetitions: int = 1, data: dict = None) -> None:
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

    def add_resources(self, resources_list: list) -> None:
        """Add wrapper for adding multiple resources"""
        for resource in resources_list:
            self.add_resource(resource)

    def add_resource(self, resource) -> None:
        """
        Add a resource such as a channel or qubit to the schedule.
        """
        assert resources.Resource.is_valid(resource)
        if resource.name in self.data["resource_dict"]:
            raise ValueError(f"Key {resource.name} is already present")

        self.data["resource_dict"][resource.name] = resource

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
        See also :attr:`~quantify_scheduler.types.ScheduleBase.timing_constraints`.

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
        else:
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
                raise ValueError(f'Label "{label}" must be unique.')

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
                raise ValueError(f'Reference "{ref_op}" does not exist in schedule.')

        operation_id = str(operation)
        self.data["operation_dict"][operation_id] = operation
        timing_constr = {
            "label": label,
            "rel_time": rel_time,
            "ref_op": ref_op,
            "ref_pt_new": ref_pt_new,
            "ref_pt": ref_pt,
            "operation_repr": operation_id,
        }
        self.data["timing_constraints"].append(timing_constr)

        return label


# pylint: disable=too-many-ancestors
class CompiledSchedule(ScheduleBase):
    """
    A schedule that contains compiled instructions ready for execution using
    the :class:`~.instrument_coordinator.InstrumentCoordinator`.

    The :class:`CompiledSchedule` differs from a :class:`.Schedule` in
    that it is considered immutable (no new operations or resources can be added), and
    that it contains :attr:`~.compiled_instructions`.

    .. tip::

        A :class:`~.CompiledSchedule` can be obtained by compiling a
        :class:`~.Schedule` using :func:`~quantify_scheduler.compilation.qcompile`.

    """

    schema_filename = "schedule.json"

    def __init__(self, schedule: Schedule) -> None:

        # validate the input data to ensure it is valid schedule data
        super().__init__()

        # ensure keys exist
        self.data["compiled_instructions"] = {}

        # deepcopy is used to prevent side effects when the
        # original (mutable) schedule is modified
        self.data.update(deepcopy(schedule.data))

    @property
    def compiled_instructions(self) -> Dict[str, Resource]:
        """
        A dictionary containing compiled instructions.

        The contents of this dictionary depend on the backend it was compiled for.
        However, we assume that the general format consists of a dictionary in which
        the keys are instrument names corresponding to components added to a
        :class:`~.instrument_coordinator.InstrumentCoordinator`, and the
        values are the instructions for that component.

        These values typically contain a combination of sequence files, waveform
        definitions, and parameters to configure on the instrument.
        """
        return self.data["compiled_instructions"]

    @classmethod
    def is_valid(cls, object_to_be_validated) -> bool:
        """
        Checks if the contents of the object_to_be_validated are valid
        according to the schema. Additionally checks if the object_to_be_validated is
        an instance of :class:`~.CompiledSchedule`
        """
        valid_schedule = super().is_valid(object_to_be_validated)
        if valid_schedule:
            return isinstance(object_to_be_validated, CompiledSchedule)

        return False


@dataclass
class AcquisitionMetadata:
    """
    Class to provide a description of the shape and type of data that a schedule will
    return when executed.

    .. note::

        The acquisition protocol, bin-mode and return types are assumed to be the same
        for all acquisitions in a schedule.
    """

    acq_protocol: str
    """The acquisition protocol that is used for all acquisitions in the schedule."""
    bin_mode: enums.BinMode
    """How the data is stored in the bins indexed by acq_channel and acq_index."""
    acq_return_type: Type
    """The datatype returned by the individual acquisitions."""
    acq_indices: Dict[int, List[int]]
    """A dictionary containing the acquisition channel as key and a list of acquisition
    indices that are used for every channel."""
