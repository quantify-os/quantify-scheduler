# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the core concepts of the scheduler."""
from __future__ import annotations

from uuid import uuid4
from collections import UserDict
from typing_extensions import Literal
import jsonschema
from quantify.utilities import general
from quantify.scheduler import resources


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
        self.data["gate_info"] = {}
        self.data["pulse_info"] = []
        self.data["acquisition_info"] = []
        self.data["logic_info"] = {}
        self._duration: float = 0

        if name is not None:
            self.data["name"] = name
        if data is not None:
            self.data.update(data)
            self._update()

    def __eq__(self, other):
        """
        Two operations are considered equal if the contents of the "data" attribute
        are identical.

        This is tested through the :code:`.hash` attribute.
        """
        return self.hash == other.hash

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
        An operation is a valid gate if it contains information on how
        to represent the operation on the gate level.
        """
        if self.data["gate_info"]:
            return True
        return False

    @property
    def valid_pulse(self) -> bool:
        """
        An operation is a valid pulse if it contains information on how
        to represent the operation on the pulse level.
        """
        if self.data["pulse_info"]:
            return True
        return False

    @property
    def valid_acquisition(self) -> bool:
        """
        An operation is a valid acquisition if it contains information on how
        to represent the operation as a acquisition on the pulse level.
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
            raise NotImplementedError

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

        operation_hash = operation.hash

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

        self.data["operation_dict"][operation_hash] = operation
        timing_constr = {
            "label": label,
            "rel_time": rel_time,
            "ref_op": ref_op,
            "ref_pt_new": ref_pt_new,
            "ref_pt": ref_pt,
            "operation_hash": operation_hash,
        }
        self.data["timing_constraints"].append(timing_constr)

        return label
