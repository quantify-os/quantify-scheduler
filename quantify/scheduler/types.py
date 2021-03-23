# -----------------------------------------------------------------------------
# Description:    Module containing the core concepts of the scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations
from uuid import uuid4
from collections import UserDict
import jsonschema
from quantify.utilities.general import make_hash, load_json_schema
from quantify.scheduler.resources import Resource, BasebandClockResource


class Operation(UserDict):
    """
    A JSON compatible data structure that contains information on
    how to represent the operation on the Gate, Pulse and/or Logical level.
    It also contains information on the :class:`~quantify.scheduler.resources.Resource` s used.

    An operation always has the following attributes

    - duration  (float) : duration of the operation in seconds (can be 0)
    - hash      (str)   : an auto generated unique identifier.
    - name      (str)   : a readable identifier, does not have to be unique

    An Operation can contain information  on several levels of abstraction.
    This information is used when different representations. Note that when
    initializing an operation  not all of this information needs to be available
    as operations are typically modified during the compilation steps.

    .. jsonschema:: schemas/operation.json


    .. note::

        Two different Operations containing the same information generate the same hash and are considered identical.
    """

    def __init__(self, name: str, data: dict = None):
        super().__init__()

        # ensure keys exist
        self.data["gate_info"] = {}
        self.data["pulse_info"] = []
        self.data["acquisition_info"] = []
        self.data["logic_info"] = {}

        if name is not None:
            self.data["name"] = name
        if data is not None:
            self.data.update(data)

    @property
    def name(self):
        return self.data["name"]

    @property
    def duration(self):
        """
        Determine the duration of the operation based on the pulses described in pulse_info.

        If the operation contains no pulse info, it is assumed to be ideal and have zero duration.
        """
        duration = 0  # default to zero duration if no pulse content is specified.

        # Iterate over all pulses and take longest duration
        for p in self.data["pulse_info"]:
            d = p["duration"] + p["t0"]
            if d > duration:
                duration = d

        return duration

    @property
    def hash(self):
        """
        A hash based on the contents of the Operation.
        """
        return make_hash(self.data)

    def add_gate_info(self, gate_operation: Operation):
        """
        Updates self.data['gate_info'] with contents of gate_operation.

        Parameters
        ----------
        gate_operation :
            an operation containing gate_info.
        """
        self.data["gate_info"].update(gate_operation.data["gate_info"])

    def add_pulse(self, pulse_operation: Operation):
        """
        Adds pulse_info of pulse_operation Operation to this Operation.

        Parameters
        ----------
        pulse_operation :
            an operation containing pulse_info.
        """
        self.data["pulse_info"] += pulse_operation.data["pulse_info"]

    def add_acquisition(self, acquisition_operation: Operation):
        """
        Adds acquisition_info of acquisition_operation Operation to this Operation.

        Parameters
        ----------
        acquisition_operation :
            an operation containing acquisition_info.
        """
        self.data["acquisition_info"] += acquisition_operation.data["acquisition_info"]

    @classmethod
    def is_valid(cls, operation):
        scheme = load_json_schema(__file__, "operation.json")
        jsonschema.validate(operation.data, scheme)
        _ = operation.hash  # test that the hash property evaluates
        return True  # if not exception was raised during validation

    @property
    def valid_gate(self):
        """
        An operation is a valid gate if it contains information on how
        to represent the operation on the gate level.
        """
        if self.data["gate_info"]:
            return True
        return False

    @property
    def valid_pulse(self):
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


class Schedule(UserDict):
    """
    A collection of :class:`~Operation` objects and timing constraints
    that define relations between the operations.

    The Schedule data structure is based on a dictionary.
    This dictionary contains:

        - `operation_dict`     : a hash table containing the unique :class:`~Operation` s added to the schedule.
        - `timing_constraints` : a list of all timing constraints added between operations.


    .. jsonschema:: schemas/schedule.json

    """

    def __init__(self, name: str, data: dict = None):
        """
        Parameters
        ----------
        name :
            name of the schedule
        data :
            a dictionary containing a pre-existing schedule.
        """

        # validate the input data to ensure it is valid schedule data
        super().__init__()

        # ensure keys exist
        self.data["operation_dict"] = {}
        self.data["timing_constraints"] = []
        self.data["resource_dict"] = {}
        self.data["name"] = "nameless"

        # This is used to define baseband pulses and is expected to always be present
        # in any schedule.
        self.add_resource(BasebandClockResource(BasebandClockResource.IDENTITY))

        if name is not None:
            self.data["name"] = name

        if data is not None:
            raise NotImplementedError

    @property
    def name(self):
        return self.data["name"]

    @property
    def operations(self):
        """
        A dictionary of all unique operations used in the schedule.
        This specifies information on *what* operation to apply *where*.

        The keys correspond to the :meth:`~Operation.hash` and values are instances of :class:`~Operation`.
        """
        return self.data["operation_dict"]

    @property
    def timing_constraints(self):
        """
        A list of dictionaries describing timing constraints between operations.

        Each item in the list is a dictionary with the following keys:
            [label, rel_time, ref_op, ref_pt_new, ref_pt, operation_hash]

        The label is used as a unique identifier that can be used as a reference for other operations
        the operation_hash refers to the hash of a unique operation in :meth:`~Schedule.operations`.
        """
        return self.data["timing_constraints"]

    @property
    def resources(self):
        """
        A dictionary containing resources. Keys are names (str),
        values are instances of :class:`~quantify.scheduler.resources.Resource` .
        """
        return self.data["resource_dict"]

    def add_resources(self, resources: list):
        for r in resources:
            self.add_resource(r)

    def add_resource(self, resource):
        """
        Add a resource such as a channel or qubit to the schedule.
        """
        assert Resource.is_valid(resource)
        if resource.name in self.data["resource_dict"]:
            raise ValueError("Key {} is already present".format(resource.name))
        else:
            self.data["resource_dict"][resource.name] = resource

    def __repr__(self):
        return 'Schedule "{}" containing ({}) {}  (unique) operations.'.format(
            self.data["name"],
            len(self.data["operation_dict"]),
            len(self.data["timing_constraints"]),
        )

    @classmethod
    def is_valid(cls, schedule):
        scheme = load_json_schema(__file__, "schedule.json")
        jsonschema.validate(schedule.data, scheme)
        return True  # if not exception was raised during validation

    def add(
        self,
        operation: Operation,
        rel_time: float = 0,
        ref_op: str = None,
        ref_pt: str = "end",
        ref_pt_new: str = "start",
        label: str = None,
    ) -> str:
        """
        Add an :class:`~Operation` to the schedule and specify timing constraints.

        Parameters
        ----------
        operation :
            The operation to add to the schedule
        rel_time :
            relative time between the the reference operation and added operation.
        ref_op :
            specifies the reference operation.
        ref_pt :
            reference point in reference operation must be one of ('start', 'center', 'end').
        ref_pt_new :
            reference point in added operation must be one of ('start', 'center', 'end').
        label :
            a label that can be used as an identifier when adding more operations.
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
