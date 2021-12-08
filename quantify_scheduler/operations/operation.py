# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the core concepts of the scheduler."""
from __future__ import annotations

import ast
import inspect
import logging
from collections import UserDict
from copy import deepcopy
from enum import Enum
from pydoc import locate
from typing import Any, Dict

import numpy as np
from quantify_core.utilities import general

from quantify_scheduler import enums
from quantify_scheduler.json_utils import JSONSchemaValMixin, lru_cache

cached_locate = lru_cache(locate)


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

        :mod:`quantify_scheduler` comes with a
        :mod:`~quantify_scheduler.operations.gate_library` and a
        :mod:`~quantify_scheduler.operations.pulse_library` , both containing common
        operations.


    **JSON schema of a valid Operation**

    .. jsonschema:: ../schemas/operation.json


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
        other

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
            key

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
        gate_operation
            an operation containing gate_info.
        """
        self.data["gate_info"].update(gate_operation.data["gate_info"])

    def add_pulse(self, pulse_operation: Operation) -> None:
        """
        Adds pulse_info of pulse_operation Operation to this Operation.

        Parameters
        ----------
        pulse_operation
            an operation containing pulse_info.
        """
        self.data["pulse_info"] += pulse_operation.data["pulse_info"]
        self._update()

    def add_acquisition(self, acquisition_operation: Operation) -> None:
        """
        Adds acquisition_info of acquisition_operation Operation to this Operation.

        Parameters
        ----------
        acquisition_operation
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
                acq_info["acq_return_type"] = cached_locate(return_type_str)

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
