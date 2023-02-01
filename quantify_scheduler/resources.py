# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Common resources for use with the quantify_scheduler."""

from __future__ import annotations

from collections import UserDict
from typing import Optional
import warnings

from quantify_scheduler.json_utils import load_json_schema, validate_json


class Resource(UserDict):
    # pylint: disable=line-too-long
    """
    A resource corresponds to a physical resource such as a port or a clock.

    .. jsonschema:: https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/quantify_scheduler/schemas/resource.json

    """

    # pylint: enable=line-too-long
    def __init__(self, name: str, data: Optional[dict] = None) -> None:
        """
        Create a new instance of Resource.

        A resource corresponds to a physical resource such as a port or a clock.

        Parameters
        ----------
        name :
            The resource name.
        data :
            The resource data dictionary, by default None
        """
        super().__init__()

        self.data["name"] = name

        if data is not None:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            self.data.update(data)

    @classmethod
    def is_valid(cls, operation: Resource) -> bool:
        """
        Validates the Resource against the schemas/resource.json fastjsonschema.

        Parameters
        ----------
        operation :

        Raises
        ------
        fastjsonschema.JsonSchemaException
            if the instance is invalid
        fastjsonschema.JsonSchemaDefinitionException
            if the schema itself is invalid

        Returns
        -------
        bool
            If the validation was successful.
        """
        scheme = load_json_schema(__file__, "resource.json")
        validate_json(operation.data, scheme)
        return True  # if not exception was raised during validation

    @property
    def name(self) -> str:
        """
        Returns the name of the Resource.

        Returns
        -------
        :
        """
        return self.data["name"]

    def __eq__(self, other: object) -> bool:
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
        instance using :code:`eval(str(operation))` only when the data dictionary has
        not been modified.

        This representation is guaranteed to be unique.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __getstate__(self):
        return {"deserialization_type": self.__class__.__name__, "data": self.data}

    def __setstate__(self, state):
        self.data = state["data"]


class ClockResource(Resource):
    """
    The ClockResource corresponds to a physical clock used to modulate pulses.
    """

    def __init__(
        self, name: str, freq: float, phase: float = 0, data: Optional[dict] = None
    ) -> None:
        """
        A clock resource used to modulate pulses.

        Parameters
        ----------
        name :
            the name of this clock
        freq :
            the frequency of the clock in Hz
        phase :
            the starting phase of the clock in deg
        """
        if data is None:
            super().__init__(name)

            self.data = {
                "name": name,
                "type": str(self.__class__.__name__),
                "freq": freq,
                "phase": phase,
            }
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(data["name"], data=data)

    def __str__(self) -> str:
        freq = self.data["freq"]
        phase = self.data["phase"]
        return f"{super().__str__()[:-1]}, freq={freq}, phase={phase})"


class BasebandClockResource(Resource):
    """
    Global identity for a virtual baseband clock
    """

    IDENTITY = "cl0.baseband"

    def __init__(self, name: str, data: Optional[dict] = None) -> None:
        """
        A clock resource for pulses that operate at baseband.

        Baseband signals are assumed to be real-valued and will not be modulated.

        Parameters
        ----------
        name :
            the name of this clock
        """
        if data is None:
            super().__init__(name)

            self.data = {
                "name": name,
                "type": str(self.__class__.__name__),
                "freq": 0,
                "phase": 0,
            }
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(data["name"], data=data)
