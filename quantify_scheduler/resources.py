# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Common resources for use with the quantify_scheduler."""

from __future__ import annotations

from collections import UserDict
from typing import Optional

from quantify_core.utilities.general import load_json_schema

from quantify_scheduler.json_utils import validate_json


class Resource(UserDict):
    """
    A resource corresponds to a physical resource such as a port or a clock.

    .. jsonschema:: schemas/resource.json
    """

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
        return f"{str(self)[:-1]}, data={self.data})"


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
            data = {
                "name": name,
                "type": str(self.__class__.__name__),
                "freq": freq,
                "phase": phase,
            }
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
            data = {
                "name": name,
                "type": str(self.__class__.__name__),
                "freq": 0,
                "phase": 0,
            }
        super().__init__(data["name"], data=data)
