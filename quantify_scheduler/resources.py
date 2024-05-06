# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Common resources for use with the quantify_scheduler."""

from __future__ import annotations

from collections import UserDict

from quantify_scheduler.helpers.collections import make_hash
from quantify_scheduler.helpers.importers import export_python_object_to_path_string
from quantify_scheduler.json_utils import load_json_schema, validate_json


class Resource(UserDict):
    """
    A resource corresponds to a physical resource such as a port or a clock.

    .. jsonschema:: https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/quantify_scheduler/schemas/resource.json

    Parameters
    ----------
    name :
        The resource name.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.data["name"] = name

    @classmethod
    def is_valid(cls, operation: Resource) -> bool:
        """
        Validates the Resource against the schemas/resource.json fastjsonschema.

        Parameters
        ----------
        operation :
            The operation to validate.

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
            The other instance to compare.

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
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": self.data,
        }

    def __setstate__(self, state):
        self.data = state["data"]

    def __hash__(self) -> int:
        return make_hash(self.data)

    @property
    def hash(self) -> str:
        """A hash based on the contents of the Operation."""
        return str(hash(self))


class ClockResource(Resource):
    """
    The ClockResource corresponds to a physical clock used to modulate pulses.

    Parameters
    ----------
    name :
        the name of this clock
    freq :
        the frequency of the clock in Hz
    phase :
        the starting phase of the clock in deg
    """

    def __init__(
        self,
        name: str,
        freq: float,
        phase: float = 0,
    ) -> None:
        super().__init__(name)

        self.data = {
            "name": name,
            "type": str(self.__class__.__name__),
            "freq": freq,
            "phase": phase,
        }

    def __str__(self) -> str:
        freq = self.data["freq"]
        phase = self.data["phase"]
        return f"{super().__str__()[:-1]}, freq={freq}, phase={phase})"


class BasebandClockResource(Resource):
    """
    Global identity for a virtual baseband clock.

    Baseband signals are assumed to be real-valued and will not be modulated.

    Parameters
    ----------
    name :
        the name of this clock
    """

    IDENTITY = "cl0.baseband"

    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.data = {
            "name": name,
            "type": str(self.__class__.__name__),
            "freq": 0,
            "phase": 0,
        }


class DigitalClockResource(Resource):
    """
    Global identity for a virtual digital clock.

    Digital clocks can only be associated with digital channels.

    Parameters
    ----------
    name :
        the name of this clock
    """

    IDENTITY = "digital"

    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.data = {
            "name": name,
            "type": str(self.__class__.__name__),
            "freq": 0,
            "phase": 0,
        }
