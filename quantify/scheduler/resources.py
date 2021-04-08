# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Common resources for use with the quantify.scheduler."""
from collections import UserDict
import jsonschema
from quantify.utilities.general import load_json_schema


class Resource(UserDict):
    """
    A resource corresponds to a physical resource such as a port or a clock.

    .. jsonschema:: schemas/resource.json
    """

    @classmethod
    def is_valid(cls, operation):
        scheme = load_json_schema(__file__, "resource.json")
        jsonschema.validate(operation.data, scheme)
        return True  # if not exception was raised during validation

    @property
    def name(self):
        return self.data["name"]


class ClockResource(Resource):
    def __init__(self, name: str, freq: float, phase: float = 0):
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

        self.data = {
            "name": name,
            "type": str(self.__class__.__name__),
            "freq": freq,
            "phase": phase,
        }


class BasebandClockResource(Resource):

    """
    Global identity for a virtual baseband clock
    """

    IDENTITY = "cl0.baseband"

    def __init__(self, name: str):
        """
        A clock resource for pulses that operate at baseband.

        Baseband signals are assumed to be real-valued and will not be modulated.

        Parameters
        ----------
        name :
            the name of this clock
        """

        self.data = {
            "name": name,
            "type": str(self.__class__.__name__),
            "freq": 0,
            "phase": 0,
        }
