# -----------------------------------------------------------------------------
# Description:    Library containing common resources for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------

'''
This module should be merged into types
'''
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
        scheme = load_json_schema(__file__, 'resource.json')
        jsonschema.validate(operation.data, scheme)
        return True  # if not exception was raised during validation

    @property
    def name(self):
        return self.data['name']


class ClockResource(Resource):

    def __init__(self, name: str, freq: float, phase: float = 0):
        """
        A clock resource used to modulate pulses.

        Parameters
        -------------
        name : str
            the name of this clock
        freq : float
            the frequency of the clock in Hz
        phase : float
            the starting phase of the clock in deg

        """

        self.data = {'name': name,
                     'type': str(self.__class__.__name__),
                     'freq': freq,
                     'phase': phase
                     }


class BasebandClockResource(Resource):

    """
    Global identity for a virtual baseband clock
    """
    IDENTITY = 'cl0.baseband'

    def __init__(self, name: str):
        """
        A clock resource for pulses that operate at baseband.

        Parameters
        -------------
        name : str
            the name of this clock

        Baseband signals are assumed to be real-valued and will not be modulated.
        """

        self.data = {'name': name,
                     'type': str(self.__class__.__name__),
                     'freq': 0,
                     'phase': 0
                     }
