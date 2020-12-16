# -----------------------------------------------------------------------------
# Description:    Library containing common resources for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------

'''
This module should be merged into types
'''


from .types import Resource


class PortResource(Resource):
    def __init__(self, name: str):
        """
        A port resource to which pulses can be scheduled.

        Parameters
        -------------
        name : str
            the name of this port

        """
        self.data = {'name': name,
                     'type': str(self.__class__.__name__)}


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
