# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the ControlStack interface."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from qcodes.instrument.base import AbstractInstrument


class AbstractControlStackComponent(AbstractInstrument):
    """The ControlStack component abstract interface."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """
        Returns if the ControlStack component is running.

        The property is_running is evaluated each time it is accessed. Example:

        .. code-block::

            while (cs.is_running):
                print('running')

        Returns
        -------
        :
            The Component's running state.
        """

    @abstractmethod
    def start(self) -> None:
        """Starts the ControlStack Component."""

    @abstractmethod
    def stop(self) -> None:
        """Stops the ControlStack Component."""

    @abstractmethod
    def prepare(self, options: Any) -> None:
        """Initializes the ControlStack Component with parameters."""

    @abstractmethod
    def retrieve_acquisition(self) -> Any:
        """Gets and returns acquisition data."""

    @abstractmethod
    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Waits until the ControlStack component has stopped running or until it has
        exceeded the amount of time to run.

        The maximum amount of time, in seconds, before it times out is set via the
        timeout_sec parameter.

        Parameters
        ----------
        timeout_sec :
            The maximum amount of time in seconds before a timeout.
        """
