# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the ControlStack interface."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

from qcodes.instrument.base import AbstractInstrument


class AbstractControlStackComponent(AbstractInstrument):
    """The ControlStack component abstract interface."""

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
