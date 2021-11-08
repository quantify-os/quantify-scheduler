# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing a Generic InstrumentCoordinator Component."""
from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

from qcodes.instrument.base import Instrument

import quantify_scheduler.instrument_coordinator.utility as util
from quantify_scheduler.helpers import time
from quantify_scheduler.instrument_coordinator.components import base

logger = logging.getLogger(__name__)

T = TypeVar("T")  # pylint: disable=invalid-name


class GenericInstrumentCoordinatorComponent(  # pylint: disable=too-many-ancestors
    Generic[T], base.InstrumentCoordinatorComponentBase
):
    """
    A Generic class which can be used for interaction with the InstrumentCoordinator.

    The GenericInstrumentCoordinatorComponent should be able to accept any type of
    qcodes instrument.
    """

    def __init__(self, instrument: Instrument, **kwargs) -> None:
        """Create a new instance of GenericInstrumentCoordinatorComponent class."""
        super().__init__(instrument, **kwargs)

    @property
    def instrument(self) -> T is Instrument:
        return super().instrument

    @property
    def is_running(self) -> bool:
        return False

    def start(self) -> None:
        self.instrument.on()
        print(f"{self.name} ON")
        time.sleep(2)
        print(f"{self.name} ON after delay")

    def stop(self) -> None:
        self.instrument.off()
        print(f"{self.name} OFF")

    def prepare(self, options: Any) -> None:
        self.instrument.reset()
        print(f"{self.name} frequency={options}")
        self.instrument.frequency(options)

    def retrieve_acquisition(self) -> Any:
        pass

    def wait_done(self, _: int = 10) -> None:
        pass
