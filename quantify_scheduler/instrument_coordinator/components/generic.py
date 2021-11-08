# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing a Generic InstrumentCoordinator Component."""
from __future__ import annotations

import logging
from typing import Any

from qcodes.instrument.base import Instrument

import quantify_scheduler.instrument_coordinator.utility as util
from quantify_scheduler.helpers import time
from quantify_scheduler.instrument_coordinator.components import base

logger = logging.getLogger(__name__)


class GenericInstrumentCoordinatorComponent(  # pylint: disable=too-many-ancestors
    base.InstrumentCoordinatorComponentBase
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
    def is_running(self) -> bool:
        """
        The is_running state refers to a state whether an instrument is capable of
        running in a program. Not to be confused with the on/off state of the
        instrument.
        """
        return True

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def prepare(
        self, param_config: Dict[str, Any], force_set_parameters: bool = False
    ) -> None:
        """
        param_config has keys which should correspond to parameter names of the
        instrument and the corresponding values to be set.
        For example, param_config = {"frequency": 6e9, "power": 13, "status": True,}
        """
        if force_set_parameters:
            for key, value in param_config.items():
                self.instrument.set(param_name=key, value=value)
        else:
            for key, value in param_config.items():
                util.lazy_set(
                    instrument=self.instrument, parameter_name=key, val=value
                )

    def retrieve_acquisition(self) -> Any:
        pass

    def wait_done(self, timeout_sec: int = 10) -> None:
        _ = timeout_sec  # Unused argument
        pass
