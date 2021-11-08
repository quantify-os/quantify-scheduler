# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing a Generic InstrumentCoordinator Component."""
from __future__ import annotations

import logging
from typing import Any, Dict

from qcodes.instrument.base import InstrumentBase

import quantify_scheduler.instrument_coordinator.utility as util
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

    # NB `_instances` also used by `Instrument` class
    _no_gc_instances: Dict[str, base.InstrumentCoordinatorComponentBase] = dict()
    default_name = "generic_instruments"

    def __new__(
        cls, hardware_config: Dict[str, Any], name: str = default_name
    ) -> base.InstrumentCoordinatorComponentBase:
        """Keeps track of the instances of this class.

        NB This is done intentionally to prevent the instances from being garbage
        collected.
        """
        instrument = InstrumentBase(name=name)
        instance = super().__new__(cls, instrument)
        cls._no_gc_instances[instrument.name] = instance
        return instance

    def __init__(
        self, hardware_config: Dict[str, Any], name: str = default_name
    ) -> None:

        instrument = InstrumentBase(name=name)
        super().__init__(instrument)

        """Create a new instance of GenericInstrumentCoordinatorComponent class."""
        generic_devices_in_hw_config = hardware_config.get("generic_devices")
        self.current_params_config = dict()

        for device, params_dict in generic_devices_in_hw_config.items():
            for param, value in params_dict.items():
                self.current_params_config[f"{device}.{param}"] = value

    @property
    def current_params(self) -> Dict[str, Any]:
        """
        Returns a dict tracking the current parameters set for the generic devices.
        """
        return self.current_params_config

    @property
    def instrument(self):
        """
        Overwrite the instrument method. There is no instrument for the
        GenericInstrumentCoordinatorComponent class.
        """
        raise NotImplementedError

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
        self, params_config: Dict[str, Any] = None, force_set_parameters: bool = False
    ) -> None:
        """
        params_config has keys which should correspond to parameter names of the
        instrument and the corresponding values to be set.
        For example, params_config = {"lo_mw_q0.frequency": 6e9,
                                      "lo_mw_q0.power": 13, "lo_mw_q0.status": True,
                                      "lo_ro_q0.frequency": 8.3e9, "lo_ro_q0.power": 16,
                                      "lo_ro_q0.status": True,
                                      "lo_spec_q0.status": False,}
        """
        self.check_update_params_config(params_config)
        self.set_params_to_devices(
            params_config=params_config, force_set_parameters=force_set_parameters
        )

    def check_update_params_config(self, params_config) -> None:
        """
        Checks if the new params_config dict has keys which are different from that
        initially specified by the hardware_config. If a key does not exist, it throws
        a KeyError. If the key exists, then it updates the current_params_config value
        corresponding to the key.
        """
        if params_config is not None:
            for key, value in params_config.items():
                instrument_name, parameter_name = key.split(".")
                if key in self.current_params_config:
                    self.current_params_config[key] = value
                else:
                    error_message = f"{key} not found in current_params_config."
                    hint_message = (
                        f"{instrument_name}:{parameter_name} possibly not "
                        + "in initial hardware_config during class initialization."
                    )
                    raise KeyError(error_message + " " + hint_message)

    def set_params_to_devices(self, params_config, force_set_parameters) -> None:
        """
        This function sets the parameters in the params_config dict to the generic
        devices set in the hardware_config. The bool force_set_parameters is used to
        change the lazy_set behaviour.
        """
        for key, value in params_config.items():
            instrument_name, parameter_name = key.split(".")
            instrument = self.find_instrument(instrument_name)
            if force_set_parameters:
                instrument.set(param_name=parameter_name, value=value)
            else:
                util.lazy_set(
                    instrument=instrument, parameter_name=parameter_name, val=value
                )

    def retrieve_acquisition(self) -> Any:
        pass

    def wait_done(self, timeout_sec: int = 10) -> None:
        _ = timeout_sec  # Unused argument
