# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing a Generic InstrumentCoordinator Component."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qcodes.instrument.base import InstrumentBase

import quantify_scheduler.instrument_coordinator.utility as util
from quantify_scheduler.instrument_coordinator.components import base

if TYPE_CHECKING:
    from xarray import Dataset

    from quantify_scheduler.schedules.schedule import CompiledSchedule

logger = logging.getLogger(__name__)

DEFAULT_NAME = "generic"


class GenericInstrumentCoordinatorComponent(base.InstrumentCoordinatorComponentBase):
    """
    A Generic class which can be used for interaction with the InstrumentCoordinator.

    The GenericInstrumentCoordinatorComponent should be able to accept any type of
    qcodes instrument. The component is meant to serve as an interface for simple
    access to instruments such as the local oscillator, or current source which needs to
    only set parameters. For now this component is not being used in any of the hardware
    backends' compilation step. This will be fixed in the next official release.
    """

    # NB `_instances` also used by `Instrument` class
    _no_gc_instances: dict[str, InstrumentBase] = dict()

    def __new__(
        cls, instrument_reference: str | InstrumentBase = DEFAULT_NAME
    ) -> base.InstrumentCoordinatorComponentBase:
        """
        Keeps track of the instances of this class.

        NB This is done intentionally to prevent the instances from being garbage
        collected.
        """
        if isinstance(instrument_reference, InstrumentBase):
            instrument = instrument_reference
        else:
            instrument = InstrumentBase(name=instrument_reference)

        instance = super().__new__(cls, instrument)
        cls._no_gc_instances[instrument.name] = instance
        return instance

    def __init__(self, instrument_reference: str | InstrumentBase = DEFAULT_NAME) -> None:
        if isinstance(instrument_reference, InstrumentBase):
            instrument = instrument_reference
        else:
            instrument = InstrumentBase(name=instrument_reference)

        super().__init__(instrument)

    @property
    def is_running(self) -> bool:
        """
        A state whether an instrument is capable of running in a program.

        Not to be confused with the on/off state of an
        instrument.
        """
        return True

    def start(self) -> None:
        """Start the instrument."""
        pass

    def stop(self) -> None:
        """Stop the instrument."""
        pass

    # Parameter name is different from base class. We ignore it because it is legacy
    # code.
    def prepare(self, params_config: dict[str, Any]) -> None:  # type: ignore
        """
        Prepare the instrument.

        params_config has keys which should correspond to parameter names of the
        instrument and the corresponding values to be set. Always ensure that the
        key to the params_config is in the format 'instrument_name.parameter_name'
        See example below.

        .. code-block:: python

            params_config = {
                             "lo_mw_q0.frequency": 6e9,
                             "lo_mw_q0.power": 13, "lo_mw_q0.status": True,
                             "lo_ro_q0.frequency": 8.3e9, "lo_ro_q0.power": 16,
                             "lo_ro_q0.status": True,
                             "lo_spec_q0.status": False,
                            }

        """
        self._set_params_to_devices(params_config=params_config)

    def _set_params_to_devices(self, params_config: dict) -> None:
        """
        Set the parameters in the params_config dict
        to the generic devices set in the hardware_config.

        The bool force_set_parameters is used to
        change the lazy_set behavior.
        """
        for key, value in params_config.items():
            if "." not in key:
                error_msg = f"Key [{key}] is not valid in the params_config."
                hint_msg = "Ensure that it is in the format " + "'instrument_name.parameter_name'"
                raise KeyError(error_msg + hint_msg)
            instrument_name, parameter_name = key.split(".", maxsplit=1)
            instrument = self.find_instrument(instrument_name)
            try:
                if self.force_set_parameters():
                    param_to_set = util.search_settable_param(
                        instrument=instrument, nested_parameter_name=parameter_name
                    )
                    param_to_set.set(value=value)
                else:
                    util.lazy_set(instrument=instrument, parameter_name=parameter_name, val=value)
            except ValueError as e:
                set_function = getattr(instrument, parameter_name)
                if callable(set_function):
                    set_function(value)
                else:
                    raise RuntimeError(
                        f"{key} is neither a parameter nor a callable function"
                    ) from e

    def retrieve_acquisition(self) -> Dataset | None:
        """Retrieve acquisition."""
        pass

    def get_hardware_log(
        self,
        compiled_schedule: CompiledSchedule,  # noqa: ARG002
    ) -> dict | None:
        """Get the hardware log."""
        return None

    def wait_done(self, timeout_sec: int = 10) -> None:
        """Wait till done."""
        _ = timeout_sec  # Unused argument
