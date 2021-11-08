# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals
from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List
from unittest.mock import ANY, call

import numpy as np
import pytest
from qcodes import Instrument, validators
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)


@pytest.fixture
def typical_zi_hardware_map() -> Dict[str, Any]:
    return json.loads(
        """
        {
            "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
            "mode": "calibration",
            "generic_devices": {
                     "lo_mw_q0": {"frequency": 6e9, "power": 13, "status": true},
                     "lo_ro_q0": {"frequency": 8.3e9, "power": 16, "status": true},
                     "lo_spec_q0": {"frequency": null, "power": null, "status": false}
                   },
            "devices": [
                {
                    "name": "hdawg0",
                    "type": "HDAWG4",
                    "clock_select": 0,
                    "ref": "int",
                    "channelgrouping": 0,
                    "channel_0": {
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "mode": "complex",
                        "modulation": {"type": "premod", "interm_freq": 0},
                        "local_oscillator": "lo_mw_q0",
                        "clock_frequency": 6e9,
                        "line_trigger_delay": 191e-9,
                        "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                        "gain1": 1,
                        "gain2": 1,
                        "latency": 12e-9,
                        "mixer_corrections": {
                            "amp_ratio": 0.950,
                            "phase_error": 90,
                            "dc_offset_I": -0.5420,
                            "dc_offset_Q": -0.3280
                        }
                    }
                },
                {
                    "name": "uhfqa0",
                    "type": "UHFQA",
                    "ref": "ext",
                    "channel_0": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "mode": "real",
                        "modulation": {"type": "premod", "interm_freq": 100e6},
                        "local_oscillator": "lo_ro_q0",
                        "clock_frequency": 6e9,
                        "triggers": [2]
                    }
                }
            ]
        }
        """
    )


@pytest.fixture
def make_generic_qcodes_instrument(mocker):
    class GenericQcodesInstrument(Instrument):
        def __init__(self, name: str, address: str):
            """
            Create an instance of the Generic instrument.

            Args:
                name: QCoDeS'name
                address: used to connect to the instrument e.g., "COM3" or "dummy"
            """
            super().__init__(name)
            self._add_qcodes_parameters_dummy()

        def _add_qcodes_parameters_dummy(self):
            """
            Used for faking communications
            """
            self.add_parameter(
                name="status",
                initial_value=False,
                vals=validators.Bool(),
                docstring="turns the output on/off",
                parameter_class=ManualParameter,
            )
            self.add_parameter(
                name="frequency",
                label="Frequency",
                unit="Hz",
                initial_value=10e6,
                docstring="The RF Frequency in Hz",
                vals=validators.Numbers(min_value=250e3, max_value=20e9),
                parameter_class=ManualParameter,
            )
            self.add_parameter(
                name="power",
                label="Power",
                unit="dBm",
                initial_value=-60.0,
                vals=validators.Numbers(min_value=-60.0, max_value=20.0),
                docstring="Signal power in dBm",
                parameter_class=ManualParameter,
            )

    def _make_generic_instrument(name, address):
        generic_instrument = GenericQcodesInstrument(name=name, address=address)
        return generic_instrument

    yield _make_generic_instrument


def test_initialize(make_generic_qcodes_instrument):
    component = make_generic_qcodes_instrument("lo_mw_q0_init", "dev1234")
    ic_component = GenericInstrumentCoordinatorComponent(component)


def test_generic_icc_prepare(
    mocker, make_generic_qcodes_instrument, typical_zi_hardware_map
):
    # Arrange
    lo_mw_q0 = make_generic_qcodes_instrument("lo_mw_q0", "dev123")
    lo_ro_q0 = make_generic_qcodes_instrument("lo_ro_q0", "dev124")
    lo_spec_q0 = make_generic_qcodes_instrument("lo_spec_q0", "dev125")

    ic_lo_mw_q0 = GenericInstrumentCoordinatorComponent(lo_mw_q0)

    generic_device_params_dict = typical_zi_hardware_map["generic_devices"]

    # Assert initial condition
    lo_mw_q0.frequency() == 1e7

    # Act
    ic_lo_mw_q0.prepare(params_config=generic_device_params_dict)

    # Assert
    lo_mw_q0.frequency() == typical_zi_hardware_map["generic_devices"]["lo_mw_q0"][
        "frequency"
    ]
