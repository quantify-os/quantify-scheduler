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
                     "lo_mw_q0": {"frequency": 7e9, "power": 13, "status": true},
                     "lo_ro_q0": {"frequency": 8.4e9, "power": 10, "status": true},
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
def make_generic_qcodes_instruments(request, mocker):
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

    lo_mw_q0 = GenericQcodesInstrument(name="lo_mw_q0", address="dev123")
    lo_ro_q0 = GenericQcodesInstrument(name="lo_ro_q0", address="dev124")
    lo_spec_q0 = GenericQcodesInstrument(name="lo_spec_q0", address="dev125")

    def cleanup_instruments():
        lo_mw_q0.close()
        lo_ro_q0.close()
        lo_spec_q0.close()

    request.addfinalizer(cleanup_instruments)

    return {"lo_mw_q0": lo_mw_q0, "lo_ro_q0": lo_ro_q0, "lo_spec_q0": lo_spec_q0}


def test_initialize(make_generic_qcodes_instruments, typical_zi_hardware_map):
    ic_component = GenericInstrumentCoordinatorComponent(
        hardware_config=typical_zi_hardware_map
    )


def test_generic_icc_prepare_lazy_set_expected(
    mocker, make_generic_qcodes_instruments, typical_zi_hardware_map
):
    # Arrange
    test_instruments = make_generic_qcodes_instruments

    ic_generic_components = GenericInstrumentCoordinatorComponent(
        hardware_config=typical_zi_hardware_map
    )

    # Test dictionary with the settings parameter for generic devices
    generic_device_params_dict = {
        "lo_mw_q0.frequency": 6e9,
        "lo_mw_q0.power": 13,
        "lo_mw_q0.status": True,
        "lo_ro_q0.frequency": 8.3e9,
        "lo_ro_q0.power": 16,
        "lo_ro_q0.status": True,
        "lo_spec_q0.status": False,
    }

    # Assert initial condition
    test_instruments["lo_mw_q0"].frequency() == 7e9

    # Act
    ic_generic_components.prepare(params_config=generic_device_params_dict)

    expected_device_params_dict = generic_device_params_dict
    expected_device_params_dict.update(
        {"lo_spec_q0.frequency": None, "lo_spec_q0.power": None}
    )

    # Assert internal dictionary is the same as expected dictionary
    assert ic_generic_components.current_params == expected_device_params_dict
    # Assert the frequency set has been changed to expected frequency
    assert (
        test_instruments["lo_mw_q0"].frequency()
        == expected_device_params_dict["lo_mw_q0.frequency"]
    )


def test_generic_icc_prepare_no_lazy_set_expected(
    mocker, make_generic_qcodes_instruments, typical_zi_hardware_map
):
    # Arrange
    test_instruments = make_generic_qcodes_instruments

    ic_generic_components = GenericInstrumentCoordinatorComponent(
        hardware_config=typical_zi_hardware_map
    )

    # Test dictionary with the settings parameter for generic devices
    generic_device_params_dict = {
        "lo_mw_q0.frequency": 6e9,
        "lo_mw_q0.power": 13,
        "lo_mw_q0.status": True,
        "lo_ro_q0.frequency": 8.3e9,
        "lo_ro_q0.power": 16,
        "lo_ro_q0.status": True,
        "lo_spec_q0.status": False,
    }

    # Assert initial condition
    test_instruments["lo_mw_q0"].frequency() == 7e9

    # Act
    ic_generic_components.prepare(
        params_config=generic_device_params_dict, force_set_parameters=True
    )

    expected_device_params_dict = generic_device_params_dict
    expected_device_params_dict.update(
        {"lo_spec_q0.frequency": None, "lo_spec_q0.power": None}
    )

    # Assert internal dictionary is the same as expected dictionary
    assert ic_generic_components.current_params == expected_device_params_dict
    # Assert the frequency set has been changed to expected frequency
    assert (
        test_instruments["lo_mw_q0"].frequency()
        == expected_device_params_dict["lo_mw_q0.frequency"]
    )


def test_generic_icc_prepare_fail_no_device(
    mocker, make_generic_qcodes_instruments, typical_zi_hardware_map
):
    # Arrange
    test_instruments = make_generic_qcodes_instruments

    ic_generic_components = GenericInstrumentCoordinatorComponent(
        hardware_config=typical_zi_hardware_map
    )

    # Test dictionary with the settings parameter for generic devices
    generic_device_params_dict = {
        "lo_mw_q0.frequency": 6e9,
        "lo_mw.power": 13,
        "lo_mw_q0.status": True,
        "lo_ro_q0.frequency": 8.3e9,
        "lo_ro_q0.power": 16,
        "lo_ro_q0.status": True,
        "lo_spec_q0.status": False,
    }

    # Act
    with pytest.raises(KeyError):
        ic_generic_components.prepare(params_config=generic_device_params_dict)
