# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals
from __future__ import annotations

import pytest
from qcodes import Instrument, validators
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)


@pytest.fixture
def make_generic_qcodes_instruments(request):
    class MockLocalOscillator(Instrument):  # pylint: disable=too-few-public-methods
        def __init__(self, name: str):
            """
            Create an instance of the Generic instrument.

            Args:
                name: QCoDeS'name
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
                initial_value=7e9,
                docstring="The RF Frequency in Hz",
                vals=validators.Numbers(min_value=250e3, max_value=20e9),
                parameter_class=ManualParameter,
            )
            self.add_parameter(
                name="power",
                label="Power",
                unit="dBm",
                initial_value=15.0,
                vals=validators.Numbers(min_value=-60.0, max_value=20.0),
                docstring="Signal power in dBm",
                parameter_class=ManualParameter,
            )

    lo_mw_q0 = MockLocalOscillator(name="lo_mw_q0")
    lo_ro_q0 = MockLocalOscillator(name="lo_ro_q0")
    lo_spec_q0 = MockLocalOscillator(name="lo_spec_q0")

    generic_icc = GenericInstrumentCoordinatorComponent()

    def cleanup_instruments():
        lo_mw_q0.close()
        lo_ro_q0.close()
        lo_spec_q0.close()
        generic_icc.close()

    request.addfinalizer(cleanup_instruments)

    return {
        "lo_mw_q0": lo_mw_q0,
        "lo_ro_q0": lo_ro_q0,
        "lo_spec_q0": lo_spec_q0,
        "generic_icc": generic_icc,
    }


def test_initialize(make_generic_qcodes_instruments):
    test_instruments = make_generic_qcodes_instruments
    generic_icc = test_instruments["generic_icc"]
    assert generic_icc.name == "ic_generic"


@pytest.mark.parametrize("force_set_parameters", [False, True])
def test_generic_icc_prepare_expected(
    make_generic_qcodes_instruments, force_set_parameters
):
    # Arrange
    test_instruments = make_generic_qcodes_instruments

    generic_icc = test_instruments["generic_icc"]

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
    generic_icc.force_set_parameters(force_set_parameters)

    # Assert initial condition
    assert test_instruments["lo_mw_q0"].frequency() == 7e9
    assert test_instruments["lo_ro_q0"].power() == 15.0

    # Act
    generic_icc.prepare(params_config=generic_device_params_dict)

    expected_device_params_dict = generic_device_params_dict

    # Assert the frequency set has been changed to expected frequency
    assert (
        test_instruments["lo_mw_q0"].frequency()
        == expected_device_params_dict["lo_mw_q0.frequency"]
    )


def test_generic_icc_prepare_fail_no_device(make_generic_qcodes_instruments):
    # Arrange
    test_instruments = make_generic_qcodes_instruments

    generic_icc = test_instruments["generic_icc"]

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
        generic_icc.prepare(params_config=generic_device_params_dict)
