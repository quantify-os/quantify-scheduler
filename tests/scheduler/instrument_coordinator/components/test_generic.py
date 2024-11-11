from __future__ import annotations

import pytest

from quantify_scheduler.helpers.mock_instruments import MockLocalOscillator
from quantify_scheduler.instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)


@pytest.fixture
def make_generic_qcodes_instruments():
    lo_mw_q0 = MockLocalOscillator(name="lo_mw_q0")
    lo_ro_q0 = MockLocalOscillator(name="lo_ro_q0")
    lo_spec_q0 = MockLocalOscillator(name="lo_spec_q0")

    generic_icc = GenericInstrumentCoordinatorComponent(instrument_reference="test_generic_icc")

    return {
        "lo_mw_q0": lo_mw_q0,
        "lo_ro_q0": lo_ro_q0,
        "lo_spec_q0": lo_spec_q0,
        "generic_icc": generic_icc,
    }


def test_initialize(make_generic_qcodes_instruments):
    test_instruments = make_generic_qcodes_instruments
    generic_icc = test_instruments["generic_icc"]
    assert generic_icc.name == "ic_test_generic_icc"


@pytest.mark.parametrize("force_set_parameters", [False, True])
def test_generic_icc_prepare_expected(make_generic_qcodes_instruments, force_set_parameters):
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
