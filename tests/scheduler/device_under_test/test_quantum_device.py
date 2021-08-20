import pytest
from quantify_scheduler.compilation import (
    validate_config,
)

# overwrite fixture locally as we need to not close the mock setup
# https://stackoverflow.com/questions/38748257/\
# disable-autouse-fixtures-on-specific-pytest-marks


@pytest.fixture(autouse=True)
def close_all_instruments():
    pass


def test_QuantumDevice_generate_device_config(mock_setup):

    quantum_device = mock_setup["quantum_device"]
    dev_cfg = quantum_device.generate_device_config()
    validate_config(dev_cfg, scheme_fn="transmon_cfg.json")

    assert {"q0", "q1"} <= set(dev_cfg["qubits"].keys())


def test_QuantumDevice_generate_hardware_config(mock_setup):

    quantum_device = mock_setup["quantum_device"]

    hardware_cfg = quantum_device.generate_hardware_config()

    # cannot validate as there is no schema exists see quantify-scheduler #181
    # validate_config(dev_cfg, scheme_fn="qblox_cfg.json")
