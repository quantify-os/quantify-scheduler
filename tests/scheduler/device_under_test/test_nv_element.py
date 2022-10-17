# pylint: disable=redefined-outer-name
import pytest
from qcodes import Instrument

from quantify_scheduler.device_under_test.nv_element import BasicElectronicNVElement
from quantify_scheduler.device_under_test.mock_setup import (
    set_up_basic_mock_nv_setup,
    set_standard_params_basic_nv,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.instrument_coordinator.instrument_coordinator import (
    InstrumentCoordinator,
)
from quantify_scheduler.backends.circuit_to_device import (
    DeviceCompilationConfig,
)

pytestmark = pytest.mark.usefixtures("close_all_instruments")


@pytest.fixture
def electronic_q0() -> BasicElectronicNVElement:
    """Fixture returning electronic qubit named qe0."""
    q0 = BasicElectronicNVElement("qe0")

    # Electronic NV element is returned
    yield q0
    # after the test, teardown...
    q0.close()


def test_qubit_name(electronic_q0: BasicElectronicNVElement):
    """Qubit name is stored correctly."""
    assert electronic_q0.name == "qe0"


def test_generate_config(electronic_q0: BasicElectronicNVElement):
    """Setting values updates the correct values in the configuration."""
    # set some values for spectroscopy
    electronic_q0.spectroscopy_operation.amplitude(1.0)
    electronic_q0.spectroscopy_operation.duration(10e-6)
    # set some values for reset
    electronic_q0.reset.amplitude(1.0)
    electronic_q0.reset.duration(10e-6)

    dev_cfg = electronic_q0.generate_device_config()

    # assert values in right place in config.
    cfg_spec_pulse = dev_cfg.elements["qe0"]["spectroscopy_operation"]
    assert cfg_spec_pulse.factory_kwargs["duration"] == 10e-6
    assert cfg_spec_pulse.factory_kwargs["amplitude"] == 1.0
    cfg_spec_pulse = dev_cfg.elements["qe0"]["reset"]
    assert cfg_spec_pulse.factory_kwargs["duration"] == 10e-6
    assert cfg_spec_pulse.factory_kwargs["amp"] == 1.0


def test_generate_device_config(electronic_q0: BasicElectronicNVElement):
    """Generating device config returns DeviceCompilationConfig."""
    dev_cfg = electronic_q0.generate_device_config()
    assert isinstance(dev_cfg, DeviceCompilationConfig)


def test_mock_mv_setup():
    """Can use mock setup multiple times after closing it."""
    # test that everything works once
    mock_nv_setup = set_up_basic_mock_nv_setup()
    assert isinstance(mock_nv_setup, dict)
    set_standard_params_basic_nv(mock_nv_setup)
    for key in mock_nv_setup:
        Instrument.find_instrument(key).close()

    # test that tear-down closes all instruments by re-executing
    mock_nv_setup = set_up_basic_mock_nv_setup()
    assert isinstance(mock_nv_setup, dict)
    set_standard_params_basic_nv(mock_nv_setup)
    for key in mock_nv_setup:
        Instrument.find_instrument(key).close()


@pytest.fixture
def dev() -> QuantumDevice:
    """Fixture returning quantum device with instrument coordinator."""
    device = QuantumDevice("dev")
    coordinator = InstrumentCoordinator("ic")
    device.instr_instrument_coordinator(coordinator.name)
    yield device
    device.close()
    coordinator.close()


def test_find_coordinator(dev: QuantumDevice):
    """Quantum device has instrument coordinator."""
    coordinator = dev.instr_instrument_coordinator.get_instr()
    assert coordinator.name == "ic"


def test_generate_device_config_part_of_device(
    electronic_q0: BasicElectronicNVElement, dev: QuantumDevice
):
    """Device config contains entry for a device element."""
    dev.add_element(electronic_q0)
    dev_cfg = dev.generate_device_config()
    assert "qe0" in dev_cfg.elements
