# pylint: disable=redefined-outer-name
import pytest
from qcodes import Instrument

from quantify_scheduler.backends.circuit_to_device import (
    DeviceCompilationConfig,
)
from quantify_scheduler.device_under_test.mock_setup import (
    set_up_basic_mock_nv_setup,
    set_standard_params_basic_nv,
)
from quantify_scheduler.device_under_test.nv_element import BasicElectronicNVElement
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.instrument_coordinator.instrument_coordinator import (
    InstrumentCoordinator,
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


def test_operation_configs_are_submodules(electronic_q0: BasicElectronicNVElement):
    """Operation configuration is not only a class attribute, but also a submodule"""
    assert "ports" in electronic_q0.submodules
    assert "clock_freqs" in electronic_q0.submodules
    assert "spectroscopy_operation" in electronic_q0.submodules
    assert "reset" in electronic_q0.submodules
    assert "measure" in electronic_q0.submodules


def test_generate_config_spectroscopy(electronic_q0: BasicElectronicNVElement):
    """Setting values updates the correct values in the config."""
    # Set values for spectroscopy
    electronic_q0.spectroscopy_operation.amplitude(1.0)
    electronic_q0.spectroscopy_operation.duration(10e-6)

    # Get device config
    dev_cfg = electronic_q0.generate_device_config()
    cfg_spec = dev_cfg.elements["qe0"]["spectroscopy_operation"]

    # Assert values are in right place
    assert cfg_spec.factory_kwargs["duration"] == 10e-6
    assert cfg_spec.factory_kwargs["amplitude"] == 1.0


def test_generate_config_reset(electronic_q0: BasicElectronicNVElement):
    """Setting values updates the correct values in the config."""
    # Set values for reset
    electronic_q0.reset.amplitude(1.0)
    electronic_q0.reset.duration(10e-6)

    # Get device config
    dev_cfg = electronic_q0.generate_device_config()
    cfg_reset = dev_cfg.elements["qe0"]["reset"]

    # Assert values are in right place
    assert cfg_reset.factory_kwargs["duration"] == 10e-6
    assert cfg_reset.factory_kwargs["amp"] == 1.0


def test_generate_config_measure(electronic_q0: BasicElectronicNVElement):
    """Setting values updates the correct values in the config."""
    # Set values for measure
    electronic_q0.measure.pulse_amplitude(1.0)
    electronic_q0.measure.pulse_duration(300e-6)
    electronic_q0.measure.acq_duration(287e-6)
    electronic_q0.measure.acq_delay(13e-6)
    electronic_q0.measure.acq_channel(7)
    electronic_q0.clock_freqs.ge0.set(470.4e12)  # 637 nm
    electronic_q0.clock_freqs.ge1.set(470.4e12 - 5e9)  # slightly detuned

    # Get device config
    dev_cfg = electronic_q0.generate_device_config()
    cfg_measure = dev_cfg.elements["qe0"]["measure"]

    # Assert values are in right place
    assert cfg_measure.factory_kwargs["pulse_amplitude"] == 1.0
    assert cfg_measure.factory_kwargs["pulse_duration"] == 300e-6
    assert cfg_measure.factory_kwargs["pulse_port"] == "qe0:optical_control"
    assert cfg_measure.factory_kwargs["pulse_clock"] == "qe0.ge0"
    assert cfg_measure.factory_kwargs["acq_duration"] == 287e-6
    assert cfg_measure.factory_kwargs["acq_delay"] == 13e-6
    assert cfg_measure.factory_kwargs["acq_channel"] == 7
    assert cfg_measure.factory_kwargs["acq_port"] == "qe0:optical_readout"
    assert cfg_measure.factory_kwargs["acq_clock"] == "qe0.ge0"
    assert cfg_measure.factory_kwargs["pulse_type"] == "SquarePulse"


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