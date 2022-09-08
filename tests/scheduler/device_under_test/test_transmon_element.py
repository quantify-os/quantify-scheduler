# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import pytest

from quantify_scheduler.compilation import validate_config
from quantify_scheduler.device_under_test.transmon_element import (
    BasicTransmonElement,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.instrument_coordinator.instrument_coordinator import (
    InstrumentCoordinator,
)
from quantify_scheduler.backends.circuit_to_device import (
    OperationCompilationConfig,
    DeviceCompilationConfig,
)

pytestmark = pytest.mark.usefixtures("close_all_instruments")


@pytest.fixture
def q0() -> BasicTransmonElement:
    coordinator = InstrumentCoordinator("ic")
    q0 = BasicTransmonElement("q0")

    # Transmon element is returned
    yield q0
    # after the test, teardown...
    q0.close()
    coordinator.close()


def test_qubit_name(q0: BasicTransmonElement):
    assert q0.name == "q0"


def test_generate_config(q0: BasicTransmonElement):
    # test that setting some values updates the correct values in the configuration
    # set some values
    q0.measure.pulse_type("SquarePulse")
    q0.measure.pulse_duration(400e-9)

    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(q0)

    q_cfg = quantum_device.generate_device_config().elements

    # assert values in right place in config.
    assert q_cfg["q0"]["measure"].factory_kwargs["pulse_type"] == "SquarePulse"
    assert q_cfg["q0"]["measure"].factory_kwargs["pulse_duration"] == 400e-9

    assert q_cfg["q0"]["Rxy"].factory_kwargs["clock"] == "q0.01"
    assert q_cfg["q0"]["Rxy"].gate_info_factory_kwargs == ["theta", "phi"]


def test_generate_device_config(q0: BasicTransmonElement):
    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(q0)

    dev_cfg = quantum_device.generate_device_config()
    assert isinstance(dev_cfg, DeviceCompilationConfig)


@pytest.fixture
def dev() -> QuantumDevice:
    dev = QuantumDevice("dev")
    yield dev
    dev.close()


@pytest.fixture
def qb0() -> BasicTransmonElement:
    qb0 = BasicTransmonElement("qb0")
    yield qb0
    qb0.close()


def test_generate_device_config(qb0):
    _ = qb0.generate_device_config()


def test_generate_device_config_part_of_device(qb0, dev):
    dev.add_element(qb0)
    _ = dev.generate_device_config()
