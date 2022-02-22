# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import pytest

from quantify_scheduler.compilation import validate_config
from quantify_scheduler.device_under_test.transmon_element import TransmonElement
from quantify_scheduler.instrument_coordinator.instrument_coordinator import (
    InstrumentCoordinator,
)
from quantify_scheduler.backends.circuit_to_device import (
    OperationCompilationConfig,
    DeviceCompilationConfig,
)

pytestmark = pytest.mark.usefixtures("close_all_instruments")


@pytest.fixture
def q0() -> TransmonElement:
    coordinator = InstrumentCoordinator("ic")
    q0 = TransmonElement("q0")
    q0.instrument_coordinator(coordinator.name)

    # Transmon element is returned
    yield q0
    # after the test, teardown...
    q0.close()
    coordinator.close()


def test_qubit_name(q0: TransmonElement):
    assert q0.name == "q0"


def test_generate_config(q0: TransmonElement):
    # test that setting some values updates the correct values in the configuration
    # set some values
    q0.ro_pulse_type("SquarePulse")
    q0.ro_pulse_duration(400e-9)

    q_cfg = q0.generate_config()

    # assert values in right place in config.
    assert q_cfg["q0"]["measure"].factory_kwargs["pulse_type"] == "SquarePulse"
    assert q_cfg["q0"]["measure"].factory_kwargs["pulse_duration"] == 400e-9

    assert q_cfg["q0"]["Rxy"].factory_kwargs["clock"] == "q0.01"
    assert q_cfg["q0"]["Rxy"].gate_info_factory_kwargs == ["theta", "phi"]


def test_generate_device_config(q0: TransmonElement):
    dev_cfg = q0.generate_device_config()
    assert isinstance(dev_cfg, DeviceCompilationConfig)
    # assert validate_config(dev_cfg, scheme_fn="transmon_cfg.json")


def test_find_coordinator(q0: TransmonElement):
    coordinator = q0.instrument_coordinator.get_instr()
    assert coordinator.name == "ic"
