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

pytestmark = pytest.mark.usefixtures("close_all_instruments")


@pytest.fixture
def q_0() -> TransmonElement:
    coordinator = InstrumentCoordinator("ic")
    q_0 = TransmonElement("q0")
    q_0.instrument_coordinator(coordinator.name)

    # Transmon element is returned
    yield q_0
    # after the test, teardown...
    q_0.close()
    coordinator.close()


def test_qubit_name(q_0: TransmonElement):
    assert q_0.name == "q0"


def test_generate_config(q_0: TransmonElement):

    # set some values
    q_0.ro_pulse_type("square")
    q_0.ro_pulse_duration(400e-9)

    q_cfg = q_0.generate_config()

    # assert values in right place in config.
    assert q_cfg["q0"]["resources"]["port_mw"] == "q0:mw"
    assert q_cfg["q0"]["params"]["ro_pulse_type"] == "square"
    assert q_cfg["q0"]["params"]["ro_pulse_duration"] == 400e-9


def test_generate_device_config(q_0: TransmonElement):
    dev_cfg = q_0.generate_device_config()
    assert validate_config(dev_cfg, scheme_fn="transmon_cfg.json")


def test_find_coordinator(q_0: TransmonElement):
    coordinator = q_0.instrument_coordinator.get_instr()
    assert coordinator.name == "ic"
