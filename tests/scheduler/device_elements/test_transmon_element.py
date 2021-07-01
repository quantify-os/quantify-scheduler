# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import pytest
from quantify_scheduler.controlstack.station import ControlStack
from quantify_scheduler.compilation import validate_config
from quantify_scheduler.device_elements.transmon_element import TransmonElement


@pytest.fixture
def q_0() -> TransmonElement:
    control_stack = ControlStack("cs")
    q_0 = TransmonElement("q0", control_stack="cs")

    # Transmon element is returned
    yield q_0
    # after the test, teardown...
    q_0.close()
    control_stack.close()


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


def test_find_control_stack(q_0: TransmonElement):
    control_stack = q_0.control_stack.get_instr()
    assert control_stack.name == "cs"
