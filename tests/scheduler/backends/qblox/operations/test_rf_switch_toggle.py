from quantify_scheduler.backends.qblox.operations.rf_switch_toggle import RFSwitchToggle
from quantify_scheduler.operations import MarkerPulse


def test_init():
    operation = RFSwitchToggle(1, "p", "digital")
    assert operation.duration == 1
    assert operation.name == "RFSwitchToggle"
    assert operation.data["pulse_info"][0]["t0"] == 0
    assert operation.data["pulse_info"][0]["port"] == "p"
    assert operation.data["pulse_info"][0]["clock"] == "digital"
    assert operation.data["pulse_info"][0]["marker_pulse"] is True
    assert not isinstance(operation, MarkerPulse)


def test_init_clock():
    operation = RFSwitchToggle(1, "p", "clock5")
    assert operation.data["pulse_info"][0]["clock"] == "clock5"
