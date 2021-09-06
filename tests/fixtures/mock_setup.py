import pytest
from quantify_core.measurement.control import MeasurementControl
from quantify_scheduler.device_under_test.transmon_element import TransmonElement
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module", autouse=False)
def mock_setup(request):
    """
    Returns a mock setup.
    """

    # importing from init_mock will execute all the code in the module which
    # will instantiate all the instruments in the mock setup.
    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator("instrument_coordinator")

    q0 = TransmonElement("q0")
    q1 = TransmonElement("q1")

    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_component(q0)
    quantum_device.add_component(q1)

    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

    mock_hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "ic_qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "mode": "complex",
            "ref": "external",
            "IP address": "192.168.0.3",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "ic_lo_mw0",
                "lo_freq": None,
                "seq0": {"port": "q0:mw", "clock": "q0.01", "interm_freq": -100e6},
            },
        },
        "ic_qrm0": {
            "name": "qrm0",
            "instrument_type": "Pulsar_QRM",
            "mode": "complex",
            "ref": "external",
            "IP address": "192.168.0.2",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "ic_lo_ro",
                "lo_freq": None,
                "seq0": {"port": "q0:res", "clock": "q0.ro", "interm_freq": 50e6},
            },
        },
        "ic_lo_ro": {"instrument_type": "LocalOscillator", "lo_freq": None, "power": 1},
        "ic_lo_mw0": {
            "instrument_type": "LocalOscillator",
            "lo_freq": None,
            "power": 1,
        },
    }

    quantum_device.hardware_config(mock_hardware_cfg)

    def cleanup_instruments():
        # NB only close the instruments this fixture is responsible for to avoid
        # hard to debug side effects
        meas_ctrl.close()
        instrument_coordinator.close()
        q0.close()
        q1.close()
        quantum_device.close()

    request.addfinalizer(cleanup_instruments)

    return {
        "meas_ctrl": meas_ctrl,
        "instrument_coordinator": instrument_coordinator,
        "q0": q0,
        "q1": q1,
        "quantum_device": quantum_device,
    }
