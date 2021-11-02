import os
import pathlib

import pytest
from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement.control import MeasurementControl
from quantify_core.utilities._tests_helpers import rmdir_recursive

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import TransmonElement
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


@pytest.fixture(scope="session", autouse=True)
def tmp_test_data_dir(request, tmp_path_factory):
    """
    This is a fixture which uses the pytest tmp_path_factory fixture
    and extends it by copying the entire contents of the test_data
    directory. After the test session is finished, then it calls
    the `cleaup_tmp` method which tears down the fixture and cleans up itself.
    """

    # disable this if you want to look at the generated datafiles for debugging.
    use_temp_dir = True
    if use_temp_dir:
        temp_data_dir = tmp_path_factory.mktemp("temp_data")

        def cleanup_tmp():
            rmdir_recursive(root_path=temp_data_dir)

        request.addfinalizer(cleanup_tmp)
    else:
        set_datadir(os.path.join(pathlib.Path.home(), "quantify_schedule_test"))
        print(f"Data directory set to: {get_datadir()}")
        temp_data_dir = get_datadir()

    return temp_data_dir


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module", autouse=False)
def mock_setup(request, tmp_test_data_dir):
    """
    Returns a mock setup.
    """
    set_datadir(tmp_test_data_dir)

    # importing from init_mock will execute all the code in the module which
    # will instantiate all the instruments in the mock setup.
    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator("instrument_coordinator")

    q0 = TransmonElement("q0")  # pylint: disable=invalid-name
    q1 = TransmonElement("q1")  # pylint: disable=invalid-name

    q0.ro_pulse_amp(0.08)
    q0.ro_freq(8.1e9)
    q0.freq_01(5.8e9)
    q0.mw_amp180(0.314)
    q0.mw_pulse_duration(20e-9)
    q0.ro_pulse_delay(20e-9)
    q0.ro_acq_delay(20e-9)

    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_component(q0)
    quantum_device.add_component(q1)

    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

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
