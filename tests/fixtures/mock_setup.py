import os
import shutil
import pathlib

import pytest
from quantify_core.data.handling import get_datadir, set_datadir
from quantify_scheduler.device_under_test.mock_setup import set_up_mock_transmon_setup


@pytest.fixture(scope="session", autouse=True)
def tmp_test_data_dir(tmp_path_factory):
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
        yield temp_data_dir
        shutil.rmtree(temp_data_dir, ignore_errors=True)
    else:
        set_datadir(os.path.join(pathlib.Path.home(), "quantify_schedule_test"))
        print(f"Data directory set to: {get_datadir()}")
        yield get_datadir()


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module", autouse=False)
def mock_setup(request, tmp_test_data_dir):
    """
    Returns a mock setup.
    """
    set_datadir(tmp_test_data_dir)

    # moved to a separate module to allow using the mock_setup in tutorials.
    mock_setup = set_up_mock_transmon_setup(include_legacy_transmon=True)

    request.addfinalizer(mock_setup["cleanup_instruments"])

    return {
        "meas_ctrl": mock_setup["meas_ctrl"],
        "instrument_coordinator": mock_setup["instrument_coordinator"],
        "q0": mock_setup["q0"],
        "q1": mock_setup["q1"],
        "q2": mock_setup["q2"],
        "q3": mock_setup["q3"],
        "q4": mock_setup["q4"],
        "edge_q2_q3": mock_setup["edge_q2_q3"],
        "quantum_device": mock_setup["quantum_device"],
    }
