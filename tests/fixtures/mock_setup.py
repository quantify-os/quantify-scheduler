# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pytest fixtures for quantify-scheduler."""

import os
import shutil
import pathlib

from typing import List

import pytest
from qcodes import Instrument

from quantify_core.data.handling import get_datadir, set_datadir
from quantify_scheduler.device_under_test.mock_setup import set_up_mock_transmon_setup

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import (
    TransmonElement,
    BasicTransmonElement,
)
from quantify_scheduler.device_under_test.sudden_nz_edge import SuddenNetZeroEdge
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


def _cleanup_instruments(instrument_names):
    for name in instrument_names:
        try:
            Instrument.find_instrument(name).close()
        except KeyError:
            pass


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
def mock_setup(tmp_test_data_dir):
    """
    Returns a mock setup.
    """

    set_datadir(tmp_test_data_dir)

    # moved to a separate module to allow using the mock_setup in tutorials.
    mock_setup = set_up_mock_transmon_setup(include_legacy_transmon=True)

    mock_instruments = {
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

    yield mock_instruments

    # NB only close the instruments this fixture is responsible for to avoid
    # hard to debug side effects
    _cleanup_instruments(mock_instruments.keys())


@pytest.fixture(scope="function")
def mock_setup_basic_transmon_elements(element_names: List[str]):
    """
    Returns a mock setup consisting of QuantumDevice and BasicTransmonElements only.
    """

    quantum_device = QuantumDevice("quantum_device")

    elements = {}
    for name in element_names:
        elements[name] = BasicTransmonElement(name)
        quantum_device.add_element(elements[name])

    mock_instruments = {"quantum_device": quantum_device, **elements}
    yield mock_instruments

    _cleanup_instruments(mock_instruments)
