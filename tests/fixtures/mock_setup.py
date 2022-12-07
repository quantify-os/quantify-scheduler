# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pytest fixtures for quantify-scheduler."""

import os
import shutil
import pathlib

from typing import Any, Dict, List, Union

import pytest
from qcodes import Instrument

from quantify_core.data.handling import get_datadir, set_datadir

from quantify_scheduler.device_under_test.mock_setup import (
    set_standard_params_basic_nv,
    set_up_basic_mock_nv_setup,
    set_up_mock_transmon_setup,
    set_standard_params_transmon,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.schemas.examples import utils

# Test hardware mappings. Note, these will change as we are updating our hardware
# mapping for the graph based compilation.
QBLOX_HARDWARE_MAPPING = utils.load_json_example_scheme("qblox_test_mapping.json")
ZHINST_HARDWARE_MAPPING = utils.load_json_example_scheme("zhinst_test_mapping.json")


def close_instruments(instrument_names: Union[List[str], Dict[str, Any]]):
    """Close all instruments in the list of names supplied.

    Parameters
    ----------
    instrument_names
        List of instrument names or dict, where keys correspond to instrument names.
    """
    for name in instrument_names:
        try:
            Instrument.find_instrument(name).close()
        except KeyError:
            pass


@pytest.fixture(scope="function", autouse=True)
def close_all_instruments_at_start():
    """
    This fixture closes all instruments at the start of each test to prevent unexpected
    KeyError from qcodes.Instrument, e.g.'Another instrument has the name: q5', that may
    arise when a previous test already created an instance of an Instrument with that
    name.
    """
    Instrument.close_all()


@pytest.fixture(scope="session", autouse=True)
def tmp_test_data_dir(tmp_path_factory):
    """
    This is a fixture which uses the pytest tmp_path_factory fixture
    and extends it by copying the entire contents of the test_data
    directory. After the test session is finished, it cleans up the temporary dir.
    """

    # disable this if you want to look at the generated datafiles for debugging.
    use_temp_dir = True
    if use_temp_dir:
        temp_data_dir = tmp_path_factory.mktemp("temp_data")
        set_datadir(temp_data_dir)
        yield temp_data_dir
        shutil.rmtree(temp_data_dir, ignore_errors=True)
    else:
        set_datadir(os.path.join(pathlib.Path.home(), "quantify_scheduler_test"))
        print(f"Data directory set to: {get_datadir()}")
        yield get_datadir()


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="function", autouse=False)
def mock_setup_basic_transmon():
    """
    Returns a mock setup for a basic 5-qubit transmon device.

    This mock setup is created using the :code:`set_up_mock_transmon_setup`
    function from the .device_under_test.mock_setup module.
    """

    # moved to a separate module to allow using the mock_setup in tutorials.
    mock_setup = set_up_mock_transmon_setup()

    mock_instruments = {
        "meas_ctrl": mock_setup["meas_ctrl"],
        "instrument_coordinator": mock_setup["instrument_coordinator"],
        "q0": mock_setup["q0"],
        "q1": mock_setup["q1"],
        "q2": mock_setup["q2"],
        "q3": mock_setup["q3"],
        "q4": mock_setup["q4"],
        "q0_q2": mock_setup["q0_q2"],
        "q1_q2": mock_setup["q1_q2"],
        "q2_q3": mock_setup["q2_q3"],
        "q2_q4": mock_setup["q2_q4"],
        "quantum_device": mock_setup["quantum_device"],
    }

    yield mock_instruments

    # NB only close the instruments this fixture is responsible for to avoid
    # hard to debug side effects
    # N.B. the keys need to correspond to the names of the instruments otherwise
    # they do not close correctly. Watch out with edges (e.g., q0_q2)
    close_instruments(mock_instruments)


@pytest.fixture(scope="function", autouse=False)
def mock_setup_basic_transmon_with_standard_params(mock_setup_basic_transmon):
    set_standard_params_transmon(mock_setup_basic_transmon)
    yield mock_setup_basic_transmon


@pytest.fixture(scope="function", autouse=False)
def mock_setup_basic_nv():
    """
    Returns a mock setup for a basic 1-qubit NV-center device.
    """
    mock_setup = set_up_basic_mock_nv_setup()
    set_standard_params_basic_nv(mock_setup)
    yield mock_setup
    close_instruments(mock_setup)


@pytest.fixture(scope="function", autouse=False)
def mock_setup_basic_nv_qblox_hardware(mock_setup_basic_nv):
    """
    Returns a mock setup for a basic 1-qubit NV-center device with qblox hardware
    config.
    """

    mock_setup_basic_nv["quantum_device"].hardware_config.set(
        utils.load_json_example_scheme("qblox_test_mapping_nv_centers.json")
    )

    yield mock_setup_basic_nv


@pytest.fixture(scope="function", autouse=False)
def device_compile_config_basic_transmon(
    mock_setup_basic_transmon_with_standard_params,
):
    """
    A config generated from a quantum device with 5 transmon qubits
    connected in a star configuration.

    The mock setup has no hardware attached to it.
    """
    # N.B. how this fixture produces the hardware config can change in the future
    # as long as it keeps doing what is described in this docstring.

    mock_setup = mock_setup_basic_transmon_with_standard_params
    yield mock_setup["quantum_device"].generate_compilation_config()


@pytest.fixture(scope="function", autouse=False)
def compile_config_basic_transmon_zhinst_hardware(
    mock_setup_basic_transmon_with_standard_params,
):
    """
    A config for a quantum device with 5 transmon qubits connected in a star
    configuration controlled using Zurich Instruments Hardware.
    """
    # N.B. how this fixture produces the hardware config will change in the future
    # as we separate the config up into a more fine grained config. For now it uses
    # the old JSON files to load settings from.
    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["quantum_device"].hardware_config(ZHINST_HARDWARE_MAPPING)

    # add the hardware config here
    yield mock_setup["quantum_device"].generate_compilation_config()


@pytest.fixture(scope="function", autouse=False)
def compile_config_basic_transmon_qblox_hardware(
    mock_setup_basic_transmon_with_standard_params,
):
    """
    A config for a quantum device with 5 transmon qubits connected in a star
    configuration controlled using Qblox Hardware.
    """
    # N.B. how this fixture produces the hardware config will change in the future
    # as we separate the config up into a more fine grained config. For now it uses
    # the old JSON files to load settings from.
    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["quantum_device"].hardware_config(QBLOX_HARDWARE_MAPPING)

    yield mock_setup["quantum_device"].generate_compilation_config()


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

    close_instruments(mock_instruments)
