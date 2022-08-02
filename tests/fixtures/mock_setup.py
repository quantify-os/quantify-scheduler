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
from quantify_core.measurement.control import MeasurementControl

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import (
    TransmonElement,
    BasicTransmonElement,
)
from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


def close_instruments(instrument_names: List[str]):
    """Close all instruments in the list of names supplied."""
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
    directory. After the test session is finished, it cleans up the temporary dir.
    """

    # disable this if you want to look at the generated datafiles for debugging.
    use_temp_dir = True
    if use_temp_dir:
        temp_data_dir = tmp_path_factory.mktemp("temp_data")
        yield temp_data_dir
        shutil.rmtree(temp_data_dir, ignore_errors=True)
    else:
        set_datadir(os.path.join(pathlib.Path.home(), "quantify_scheduler_test"))
        print(f"Data directory set to: {get_datadir()}")
        yield get_datadir()


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module", autouse=False)
def mock_setup(tmp_test_data_dir):
    """
    Returns a mock setup.
    """
    set_datadir(tmp_test_data_dir)

    # importing from init_mock will execute all the code in the module which
    # will instantiate all the instruments in the mock setup.
    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator(
        name="instrument_coordinator", add_default_generic_icc=False
    )

    q0 = TransmonElement("q0")
    q1 = TransmonElement("q1")
    q2 = BasicTransmonElement("q2")
    q3 = BasicTransmonElement("q3")
    q4 = BasicTransmonElement("q4")
    q5 = BasicTransmonElement("q5")

    edge_q2_q3 = CompositeSquareEdge(
        parent_element_name=q2.name, child_element_name=q3.name
    )

    q0.ro_pulse_amp(0.08)
    q0.ro_freq(8.1e9)
    q0.freq_01(5.8e9)
    q0.freq_12(5.45e9)
    q0.mw_amp180(0.314)
    q0.mw_pulse_duration(20e-9)
    q0.ro_pulse_delay(20e-9)
    q0.ro_acq_delay(20e-9)

    q1.ro_freq(8.64e9)
    q1.freq_01(6.4e9)
    q1.freq_12(5.05e9)

    edge_q2_q3.cz.q2_phase_correction(44)
    edge_q2_q3.cz.q3_phase_correction(63)

    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(q0)
    quantum_device.add_element(q1)
    quantum_device.add_element(q2)
    quantum_device.add_element(q3)
    quantum_device.add_element(q4)
    quantum_device.add_element(q5)
    quantum_device.add_edge(edge_q2_q3)

    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

    mock_instruments = {
        "meas_ctrl": meas_ctrl,
        "instrument_coordinator": instrument_coordinator,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q5": q5,
        "q2-q3": edge_q2_q3,
        "quantum_device": quantum_device,
    }
    yield mock_instruments

    # NB only close the instruments this fixture is responsible for to avoid
    # hard to debug side effects
    close_instruments(mock_instruments)


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
