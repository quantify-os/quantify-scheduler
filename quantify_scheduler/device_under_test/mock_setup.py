# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Code to set up a mock setup for use in tutorials and testing.
"""

from typing import Dict
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import (
    TransmonElement,
    BasicTransmonElement,
)
from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_core.measurement.control import MeasurementControl


def set_up_mock_transmon_setup_legacy() -> Dict:
    """
    Sets up a system containing 5 transmon qubits. In this legacy setup only qubits
    2 and 3 are connected.

    .. code-block::

        q0    q1

           q2
          /
        q3    q4


    Returns a dictionary containing the instruments that are instantiated as part of
    this setup. The keys corresponds to the names of the instruments.
    """

    # importing from init_mock will execute all the code in the module which
    # will instantiate all the instruments in the mock setup.
    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator(
        name="instrument_coordinator", add_default_generic_icc=False
    )

    # used in parts of the old test suite
    q0 = TransmonElement("q0")  # pylint: disable=invalid-name
    q1 = TransmonElement("q1")  # pylint: disable=invalid-name
    q2 = BasicTransmonElement("q2")  # pylint: disable=invalid-name
    q3 = BasicTransmonElement("q3")  # pylint: disable=invalid-name
    q4 = BasicTransmonElement("q4")  # pylint: disable=invalid-name

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
    quantum_device.add_edge(edge_q2_q3)
    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

    # rational of the dict format is that this function is historically used as part
    # of a fixture and by providing this dict, a cleanup instruments function can
    # iterate over these keys to close all individual instruments and avoid
    # statefull behavior in the tests.
    return {
        "meas_ctrl": meas_ctrl,
        "instrument_coordinator": instrument_coordinator,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q2-q3": edge_q2_q3,
        "quantum_device": quantum_device,
    }


def set_up_mock_transmon_setup() -> Dict:
    """
    Sets up a system containing 5 transmon qubits connected in a star shape.

    .. code-block::

        q0    q1
          \  /
           q2
          /  \
        q3    q4

    Returns a dictionary containing the instruments that are instantiated as part of
    this setup. The keys corresponds to the names of the instruments.
    """

    # importing from init_mock will execute all the code in the module which
    # will instantiate all the instruments in the mock setup.
    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator(
        name="instrument_coordinator", add_default_generic_icc=False
    )

    q0 = BasicTransmonElement("q0")
    q1 = BasicTransmonElement("q1")

    q2 = BasicTransmonElement("q2")  # pylint: disable=invalid-name
    q3 = BasicTransmonElement("q3")  # pylint: disable=invalid-name
    q4 = BasicTransmonElement("q4")  # pylint: disable=invalid-name

    edge_q0_q2 = CompositeSquareEdge(
        parent_element_name=q0.name, child_element_name=q2.name
    )
    edge_q1_q2 = CompositeSquareEdge(
        parent_element_name=q1.name, child_element_name=q2.name
    )

    edge_q2_q3 = CompositeSquareEdge(
        parent_element_name=q2.name, child_element_name=q3.name
    )
    edge_q2_q4 = CompositeSquareEdge(
        parent_element_name=q2.name, child_element_name=q4.name
    )

    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(q0)
    quantum_device.add_element(q1)
    quantum_device.add_element(q2)
    quantum_device.add_element(q3)
    quantum_device.add_element(q4)
    quantum_device.add_edge(edge_q0_q2)
    quantum_device.add_edge(edge_q1_q2)
    quantum_device.add_edge(edge_q2_q3)
    quantum_device.add_edge(edge_q2_q4)
    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

    # rational of the dict format is that this function is historically used as part
    # of a fixture and by providing this dict, a cleanup instruments function can
    # iterate over these keys to close all individual instruments and avoid
    # statefull behavior in the tests.
    return {
        "meas_ctrl": meas_ctrl,
        "instrument_coordinator": instrument_coordinator,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q0-q2": edge_q0_q2,
        "q1-q2": edge_q1_q2,
        "q2-q3": edge_q2_q3,
        "q2-q4": edge_q2_q4,
        "quantum_device": quantum_device,
    }


def set_standard_params_transmon(mock_setup):
    """
    Sets somewhat standard parameters to the mock setup generated above.
    These parameters serve so that the quantum-device is capable of generating
    a configuration that can be used for compiling schedules.

    In normal use, unknown parameters are set as 'nan' values, forcing the user to
    set these. However for testing purposes it can be useful to set some semi-random
    values. The values here are chosen to reflect typical values as used in practical
    experiments.
    """

    q0 = mock_setup["q0"]
    q0.rxy.amp180(0.45)
    q0.clock_freqs.f01(7.3e9)
    q0.clock_freqs.f12(7.0e9)
    q0.clock_freqs.readout(8.0e9)
    q0.measure.acq_delay(100e-9)

    q1 = mock_setup["q1"]
    q1.rxy.amp180(0.325)
    q1.clock_freqs.f01(7.25e9)
    q1.clock_freqs.f12(6.89e9)
    q1.clock_freqs.readout(8.3e9)
    q1.measure.acq_delay(100e-9)

    q2 = mock_setup["q2"]
    # controlled by a QCM-RF max output amp is 0.25V
    q2.rxy.amp180(0.213)
    q2.clock_freqs.f01(6.33e9)
    q2.clock_freqs.f12(6.09e9)
    q2.clock_freqs.readout(8.5e9)
    q2.measure.acq_delay(100e-9)

    q3 = mock_setup["q3"]
    q3.rxy.amp180(0.215)
    q3.clock_freqs.f01(5.71e9)
    q3.clock_freqs.f12(5.48e9)
    q3.clock_freqs.readout(8.7e9)
    q3.measure.acq_delay(100e-9)

    q4 = mock_setup["q4"]
    q4.rxy.amp180(0.208)
    q4.clock_freqs.f01(5.68e9)
    q4.clock_freqs.f12(5.41e9)
    q4.clock_freqs.readout(9.1e9)
    q4.measure.acq_delay(100e-9)
