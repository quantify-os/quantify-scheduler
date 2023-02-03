# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=(invalid-name)
"""
Code to set up a mock setup for use in tutorials and testing.
"""

from typing import Any, Dict

from quantify_core.measurement.control import MeasurementControl
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

from quantify_scheduler.device_under_test.nv_element import BasicElectronicNVElement
from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


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

    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator(
        name="instrument_coordinator", add_default_generic_icc=False
    )

    q0 = BasicTransmonElement("q0")
    q1 = BasicTransmonElement("q1")
    q2 = BasicTransmonElement("q2")
    q3 = BasicTransmonElement("q3")
    q4 = BasicTransmonElement("q4")

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

    # Rationale of the dict format is that all instruments can be cleaned up by
    # iterating over the values and calling close. It also avoids that the instruments
    # get garbage-collected while their name is still recorded and used to refer to the
    # instruments.
    return {
        "meas_ctrl": meas_ctrl,
        "instrument_coordinator": instrument_coordinator,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q0_q2": edge_q0_q2,
        "q1_q2": edge_q1_q2,
        "q2_q3": edge_q2_q3,
        "q2_q4": edge_q2_q4,
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


def set_up_basic_mock_nv_setup() -> Dict:
    """Sets up a system containing 1 electronic qubit in an NV center.

    After usage, close all instruments.

    Returns
    -------
        All instruments created. Containing a "quantum_device", electronic qubit "qe0",
        "meas_ctrl" and "instrument_coordinator".
    """

    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator(
        name="instrument_coordinator", add_default_generic_icc=False
    )

    qe0 = BasicElectronicNVElement("qe0")
    qe1 = BasicElectronicNVElement("qe1")
    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(qe0)
    quantum_device.add_element(qe1)
    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

    # Rationale of the dict format is that all instruments can be cleaned up by
    # iterating over the values and calling close. It also avoids that the instruments
    # get garbage-collected while their name is still recorded and used to refer to the
    # instruments.
    return {
        "meas_ctrl": meas_ctrl,
        "instrument_coordinator": instrument_coordinator,
        "qe0": qe0,
        "qe1": qe1,
        "quantum_device": quantum_device,
    }


def set_standard_params_basic_nv(mock_nv_device: Dict[str, Any]) -> None:
    """
    Sets somewhat standard parameters to the mock setup generated above.
    These parameters serve so that the quantum-device is capable of generating
    a configuration that can be used for compiling schedules.

    In normal use, unknown parameters are set as 'nan' values, forcing the user to
    set these. However for testing purposes it can be useful to set some semi-random
    values. The values here are chosen to reflect typical values as used in practical
    experiments. All amplitudes for pulses are set to 1e-3.
    """

    quantum_device = mock_nv_device["quantum_device"]
    qe0: BasicElectronicNVElement = quantum_device.get_element("qe0")
    qe0.clock_freqs.f01.set(3.592e9)
    qe0.clock_freqs.spec.set(2.2e9)
    qe0.clock_freqs.ionization.set(564e12)
    qe0.clock_freqs.ge0.set(470.4e12)
    qe0.clock_freqs.ge1.set(470.4e12 - 5e9)

    qe0.charge_reset.amplitude(1e-3)
    qe0.cr_count.readout_pulse_amplitude(1e-3)
    qe0.cr_count.spinpump_pulse_amplitude(1e-3)
    qe0.reset.amplitude(1e-3)
    qe0.measure.pulse_amplitude(1e-3)
    qe0.spectroscopy_operation.amplitude.set(1e-3)

    qe1: BasicElectronicNVElement = quantum_device.get_element("qe1")
    qe1.clock_freqs.f01.set(4.874e9)
    qe1.clock_freqs.spec.set(1.4e9)
    qe1.clock_freqs.ionization.set(420e12)
    qe1.clock_freqs.ge0.set(470.4e12)
    qe1.clock_freqs.ge1.set(470.4e12 - 5e9)

    qe1.charge_reset.amplitude(1e-3)
    qe1.cr_count.readout_pulse_amplitude(1e-3)
    qe1.cr_count.spinpump_pulse_amplitude(1e-3)
    qe1.reset.amplitude(1e-3)
    qe1.measure.pulse_amplitude(1e-3)
    qe1.spectroscopy_operation.amplitude.set(1e-3)
