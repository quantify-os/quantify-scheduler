# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Code to set up a mock setup for use in tutorials and testing.
"""

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import (
    TransmonElement,
    BasicTransmonElement,
)
from quantify_scheduler.device_under_test.sudden_nz_edge import SuddenNetZeroEdge
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_core.measurement.control import MeasurementControl


def set_up_mock_transmon_setup(include_legacy_transmon: bool = False):
    """
    Sets up a system containing 5 transmon qubits connected in a star shape.

    .. code-block::

        q0    q1
          \  /
           q2
          /  \
        q3    q4

    """

    # importing from init_mock will execute all the code in the module which
    # will instantiate all the instruments in the mock setup.
    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator(
        name="instrument_coordinator", add_default_generic_icc=False
    )

    if include_legacy_transmon:
        q0 = TransmonElement("q0")  # pylint: disable=invalid-name
        q1 = TransmonElement("q1")  # pylint: disable=invalid-name
    else:
        q0 = BasicTransmonElement("q0")
        q1 = BasicTransmonElement("q1")

    q2 = BasicTransmonElement("q2")  # pylint: disable=invalid-name
    q3 = BasicTransmonElement("q3")  # pylint: disable=invalid-name
    q4 = BasicTransmonElement("q4")  # pylint: disable=invalid-name

    edge_q0_q2 = SuddenNetZeroEdge(
        parent_element_name=q0.name, child_element_name=q2.name
    )
    edge_q1_q2 = SuddenNetZeroEdge(
        parent_element_name=q1.name, child_element_name=q2.name
    )

    edge_q2_q3 = SuddenNetZeroEdge(
        parent_element_name=q2.name, child_element_name=q3.name
    )
    edge_q2_q4 = SuddenNetZeroEdge(
        parent_element_name=q2.name, child_element_name=q4.name
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

    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(q0)
    quantum_device.add_element(q1)
    quantum_device.add_element(q2)
    quantum_device.add_element(q3)
    quantum_device.add_element(q4)
    quantum_device.add_edge(edge_q2_q3)

    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

    def cleanup_instruments():
        # NB only close the instruments this fixture is responsible for to avoid
        # hard to debug side effects
        meas_ctrl.close()
        instrument_coordinator.close()
        q0.close()
        q1.close()
        q2.close()
        q3.close()
        q4.close()
        edge_q2_q3.close()
        quantum_device.close()

    return {
        "meas_ctrl": meas_ctrl,
        "instrument_coordinator": instrument_coordinator,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "edge_q2_q3": edge_q2_q3,
        "quantum_device": quantum_device,
        "cleanup_instruments": cleanup_instruments,
    }
