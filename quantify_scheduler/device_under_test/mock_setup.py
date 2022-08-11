
from typing import Dict
import os

from quantify_core.measurement.control import MeasurementControl
from quantify_core.utilities import general

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.nv_element import (
    BasicElectronicNVElement,
)
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


def set_up_basic_mock_nv_setup() -> QuantumDevice:
    """Sets up a system containing 1 electronic qubit in an NV center.

    Returns
    -------
        QuantumDevice containing a qubit "qe0", MeasurementControl and
        InstrumentCoordinator.
    """

    meas_ctrl = MeasurementControl("meas_ctrl")
    instrument_coordinator = InstrumentCoordinator(
        name="instrument_coordinator", add_default_generic_icc=False
    )

    q0 = BasicElectronicNVElement("qe0")
    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(q0)
    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

    return quantum_device


def set_standard_params_basic_nv(mock_nv_device: QuantumDevice) -> None:
    """
    Sets somewhat standard parameters to the mock setup generated above.
    These parameters serve so that the quantum-device is capable of generating
    a configuration that can be used for compiling schedules.

    In normal use, unknown parameters are set as 'nan' values, forcing the user to
    set these. However for testing purposes it can be useful to set some semi-random
    values. The values here are chosen to reflect typical values as used in practical
    experiments.
    """

    qe0 = mock_nv_device.get_element("qe0")
    qe0.spectroscopy_pulse.amplitude.set(0.1)
    qe0.clock_freqs.f01.set(3.592e9)
    qe0.clock_freqs.spec.set(2.2e9)

    abs_path = os.path.abspath("quantify_scheduler/schemas/examples/qblox_test_mapping_nv_centers.json")
    qblox_hardware_config = general.load_json_safe(abs_path)
    mock_nv_device.hardware_config.set(qblox_hardware_config)
