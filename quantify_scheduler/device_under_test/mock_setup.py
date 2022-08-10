
from typing import Dict
from quantify_core.measurement.control import MeasurementControl
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

    qblox_hardware_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            # QCM-RF for microwave control
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module6": {
                "instrument_type": "QCM_RF",
                "complex_output_0": {
                    "line_gain_db": 0,
                    "lo_freq": None,
                    "dc_mixer_offset_I": 0.0,
                    "dc_mixer_offset_Q": 0.0,
                    "seq0": {
                        "interm_freq": 200.0e6,
                        "mixer_amp_ratio": 0.9999,
                        "mixer_phase_error_deg": -4.2,
                        "port": "qe0:mw",
                        "clock": "qe0.spec",
                    },
                },
            },
        },
    }
    mock_nv_device.hardware_config.set(qblox_hardware_config)
