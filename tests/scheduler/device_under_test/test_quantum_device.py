# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

from quantify_scheduler.compilation import validate_config


def test_QuantumDevice_generate_device_config(mock_setup: dict) -> None:

    quantum_device = mock_setup["quantum_device"]
    dev_cfg = quantum_device.generate_device_config()
    validate_config(dev_cfg, scheme_fn="transmon_cfg.json")

    assert {"q0", "q1"} <= set(dev_cfg["qubits"].keys())


def test_QuantumDevice_generate_hardware_config(mock_setup: dict) -> None:

    quantum_device = mock_setup["quantum_device"]

    mock_hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "ic_qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "mode": "complex",
            "ref": "external",
            "IP address": "192.168.0.3",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "ic_lo_mw0",
                "lo_freq": None,
                "seq0": {"port": "q0:mw", "clock": "q0.01", "interm_freq": -100e6},
            },
        },
        "ic_qrm0": {
            "name": "qrm0",
            "instrument_type": "Pulsar_QRM",
            "mode": "complex",
            "ref": "external",
            "IP address": "192.168.0.2",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "ic_lo_ro",
                "lo_freq": None,
                "seq0": {"port": "q0:res", "clock": "q0.ro", "interm_freq": 50e6},
            },
        },
        "ic_lo_ro": {"instrument_type": "LocalOscillator", "lo_freq": None, "power": 1},
        "ic_lo_mw0": {
            "instrument_type": "LocalOscillator",
            "lo_freq": None,
            "power": 1,
        },
    }

    quantum_device.hardware_config(mock_hardware_cfg)

    _ = quantum_device.generate_hardware_config()

    # cannot validate as there is no schema exists see quantify-scheduler #181
    # validate_config(dev_cfg, scheme_fn="qblox_cfg.json")
