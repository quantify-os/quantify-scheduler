# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

import os
import pytest
import re
from unittest.mock import patch
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice


def test_generate_device_config(mock_setup_basic_transmon: dict) -> None:
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    # N.B. the validation of the generated config is happening inside the
    # device object itself using the pydantic dataclass. Invoking the function
    # tests this directly.
    dev_cfg = quantum_device.generate_device_config()

    assert {"q0", "q1", "q2", "q3"} <= set(dev_cfg.elements.keys())
    # Ensure that we also check that the edges are being configured
    assert "q2_q3" in dev_cfg.edges


def test_generate_hardware_config(
    mock_setup_basic_transmon: dict,
) -> None:
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    mock_hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "ic_qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "mode": "complex",
            "ref": "external",
            "IP address": "192.168.0.3",
            "complex_output_0": {
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


@pytest.fixture
def dev():
    dev = QuantumDevice("dev")
    yield dev
    dev.close()


@pytest.fixture
def meas_ctrl():
    test_mc = QuantumDevice("test_mc")
    yield test_mc
    test_mc.close()


def test_adding_non_element_raises(dev, meas_ctrl):
    with pytest.raises(TypeError):
        dev.add_element(meas_ctrl)


def test_invalid_device_element_name():
    invalid_name = "q_0"
    with pytest.raises(ValueError):
        DeviceElement(invalid_name)


def test_wrong_scheduling_strategy(mock_setup_basic_transmon_with_standard_params):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    # Assert that a validation error is raised for scheduling strategy other_strategy
    with pytest.raises(ValueError):
        quantum_device.scheduling_strategy("other_strategy")


@pytest.mark.parametrize("to_file", (True, False))
def test_quantum_device_serialization(
    mock_setup_basic_transmon_with_standard_params,
    tmpdir,
    to_file,
):
    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]

    # Modify default values to ensure args are read correctly within __init__()
    amp180_test = 0.250
    q2_phase_correction_test = 44
    cfg_sched_repetitions_test = 512

    q2 = mock_setup["q2"]
    q2.rxy.amp180(amp180_test)

    edge_q2_q3 = mock_setup["q2_q3"]
    edge_q2_q3.cz.q2_phase_correction(q2_phase_correction_test)

    quantum_device.cfg_sched_repetitions(cfg_sched_repetitions_test)

    elements_list = quantum_device.elements()
    edges_list = quantum_device.edges()

    # Serialize, close all instruments, deserialize
    if to_file:
        path_serialized_quantum_device = quantum_device.to_json_file(path=tmpdir)

        # Ensure exceptions are thrown when trying to serialize after closing instruments
        # (filename is modified to prevent rewriting of file containing the serialize device)
        mock_filename = os.path.join(tmpdir, "mock_filename")
        with patch("os.path.join", return_value=mock_filename):
            edge_q2_q3.close()
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    f"Cannot serialize 'quantum_device'. Instruments '['q2_q3']' have "
                    f"been closed and their information cannot be retrieved any longer. "
                    f"If you do not wish to include these in the "
                    f"serialization, please remove using `QuantumDevice.remove_element` or "
                    f"`QuantumDevice.remove_edge`."
                ),
            ):
                _ = quantum_device.to_json_file(path=tmpdir)

            QuantumDevice.close_all()  # This closes *any* open instrument
            with pytest.raises(
                RuntimeError,
                match=f"Cannot serialize 'quantum_device'. All attached instruments have been "
                f"closed and their information cannot be retrieved any longer.",
            ):
                _ = quantum_device.to_json_file(path=tmpdir)

        assert path_serialized_quantum_device.__class__ is str

        deserialized_quantum_device = QuantumDevice.from_json_file(
            path_serialized_quantum_device
        )

    else:
        serialized_quantum_device = quantum_device.to_json()

        # Ensure exceptions are thrown when trying to serialize after closing instruments
        edge_q2_q3.close()
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                f"Cannot serialize 'quantum_device'. Instruments '['q2_q3']' have "
                f"been closed and their information cannot be retrieved any longer. "
                f"If you do not wish to include these in the "
                f"serialization, please remove using `QuantumDevice.remove_element` or "
                f"`QuantumDevice.remove_edge`."
            ),
        ):
            _ = quantum_device.to_json()

        QuantumDevice.close_all()  # This closes *any* open instrument
        with pytest.raises(
            RuntimeError,
            match=f"Cannot serialize 'quantum_device'. All attached instruments have been "
            f"closed and their information cannot be retrieved any longer.",
        ):
            _ = quantum_device.to_json()

        assert serialized_quantum_device.__class__ is str

        deserialized_quantum_device = QuantumDevice.from_json(serialized_quantum_device)

        # Ensure QuantumDevice can be deserialized again after closing instruments
        QuantumDevice.close_all()
        deserialized_quantum_device = QuantumDevice.from_json(serialized_quantum_device)

    assert deserialized_quantum_device.__class__ is QuantumDevice

    assert (
        deserialized_quantum_device.get_element("q2").rxy.parameters["amp180"]()
        == amp180_test
    )
    assert (
        deserialized_quantum_device.get_edge("q2_q3").cz.parameters[
            "q2_phase_correction"
        ]()
        == q2_phase_correction_test
    )
    assert (
        deserialized_quantum_device.cfg_sched_repetitions()
        == cfg_sched_repetitions_test
    )

    assert deserialized_quantum_device.elements() == elements_list
    assert deserialized_quantum_device.edges() == edges_list

    assert (
        deserialized_quantum_device.snapshot()["parameters"]["elements"]["value"]
        == elements_list
    )

    assert (
        deserialized_quantum_device.snapshot()["parameters"]["edges"]["value"]
    ) == edges_list
