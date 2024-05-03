import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from quantify_core.data.handling import (
    create_exp_folder,
    gen_tuid,
    get_datadir,
    snapshot,
)
from quantify_core.utilities.experiment_helpers import load_settings_onto_instrument
from quantify_core.utilities.general import save_json
from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.device_under_test.edge import Edge
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement


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
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "external",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                    "2": {"instrument_type": "QRM"},
                },
            },
            "iq_mixer_ic_lo_mw0": {"instrument_type": "IQMixer"},
            "iq_mixer_ic_lo_ro": {"instrument_type": "IQMixer"},
            "ic_lo_ro": {"instrument_type": "LocalOscillator", "power": 1},
            "ic_lo_mw0": {"instrument_type": "LocalOscillator", "power": 1},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:mw-q0.01": {"lo_freq": None, "interm_freq": -100000000.0},
                "q0:res-q0.ro": {"lo_freq": None, "interm_freq": 50000000.0},
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "iq_mixer_ic_lo_mw0.if"],
                ["ic_lo_mw0.output", "iq_mixer_ic_lo_mw0.lo"],
                ["iq_mixer_ic_lo_mw0.rf", "q0:mw"],
                ["cluster0.module2.complex_output_0", "iq_mixer_ic_lo_ro.if"],
                ["ic_lo_ro.output", "iq_mixer_ic_lo_ro.lo"],
                ["iq_mixer_ic_lo_ro.rf", "q0:res"],
            ]
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


@pytest.fixture
def meas_ctrl():
    test_mc = QuantumDevice("test_mc")
    yield test_mc


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


def test_add_and_get_unbound_device_elements(dev):
    DeviceElement("unbound")

    for i in range(25):
        dev.add_element(DeviceElement(f"elem{i}"))

    # Entries in WeakValueDictionary Instrument._all_instruments are automatically
    # discarded when no strong reference to the value exists anymore;
    # Instrument.find_instrument is called by get_element
    for i in range(25):
        dev.get_element(f"elem{i}")

    with pytest.raises(KeyError, match="Instrument with name unbound does not exist"):
        DeviceElement.find_instrument("unbound")


def test_add_and_get_unbound_edges(dev):
    Edge(parent_element_name="parent", child_element_name="child")

    for i in range(25):
        dev.add_edge(
            Edge(parent_element_name=f"parent{i}", child_element_name=f"child{i}")
        )

    # Entries in WeakValueDictionary Instrument._all_instruments are automatically
    # discarded when no strong reference to the value exists anymore;
    # Instrument.find_instrument is called by get_edge
    for i in range(25):
        dev.get_edge(f"parent{i}_child{i}")

    with pytest.raises(
        KeyError, match="Instrument with name parent_child does not exist"
    ):
        Edge.find_instrument("parent_child")


@pytest.mark.parametrize("to_file", (True, False))
def test_quantum_device_serialization(
    mock_setup_basic_transmon_with_standard_params,
    to_file,
):
    # Prepare to be serialized quantum device
    amp180_test = 0.250
    q2_phase_correction_test = 44
    cfg_sched_repetitions_test = 512

    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]

    q2 = mock_setup["q2"]
    q2.rxy.amp180(amp180_test)

    edge_q2_q3 = mock_setup["q2_q3"]
    edge_q2_q3.cz.q2_phase_correction(q2_phase_correction_test)

    quantum_device.cfg_sched_repetitions(cfg_sched_repetitions_test)

    elements_list = list(quantum_device.elements())
    edges_list = list(quantum_device.edges())

    # Serialize, close all instruments, deserialize
    if to_file:
        path_serialized_quantum_device = quantum_device.to_json_file(path=get_datadir())

        # Ensure exceptions are thrown when trying to serialize after closing instruments
        # (filename is modified to prevent rewriting of file containing the serialized device)
        mock_filename = os.path.join(get_datadir(), "mock_filename")
        with patch("os.path.join", return_value=mock_filename):
            edge_q2_q3.close()
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Cannot serialize 'quantum_device'. Instruments '['q2_q3']' have "
                    "been closed and their information cannot be retrieved any longer. "
                    "If you do not wish to include these in the "
                    "serialization, please remove using `QuantumDevice.remove_element` or "
                    "`QuantumDevice.remove_edge`."
                ),
            ):
                _ = quantum_device.to_json_file(path=get_datadir())

            QuantumDevice.close_all()  # This closes *any* open instrument
            with pytest.raises(
                RuntimeError,
                match="Cannot serialize 'quantum_device'. All attached instruments have been "
                "closed and their information cannot be retrieved any longer.",
            ):
                _ = quantum_device.to_json_file(path=get_datadir())

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
                "Cannot serialize 'quantum_device'. Instruments '['q2_q3']' have "
                "been closed and their information cannot be retrieved any longer. "
                "If you do not wish to include these in the "
                "serialization, please remove using `QuantumDevice.remove_element` or "
                "`QuantumDevice.remove_edge`."
            ),
        ):
            _ = quantum_device.to_json()

        QuantumDevice.close_all()  # This closes *any* open instrument
        with pytest.raises(
            RuntimeError,
            match="Cannot serialize 'quantum_device'. All attached instruments have been "
            "closed and their information cannot be retrieved any longer.",
        ):
            _ = quantum_device.to_json()

        assert serialized_quantum_device.__class__ is str

        # Ensure QuantumDevice can be deserialized again after closing instruments
        deserialized_quantum_device = QuantumDevice.from_json(serialized_quantum_device)
        QuantumDevice.close_all()
        deserialized_quantum_device = QuantumDevice.from_json(serialized_quantum_device)

    assert deserialized_quantum_device.__class__ is QuantumDevice

    assert deserialized_quantum_device.get_element("q2").rxy.amp180() == amp180_test
    assert (
        deserialized_quantum_device.get_edge("q2_q3").cz.q2_phase_correction()
        == q2_phase_correction_test
    )
    assert (
        deserialized_quantum_device.cfg_sched_repetitions()
        == cfg_sched_repetitions_test
    )

    assert deserialized_quantum_device.elements() == elements_list
    assert deserialized_quantum_device.edges() == edges_list


def test_quantum_device_serialization_via_snapshot(
    mock_setup_basic_transmon_with_standard_params,
):
    # Prepare to be serialized quantum device
    amp180_test = 0.250
    q2_phase_correction_test = 44
    cfg_sched_repetitions_test = 512

    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]

    q2 = mock_setup["q2"]
    q2.rxy.amp180(amp180_test)

    edge_q2_q3 = mock_setup["q2_q3"]
    edge_q2_q3.cz.q2_phase_correction(q2_phase_correction_test)

    quantum_device.cfg_sched_repetitions(cfg_sched_repetitions_test)

    elements_list = list(quantum_device.elements())
    edges_list = list(quantum_device.edges())

    # Create snapshot and write to disk such that load_settings_onto_instrument can find it
    tuid = gen_tuid()
    save_json(
        directory=Path(create_exp_folder(tuid=tuid)),
        filename="snapshot.json",
        data=snapshot(),
    )

    QuantumDevice.close_all()

    # In deserializing via snapshot, all instruments first need to be recreated
    # Only including a few of these "empty" instruments here for purpose of testing
    deserialized_q2 = BasicTransmonElement("q2")
    deserialized_q3 = BasicTransmonElement("q3")
    deserialized_edge_q2_q3 = CompositeSquareEdge(
        parent_element_name=deserialized_q2.name,
        child_element_name=deserialized_q3.name,
    )
    deserialized_quantum_device = QuantumDevice(name="quantum_device")

    # Perform deserialization of snapshot into just created "empty" instruments
    for instrument in [
        deserialized_q2,
        deserialized_q3,
        deserialized_edge_q2_q3,
        deserialized_quantum_device,
    ]:
        load_settings_onto_instrument(instrument=instrument, tuid=tuid)

    # Assert equality of deserialized and original quantum device
    assert deserialized_quantum_device.get_element("q2").rxy.amp180() == amp180_test
    assert (
        deserialized_quantum_device.get_edge("q2_q3").cz.q2_phase_correction()
        == q2_phase_correction_test
    )
    assert (
        deserialized_quantum_device.cfg_sched_repetitions()
        == cfg_sched_repetitions_test
    )

    assert deserialized_quantum_device.elements() == elements_list
    assert deserialized_quantum_device.edges() == edges_list
