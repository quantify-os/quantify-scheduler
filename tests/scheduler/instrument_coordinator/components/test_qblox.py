# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from cluster import cluster
from pulsar_qcm import pulsar_qcm
from pulsar_qrm import pulsar_qrm
from quantify_core.data.handling import set_datadir  # pylint: disable=no-name-in-module

import quantify_scheduler.schemas.examples as es
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.instrument_coordinator.components import qblox

pytestmark = pytest.mark.usefixtures("close_all_instruments")

esp = inspect.getfile(es)

cfg_f = Path(esp).parent / "transmon_test_config.json"
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = Path(esp).parent / "qblox_test_mapping.json"
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)


@pytest.fixture(name="make_cluster")
def fixture_make_cluster(mocker):
    def _make_cluster(name: str = "cluster0") -> qblox.ClusterComponent:
        cluster0 = cluster.cluster_dummy(name)
        component = qblox.ClusterComponent(cluster0)
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.arm_sequencer")
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.start_sequencer")
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.stop_sequencer")

        qcm0 = cluster.cluster_qcm_dummy(f"{name}_qcm0")
        qcm1 = cluster.cluster_qcm_dummy(f"{name}_qcm1")
        component.add_modules(qcm0, qcm1)

        return component

    yield _make_cluster


@pytest.fixture(name="make_qcm")
def fixture_make_qcm(mocker):
    def _make_qcm(
        name: str = "qcm0", serial: str = "dummy"
    ) -> qblox.PulsarQCMComponent:
        mocker.patch(
            "pulsar_qcm.pulsar_qcm_scpi_ifc.pulsar_qcm_scpi_ifc._get_lo_hw_present",
            return_value=False,
        )
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.arm_sequencer")
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.start_sequencer")
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.stop_sequencer")
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc._set_reference_source")

        qcm = pulsar_qcm.pulsar_qcm_dummy(name)
        qcm._serial = serial

        component = qblox.PulsarQCMComponent(qcm)
        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qcm)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value={"status": "ARMED"},
        )

        return component

    yield _make_qcm


@pytest.fixture(name="make_qrm")
def fixture_make_qrm(mocker):
    def _make_qrm(
        name: str = "qrm0", serial: str = "dummy"
    ) -> qblox.PulsarQRMComponent:
        mocker.patch(
            "pulsar_qrm.pulsar_qrm_scpi_ifc.pulsar_qrm_scpi_ifc._get_lo_hw_present",
            return_value=False,
        )
        mocker.patch("pulsar_qrm.pulsar_qrm_ifc.pulsar_qrm_ifc.arm_sequencer")
        mocker.patch("pulsar_qrm.pulsar_qrm_ifc.pulsar_qrm_ifc.start_sequencer")
        mocker.patch("pulsar_qrm.pulsar_qrm_ifc.pulsar_qrm_ifc.stop_sequencer")
        mocker.patch("pulsar_qrm.pulsar_qrm_ifc.pulsar_qrm_ifc._set_reference_source")

        qrm = pulsar_qrm.pulsar_qrm_dummy(name)
        qrm._serial = serial

        component = qblox.PulsarQRMComponent(qrm)
        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qrm)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value={"status": "ARMED"},
        )
        mocker.patch.object(
            component.instrument,
            "get_acquisitions",
            return_value={
                "0": {
                    "index": 0,
                    "acquisition": {
                        "bins": {
                            "integration": {"path0": [0], "path1": [0]},
                            "threshold": [0.12],
                            "avg_cnt": [1],
                        }
                    },
                }
            },
        )

        return component

    yield _make_qrm


@pytest.fixture(name="mock_acquisition_data")
def fixture_mock_acquisition_data():
    acq_channel, acq_index_len = 0, 10  # mock 1 channel, N indices
    avg_count = 10
    data = {
        str(acq_channel): {
            "index": acq_channel,
            "acquisition": {
                "scope": {
                    "path0": {
                        "data": [0.0] * 2 ** 14,
                        "out-of-range": False,
                        "avg_count": avg_count,
                    },
                    "path1": {
                        "data": [0.0] * 2 ** 14,
                        "out-of-range": False,
                        "avg_count": avg_count,
                    },
                },
                "bins": {
                    "integration": {
                        "path0": [0.0] * acq_index_len,
                        "path1": [0.0] * acq_index_len,
                    },
                    "threshold": [0.12] * acq_index_len,
                    "avg_cnt": [avg_count] * acq_index_len,
                },
            },
        }
    }
    yield data


@pytest.fixture
def make_qcm_rf(mocker):
    def _make_qcm_rf(
        name: str = "qcm_rf0", serial: str = "dummy"
    ) -> qblox.PulsarQCMRFComponent:
        mocker.patch(
            "pulsar_qcm.pulsar_qcm_scpi_ifc.pulsar_qcm_scpi_ifc._get_lo_hw_present",
            return_value=True,
        )
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.arm_sequencer")
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.start_sequencer")
        mocker.patch("pulsar_qcm.pulsar_qcm_ifc.pulsar_qcm_ifc.stop_sequencer")

        qcm_rf = pulsar_qcm.pulsar_qcm_dummy(name)
        qcm_rf._serial = serial

        component = qblox.PulsarQCMRFComponent(qcm_rf)
        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qcm_rf)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value={"status": "ARMED"},
        )

        return component

    yield _make_qcm_rf


@pytest.fixture
def make_qrm_rf(mocker):
    def _make_qrm_rf(
        name: str = "qrm_rf0", serial: str = "dummy"
    ) -> qblox.PulsarQRMRFComponent:
        mocker.patch(
            "pulsar_qrm.pulsar_qrm_scpi_ifc.pulsar_qrm_scpi_ifc._get_lo_hw_present",
            return_value=True,
        )
        mocker.patch("pulsar_qrm.pulsar_qrm_ifc.pulsar_qrm_ifc.arm_sequencer")
        mocker.patch("pulsar_qrm.pulsar_qrm_ifc.pulsar_qrm_ifc.start_sequencer")
        mocker.patch("pulsar_qrm.pulsar_qrm_ifc.pulsar_qrm_ifc.stop_sequencer")

        qrm_rf = pulsar_qrm.pulsar_qrm_dummy(name)
        qrm_rf._serial = serial

        component = qblox.PulsarQRMRFComponent(qrm_rf)
        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qrm_rf)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value={"status": "ARMED"},
        )
        mocker.patch.object(
            component.instrument,
            "get_acquisitions",
            return_value={
                "0": {
                    "index": 0,
                    "acquisition": {
                        "bins": {
                            "integration": {"path0": [0], "path1": [0]},
                            "threshold": [0.12],
                            "avg_cnt": [1],
                        }
                    },
                }
            },
        )

        return component

    yield _make_qrm_rf


def test_initialize_pulsar_qcm_component(make_qcm):
    make_qcm("qblox_qcm0", "1234")


def test_initialize_pulsar_qrm_component(make_qrm):
    make_qrm("qblox_qrm0", "1234")


def test_initialize_pulsar_qcm_rf_component(make_qcm_rf):
    make_qcm_rf("qblox_qcm_rf0", "1234")


def test_initialize_pulsar_qrm_rf_component(make_qrm_rf):
    make_qrm_rf("qblox_qrm_rf0", "1234")


def test_initialize_cluster_component(make_cluster):
    make_cluster("cluster0")


def test_prepare(close_all_instruments, schedule_with_measurement, make_qcm, make_qrm):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm.prepare(prog["qcm0"])
        qrm.prepare(prog["qrm0"])

    # Assert
    qcm.instrument.arm_sequencer.assert_called_with(sequencer=0)
    qrm.instrument.arm_sequencer.assert_called_with(sequencer=0)


def test_prepare_force_set(
    close_all_instruments, schedule_with_measurement, make_qcm, make_qrm
):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")

    qcm.instrument.reference_source("internal")
    qcm.instrument._set_reference_source.reset_mock()

    qrm.instrument.reference_source("external")
    qrm.instrument._set_reference_source.reset_mock()

    # Act
    qcm.force_set_parameters(True)
    qrm.force_set_parameters(True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm.prepare(prog["qcm0"])
        qrm.prepare(prog["qrm0"])

    # Assert
    qcm.instrument._set_reference_source.assert_called()
    qrm.instrument._set_reference_source.assert_called()


def test_prepare_lazy(
    close_all_instruments, schedule_with_measurement, make_qcm, make_qrm
):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")

    qcm.instrument.reference_source("internal")
    qcm.instrument._set_reference_source.reset_mock()

    qrm.instrument.reference_source("external")
    qrm.instrument._set_reference_source.reset_mock()

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm.prepare(prog["qcm0"])
        qrm.prepare(prog["qrm0"])

    # Assert
    qcm.instrument._set_reference_source.assert_not_called()
    qrm.instrument._set_reference_source.assert_not_called()


def test_prepare_rf(
    close_all_instruments, schedule_with_measurement_q2, make_qcm_rf, make_qrm_rf
):
    # Arrange
    qcm: qblox.PulsarQCMRFComponent = make_qcm_rf("qcm_rf0", "1234")
    qrm: qblox.PulsarQRMRFComponent = make_qrm_rf("qrm_rf0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement_q2, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm.prepare(prog["qcm_rf0"])
        qrm.prepare(prog["qrm_rf0"])

    # Assert
    qcm.instrument.arm_sequencer.assert_called_with(sequencer=0)
    qrm.instrument.arm_sequencer.assert_called_with(sequencer=0)


def test_prepare_exception_qcm(close_all_instruments, make_qcm):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm("qcm0", "1234")

    invalid_config = {"idontexist": "this is not used"}

    # Act
    with pytest.raises(KeyError) as execinfo:
        qcm.prepare(invalid_config)

    # Assert
    assert execinfo.value.args[0] == (
        "Invalid program. Attempting to access non-existing sequencer with"
        ' name "idontexist".'
    )


def test_prepare_exception_qrm(close_all_instruments, make_qrm):
    # Arrange
    qrm: qblox.PulsarQRMComponent = make_qrm("qcm0", "1234")

    invalid_config = {"idontexist": "this is not used"}

    # Act
    with pytest.raises(KeyError) as execinfo:
        qrm.prepare(invalid_config)

    # Assert
    assert execinfo.value.args[0] == (
        "Invalid program. Attempting to access non-existing sequencer with"
        ' name "idontexist".'
    )


def test_prepare_exception_qcm_rf(close_all_instruments, make_qcm_rf):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm_rf("qcm_rf0", "1234")

    invalid_config = {"idontexist": "this is not used"}

    # Act
    with pytest.raises(KeyError) as execinfo:
        qcm.prepare(invalid_config)

    # Assert
    assert execinfo.value.args[0] == (
        "Invalid program. Attempting to access non-existing sequencer with"
        ' name "idontexist".'
    )


def test_prepare_exception_qrm_rf(close_all_instruments, make_qrm_rf):
    # Arrange
    qrm: qblox.PulsarQRMComponent = make_qrm_rf("qcm_rf0", "1234")

    invalid_config = {"idontexist": "this is not used"}

    # Act
    with pytest.raises(KeyError) as execinfo:
        qrm.prepare(invalid_config)

    # Assert
    assert execinfo.value.args[0] == (
        "Invalid program. Attempting to access non-existing sequencer with"
        ' name "idontexist".'
    )


def test_is_running(make_qrm):
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    assert not qrm.is_running


def test_wait_done(make_qrm):
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    qrm.wait_done()


def test_retrieve_acquisition_qcm(close_all_instruments, make_qcm):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm("qcm0", "1234")

    # Act
    acq = qcm.retrieve_acquisition()

    # Assert
    assert acq is None


def test_retrieve_acquisition_qrm(
    close_all_instruments, schedule_with_measurement, make_qrm
):
    # Arrange
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)
        compiled_schedule = qcompile(
            schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]
        prog = dict(prog)

        qrm.prepare(prog[qrm.instrument.name])
        qrm.start()
        acq = qrm.retrieve_acquisition()

        # Assert
        assert len(acq[(0, 0)]) == 2


def test_retrieve_acquisition_qcm_rf(close_all_instruments, make_qcm_rf):
    # Arrange
    qcm_rf: qblox.PulsarQCMRFComponent = make_qcm_rf("qcm_rf0", "1234")

    # Act
    acq = qcm_rf.retrieve_acquisition()

    # Assert
    assert acq is None


def test_retrieve_acquisition_qrm_rf(
    close_all_instruments, schedule_with_measurement_q2, make_qrm_rf
):
    # Arrange
    qrm_rf: qblox.PulsarQRMComponent = make_qrm_rf("qrm_rf0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)
        compiled_schedule = qcompile(
            schedule_with_measurement_q2, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]
        prog = dict(prog)

        qrm_rf.prepare(prog[qrm_rf.instrument.name])
        qrm_rf.start()
        acq = qrm_rf.retrieve_acquisition()

    # Assert
    assert len(acq[(0, 0)]) == 2


def test_start_qcm_qrm(
    close_all_instruments, schedule_with_measurement, make_qcm, make_qrm
):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm.prepare(prog["qcm0"])
        qrm.prepare(prog["qrm0"])

        qcm.start()
        qrm.start()

    # Assert
    qcm.instrument.start_sequencer.assert_called()
    qrm.instrument.start_sequencer.assert_called()


def test_start_qcm_qrm_rf(
    close_all_instruments, schedule_with_measurement_q2, make_qcm_rf, make_qrm_rf
):
    # Arrange
    qcm_rf: qblox.PulsarQCMRFComponent = make_qcm_rf("qcm_rf0", "1234")
    qrm_rf: qblox.PulsarQRMRFComponent = make_qrm_rf("qrm_rf0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement_q2, DEVICE_CFG, HARDWARE_MAPPING
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm_rf.prepare(prog["qcm_rf0"])
        qrm_rf.prepare(prog["qrm_rf0"])

        qcm_rf.start()
        qrm_rf.start()

    # Assert
    qcm_rf.instrument.start_sequencer.assert_called()
    qrm_rf.instrument.start_sequencer.assert_called()


def test_stop_qcm_qrm(close_all_instruments, make_qcm, make_qrm):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")

    # Act
    qcm.stop()
    qrm.stop()

    # Assert
    qcm.instrument.stop_sequencer.assert_called()
    qrm.instrument.stop_sequencer.assert_called()


def test_stop_qcm_qrm_rf(close_all_instruments, make_qcm, make_qrm):
    # Arrange
    qcm_rf: qblox.PulsarQCMRFComponent = make_qcm("qcm_rf0", "1234")
    qrm_rf: qblox.PulsarQRMRFComponent = make_qrm("qrm_rf0", "1234")

    # Act
    qcm_rf.stop()
    qrm_rf.stop()

    # Assert
    qcm_rf.instrument.stop_sequencer.assert_called()
    qrm_rf.instrument.stop_sequencer.assert_called()


# ------------------- _QRMAcquisitionManager -------------------


def test_qrm_acquisition_manager__init__(make_qrm):
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, dict(), None
    )


def test_get_threshold_data(make_qrm, mock_acquisition_data):
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, dict(), None
    )
    data = acq_manager._get_threshold_data(mock_acquisition_data, 0, 0)
    assert data == 0.12


def test_get_integration_data(make_qrm, mock_acquisition_data):
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, dict(), None
    )
    data = acq_manager._get_integration_data(mock_acquisition_data, acq_channel=0)
    np.testing.assert_array_equal(data[0], np.array([0.0] * 10))
    np.testing.assert_array_equal(data[1], np.array([0.0] * 10))


def test_get_scope_channel_and_index(make_qrm):
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): ("seq0", "trace"),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    result = acq_manager._get_scope_channel_and_index()
    assert result == (0, 0)


def test_get_scope_channel_and_index_exception(make_qrm):
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): ("seq0", "trace"),
        qblox.AcquisitionIndexing(acq_index=1, acq_channel=0): ("seq0", "trace"),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    with pytest.raises(RuntimeError) as execinfo:
        acq_manager._get_scope_channel_and_index()

    assert (
        execinfo.value.args[0]
        == "A scope mode acquisition is defined for both acq_channel 0 with "
        "acq_index 0 as well as acq_channel 0 with acq_index 1. Only a single "
        "trace acquisition is allowed per QRM."
    )


def test_get_protocol(make_qrm):
    answer = "trace"
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): ("seq0", answer),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    assert acq_manager._get_protocol(0, 0) == answer


def test_get_sequencer_index(make_qrm):
    answer = 0
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): (
            f"seq{answer}",
            "trace",
        ),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    assert acq_manager._get_sequencer_index(0, 0) == answer
