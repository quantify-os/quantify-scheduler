# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox instrument coordinator components."""
import inspect
import json
import logging
import tempfile
from copy import deepcopy
from operator import countOf
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
from qblox_instruments import (
    Cluster,
    ClusterType,
    Pulsar,
    PulsarType,
    SequencerState,
    SequencerStatus,
    SequencerStatusFlags,
)

from quantify_core.data.handling import set_datadir  # pylint: disable=no-name-in-module

from quantify_scheduler.compilation import qcompile
from quantify_scheduler.instrument_coordinator.components import qblox

from tests.fixtures.mock_setup import close_instruments


@pytest.fixture
def make_cluster_component(mocker):
    cluster_component: qblox.ClusterComponent = None

    def _make_cluster_component(
        name: str = "cluster0",
        sequencer_status: SequencerStatus = SequencerStatus.ARMED,
        sequencer_flags: Optional[List[SequencerStatusFlags]] = None,
    ) -> qblox.ClusterComponent:

        mocker.patch("qblox_instruments.native.cluster.Cluster.arm_sequencer")
        mocker.patch("qblox_instruments.native.cluster.Cluster.start_sequencer")
        mocker.patch("qblox_instruments.native.cluster.Cluster.stop_sequencer")

        close_instruments([name])
        cluster = Cluster(
            name=name,
            dummy_cfg={
                "1": ClusterType.CLUSTER_QCM,
                "2": ClusterType.CLUSTER_QCM_RF,
                "3": ClusterType.CLUSTER_QRM,
                "4": ClusterType.CLUSTER_QRM_RF,
            },
        )
        nonlocal cluster_component
        cluster_component = qblox.ClusterComponent(cluster)

        mocker.patch.object(cluster, "reference_source", wraps=cluster.reference_source)

        for component in cluster_component._cluster_modules.values():
            mocker.patch.object(
                component.instrument_channel,
                "get_sequencer_state",
                return_value=SequencerState(
                    sequencer_status, sequencer_flags if sequencer_flags else []
                ),
            )

        return cluster_component

    yield _make_cluster_component

    if cluster_component:
        for component in cluster_component._cluster_modules.values():
            component.close()
        cluster_component.close()


@pytest.fixture
def make_qcm_component(mocker):
    component: qblox.PulsarQCMComponent = None

    def _make_qcm_component(
        name: str = "qcm0",
        serial: str = "dummy",
        sequencer_status: SequencerStatus = SequencerStatus.ARMED,
        sequencer_flags: Optional[List[SequencerStatusFlags]] = None,
    ) -> qblox.PulsarQCMComponent:

        mocker.patch("qblox_instruments.native.pulsar.Pulsar.arm_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.start_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.stop_sequencer")
        mocker.patch(
            "qblox_instruments.scpi.pulsar_qcm.PulsarQcm._set_reference_source"
        )

        close_instruments([name])
        qcm = Pulsar(name=name, dummy_type=PulsarType.PULSAR_QCM)
        qcm._serial = serial

        nonlocal component
        component = qblox.PulsarQCMComponent(qcm)

        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qcm)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value=SequencerState(
                sequencer_status, sequencer_flags if sequencer_flags else []
            ),
        )

        return component

    yield _make_qcm_component

    if component:
        component.close()


@pytest.fixture
def make_qrm_component(mocker):
    component: qblox.PulsarQRMComponent = None

    def _make_qrm_component(
        name: str = "qrm0",
        serial: str = "dummy",
        sequencer_status: SequencerStatus = SequencerStatus.ARMED,
        sequencer_flags: Optional[List[SequencerStatusFlags]] = None,
        patch_acquisitions: bool = False,
    ) -> qblox.PulsarQRMComponent:

        mocker.patch("qblox_instruments.native.pulsar.Pulsar.arm_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.start_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.stop_sequencer")
        mocker.patch(
            "qblox_instruments.scpi.pulsar_qrm.PulsarQrm._set_reference_source"
        )
        if patch_acquisitions:
            mocker.patch(
                "qblox_instruments.native.pulsar.Pulsar.store_scope_acquisition"
            )

        close_instruments([name])
        qrm = Pulsar(name=name, dummy_type=PulsarType.PULSAR_QRM)
        qrm._serial = serial

        nonlocal component
        component = qblox.PulsarQRMComponent(qrm)

        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qrm)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value=SequencerState(
                sequencer_status, sequencer_flags if sequencer_flags else []
            ),
        )

        if patch_acquisitions:
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

    yield _make_qrm_component

    if component:
        component.close()


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
                        "data": [0.0] * 2**14,
                        "out-of-range": False,
                        "avg_count": avg_count,
                    },
                    "path1": {
                        "data": [0.0] * 2**14,
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
    component: qblox.QCMRFComponent = None

    def _make_qcm_rf(
        name: str = "qcm_rf0",
        serial: str = "dummy",
        sequencer_status: SequencerStatus = SequencerStatus.ARMED,
        sequencer_flags: Optional[List[SequencerStatusFlags]] = None,
    ) -> qblox.QCMRFComponent:

        mocker.patch("qblox_instruments.native.pulsar.Pulsar.arm_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.start_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.stop_sequencer")

        close_instruments([name])
        qcm_rf = Pulsar(name=name, dummy_type=PulsarType._PULSAR_QCM_RF)
        qcm_rf._serial = serial

        nonlocal component
        component = qblox.QCMRFComponent(qcm_rf)

        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qcm_rf)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value=SequencerState(
                sequencer_status, sequencer_flags if sequencer_flags else []
            ),
        )

        return component

    yield _make_qcm_rf

    if component:
        component.close()


@pytest.fixture
def make_qrm_rf(mocker):
    component: qblox.QRMRFComponent = None

    def _make_qrm_rf(
        name: str = "qrm_rf0",
        serial: str = "dummy",
        sequencer_status: SequencerStatus = SequencerStatus.ARMED,
        sequencer_flags: Optional[List[SequencerStatusFlags]] = None,
    ) -> qblox.QRMRFComponent:

        mocker.patch("qblox_instruments.native.pulsar.Pulsar.arm_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.start_sequencer")
        mocker.patch("qblox_instruments.native.pulsar.Pulsar.stop_sequencer")

        close_instruments([name])
        qrm_rf = Pulsar(name=name, dummy_type=PulsarType._PULSAR_QRM_RF)
        qrm_rf._serial = serial

        nonlocal component
        component = qblox.QRMRFComponent(qrm_rf)

        mocker.patch.object(component.instrument_ref, "get_instr", return_value=qrm_rf)
        mocker.patch.object(
            component.instrument,
            "get_sequencer_state",
            return_value=SequencerState(
                sequencer_status, sequencer_flags if sequencer_flags else []
            ),
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

    if component:
        component.close()


def test_sequencer_state_flag_info():
    assert len(SequencerStatusFlags) == len(
        qblox._SEQUENCER_STATE_FLAG_INFO
    ), "Verify all flags are represented"

    assert (
        countOf(
            [info.logging_level for info in qblox._SEQUENCER_STATE_FLAG_INFO.values()],
            logging.DEBUG,
        )
        == 3
    ), "Verify no new flags were implicitly added (possibly update count)"


def test_initialize_pulsar_qcm_component(make_qcm_component):
    make_qcm_component("qblox_qcm0", "1234")


def test_initialize_pulsar_qrm_component(make_qrm_component):
    make_qrm_component("qblox_qrm0", "1234")


def test_initialize_pulsar_qcm_rf_component(make_qcm_rf):
    make_qcm_rf("qblox_qcm_rf0", "1234")


def test_initialize_pulsar_qrm_rf_component(make_qrm_rf):
    make_qrm_rf("qblox_qrm_rf0", "1234")


def test_initialize_cluster_component(make_cluster_component):
    make_cluster_component("cluster0")


@pytest.mark.parametrize(
    "set_reference_source, force_set_parameters",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_prepare(
    schedule_with_measurement,
    load_example_transmon_config,
    load_example_qblox_hardware_config,
    make_qcm_component,
    make_qrm_component,
    set_reference_source,
    force_set_parameters,
):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm_component("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")

    if set_reference_source:
        qcm.instrument.reference_source("internal")
        qcm.instrument._set_reference_source.reset_mock()

        qrm.instrument.reference_source("external")
        qrm.instrument._set_reference_source.reset_mock()

    # Act
    qcm.force_set_parameters(force_set_parameters)
    qrm.force_set_parameters(force_set_parameters)

    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement,
            load_example_transmon_config,
            load_example_qblox_hardware_config,
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm.prepare(prog["qcm0"])
        qrm.prepare(prog["qrm0"])

    # Assert
    if not set_reference_source:
        qcm.instrument.arm_sequencer.assert_called_with(sequencer=0)
        qrm.instrument.arm_sequencer.assert_called_with(sequencer=0)
    else:
        if force_set_parameters:
            qcm.instrument._set_reference_source.assert_called()
            qrm.instrument._set_reference_source.assert_called()
        else:
            qcm.instrument._set_reference_source.assert_not_called()
            qrm.instrument._set_reference_source.assert_not_called()


@pytest.mark.parametrize("force_set_parameters", [False, True])
def test_prepare_ref_source_cluster(
    make_basic_schedule,
    load_legacy_transmon_config,
    load_example_qblox_hardware_config,
    make_cluster_component,
    force_set_parameters,
):
    # Arrange
    cluster_name = "cluster0"
    cluster: qblox.ClusterComponent = make_cluster_component(cluster_name)

    cluster.force_set_parameters(force_set_parameters)
    cluster.instrument.reference_source("internal")  # Put it in a known state

    sched = make_basic_schedule("q4")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            sched, load_legacy_transmon_config, load_example_qblox_hardware_config
        )
        compiled_schedule2 = deepcopy(compiled_schedule)
        prog = compiled_schedule["compiled_instructions"]

        cluster.prepare(prog[cluster_name])

    # Assert it's only set in initialization
    cluster.instrument.reference_source.assert_called_once()
    assert compiled_schedule == compiled_schedule2


def test_prepare_rf(
    schedule_with_measurement_q2,
    load_legacy_transmon_config,
    load_example_qblox_hardware_config,
    make_qcm_rf,
    make_qrm_rf,
):
    # Arrange
    qcm: qblox.QCMRFComponent = make_qcm_rf("qcm_rf0", "1234")
    qrm: qblox.QRMRFComponent = make_qrm_rf("qrm_rf0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement_q2,
            load_legacy_transmon_config,
            load_example_qblox_hardware_config,
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm.prepare(prog["qcm_rf0"])
        qrm.prepare(prog["qrm_rf0"])

    # Assert
    qcm.instrument.arm_sequencer.assert_called_with(sequencer=0)
    qrm.instrument.arm_sequencer.assert_called_with(sequencer=0)


def test_prepare_exception_qcm(close_all_instruments, make_qcm_component):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm_component("qcm0", "1234")

    invalid_config = {"idontexist": "this is not used"}

    # Act
    with pytest.raises(KeyError) as execinfo:
        qcm.prepare(invalid_config)

    # Assert
    assert execinfo.value.args[0] == (
        "Invalid program. Attempting to access non-existing sequencer with"
        ' name "idontexist".'
    )


def test_prepare_exception_qrm(close_all_instruments, make_qrm_component):
    # Arrange
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qcm0", "1234")

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
    qcm: qblox.QCMRFComponent = make_qcm_rf("qcm_rf0", "1234")

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
    qrm: qblox.QRMRFComponent = make_qrm_rf("qcm_rf0", "1234")

    invalid_config = {"idontexist": "this is not used"}

    # Act
    with pytest.raises(KeyError) as execinfo:
        qrm.prepare(invalid_config)

    # Assert
    assert execinfo.value.args[0] == (
        "Invalid program. Attempting to access non-existing sequencer with"
        ' name "idontexist".'
    )


@pytest.mark.parametrize(
    "sequencer_status",
    [SequencerStatus.ARMED, SequencerStatus.RUNNING, SequencerStatus.STOPPED],
)
def test_is_running(make_qrm_component, sequencer_status):
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234", sequencer_status)
    assert qrm.is_running is (sequencer_status is SequencerStatus.RUNNING)


@pytest.mark.parametrize(
    "sequencer_status",
    [SequencerStatus.ARMED, SequencerStatus.RUNNING, SequencerStatus.STOPPED],
)
def test_is_running_cluster(make_cluster_component, sequencer_status):
    cluster: qblox.ClusterComponent = make_cluster_component(
        "cluster0", sequencer_status
    )
    assert cluster.is_running is (sequencer_status is SequencerStatus.RUNNING)


@pytest.mark.parametrize(
    "sequencer_flags",
    [[], [SequencerStatusFlags.ACQ_SCOPE_OVERWRITTEN_PATH_0]],
)
def test_wait_done(make_qrm_component, sequencer_flags):
    qrm: qblox.PulsarQRMComponent = make_qrm_component(
        "qrm0", "1234", SequencerStatus.ARMED, sequencer_flags
    )
    qrm.wait_done()


def test_wait_done_cluster(make_cluster_component):
    cluster: qblox.ClusterComponent = make_cluster_component("cluster0")
    cluster.wait_done()


def test_retrieve_acquisition_qcm(close_all_instruments, make_qcm_component):
    qcm: qblox.PulsarQCMComponent = make_qcm_component("qcm0", "1234")

    assert qcm.retrieve_acquisition() is None


def test_retrieve_acquisition_qrm(
    schedule_with_measurement,
    load_example_transmon_config,
    load_example_qblox_hardware_config,
    make_qrm_component,
):
    # Arrange
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)
        compiled_schedule = qcompile(
            schedule_with_measurement,
            load_example_transmon_config,
            load_example_qblox_hardware_config,
        )
        prog = compiled_schedule["compiled_instructions"]
        prog = dict(prog)

        qrm.prepare(prog[qrm.instrument.name])
        qrm.start()
        acq = qrm.retrieve_acquisition()

    # Assert
    assert len(acq[(0, 0)]) == 2


def test_retrieve_acquisition_qcm_rf(close_all_instruments, make_qcm_rf):
    qcm_rf: qblox.QCMRFComponent = make_qcm_rf("qcm_rf0", "1234")

    assert qcm_rf.retrieve_acquisition() is None


def test_retrieve_acquisition_qrm_rf(
    schedule_with_measurement_q2,
    load_legacy_transmon_config,
    load_example_qblox_hardware_config,
    make_qrm_rf,
):
    # Arrange
    qrm_rf: qblox.QRMRFComponent = make_qrm_rf("qrm_rf0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)
        compiled_schedule = qcompile(
            schedule_with_measurement_q2,
            load_legacy_transmon_config,
            load_example_qblox_hardware_config,
        )
        prog = compiled_schedule["compiled_instructions"]
        prog = dict(prog)

        qrm_rf.prepare(prog[qrm_rf.instrument.name])
        qrm_rf.start()
        acq = qrm_rf.retrieve_acquisition()

    # Assert
    assert len(acq[(0, 0)]) == 2


def test_retrieve_acquisition_cluster(
    make_schedule_with_measurement,
    load_legacy_transmon_config,
    load_example_qblox_hardware_config,
    make_cluster_component,
):
    # Arrange
    cluster_name = "cluster0"
    cluster: qblox.ClusterComponent = make_cluster_component(cluster_name)

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)
        compiled_schedule = qcompile(
            make_schedule_with_measurement("q4"),
            load_legacy_transmon_config,
            load_example_qblox_hardware_config,
        )
        prog = compiled_schedule["compiled_instructions"]
        prog = dict(prog)

        cluster.prepare(prog[cluster_name])
        cluster.start()
        acq = cluster.retrieve_acquisition()

    # Assert
    assert acq is not None


def test_start_qcm_qrm(
    schedule_with_measurement,
    load_example_transmon_config,
    load_example_qblox_hardware_config,
    make_qcm_component,
    make_qrm_component,
):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm_component("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement,
            load_example_transmon_config,
            load_example_qblox_hardware_config,
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
    schedule_with_measurement_q2,
    load_legacy_transmon_config,
    load_example_qblox_hardware_config,
    make_qcm_rf,
    make_qrm_rf,
):
    # Arrange
    qcm_rf: qblox.QCMRFComponent = make_qcm_rf("qcm_rf0", "1234")
    qrm_rf: qblox.QRMRFComponent = make_qrm_rf("qrm_rf0", "1234")

    # Act
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_datadir(tmp_dir)

        compiled_schedule = qcompile(
            schedule_with_measurement_q2,
            load_legacy_transmon_config,
            load_example_qblox_hardware_config,
        )
        prog = compiled_schedule["compiled_instructions"]

        qcm_rf.prepare(prog["qcm_rf0"])
        qrm_rf.prepare(prog["qrm_rf0"])

        qcm_rf.start()
        qrm_rf.start()

    # Assert
    qcm_rf.instrument.start_sequencer.assert_called()
    qrm_rf.instrument.start_sequencer.assert_called()


def test_stop_qcm_qrm(close_all_instruments, make_qcm_component, make_qrm_component):
    # Arrange
    qcm: qblox.PulsarQCMComponent = make_qcm_component("qcm0", "1234")
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")

    # Act
    qcm.stop()
    qrm.stop()

    # Assert
    qcm.instrument.stop_sequencer.assert_called()
    qrm.instrument.stop_sequencer.assert_called()


def test_stop_qcm_qrm_rf(close_all_instruments, make_qcm_rf, make_qrm_rf):
    # Arrange
    qcm_rf: qblox.QCMRFComponent = make_qcm_rf("qcm_rf0", "1234")
    qrm_rf: qblox.QRMRFComponent = make_qrm_rf("qrm_rf0", "1234")

    # Act
    qcm_rf.stop()
    qrm_rf.stop()

    # Assert
    qcm_rf.instrument.stop_sequencer.assert_called()
    qrm_rf.instrument.stop_sequencer.assert_called()


def test_stop_cluster(close_all_instruments, make_cluster_component):
    # Arrange
    cluster: qblox.ClusterComponent = make_cluster_component("cluster0")

    # Act
    cluster.stop()

    # Assert
    cluster.instrument.stop_sequencer.assert_called()


# ------------------- _QRMAcquisitionManager -------------------
def test_qrm_acquisition_manager__init__(make_qrm_component):
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")
    qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, dict(), None
    )


def test_get_threshold_data(make_qrm_component, mock_acquisition_data):
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, dict(), None
    )
    data = acq_manager._get_threshold_data(mock_acquisition_data, 0, 0)
    assert data == 0.12


def test_get_integration_data(make_qrm_component, mock_acquisition_data):
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, dict(), None
    )
    data = acq_manager._get_integration_data(mock_acquisition_data, acq_channel=0)
    np.testing.assert_array_equal(data[0], np.array([0.0] * 10))
    np.testing.assert_array_equal(data[1], np.array([0.0] * 10))


def test_store_scope_acquisition(make_qrm_component):
    # Arrange
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): ("seq0", "trace"),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm_component(
        name="qrm0", serial="1234", patch_acquisitions=True
    )
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    acq_manager.scope_mode_sequencer = "seq0"

    # Act
    acq_manager._store_scope_acquisition()

    # Assert
    qrm.instrument.store_scope_acquisition.assert_called_once()


def test_get_scope_channel_and_index(make_qrm_component):
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): ("seq0", "trace"),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    result = acq_manager._get_scope_channel_and_index()
    assert result == (0, 0)


def test_get_scope_channel_and_index_exception(make_qrm_component):
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): ("seq0", "trace"),
        qblox.AcquisitionIndexing(acq_index=1, acq_channel=0): ("seq0", "trace"),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")
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


def test_get_protocol(make_qrm_component):
    answer = "trace"
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): ("seq0", answer),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    assert acq_manager._get_protocol(0, 0) == answer


def test_get_sequencer_index(make_qrm_component):
    answer = 0
    acq_mapping = {
        qblox.AcquisitionIndexing(acq_index=0, acq_channel=0): (
            f"seq{answer}",
            "trace",
        ),
    }
    qrm: qblox.PulsarQRMComponent = make_qrm_component("qrm0", "1234")
    acq_manager = qblox._QRMAcquisitionManager(
        qrm, qrm._hardware_properties.number_of_sequencers, acq_mapping, None
    )
    assert acq_manager._get_sequencer_index(0, 0) == answer
