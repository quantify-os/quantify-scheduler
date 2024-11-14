# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox instrument coordinator components."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from copy import copy, deepcopy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from qblox_instruments import (
    Cluster,
    ClusterType,
    DummyBinnedAcquisitionData,
    SequencerStates,
    SequencerStatus,
    SequencerStatuses,
    SequencerStatusFlags,
)
from qcodes.instrument import Instrument, InstrumentChannel, InstrumentModule

from quantify_core.data.handling import get_datadir
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.enums import BinMode, TimeRef, TimeSource
from quantify_scheduler.helpers.qblox_dummy_instrument import (
    start_dummy_cluster_armed_sequencers,
)
from quantify_scheduler.instrument_coordinator.components import qblox
from quantify_scheduler.operations.acquisition_library import (
    SSBIntegrationComplex,
    Timetag,
    TimetagTrace,
    Trace,
    TriggerCount,
)
from quantify_scheduler.operations.gate_library import X90, Measure, Reset
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    SquarePulse,
)
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import AcquisitionMetadata, Schedule
from quantify_scheduler.schemas.examples import utils

EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER = utils.load_json_example_scheme(
    "qblox_hardware_config_nv_center.json"
)


def patch_qtm_parameters(mocker, module):
    """
    Patch QTM QCoDeS parameters.

    This is necessary until the QTM dummy is available in qblox-instruments. (SE-499)
    """
    for seq_no in range(8):
        mocker.patch.object(module[f"sequencer{seq_no}"].parameters["sync_en"], "set")
        mocker.patch.object(module[f"sequencer{seq_no}"].parameters["sequence"], "set")
        mocker.patch.object(module[f"io_channel{seq_no}"].parameters["out_mode"], "set")
        mocker.patch.object(module[f"io_channel{seq_no}"].parameters["out_mode"], "get")
        mocker.patch.object(module[f"io_channel{seq_no}"].parameters["in_trigger_en"], "set")
        mocker.patch.object(module[f"io_channel{seq_no}"].parameters["in_threshold_primary"], "set")
        mocker.patch.object(
            module[f"io_channel{seq_no}"].parameters["binned_acq_time_ref"],
            "set",
        )
        mocker.patch.object(
            module[f"io_channel{seq_no}"].parameters["binned_acq_time_source"],
            "set",
        )
        mocker.patch.object(
            module[f"io_channel{seq_no}"].parameters["binned_acq_on_invalid_time_delta"],
            "set",
        )
        mocker.patch.object(
            module[f"io_channel{seq_no}"].parameters["scope_trigger_mode"],
            "set",
        )
        mocker.patch.object(
            module[f"io_channel{seq_no}"].parameters["scope_mode"],
            "set",
        )


@pytest.fixture
def make_cluster_component(mocker):
    cluster_component: qblox.ClusterComponent = None

    default_modules = {
        "1": "QCM",
        "2": "QCM_RF",
        "3": "QRM",
        "4": "QRM_RF",
        "5": "QTM",
        "7": "QCM",
        "10": "QCM",  # for flux pulsing q0_q3
        "12": "QCM",  # for flux pulsing q4
    }

    def _make_cluster_component(
        name: str = "cluster0",
        modules: dict = default_modules,
        sequencer_status: SequencerStatuses = SequencerStatuses.OKAY,
        sequencer_state: SequencerStates = SequencerStates.ARMED,
        info_flags: list[SequencerStatusFlags] | None = None,
        warn_flags: list[SequencerStatusFlags] | None = None,
        err_flags: list[SequencerStatusFlags] | None = None,
        sequencer_logs: list[str] | None = None,
    ) -> qblox.ClusterComponent:
        qblox_types = {
            "QCM": ClusterType.CLUSTER_QCM,
            "QCM_RF": ClusterType.CLUSTER_QCM_RF,
            "QRM": ClusterType.CLUSTER_QRM,
            "QRM_RF": ClusterType.CLUSTER_QRM_RF,
            "QTM": ClusterType.CLUSTER_QTM,
        }
        cluster = Cluster(
            name=name,
            dummy_cfg={
                slot_idx: qblox_types[module_type] for slot_idx, module_type in modules.items()
            },
        )

        nonlocal cluster_component
        cluster_component = qblox.ClusterComponent(cluster)

        mocker.patch.object(cluster, "reference_source", wraps=cluster.reference_source)

        mocker.patch.object(
            cluster,
            "start_sequencer",
            wraps=lambda: start_dummy_cluster_armed_sequencers(cluster_component),
        )

        mocker.patch.object(cluster, "stop_sequencer", wraps=cluster.stop_sequencer)

        for comp in cluster_component._cluster_modules.values():
            instrument = comp.instrument
            mocker.patch.object(instrument, "arm_sequencer", wraps=instrument.arm_sequencer)
            mocker.patch.object(instrument, "start_sequencer", wraps=instrument.start_sequencer)
            mocker.patch.object(instrument, "stop_sequencer", wraps=instrument.stop_sequencer)
            if not instrument.is_rf_type and not instrument.is_qtm_type:
                mocker.patch.object(instrument, "out0_offset", wraps=instrument.out0_offset)
            mocker.patch.object(instrument, "set", wraps=instrument.set)
            mocker.patch.object(
                instrument,
                "get_sequencer_status",
                return_value=SequencerStatus(
                    sequencer_status,
                    sequencer_state,
                    info_flags if info_flags else [],
                    warn_flags if warn_flags else [],
                    err_flags if err_flags else [],
                    sequencer_logs if sequencer_logs else [],
                ),
            )
            if not instrument.is_qtm_type:
                mocker.patch.object(
                    instrument,
                    "store_scope_acquisition",
                    wraps=instrument.store_scope_acquisition,
                )
            else:
                patch_qtm_parameters(mocker, instrument)

        return cluster_component

    yield _make_cluster_component


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
def mock_qtm_acquisition_data():
    acq_channel, acq_index_len = 0, 10  # mock 1 channel, N indices
    avg_count = 10
    data = {
        str(acq_channel): {
            "index": acq_channel,
            "acquisition": {
                "bins": {
                    "count": [4.0] * acq_index_len,
                    "timedelta": [236942.66] * acq_index_len,
                    "threshold": [1.0] * acq_index_len,
                    "avg_cnt": [avg_count] * acq_index_len,
                },
            },
        }
    }
    yield data


def test_initialize_cluster_component(make_cluster_component):
    modules = {"1": "QCM", "2": "QRM", "3": "QCM_RF", "4": "QRM_RF"}
    cluster = make_cluster_component(name="cluster0", modules=modules)

    cluster = cluster.instrument
    is_qcm_qrm_rf = {
        "1": (True, False, False),
        "2": (False, True, False),
        "3": (True, False, True),
        "4": (False, True, True),
    }

    for module in cluster.modules:
        if (slot_idx := str(module.slot_idx)) in is_qcm_qrm_rf:
            assert (
                module.is_qcm_type,
                module.is_qrm_type,
                module.is_rf_type,
            ) == is_qcm_qrm_rf[slot_idx]


def test_reset_qcodes_settings(
    hardware_cfg_cluster,
    make_cluster_component,
    mock_setup_basic_transmon_with_standard_params,
):
    # Arrange
    cluster_name = "cluster0"
    qcm_name = f"{cluster_name}_module1"
    qrm_name = f"{cluster_name}_module5"
    cluster = make_cluster_component(name=cluster_name, modules={"1": "QCM", "5": "QRM"})
    qcm = cluster._cluster_modules[qcm_name]
    qrm = cluster._cluster_modules[qrm_name]

    hardware_cfg = deepcopy(hardware_cfg_cluster)

    # Set some AWG offsets and gains directly (not through hardware settings).
    # These should be reset when `prepare` is called.
    for path in (0, 1):
        qcm.instrument["sequencer0"].set(f"offset_awg_path{path}", 0.1234)
        qcm.instrument["sequencer0"].set(f"gain_awg_path{path}", 0.4321)

    for seq in (0, 1):
        for path in (0, 1):
            qrm.instrument[f"sequencer{seq}"].set(f"offset_awg_path{path}", 0.6789)
            qrm.instrument[f"sequencer{seq}"].set(f"gain_awg_path{path}", 0.9876)

    # Act
    hardware_cfg["hardware_options"]["sequencer_options"] = {
        "q0:mw-q0.01": {
            "init_offset_awg_path_I": 0.25,
            "init_offset_awg_path_Q": 0.33,
        },
        "q0:fl-cl0.baseband": {"init_gain_awg_path_I": 0.5},
        "q1:fl-cl0.baseband": {"init_gain_awg_path_Q": -0.5},
    }

    schedule = Schedule("Schedule")
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q0:mw",
            clock="q0.01",
            t0=1e-6,
        )
    )
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q0:fl",
            clock="cl0.baseband",
            t0=1e-6,
        )
    )
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q1:fl",
            clock="cl0.baseband",
            t0=1e-6,
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    config = quantum_device.generate_compilation_config()
    compiled_schedule = SerialCompiler(name="compiler").compile(schedule=schedule, config=config)
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    qcm.prepare(prog[qcm_name])
    qrm.prepare(prog[qrm_name])

    # Assert
    qcm_offset = defaultdict(lambda: 0.0)
    qcm_gain = defaultdict(lambda: 1.0)
    qcm_offset[0] = 0.25
    qcm_offset[1] = 0.33
    for path in (0, 1):
        assert qcm.instrument["sequencer0"].parameters[
            f"offset_awg_path{path}"
        ].get() == pytest.approx(qcm_offset[path])
        assert qcm.instrument["sequencer0"].parameters[
            f"gain_awg_path{path}"
        ].get() == pytest.approx(qcm_gain[path])

    qrm_offset = defaultdict(lambda: 0.0)
    qrm_gain = defaultdict(lambda: 1.0)
    qrm_gain["seq0_path0"] = 0.5
    qrm_gain["seq1_path1"] = -0.5
    for seq in (0, 1):
        for path in (0, 1):
            assert qrm.instrument[f"sequencer{seq}"].parameters[
                f"offset_awg_path{path}"
            ].get() == pytest.approx(qrm_offset[f"seq{seq}_path{path}"])
            assert qrm.instrument[f"sequencer{seq}"].parameters[
                f"gain_awg_path{path}"
            ].get() == pytest.approx(qrm_gain[f"seq{seq}_path{path}"])


def test_marker_override_false(
    schedule_with_measurement_q2,
    hardware_cfg_rf,
    make_cluster_component,
    mock_setup_basic_transmon_with_standard_params,
):
    # Arrange
    cluster = make_cluster_component(name="cluster0", sequencer_status=SequencerStates.IDLE)

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["q2"].clock_freqs.readout(7.5e9)
    mock_setup["q2"].clock_freqs.f01(6.03e9)

    all_modules = {module.name: module for module in cluster.instrument.modules}
    qcm_rf_module = all_modules["cluster0_module2"]
    qrm_rf_module = all_modules["cluster0_module4"]

    # These should be reset when `prepare` is called.
    qcm_rf_module["sequencer0"].set("marker_ovr_en", True)
    qrm_rf_module["sequencer0"].set("marker_ovr_en", True)

    mock_setup["quantum_device"].hardware_config(hardware_cfg_rf)
    compiled_schedule = SerialCompiler(name="compiler").compile(
        schedule=schedule_with_measurement_q2,
        config=mock_setup["quantum_device"].generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"]

    cluster.prepare(prog[cluster.instrument.name])

    # Assert
    assert qcm_rf_module["sequencer0"].parameters["marker_ovr_en"].get() is False
    assert qrm_rf_module["sequencer0"].parameters["marker_ovr_en"].get() is False


def test_init_qcodes_settings(
    mocker,
    hardware_cfg_cluster,
    make_cluster_component,
    mock_setup_basic_transmon_with_standard_params,
):
    # Arrange
    cluster_name = "cluster0"
    qcm_name = f"{cluster_name}_module1"
    qrm_name = f"{cluster_name}_module5"
    cluster = make_cluster_component(name=cluster_name, modules={"1": "QCM", "5": "QRM"})
    qcm = cluster._cluster_modules[qcm_name]
    qrm = cluster._cluster_modules[qrm_name]

    for dev in (qcm, qrm):
        for seq in range(qcm._hardware_properties.number_of_sequencers):
            mocker.patch.object(
                dev.instrument[f"sequencer{seq}"].parameters["offset_awg_path0"], "set"
            )
            mocker.patch.object(
                dev.instrument[f"sequencer{seq}"].parameters["offset_awg_path1"], "set"
            )
            mocker.patch.object(
                dev.instrument[f"sequencer{seq}"].parameters["gain_awg_path0"], "set"
            )
            mocker.patch.object(
                dev.instrument[f"sequencer{seq}"].parameters["gain_awg_path1"], "set"
            )
            mocker.patch.object(dev.instrument[f"sequencer{seq}"].parameters["sync_en"], "set")

    hardware_cfg = deepcopy(hardware_cfg_cluster)

    # Act
    hardware_cfg["hardware_options"]["sequencer_options"] = {
        "q0:mw-q0.01": {"init_offset_awg_path_I": 0.25, "init_offset_awg_path_Q": 0.33},
        "q0:fl-cl0.baseband": {"init_gain_awg_path_I": 0.5},
        "q1:fl-cl0.baseband": {"init_gain_awg_path_Q": -0.5},
    }

    schedule = Schedule("Schedule")
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q0:mw",
            clock="q0.01",
            t0=1e-6,
        )
    )
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q0:fl",
            clock="cl0.baseband",
            t0=1e-6,
        )
    )
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q1:fl",
            clock="cl0.baseband",
            t0=1e-6,
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    config = quantum_device.generate_compilation_config()
    compiled_schedule = SerialCompiler(name="compiler").compile(schedule=schedule, config=config)
    prog = compiled_schedule["compiled_instructions"]

    qcm.prepare(prog[cluster_name][qcm_name])
    qrm.prepare(prog[cluster_name][qrm_name])

    # Assert
    qcm_offset = defaultdict(lambda: 0.0)
    qcm_gain = defaultdict(lambda: 1.0)
    qcm_offset[0] = 0.25
    qcm_offset[1] = 0.33
    for path in (0, 1):
        qcm.instrument["sequencer0"].parameters[
            f"offset_awg_path{path}"
        ].set.assert_called_once_with(qcm_offset[path])
        qcm.instrument["sequencer0"].parameters[f"gain_awg_path{path}"].set.assert_called_once_with(
            qcm_gain[path]
        )

    qcm.instrument["sequencer0"].parameters["sync_en"].set.assert_called_with(True)
    qrm.instrument["sequencer0"].parameters["sync_en"].set.assert_called_with(True)

    qrm_offset = defaultdict(lambda: 0.0)
    qrm_gain = defaultdict(lambda: 1.0)
    qrm_gain["seq0_path0"] = 0.5
    qrm_gain["seq1_path1"] = -0.5
    for seq in (0, 1):
        for path in (0, 1):
            qrm.instrument[f"sequencer{seq}"].parameters[
                f"offset_awg_path{path}"
            ].set.assert_called_once_with(qrm_offset[f"seq{seq}_path{path}"])
            qrm.instrument[f"sequencer{seq}"].parameters[
                f"gain_awg_path{path}"
            ].set.assert_called_once_with(qrm_gain[f"seq{seq}_path{path}"])


@pytest.mark.parametrize(
    "set_offset, force_set_parameters",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_prepare_baseband(  # noqa: PLR0915
    mocker,
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_cluster,
    make_cluster_component,
    set_offset,
    force_set_parameters,
):
    # Arrange
    modules = {
        "1": "QCM",
        "2": "QCM_RF",
        "3": "QRM",
        "4": "QRM_RF",
        "5": "QRM",
        "6": "QRM",
    }

    cluster_name = "cluster0"
    cluster = make_cluster_component(
        name=cluster_name,
        modules=modules,
    )

    qcm_name = f"{cluster_name}_module1"
    qrm_name = f"{cluster_name}_module3"
    qrm2_name = f"{cluster_name}_module5"
    qcm0 = cluster._cluster_modules[qcm_name]
    qrm0 = cluster._cluster_modules[qrm_name]
    qrm2 = cluster._cluster_modules[qrm2_name]

    mocker.patch.object(qcm0.instrument.parameters["out0_offset"], "set")
    mocker.patch.object(qcm0.instrument.parameters["out1_offset"], "set")
    mocker.patch.object(qcm0.instrument.parameters["out2_offset"], "set")
    mocker.patch.object(qcm0.instrument.parameters["out3_offset"], "set")

    mocker.patch.object(qrm0.instrument.parameters["out0_offset"], "set")
    mocker.patch.object(qrm0.instrument.parameters["out1_offset"], "set")
    mocker.patch.object(qrm0.instrument.parameters["in0_gain"], "set")
    mocker.patch.object(qrm0.instrument.parameters["in1_gain"], "set")

    mocker.patch.object(qrm2.instrument.parameters["in0_gain"], "set")
    mocker.patch.object(qrm2.instrument.parameters["in1_gain"], "set")

    if set_offset:
        qcm0.instrument.out0_offset.set(0.0)
        qcm0.instrument.out0_offset.set.reset_mock()

        qrm0.instrument.out0_offset.set(0.0)
        qrm0.instrument.out0_offset.set.reset_mock()

        qrm2.instrument.out0_offset.set(0.0)
        qrm2.instrument.out0_offset.set.reset_mock()

    qcm0.force_set_parameters(force_set_parameters)
    qrm0.force_set_parameters(force_set_parameters)
    qrm2.force_set_parameters(force_set_parameters)

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_cluster)
    quantum_device.get_element("q0").clock_freqs.readout(7.5e9)

    schedule = Schedule("Schedule with measurement")
    schedule.add(Reset("q0", "q1"))
    schedule.add(X90("q0"))
    schedule.add(X90("q1"))
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q0:fl",
            clock="cl0.baseband",
            t0=1e-6,
        )
    )
    schedule.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q1:fl",
            clock="cl0.baseband",
            t0=1e-6,
        )
    )
    schedule.add(Measure("q0"))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule, config=quantum_device.generate_compilation_config()
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)

    # Assert
    if set_offset:
        if force_set_parameters:
            qcm0.instrument.set.assert_called()
            qrm0.instrument.set.assert_called()
            qrm2.instrument.set.assert_called()
        else:
            qcm0.instrument.out0_offset.set.assert_not_called()
            qcm0.instrument.set.assert_not_called()
            qrm0.instrument.out0_offset.set.assert_not_called()
            qrm0.instrument.set.assert_not_called()
            qrm2.instrument.out0_offset.set.assert_not_called()
            qrm2.instrument.set.assert_not_called()

    for qcodes_param, hw_config_param in [
        ("out0_offset", ["q0:mw-q0.01", "dc_offset_i"]),
        ("out1_offset", ["q0:mw-q0.01", "dc_offset_q"]),
        ("out2_offset", ["q1:mw-q1.01", "dc_offset_i"]),
        ("out3_offset", ["q1:mw-q1.01", "dc_offset_q"]),
    ]:
        qcm0.instrument.parameters[qcodes_param].set.assert_any_call(
            hardware_cfg_cluster["hardware_options"]["mixer_corrections"][hw_config_param[0]][
                hw_config_param[1]
            ]
        )

    for qcodes_param, hw_config_param in [
        ("out0_offset", ["q0:res-q0.ro", "dc_offset_i"]),
        ("out1_offset", ["q0:res-q0.ro", "dc_offset_q"]),
    ]:
        qrm0.instrument.parameters[qcodes_param].set.assert_any_call(
            hardware_cfg_cluster["hardware_options"]["mixer_corrections"][hw_config_param[0]][
                hw_config_param[1]
            ]
        )

    for qcodes_param, hw_config_param in [
        ("in0_gain", ["q0:res-q0.ro", "gain_I"]),
        ("in1_gain", ["q0:res-q0.ro", "gain_Q"]),
    ]:
        qrm0.instrument.parameters[qcodes_param].set.assert_any_call(
            hardware_cfg_cluster["hardware_options"]["input_gain"][hw_config_param[0]][
                hw_config_param[1]
            ]
        )

    for qcodes_param, portclock in [
        ("in0_gain", "q0:fl-cl0.baseband"),
        ("in1_gain", "q1:fl-cl0.baseband"),
    ]:
        qrm2.instrument.parameters[qcodes_param].set.assert_any_call(
            hardware_cfg_cluster["hardware_options"]["input_gain"][portclock]
        )


@pytest.mark.parametrize("force_set_parameters", [False, True])
def test_prepare_rf(
    mocker,
    mock_setup_basic_transmon,
    qblox_hardware_config_transmon,
    make_cluster_component,
    force_set_parameters,
):
    # Arrange
    cluster_name = "cluster0"
    ic_cluster = make_cluster_component(cluster_name)

    qcm_rf = ic_cluster.instrument.module2
    mocker.patch.object(qcm_rf.parameters["out0_att"], "set")
    mocker.patch.object(qcm_rf.parameters["out1_att"], "set")
    mocker.patch.object(qcm_rf["sequencer0"].parameters["sync_en"], "set")

    qrm_rf = ic_cluster.instrument.module4
    mocker.patch.object(qrm_rf.parameters["out0_att"], "set")
    mocker.patch.object(qrm_rf.parameters["in0_att"], "set")
    mocker.patch.object(qrm_rf["sequencer0"].parameters["sync_en"], "set")

    ic_cluster.force_set_parameters(force_set_parameters)
    ic_cluster.instrument.reference_source("internal")  # Put it in a known state

    sched = Schedule("pulse_sequence")
    sched.add(
        SquarePulse(port="q5:mw", clock="q5.01", amp=0.25, duration=12e-9),
        ref_pt="start",
    )
    sched.add(
        SquarePulse(port="q6:mw", clock="q6.01", amp=0.25, duration=12e-9),
        ref_pt="start",
    )
    sched.add(
        SquarePulse(port="q0:res", clock="q0.ro", amp=0.25, duration=12e-9),
        ref_pt="start",
    )
    sched.add(SSBIntegrationComplex(duration=1e-6, port="q0:res", clock="q0.ro"))
    sched.add_resource(ClockResource("q5.01", freq=5e9))
    sched.add_resource(ClockResource("q6.01", freq=5.3e9))
    sched.add_resource(ClockResource("q0.ro", freq=8e9))

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_transmon)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )
    compiled_schedule_before_prepare = deepcopy(compiled_schedule)

    prog = compiled_schedule["compiled_instructions"]
    ic_cluster.prepare(prog[cluster_name])

    # Assert
    assert compiled_schedule == compiled_schedule_before_prepare

    # Assert it's only set in initialization
    ic_cluster.instrument.reference_source.assert_called_once()

    for qcodes_param, hw_options_param in [
        ("out0_att", ["output_att", "q5:mw-q5.01"]),
        ("out1_att", ["output_att", "q6:mw-q6.01"]),
    ]:
        qcm_rf.parameters[qcodes_param].set.assert_any_call(
            qblox_hardware_config_transmon["hardware_options"][hw_options_param[0]][
                hw_options_param[1]
            ]
        )
    qcm_rf["sequencer0"].parameters["sync_en"].set.assert_called_with(True)

    for qcodes_param, hw_options_param in [
        ("out0_att", ["output_att", "q0:res-q0.ro"]),
        ("in0_att", ["input_att", "q0:res-q0.ro"]),
    ]:
        qrm_rf.parameters[qcodes_param].set.assert_any_call(
            qblox_hardware_config_transmon["hardware_options"][hw_options_param[0]][
                hw_options_param[1]
            ]
        )
    qrm_rf["sequencer0"].parameters["sync_en"].set.assert_called_with(True)


@pytest.mark.parametrize("force_set_parameters", [False, True])
def test_prepare_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
    force_set_parameters,
):
    # Arrange
    out_seq_no = 0
    in_seq_no = 4
    cluster_name = "cluster0"
    ic_cluster = make_cluster_component(cluster_name)

    qtm = ic_cluster.instrument.module5

    ic_cluster.force_set_parameters(force_set_parameters)
    ic_cluster.instrument.reference_source("internal")  # Put it in a known state

    sched = Schedule("pulse_sequence")
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(TriggerCount(duration=1e-6, port="qe1:optical_readout", clock="qe1.ge0"))

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )
    compiled_schedule_before_prepare = deepcopy(compiled_schedule)

    prog = compiled_schedule["compiled_instructions"]
    ic_cluster.prepare(prog[cluster_name])

    # Assert
    assert compiled_schedule == compiled_schedule_before_prepare

    # Assert it's only set in initialization
    ic_cluster.instrument.reference_source.assert_called_once()

    qtm[f"sequencer{in_seq_no}"].parameters["sync_en"].set.assert_called_with(True)
    qtm[f"sequencer{in_seq_no}"].parameters["sequence"].set.assert_called_once()
    qtm[f"sequencer{out_seq_no}"].parameters["sync_en"].set.assert_called_with(True)
    qtm[f"sequencer{out_seq_no}"].parameters["sequence"].set.assert_called_once()


def test_prepare_exception(make_cluster_component):
    # Arrange
    cluster = make_cluster_component(name="cluster0", sequencer_status=SequencerStates.IDLE)

    invalid_config = {"sequencers": {"idontexist": "this is not used"}}

    # Act
    for module_name in cluster._cluster_modules:
        # Act
        with pytest.raises(KeyError) as execinfo:
            cluster._cluster_modules[module_name].prepare(invalid_config)

        # Assert
        assert execinfo.value.args[0] == (
            "Invalid program. Attempting to access non-existing sequencer with"
            ' name "idontexist".'
        )


@pytest.mark.parametrize(
    "sequencer_state",
    [SequencerStates.ARMED, SequencerStates.RUNNING, SequencerStates.STOPPED],
)
def test_is_running(make_cluster_component, sequencer_state):
    cluster = make_cluster_component(name="cluster0", sequencer_state=sequencer_state)
    assert cluster.is_running is (sequencer_state is SequencerStates.RUNNING)


@pytest.mark.parametrize(
    "warn_flags, err_flags",
    [
        ([], []),
        (
            [SequencerStatusFlags.ACQ_SCOPE_OVERWRITTEN_PATH_0],
            [SequencerStatusFlags.SEQUENCE_PROCESSOR_Q1_ILLEGAL_INSTRUCTION],
        ),
        ([SequencerStatusFlags.ACQ_SCOPE_OVERWRITTEN_PATH_0], []),
    ],
)
def test_wait_done(make_cluster_component, warn_flags, err_flags):
    cluster = make_cluster_component(
        name="cluster0",
        sequencer_status=SequencerStates.ARMED,
        warn_flags=warn_flags,
        err_flags=err_flags,
    )
    cluster.wait_done()


def test_retrieve_acquisition(
    mock_setup_basic_transmon_with_standard_params,
    make_cluster_component,
    hardware_cfg_cluster_test_component,
):
    cluster_name = "cluster0"
    qcm_name = f"{cluster_name}_module1"
    qcm_rf_name = f"{cluster_name}_module2"
    qrm_name = f"{cluster_name}_module3"
    qrm_rf_name = f"{cluster_name}_module4"

    cluster = make_cluster_component(
        name=cluster_name,
        modules={"1": "QCM", "2": "QCM_RF", "3": "QRM", "4": "QRM_RF"},
    )
    qcm = cluster._cluster_modules[qcm_name]
    qcm_rf = cluster._cluster_modules[qcm_rf_name]
    qrm = cluster._cluster_modules[qrm_name]
    qrm_rf = cluster._cluster_modules[qrm_rf_name]

    dummy_data = [
        DummyBinnedAcquisitionData(data=(100.0, 200.0), thres=0, avg_cnt=0),
    ]
    expected_dataset = xr.Dataset(
        {0: (["acq_index_0"], [0.1 + 0.2j], {"acq_protocol": "SSBIntegrationComplex"})},
        coords={"acq_index_0": [0]},
    )

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["quantum_device"].hardware_config(hardware_cfg_cluster_test_component)
    mock_setup["q0"].clock_freqs.readout(4.5e8)
    mock_setup["q2"].clock_freqs.readout(7.3e9)

    schedule = Schedule("Retrieve acq sched")

    schedule.add(Measure("q0"))
    schedule.add(Measure("q2"))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule,
        config=mock_setup["quantum_device"].generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    # Baseband
    qrm.instrument.set_dummy_binned_acquisition_data(
        sequencer=0, acq_index_name="0", data=dummy_data
    )

    cluster.prepare(prog)
    cluster.start()

    assert qcm.retrieve_acquisition() is None

    xr.testing.assert_identical(qrm.retrieve_acquisition(), expected_dataset)

    # RF
    qrm_rf.instrument.set_dummy_binned_acquisition_data(
        sequencer=0, acq_index_name="0", data=dummy_data
    )

    cluster.prepare(prog)
    cluster.start()
    assert qcm_rf.retrieve_acquisition() is None
    xr.testing.assert_identical(qrm_rf.retrieve_acquisition(), expected_dataset)


def test_retrieve_acquisition_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    # Dummy data taken directly from hardware test, does not correspond to schedule below
    dummy_data = {
        "0": {
            "index": 0,
            "acquisition": {
                "bins": {
                    "count": [
                        28.0,
                        28.0,
                        29.0,
                        28.0,
                        27.0,
                        30.0,
                        27.0,
                        28.0,
                        29.0,
                        28.0,
                    ],
                    "timedelta": [
                        1898975.0,
                        326098.0,
                        809414.0,
                        2333191.0,
                        760258.0,
                        203253.0,
                        2767205.0,
                        154074.0,
                        637301.0,
                        104949.0,
                    ],
                    "threshold": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "avg_cnt": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                }
            },
        }
    }

    count = np.array(dummy_data["0"]["acquisition"]["bins"]["count"]).astype(int)
    dataarray = xr.DataArray(
        [count],
        dims=["repetition", "acq_index_0"],
        coords={"repetition": [0], "acq_index_0": range(len(count))},
        attrs={"acq_protocol": "TriggerCount"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq")
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(TriggerCount(duration=1e-6, port="qe1:optical_readout", clock="qe1.ge0"))

    mocker.patch.object(
        cluster.instrument.module5,
        "get_acquisitions",
        return_value=dummy_data,
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qtm.retrieve_acquisition(), expected_dataset)


def test_timetag_acquisition_qtm_average(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    qtm_instrument = cluster.instrument.module5

    # Dummy data does not necessarily correspond to schedule below
    dataarray = xr.DataArray(
        np.array([-65145 / 2048]),
        dims=["acq_index_0"],
        coords={"acq_index_0": [0]},
        attrs={"acq_protocol": "Timetag"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq", repetitions=3)
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(
        Timetag(
            duration=1e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.SECOND,
            time_ref=TimeRef.END,
            bin_mode=BinMode.AVERAGE,
        )
    )

    mocker.patch.object(
        cluster.instrument.module5,
        "get_acquisitions",
        return_value={
            "0": {
                "index": 0,
                "acquisition": {
                    "bins": {
                        "count": [5],
                        "timedelta": [-65145],
                        "threshold": [1.0],
                        "avg_cnt": [3],
                    }
                },
            }
        },
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qtm.retrieve_acquisition(), expected_dataset)

    qtm_instrument.io_channel4.binned_acq_time_source.set.assert_called_with(str(TimeSource.SECOND))
    qtm_instrument.io_channel4.binned_acq_time_ref.set.assert_called_with(str(TimeRef.END))


def test_timetag_acquisition_qtm_append(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    # qblox-instruments test assembler does not work for QTM commands yet.
    qtm_instrument = cluster.instrument.module5

    # Dummy data does not necessarily correspond to schedule below
    raw_timetags = [65145, 46403, 34199]
    dataarray = xr.DataArray(
        np.array([t / 2048 for t in raw_timetags]).reshape(3, 1),
        dims=["repetition", "acq_index_0"],
        coords={"acq_index_0": [0]},
        attrs={"acq_protocol": "Timetag"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq", repetitions=3)
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(
        Timetag(
            duration=1e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.SECOND,
            time_ref=TimeRef.START,
            bin_mode=BinMode.APPEND,
        )
    )

    mocker.patch.object(
        cluster.instrument.module5,
        "get_acquisitions",
        return_value={
            "0": {
                "index": 0,
                "acquisition": {
                    "bins": {
                        "count": [5, 2, 7],
                        "timedelta": raw_timetags,
                        "threshold": [1.0, 1.0, 1.0],
                        "avg_cnt": [1, 1, 1],
                    }
                },
            }
        },
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qtm.retrieve_acquisition(), expected_dataset)

    qtm_instrument.io_channel4.binned_acq_time_source.set.assert_called_with(str(TimeSource.SECOND))
    qtm_instrument.io_channel4.binned_acq_time_ref.set.assert_called_with(str(TimeRef.START))


def test_set_in_threshold_primary(
    mock_setup_basic_nv,
    make_cluster_component,
):
    cluster_name = "cluster0"
    cluster = make_cluster_component(cluster_name)
    qtm_instrument = cluster.instrument.module5

    quantum_device = mock_setup_basic_nv["quantum_device"]
    hardware_cfg = EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER.copy()
    hardware_cfg["hardware_options"]["digitization_thresholds"]["qe1:optical_readout-qe1.ge0"][
        "in_threshold_primary"
    ] = 0.3
    quantum_device.hardware_config(hardware_cfg)

    sched = Schedule("digital_pulse_and_acq", repetitions=3)
    sched.add(
        Timetag(
            duration=1e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_source=TimeSource.SECOND,
            time_ref=TimeRef.START,
            bin_mode=BinMode.APPEND,
        )
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    qtm_instrument.io_channel4.in_threshold_primary.set.assert_called_with(0.3)


def test_retrieve_trace_acquisition_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    # Dummy data does not necessarily correspond to schedule below
    dummy_data = [0] * 1000

    dataarray = xr.DataArray(
        np.array(dummy_data, dtype=int).reshape(1, -1),
        dims=["acq_index_0", "trace_index_0"],
        coords={"acq_index_0": [0], "trace_index_0": list(range(1000))},
        attrs={"acq_protocol": "Trace"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq")
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(
        Trace(
            duration=1e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            bin_mode=BinMode.FIRST,
        )
    )

    mocker.patch.object(
        cluster.instrument.module5,
        "get_acquisitions",
        return_value={
            "0": {
                "index": 0,
                "acquisition": {
                    "bins": {
                        "count": [],
                        "timedelta": [],
                        "threshold": [],
                        "avg_cnt": [],
                    }
                },
            }
        },
    )
    mocker.patch.object(
        cluster.instrument.module5.io_channel4,
        "get_scope_data",
        return_value=dummy_data,
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qtm.retrieve_acquisition(), expected_dataset)


def test_retrieve_timetag_trace_acquisition_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    # Dummy data taken directly from hardware test, does not necessarily correspond to
    # schedule below.
    dummy_data = {
        "0": {
            "index": 0,
            "acquisition": {
                "bins": {
                    "count": [
                        28.0,
                    ],
                    "timedelta": [
                        1898975.0,
                    ],
                    "threshold": [1.0],
                    "avg_cnt": [4],
                }
            },
        }
    }
    dummy_scope_data = [
        ["OPEN", 322053621179604992],
        ["RISE", 322053621179644241],
        ["RISE", 322053621181692230],
        ["RISE", 322053621185788284],
        ["RISE", 322053621191932191],
        ["CLOSE", 322053621200494592],
    ]

    timedelta = dummy_data["0"]["acquisition"]["bins"]["timedelta"][0]
    rel_times = np.array(
        [(timedelta + data[1] - dummy_scope_data[1][1]) / 2048 for data in dummy_scope_data[1:-1]]
    )

    dataarray = xr.DataArray(
        rel_times.reshape((1, 1, 4)),
        dims=["repetition", "acq_index_0", "trace_index_0"],
        coords={"acq_index_0": [0], "trace_index_0": list(range(4))},
        attrs={"acq_protocol": "TimetagTrace"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq")
    sched.add(
        TimetagTrace(
            duration=10e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_ref=TimeRef.START,
        )
    )

    mocker.patch.object(
        cluster.instrument.module5,
        "get_acquisitions",
        return_value=dummy_data,
    )
    mocker.patch.object(
        cluster.instrument.module5.io_channel4,
        "get_scope_data",
        return_value=dummy_scope_data,
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qtm.retrieve_acquisition(), expected_dataset)


def test_multiple_retrieve_timetag_trace_acquisition_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    # Dummy data taken directly from hardware test, does not necessarily correspond to
    # schedule below.
    dummy_data = {
        "0": {
            "index": 0,
            "acquisition": {
                "bins": {
                    "count": [
                        4.0,
                        3.0,
                    ],
                    "timedelta": [
                        1898975.0,
                        1898980.0,
                    ],
                    "threshold": [1.0, 1.0],
                    "avg_cnt": [4, 3],
                }
            },
        }
    }
    dummy_scope_data = [
        ["OPEN", 322053621179604992],
        ["RISE", 322053621179644241],
        ["RISE", 322053621181692230],
        ["RISE", 322053621185788284],
        ["RISE", 322053621191932191],
        ["CLOSE", 322053621200494592],
        ["OPEN", 322053621179604992],
        ["RISE", 322053621181692230],
        ["RISE", 322053621185788284],
        ["RISE", 322053621191932191],
        ["CLOSE", 322053621200494592],
    ]

    timedelta_0 = dummy_data["0"]["acquisition"]["bins"]["timedelta"][0]
    rel_times_0 = [
        (timedelta_0 + data[1] - dummy_scope_data[1][1]) / 2048 for data in dummy_scope_data[1:5]
    ]
    timedelta_1 = dummy_data["0"]["acquisition"]["bins"]["timedelta"][1]
    rel_times_1 = [
        (timedelta_1 + data[1] - dummy_scope_data[7][1]) / 2048 for data in dummy_scope_data[7:-1]
    ]
    rel_times = np.array([rel_times_0, rel_times_1 + [np.nan]])

    dataarray = xr.DataArray(
        rel_times.reshape((2, 1, 4)),
        dims=["repetition", "acq_index_0", "trace_index_0"],
        coords={"acq_index_0": [0], "trace_index_0": list(range(4))},
        attrs={"acq_protocol": "TimetagTrace"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq", repetitions=2)
    sched.add(
        TimetagTrace(
            duration=10e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            time_ref=TimeRef.START,
        )
    )

    mocker.patch.object(
        cluster.instrument.module5,
        "get_acquisitions",
        return_value=dummy_data,
    )
    mocker.patch.object(
        cluster.instrument.module5.io_channel4,
        "get_scope_data",
        return_value=dummy_scope_data,
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qtm.retrieve_acquisition(), expected_dataset)


def test_start_baseband(
    schedule_with_measurement,
    hardware_cfg_cluster,
    mock_setup_basic_transmon_with_standard_params,
    make_cluster_component,
):
    cluster_name = "cluster0"
    qcm_name = f"{cluster_name}_module1"
    qrm_name = f"{cluster_name}_module3"

    cluster = make_cluster_component(cluster_name)
    qcm = cluster._cluster_modules[qcm_name]
    qrm = cluster._cluster_modules[qrm_name]

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_cluster)
    quantum_device.get_element("q0").clock_freqs.readout(7.5e9)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule_with_measurement, config=quantum_device.generate_compilation_config()
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    qcm.prepare(prog[qcm_name])
    qrm.prepare(prog[qrm_name])

    qcm.start()
    qrm.start()

    # Assert
    qcm.instrument.arm_sequencer.assert_called_with(sequencer=0)
    qrm.instrument.arm_sequencer.assert_called_with(sequencer=0)

    qcm.instrument.start_sequencer.assert_called()
    qrm.instrument.start_sequencer.assert_called()


def test_start_cluster(
    mock_setup_basic_transmon_with_standard_params,
    schedule_with_measurement_q2,
    hardware_cfg_rf,
    make_cluster_component,
):
    # Arrange
    cluster_name = "cluster0"
    cluster = make_cluster_component(cluster_name)

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["quantum_device"].hardware_config(hardware_cfg_rf)
    mock_setup["q2"].clock_freqs.readout(7.3e9)
    mock_setup["q2"].clock_freqs.f01(6.03e9)
    compilation_config = mock_setup["quantum_device"].generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule_with_measurement_q2, config=compilation_config
    )
    prog = compiled_schedule["compiled_instructions"]

    qcm_rf = cluster._cluster_modules[f"{cluster_name}_module2"]
    qrm_rf = cluster._cluster_modules[f"{cluster_name}_module4"]

    cluster.prepare(prog[cluster_name])

    cluster.start()

    # Assert
    qcm_rf.instrument.arm_sequencer.assert_called_with(sequencer=0)
    qrm_rf.instrument.arm_sequencer.assert_called_with(sequencer=0)

    cluster.instrument.start_sequencer.assert_called()


def test_start_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq")
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(IdlePulse(duration=4e-9))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(sched, config=quantum_device.generate_compilation_config())

    prog = compiled_schedule["compiled_instructions"][cluster_name]

    qtm.prepare(prog[qtm_name])

    qtm.start()

    # Assert
    qtm.instrument.arm_sequencer.assert_called_with(sequencer=0)

    qtm.instrument.start_sequencer.assert_called()


def test_stop_cluster(make_cluster_component):
    # Arrange
    cluster = make_cluster_component("cluster0")

    # Act
    cluster.stop()

    # Assert
    cluster.instrument.stop_sequencer.assert_called()


# ------------------- _QRMAcquisitionManager -------------------
def test_qrm_acquisition_manager__init__(make_cluster_component):
    cluster = make_cluster_component("cluster0")
    qblox._QRMAcquisitionManager(
        parent=cluster._cluster_modules["cluster0_module1"],
        acquisition_metadata=dict(),
        scope_mode_sequencer_and_qblox_acq_index=None,
        acquisition_duration={},
        seq_name_to_idx_map={},
    )


def test_qtm_acquisition_manager__init__(make_cluster_component):
    cluster = make_cluster_component("cluster0")
    qblox._QTMAcquisitionManager(
        parent=cluster._cluster_modules["cluster0_module5"],
        acquisition_metadata=dict(),
        acquisition_duration={},
        seq_name_to_idx_map={},
    )


def test_get_integration_data(make_cluster_component, mock_acquisition_data):
    cluster = make_cluster_component("cluster0")
    acq_manager = qblox._QRMAcquisitionManager(
        parent=cluster._cluster_modules["cluster0_module1"],
        acquisition_metadata=dict(),
        scope_mode_sequencer_and_qblox_acq_index=None,
        acquisition_duration={0: 10},
        seq_name_to_idx_map={"seq0": 0},
    )
    acq_metadata = AcquisitionMetadata(
        "SSBIntegrationComplex", BinMode.AVERAGE, complex, {0: [0]}, 1
    )
    formatted_acquisitions = acq_manager._get_integration_data(
        acq_indices=list(range(10)),
        hardware_retrieved_acquisitions=mock_acquisition_data,
        acquisition_metadata=acq_metadata,
        acq_duration=10,
        qblox_acq_index=0,
        acq_channel=0,
    )

    np.testing.assert_almost_equal(formatted_acquisitions.values, [0.0] * 10)


def test_instrument_module():
    """
    InstrumentModule is treated like InstrumentChannel.
    """

    # Arrange
    instrument = Instrument("test_instr")
    instrument_module = InstrumentModule(instrument, "test_instr_module")
    instrument_module.is_qcm_type = True
    instrument_module.is_rf_type = False
    instrument_module.is_qtm_type = False

    # Act
    component = qblox._QCMComponent(instrument_module)

    # Assert
    assert component.instrument is instrument_module


def test_instrument_channel():
    """InstrumentChannel is added as InstrumentModule."""

    # Arrange
    instrument = Instrument("test_instr")
    instrument_channel = InstrumentChannel(instrument, "test_instr_channel")
    instrument_channel.is_qcm_type = True
    instrument_channel.is_rf_type = False
    instrument_channel.is_qtm_type = False

    # Act
    component = qblox._QCMComponent(instrument_channel)

    # Assert
    assert component.instrument is instrument_channel


def test_get_hardware_log_component_base(
    example_ip,
    hardware_cfg_cluster,
    make_cluster_component,
    mocker,
    mock_qblox_instruments_config_manager,
    mock_setup_basic_transmon_with_standard_params,
):
    cluster = make_cluster_component("cluster0")
    module4 = cluster._cluster_modules["cluster0_module4"]
    module4.instrument.get_ip_config = MagicMock(return_value=example_ip)

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    hardware_cfg = hardware_cfg_cluster
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    sched = Schedule("sched")
    sched.add(Measure("q1"))

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )

    # Create compiled schedule for module
    compiled_sched.compiled_instructions["cluster0_module4"] = deepcopy(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module4"]
    )
    module4_log = module4.get_hardware_log(compiled_sched)
    assert module4_log["app_log"] == "Mock hardware log for app"


def test_get_hardware_log_cluster_component(
    example_ip,
    hardware_cfg_cluster,
    make_cluster_component,
    mocker,
    mock_qblox_instruments_config_manager,
    mock_setup_basic_transmon_with_standard_params,
):
    cluster0 = make_cluster_component("cluster0")
    cluster1 = make_cluster_component("cluster1")

    cluster0.instrument.get_ip_config = MagicMock(return_value=example_ip)

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    hardware_cfg = hardware_cfg_cluster
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    sched = Schedule("sched")
    sched.add(Measure("q1"))

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )

    cluster0_log = cluster0.get_hardware_log(compiled_sched)
    cluster1_log = cluster1.get_hardware_log(compiled_sched)

    source = "app"
    assert cluster0_log["cluster0_cmm"][f"{source}_log"] == f"Mock hardware log for {source}"
    assert cluster0_log["cluster0_module4"][f"{source}_log"] == f"Mock hardware log for {source}"
    assert "cluster0_module17" not in cluster0_log
    assert "serial_number" in cluster0_log["cluster0_idn"]
    assert "IDN" in cluster0_log["cluster0_mods_info"]

    # Assert an instrument not included in the compiled schedule (cluster1)
    # does not produce a log.
    assert cluster1_log is None


def test_download_log(
    example_ip,
    mocker,
    mock_qblox_instruments_config_manager,
):
    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    config_manager = qblox._get_configuration_manager(example_ip)
    cluster_logs = qblox._download_log(config_manager=config_manager, is_cluster=True)

    for source in ["cfg_man", "app", "system"]:
        assert cluster_logs[f"{source}_log"] == f"Mock hardware log for {source}"

    # Assert files are deleted after retrieving hardware logs
    for source in ["app", "system", "cfg_man"]:
        assert source not in os.listdir(get_datadir())

    # Assert error is raised if download_log does not create to file
    config_manager.download_log = MagicMock(return_value=None)
    with pytest.raises(RuntimeError):
        qblox._download_log(config_manager=config_manager, is_cluster=True)


def test_get_instrument_ip(make_cluster_component, example_ip):
    cluster0 = make_cluster_component("cluster0")

    with pytest.raises(ValueError):
        qblox._get_instrument_ip(cluster0)

    cluster0.instrument.get_ip_config = MagicMock(return_value=example_ip)
    assert qblox._get_instrument_ip(cluster0) == example_ip

    cluster0.instrument.get_ip_config = MagicMock(return_value=f"{example_ip}/23")
    assert qblox._get_instrument_ip(cluster0) == example_ip


def test_get_configuration_manager(
    example_ip,
    mocker,
    mock_qblox_instruments_config_manager,
):
    with pytest.raises(RuntimeError) as error:
        # The .test domain is guaranteed not to be registered.
        # https://en.wikipedia.org/wiki/.test
        qblox._get_configuration_manager("bad_ip.test")
    assert "Note: qblox-instruments" in str(error)

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )
    assert hasattr(qblox._get_configuration_manager(example_ip), "download_log")


@pytest.mark.parametrize(
    ("module_type, channel_name, channel_map_parameters"),
    [
        (
            "QCM",
            "complex_output_0",
            {
                "connect_out0": "I",
                "connect_out1": "Q",
                "connect_out2": "off",
                "connect_out3": "off",
            },
        ),
        (
            "QCM",
            "complex_output_1",
            {
                "connect_out0": "off",
                "connect_out1": "off",
                "connect_out2": "I",
                "connect_out3": "Q",
            },
        ),
        (
            "QCM",
            "real_output_0",
            {
                "connect_out0": "I",
                "connect_out1": "off",
                "connect_out2": "off",
                "connect_out3": "off",
            },
        ),
        (
            "QCM",
            "real_output_1",
            {
                "connect_out0": "off",
                "connect_out1": "I",
                "connect_out2": "off",
                "connect_out3": "off",
            },
        ),
        (
            "QCM",
            "real_output_2",
            {
                "connect_out0": "off",
                "connect_out1": "off",
                "connect_out2": "I",
                "connect_out3": "off",
            },
        ),
        (
            "QCM",
            "real_output_3",
            {
                "connect_out0": "off",
                "connect_out1": "off",
                "connect_out2": "off",
                "connect_out3": "I",
            },
        ),
        (
            "QRM",
            "complex_output_0",
            {
                "connect_out0": "I",
                "connect_out1": "Q",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "complex_input_0",
            {
                "connect_out0": "off",
                "connect_out1": "off",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "real_output_0",
            {
                "connect_out0": "I",
                "connect_out1": "off",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "real_output_1",
            {
                "connect_out0": "off",
                "connect_out1": "I",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "real_input_0",
            {
                "connect_out0": "off",
                "connect_out1": "off",
                "connect_acq_I": "in0",
                "connect_acq_Q": "off",
            },
        ),
        (
            "QRM",
            "real_input_1",
            {
                "connect_out0": "off",
                "connect_out1": "off",
                "connect_acq_I": "off",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QCM_RF",
            "complex_output_0",
            {
                "connect_out0": "IQ",
                "connect_out1": "off",
            },
        ),
        (
            "QCM_RF",
            "complex_output_1",
            {
                "connect_out0": "off",
                "connect_out1": "IQ",
            },
        ),
        (
            "QRM_RF",
            "complex_output_0",
            {
                "connect_out0": "IQ",
                "connect_acq": "in0",
            },
        ),
        (
            "QRM_RF",
            "complex_input_0",
            {
                "connect_out0": "off",
                "connect_acq": "in0",
            },
        ),
    ],
)
def test_channel_map(
    make_cluster_component,
    module_type,
    channel_name,
    channel_map_parameters,
):
    # Indices according to `make_cluster_component` instrument setup
    module_idx = {"QCM": 1, "QCM_RF": 2, "QRM": 3, "QRM_RF": 4}
    test_module_name = f"cluster0_module{module_idx[module_type]}"

    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {module_idx[module_type]: {"instrument_type": module_type}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [[f"cluster0.module{module_idx[module_type]}.{channel_name}", "q5:mw"]]
        },
    }

    if "RF" in module_type:
        hardware_config["hardware_options"] = {
            "modulation_frequencies": {"q5:mw-q5.01": {"interm_freq": 3e5}}
        }
        freq_01 = 5e9
    else:
        freq_01 = 4.33e8

    q5 = BasicTransmonElement("q5")

    q5.rxy.amp180(0.213)
    q5.clock_freqs.f01(freq_01)
    q5.clock_freqs.f12(6.09e9)
    q5.clock_freqs.readout(8.5e9)
    q5.measure.acq_delay(100e-9)

    schedule = Schedule("test_channel_map")
    schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q5:mw", clock="q5.01"))

    quantum_device = QuantumDevice("basic_transmon_quantum_device")
    quantum_device.add_element(q5)
    quantum_device.hardware_config(hardware_config)

    compiled_schedule = SerialCompiler(name="compiler").compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    prog = compiled_schedule["compiled_instructions"]

    cluster = make_cluster_component("cluster0")
    cluster.prepare(prog[cluster.instrument.name])

    all_modules = {module.name: module for module in cluster.instrument.modules}
    module = all_modules[test_module_name]

    all_sequencers = {sequencer.name: sequencer for sequencer in module.sequencers}
    sequencer = all_sequencers[f"{test_module_name}_sequencer0"]

    for key, value in channel_map_parameters.items():
        assert sequencer.parameters[key].get() == value


@pytest.mark.parametrize(
    ("module_type, channel_name, channel_name_measure, channel_map_parameters"),
    [
        (
            "QRM",
            "complex_output_0",
            ["complex_input_0"],
            {
                "connect_out0": "I",
                "connect_out1": "Q",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "complex_output_0",
            ["real_input_0", "real_input_1"],
            {
                "connect_out0": "I",
                "connect_out1": "Q",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "real_output_0",
            ["real_input_0"],
            {
                "connect_out0": "I",
                "connect_out1": "off",
                "connect_acq_I": "in0",
                "connect_acq_Q": "off",
            },
        ),
        (
            "QRM",
            "real_output_0",
            ["real_input_1"],
            {
                "connect_out0": "I",
                "connect_out1": "off",
                "connect_acq_I": "off",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "real_output_1",
            ["real_input_0"],
            {
                "connect_out0": "off",
                "connect_out1": "I",
                "connect_acq_I": "in0",
                "connect_acq_Q": "off",
            },
        ),
        (
            "QRM",
            "real_output_1",
            ["real_input_1"],
            {
                "connect_out0": "off",
                "connect_out1": "I",
                "connect_acq_I": "off",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "real_output_0",
            ["real_input_0", "real_input_1"],
            {
                "connect_out0": "I",
                "connect_out1": "off",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM",
            "real_output_1",
            ["real_input_0", "real_input_1"],
            {
                "connect_out0": "off",
                "connect_out1": "I",
                "connect_acq_I": "in0",
                "connect_acq_Q": "in1",
            },
        ),
        (
            "QRM_RF",
            "complex_output_0",
            ["complex_input_0"],
            {
                "connect_out0": "IQ",
                "connect_acq": "in0",
            },
        ),
    ],
)
def test_channel_map_measure(
    make_cluster_component,
    module_type,
    channel_name,
    channel_name_measure,
    channel_map_parameters,
):
    # Indices according to `make_cluster_component` instrument setup
    module_idx = {"QRM": 3, "QRM_RF": 4}
    test_module_name = f"cluster0_module{module_idx[module_type]}"

    hardware_config = {
        "version": "0.2",
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {module_idx[module_type]: {"instrument_type": module_type}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                [f"cluster0.module{module_idx[module_type]}.{channel_name}", "q5:res"],
            ]
        },
    }

    for name in channel_name_measure:
        hardware_config["connectivity"]["graph"].append(
            [
                f"cluster0.module{module_idx[module_type]}.{name}",
                "q5:res",
            ],
        )

    if "RF" in module_type:
        hardware_config["hardware_options"] = {
            "modulation_frequencies": {"q5:res-q5.ro": {"interm_freq": 3e5}}
        }
        freq_01 = 5e9
        readout = 8.5e9
    else:
        freq_01 = 4.33e8
        readout = 4.5e8

    q5 = BasicTransmonElement("q5")

    q5.rxy.amp180(0.213)
    q5.clock_freqs.f01(freq_01)
    q5.clock_freqs.f12(6.09e9)
    q5.clock_freqs.readout(readout)
    q5.measure.acq_delay(100e-9)

    schedule = Schedule("test_channel_map")
    schedule.add(Measure("q5"))

    quantum_device = QuantumDevice("basic_transmon_quantum_device")
    quantum_device.add_element(q5)
    quantum_device.hardware_config(hardware_config)

    compiled_schedule = SerialCompiler(name="compiler").compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    prog = compiled_schedule["compiled_instructions"]

    cluster = make_cluster_component("cluster0")
    cluster.prepare(prog[cluster.instrument.name])

    all_modules = {module.name: module for module in cluster.instrument.modules}
    module = all_modules[test_module_name]

    all_sequencers = {sequencer.name: sequencer for sequencer in module.sequencers}
    sequencer = all_sequencers[f"{test_module_name}_sequencer0"]

    for key, value in channel_map_parameters.items():
        assert sequencer.parameters[key].get() == value


@pytest.mark.parametrize(
    ("slot_idx, module_type"),
    [
        (
            1,
            "QCM",
        ),
        (
            2,
            "QCM_RF",
        ),
        (
            3,
            "QRM",
        ),
        (
            4,
            "QRM_RF",
        ),
    ],
)
def test_channel_map_off_with_marker_pulse(make_cluster_component, slot_idx, module_type):
    cluster_name = "cluster0"
    module_name = f"{cluster_name}_module{slot_idx}"

    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            cluster_name: {
                "instrument_type": "Cluster",
                "modules": {slot_idx: {"instrument_type": module_type, "digital_output_0": {}}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {"graph": [[f"cluster0.module{slot_idx}.digital_output_0", "q0:switch"]]},
    }

    # Setup objects needed for experiment
    quantum_device = QuantumDevice("quantum_device")
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test MarkerPulse compilation")
    schedule.add(
        MarkerPulse(
            duration=500e-9,
            port="q0:switch",
        ),
    )
    schedule.add(IdlePulse(4e-9))

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    prog = compiled_schedule["compiled_instructions"]

    # Assert channel map parameters are still defaults
    cluster = make_cluster_component(cluster_name)
    cluster.prepare(prog[cluster_name])

    all_modules = {module.name: module for module in cluster.instrument.modules}
    module = all_modules[module_name]

    all_sequencers = {sequencer.name: sequencer for sequencer in module.sequencers}
    seq0 = all_sequencers[f"{module_name}_sequencer0"]

    for param_name, param in seq0.parameters.items():
        if "connect" in param_name:
            assert param.get() == "off"


def test_amc_setting_is_set_on_instrument(
    mocker,
    mock_setup_basic_transmon_with_standard_params,
    schedule_with_measurement_q2,
    hardware_cfg_rf,
    make_cluster_component,
):
    hardware_config = copy(hardware_cfg_rf)
    hardware_config["hardware_options"]["mixer_corrections"] = {
        "q2:mw-q2.01": {
            "auto_lo_cal": "on_lo_freq_change",
            "auto_sideband_cal": "off",
        },
        "q2:res-q2.ro": {
            "auto_lo_cal": "on_lo_interm_freq_change",
            "auto_sideband_cal": "on_interm_freq_change",
        },
    }

    cluster_name = "cluster0"
    cluster = make_cluster_component(cluster_name)

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["quantum_device"].hardware_config(hardware_cfg_rf)
    mock_setup["q2"].clock_freqs.readout(7.3e9)
    mock_setup["q2"].clock_freqs.f01(6.03e9)
    compilation_config = mock_setup["quantum_device"].generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule_with_measurement_q2, config=compilation_config
    )
    prog = compiled_schedule["compiled_instructions"]

    qcm_rf = cluster._cluster_modules[f"{cluster_name}_module2"]
    qrm_rf = cluster._cluster_modules[f"{cluster_name}_module4"]

    mocker.patch.object(qcm_rf.instrument, "out0_lo_cal")
    mocker.patch.object(qcm_rf.instrument, "out1_lo_cal")
    mocker.patch.object(qcm_rf.instrument.sequencer0, "sideband_cal")
    mocker.patch.object(qrm_rf.instrument, "out0_in0_lo_cal")
    mocker.patch.object(qrm_rf.instrument.sequencer0, "sideband_cal")

    # Call it twice to check that calibration is only done once.
    cluster.prepare(prog[cluster_name])
    cluster.prepare(prog[cluster_name])

    qcm_rf.instrument.out0_lo_cal.assert_called_once()
    qcm_rf.instrument.out1_lo_cal.assert_not_called()  # Not used in schedule
    qcm_rf.instrument.sequencer0.sideband_cal.assert_not_called()  # Turned off
    qrm_rf.instrument.out0_in0_lo_cal.assert_called_once()
    qrm_rf.instrument.sequencer0.sideband_cal.assert_called_once()


def test_amc_setting_is_set_on_instrument_change_frequency(
    mocker,
    mock_setup_basic_transmon_with_standard_params,
    schedule_with_measurement_q2,
    hardware_cfg_rf,
    make_cluster_component,
):
    hardware_config = copy(hardware_cfg_rf)
    hardware_config["hardware_options"]["mixer_corrections"] = {
        "q2:mw-q2.01": {
            "auto_lo_cal": "on_lo_freq_change",
            "auto_sideband_cal": "off",
        },
        "q2:res-q2.ro": {
            "auto_lo_cal": "on_lo_interm_freq_change",
            "auto_sideband_cal": "on_interm_freq_change",
        },
    }

    cluster_name = "cluster0"
    cluster = make_cluster_component(cluster_name)

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["quantum_device"].hardware_config(hardware_cfg_rf)
    mock_setup["q2"].clock_freqs.readout(7.3e9)
    mock_setup["q2"].clock_freqs.f01(6.03e9)
    compilation_config = mock_setup["quantum_device"].generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule_with_measurement_q2, config=compilation_config
    )
    prog = compiled_schedule["compiled_instructions"]

    qcm_rf = cluster._cluster_modules[f"{cluster_name}_module2"]
    qrm_rf = cluster._cluster_modules[f"{cluster_name}_module4"]

    mocker.patch.object(qcm_rf.instrument, "out0_lo_cal")
    mocker.patch.object(qcm_rf.instrument, "out1_lo_cal")
    mocker.patch.object(qcm_rf.instrument.sequencer0, "sideband_cal")
    mocker.patch.object(qrm_rf.instrument, "out0_in0_lo_cal")
    mocker.patch.object(qrm_rf.instrument.sequencer0, "sideband_cal")

    # First round:
    # QRM has LO frequency 7.2e9, NCO frequency 1e8
    # QCM has LO frequency 5.98e9, NCO frequency 5e7
    cluster.prepare(prog[cluster_name])
    cluster.start()

    # Run a second time with different frequencies
    mock_setup["q2"].clock_freqs.readout(7.2e9)
    mock_setup["q2"].clock_freqs.f01(6.04e9)
    compilation_config = mock_setup["quantum_device"].generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule_with_measurement_q2, config=compilation_config
    )
    prog = compiled_schedule["compiled_instructions"]

    # Check calls and make fresh mocks for second round (only for methods expected to be
    # called twice).
    qcm_rf.instrument.out0_lo_cal.assert_called_once()
    qrm_rf.instrument.out0_in0_lo_cal.assert_called_once()
    qrm_rf.instrument.sequencer0.sideband_cal.assert_called_once()
    mocker.patch.object(qcm_rf.instrument, "out0_lo_cal")
    mocker.patch.object(qrm_rf.instrument, "out0_in0_lo_cal")
    mocker.patch.object(qrm_rf.instrument.sequencer0, "sideband_cal")

    # Second round:
    # QRM has LO frequency 7.2e9, NCO frequency 0.0
    # QCM has LO frequency 5.99e9, NCO frequency 5e7
    cluster.prepare(prog[cluster_name])
    cluster.start()

    qcm_rf.instrument.out0_lo_cal.assert_called_once()
    qcm_rf.instrument.out1_lo_cal.assert_not_called()
    qcm_rf.instrument.sequencer0.sideband_cal.assert_not_called()  # Turned off
    qrm_rf.instrument.out0_in0_lo_cal.assert_called_once()
    qrm_rf.instrument.sequencer0.sideband_cal.assert_called_once()


def test_missing_acq_index(
    mocker,
    mock_setup_basic_transmon_with_standard_params,
    make_cluster_component,
    hardware_cfg_cluster_test_component,
):
    cluster_name = "cluster0"
    qrm_name = f"{cluster_name}_module3"

    cluster = make_cluster_component(
        name=cluster_name,
        modules={"1": "QCM", "2": "QCM_RF", "3": "QRM", "4": "QRM_RF"},
    )
    qrm = cluster._cluster_modules[qrm_name]

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["quantum_device"].hardware_config(hardware_cfg_cluster_test_component)
    mock_setup["q0"].clock_freqs.readout(4.5e8)
    mock_setup["q2"].clock_freqs.readout(7.3e9)

    schedule = Schedule("Retrieve acq sched")

    schedule.add(Measure("q0"))
    schedule.add(Measure("q2"))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule,
        config=mock_setup["quantum_device"].generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    mocker.patch.object(
        qrm.instrument,
        "get_acquisitions",
        return_value={},
    )

    cluster.prepare(prog)
    cluster.start()

    with pytest.raises(
        KeyError,
        match=re.escape(
            "The acquisition data retrieved from the hardware does not contain data "
            r"for acquisition channel 0 (referred to by Qblox acquisition index 0).\n"
            r"hardware_retrieved_acquisitions={}"
        ),
    ):
        _ = qrm.retrieve_acquisition()


@pytest.mark.parametrize(
    ["protocol", "bin_mode"],
    [
        ("Trace", BinMode.APPEND),
        ("SSBIntegrationComplex", "foo"),
        ("ThresholdedAcquisition", "foo"),
        ("TriggerCount", "foo"),
    ],
)
def test_unsupported_bin_modes_qrm(
    make_cluster_component, mock_acquisition_data, protocol, bin_mode
):
    cluster = make_cluster_component("cluster0")
    acq_manager = qblox._QRMAcquisitionManager(
        parent=cluster._cluster_modules["cluster0_module1"],
        acquisition_metadata=dict(),
        scope_mode_sequencer_and_qblox_acq_index=(0, 0),
        acquisition_duration={0: 10},
        seq_name_to_idx_map={"seq0": 0},
    )
    acq_metadata = AcquisitionMetadata(protocol, bin_mode, np.ndarray, {0: [0]}, 1)
    acq_indices = [0] if protocol == "Trace" else list(range(10))
    get_data_fn_map = {
        "Trace": acq_manager._get_scope_data,
        "SSBIntegrationComplex": acq_manager._get_integration_data,
        "ThresholdedAcquisition": acq_manager._get_threshold_data,
        "TriggerCount": acq_manager._get_trigger_count_data,
    }
    with pytest.raises(
        RuntimeError,
        match=f"{protocol} acquisition protocol does not support bin mode {bin_mode}",
    ):
        _ = get_data_fn_map[protocol](
            acq_indices=acq_indices,
            hardware_retrieved_acquisitions=mock_acquisition_data,
            acquisition_metadata=acq_metadata,
            acq_duration=10,
            qblox_acq_index=0,
            acq_channel=0,
        )


@pytest.mark.parametrize(
    ["protocol", "bin_mode"],
    [
        ("TriggerCount", BinMode.AVERAGE),
        ("Timetag", "foo"),
        ("TimetagTrace", BinMode.AVERAGE),
        ("Trace", BinMode.APPEND),
    ],
)
def test_unsupported_bin_modes_qtm(
    make_cluster_component, mock_qtm_acquisition_data, protocol, bin_mode
):
    cluster = make_cluster_component("cluster0")
    acq_manager = qblox._QTMAcquisitionManager(
        parent=cluster._cluster_modules["cluster0_module5"],
        acquisition_metadata=dict(),
        acquisition_duration={0: 10},
        seq_name_to_idx_map={"seq0": 0},
    )
    acq_metadata = AcquisitionMetadata(protocol, bin_mode, np.ndarray, {0: [0]}, 1)
    get_data_fn_map = {
        "TriggerCount": acq_manager._get_trigger_count_data,
        "Timetag": acq_manager._get_timetag_data,
        "TimetagTrace": acq_manager._get_timetag_trace_data,
        "Trace": acq_manager._get_digital_trace_data,
    }
    with pytest.raises(
        RuntimeError,
        match=f"{protocol} acquisition protocol does not support bin mode {bin_mode}",
    ):
        _ = get_data_fn_map[protocol](
            acq_indices=list(range(10)),
            hardware_retrieved_acquisitions=mock_qtm_acquisition_data,
            acquisition_metadata=acq_metadata,
            acq_duration=10,
            qblox_acq_index=0,
            acq_channel=0,
        )


class MockModuleComponent(qblox._ModuleComponentBase):
    """
    A Mock Module Component for testing purposes. n.b. Mock() doesn't work well with
    abstract base classes, so we need to implement this.
    """

    def _configure_global_settings(self):  # type: ignore incompatible override abstract method
        pass

    @property
    def _hardware_properties(self):
        return MagicMock(number_of_sequencers=4)

    def retrieve_acquisition(self):
        pass


@pytest.fixture
def mock_module_component():
    mock_module = MagicMock()
    mock_module.name = "test_module"

    component = MockModuleComponent(mock_module)
    return component


# _ModuleComponentBase._set_parameter temporarily catches value errors related
# to realtime predistortion (RTP) filters. This was a request from Orange until
# we have official RTP support in Qblox instruments. This test can be removed
# when that is the case.
def test_set_parameter_value_error_is_passed(mock_module_component):
    mock_instrument = MagicMock()

    with patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.search_settable_param"
    ) as mock_search_settable_param:
        mock_search_settable_param.side_effect = ValueError("Test error")

        # Test case where the ValueError should be passed through
        with pytest.raises(ValueError, match="Test error"):
            mock_module_component._set_parameter(
                mock_instrument, "some_invalid_param", "some_value"
            )

        # Test case where ValueError should NOT be passed because it's handled internally
        mock_module_component._set_parameter(mock_instrument, "out0_bt_config", "bypassed")
        mock_module_component._set_parameter(mock_instrument, "out0_bt_time_constant", "some_value")

        assert mock_search_settable_param.called
