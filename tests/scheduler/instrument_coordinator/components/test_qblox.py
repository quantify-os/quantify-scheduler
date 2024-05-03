# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox instrument coordinator components."""
import os
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional
from unittest.mock import MagicMock

import xarray as xr
import numpy as np
import pytest
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

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.enums import BinMode
from quantify_scheduler.instrument_coordinator.components import qblox
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.gate_library import Reset, Measure, X90
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    SquarePulse,
)
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import AcquisitionMetadata, Schedule
from quantify_scheduler.helpers.qblox_dummy_instrument import (
    start_dummy_cluster_armed_sequencers,
)


@pytest.fixture
def make_cluster_component(mocker):
    cluster_component: qblox.ClusterComponent = None

    default_modules = {
        "1": "QCM",
        "2": "QCM_RF",
        "3": "QRM",
        "4": "QRM_RF",
        "7": "QCM",
        "10": "QCM",  # for flux pulsing q0_q3
        "12": "QCM",  # for flux pulsing q4
    }

    def _make_cluster_component(
        name: str = "cluster0",
        modules: dict = default_modules,
        sequencer_status: SequencerStatuses = SequencerStatuses.OKAY,
        sequencer_state: SequencerStates = SequencerStates.ARMED,
        info_flags: Optional[List[SequencerStatusFlags]] = None,
        warn_flags: Optional[List[SequencerStatusFlags]] = None,
        err_flags: Optional[List[SequencerStatusFlags]] = None,
        sequencer_logs: Optional[List[str]] = None,
    ) -> qblox.ClusterComponent:
        qblox_types = {
            "QCM": ClusterType.CLUSTER_QCM,
            "QCM_RF": ClusterType.CLUSTER_QCM_RF,
            "QRM": ClusterType.CLUSTER_QRM,
            "QRM_RF": ClusterType.CLUSTER_QRM_RF,
        }
        cluster = Cluster(
            name=name,
            dummy_cfg={
                slot_idx: qblox_types[module_type]
                for slot_idx, module_type in modules.items()
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
            mocker.patch.object(
                instrument, "arm_sequencer", wraps=instrument.arm_sequencer
            )
            mocker.patch.object(
                instrument, "start_sequencer", wraps=instrument.start_sequencer
            )
            mocker.patch.object(
                instrument, "stop_sequencer", wraps=instrument.stop_sequencer
            )
            if not instrument.is_rf_type:
                mocker.patch.object(
                    instrument, "out0_offset", wraps=instrument.out0_offset
                )
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
            mocker.patch.object(
                instrument,
                "store_scope_acquisition",
                wraps=instrument.store_scope_acquisition,
            )

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
    cluster = make_cluster_component(
        name=cluster_name, modules={"1": "QCM", "5": "QRM"}
    )
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

    schedule = Schedule(f"Schedule")
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
    compiled_schedule = SerialCompiler(name="compiler").compile(
        schedule=schedule, config=config
    )
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
    cluster = make_cluster_component(
        name="cluster0", sequencer_status=SequencerStates.IDLE
    )

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
    cluster = make_cluster_component(
        name=cluster_name, modules={"1": "QCM", "5": "QRM"}
    )
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
            mocker.patch.object(
                dev.instrument[f"sequencer{seq}"].parameters["sync_en"], "set"
            )

    hardware_cfg = deepcopy(hardware_cfg_cluster)

    # Act
    hardware_cfg["hardware_options"]["sequencer_options"] = {
        "q0:mw-q0.01": {"init_offset_awg_path_I": 0.25, "init_offset_awg_path_Q": 0.33},
        "q0:fl-cl0.baseband": {"init_gain_awg_path_I": 0.5},
        "q1:fl-cl0.baseband": {"init_gain_awg_path_Q": -0.5},
    }

    schedule = Schedule(f"Schedule")
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
    compiled_schedule = SerialCompiler(name="compiler").compile(
        schedule=schedule, config=config
    )
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
        qcm.instrument["sequencer0"].parameters[
            f"gain_awg_path{path}"
        ].set.assert_called_once_with(qcm_gain[path])

    qcm.instrument["sequencer0"].parameters[f"sync_en"].set.assert_called_with(True)
    qrm.instrument["sequencer0"].parameters[f"sync_en"].set.assert_called_with(True)

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


def test_invalid_init_qcodes_settings(
    mocker,
    schedule_with_measurement,
    hardware_cfg_cluster_legacy,
    make_cluster_component,
    mock_setup_basic_transmon_with_standard_params,
):
    # Arrange
    cluster_name = "cluster0"
    qcm_name = f"{cluster_name}_module1"
    cluster = make_cluster_component(cluster_name)
    qcm = cluster._cluster_modules[qcm_name]

    for seq in range(qcm._hardware_properties.number_of_sequencers):
        mocker.patch.object(
            qcm.instrument[f"sequencer{seq}"].parameters["offset_awg_path0"],
            "set",
        )
        mocker.patch.object(
            qcm.instrument[f"sequencer{seq}"].parameters["offset_awg_path1"],
            "set",
        )

    hardware_cfg = deepcopy(hardware_cfg_cluster_legacy)

    # Act
    hardware_cfg[cluster_name]["cluster0_module1"]["complex_output_0"][
        "portclock_configs"
    ][0]["init_offset_awg_path_I"] = 1.25

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    config = quantum_device.generate_compilation_config()
    with pytest.raises(ValueError):
        _ = SerialCompiler(name="compiler").compile(
            schedule=schedule_with_measurement, config=config
        )


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

    schedule = Schedule(f"Schedule with measurement")
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
            hardware_cfg_cluster["hardware_options"]["mixer_corrections"][
                hw_config_param[0]
            ][hw_config_param[1]]
        )

    for qcodes_param, hw_config_param in [
        ("out0_offset", ["q0:res-q0.ro", "dc_offset_i"]),
        ("out1_offset", ["q0:res-q0.ro", "dc_offset_q"]),
    ]:
        qrm0.instrument.parameters[qcodes_param].set.assert_any_call(
            hardware_cfg_cluster["hardware_options"]["mixer_corrections"][
                hw_config_param[0]
            ][hw_config_param[1]]
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
    hardware_compilation_config_qblox_example,
    make_cluster_component,
    force_set_parameters,
):
    # Arrange
    cluster_name = "cluster0"
    ic_cluster = make_cluster_component(cluster_name)

    qcm_rf = ic_cluster.instrument.module2
    mocker.patch.object(qcm_rf.parameters["out0_att"], "set")
    mocker.patch.object(qcm_rf.parameters["out1_att"], "set")
    mocker.patch.object(qcm_rf[f"sequencer0"].parameters["sync_en"], "set")

    qrm_rf = ic_cluster.instrument.module4
    mocker.patch.object(qrm_rf.parameters["out0_att"], "set")
    mocker.patch.object(qrm_rf.parameters["in0_att"], "set")
    mocker.patch.object(qrm_rf[f"sequencer0"].parameters["sync_en"], "set")

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
    quantum_device.hardware_config(hardware_compilation_config_qblox_example)

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
            hardware_compilation_config_qblox_example["hardware_options"][
                hw_options_param[0]
            ][hw_options_param[1]]
        )
    qcm_rf["sequencer0"].parameters[f"sync_en"].set.assert_called_with(True)

    for qcodes_param, hw_options_param in [
        ("out0_att", ["output_att", "q0:res-q0.ro"]),
        ("in0_att", ["input_att", "q0:res-q0.ro"]),
    ]:
        qrm_rf.parameters[qcodes_param].set.assert_any_call(
            hardware_compilation_config_qblox_example["hardware_options"][
                hw_options_param[0]
            ][hw_options_param[1]]
        )
    qrm_rf["sequencer0"].parameters[f"sync_en"].set.assert_called_with(True)


def test_prepare_exception(make_cluster_component):
    # Arrange
    cluster = make_cluster_component(
        name="cluster0", sequencer_status=SequencerStates.IDLE
    )

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

    schedule = Schedule(f"Retrieve acq sched")

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
    module1 = cluster._cluster_modules["cluster0_module1"]
    module1.instrument.get_ip_config = MagicMock(return_value=example_ip)

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
    sched.add(Reset("q1"))

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )

    # Create compiled schedule for module
    compiled_sched.compiled_instructions["cluster0_module1"] = deepcopy(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]
    )
    module1_log = module1.get_hardware_log(compiled_sched)
    assert module1_log["app_log"] == f"Mock hardware log for app"


def test_get_hardware_log_cluster_component(
    example_ip,
    hardware_cfg_qcm_rf,
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

    hardware_cfg = hardware_cfg_qcm_rf
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    sched = Schedule("sched")
    sched.add(Reset("q1"))

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )

    cluster0_log = cluster0.get_hardware_log(compiled_sched)
    cluster1_log = cluster1.get_hardware_log(compiled_sched)

    source = "app"
    assert (
        cluster0_log["cluster0_cmm"][f"{source}_log"]
        == f"Mock hardware log for {source}"
    )
    assert (
        cluster0_log["cluster0_module1"][f"{source}_log"]
        == f"Mock hardware log for {source}"
    )
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
        qblox._get_configuration_manager("bad_ip")
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
            "graph": [
                [f"cluster0.module{module_idx[module_type]}.{channel_name}", "q5:mw"]
            ]
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
def test_channel_map_off_with_marker_pulse(
    make_cluster_component, slot_idx, module_type
):
    cluster_name = "cluster0"
    module_name = f"{cluster_name}_module{slot_idx}"

    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            cluster_name: {
                "instrument_type": "Cluster",
                "modules": {
                    slot_idx: {"instrument_type": module_type, "digital_output_0": {}}
                },
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [[f"cluster0.module{slot_idx}.digital_output_0", "q0:switch"]]
        },
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
