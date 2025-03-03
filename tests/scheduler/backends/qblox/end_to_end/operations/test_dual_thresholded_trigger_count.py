# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from quantify_scheduler import Schedule, SerialCompiler
from quantify_scheduler.operations import (
    ConditionalOperation,
    DualThresholdedTriggerCount,
    IdlePulse,
    MarkerPulse,
    X,
)
from quantify_scheduler.schemas.examples import utils
from tests.scheduler.instrument_coordinator.components.test_qblox import (
    make_cluster_component,
)

EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER = utils.load_json_example_scheme(
    "qblox_hardware_config_nv_center.json"
)


@dataclass
class ThresholdsAndLabels:
    threshold_low: int
    threshold_high: int
    label_low: str | None = None
    label_mid: str | None = None
    label_high: str | None = None
    label_invalid: str | None = None


@pytest.fixture
def dual_thresholded_cond_playback_sched():
    def make_schedule(qubit: str, thresholds_and_labels: ThresholdsAndLabels):
        schedule = Schedule("test")
        schedule.add(
            DualThresholdedTriggerCount(
                port=f"{qubit}:optical_readout",
                clock=f"{qubit}.ge0",
                duration=1e-6,
                threshold_low=thresholds_and_labels.threshold_low,
                threshold_high=thresholds_and_labels.threshold_high,
                label_low=thresholds_and_labels.label_low,
                label_mid=thresholds_and_labels.label_mid,
                label_high=thresholds_and_labels.label_high,
                label_invalid=thresholds_and_labels.label_invalid,
            )
        )
        conditional_added = False
        for label in (
            thresholds_and_labels.label_low,
            thresholds_and_labels.label_mid,
            thresholds_and_labels.label_high,
            thresholds_and_labels.label_invalid,
        ):
            if label is None:
                continue
            conditional_added = True
            schedule.add(
                ConditionalOperation(body=X(qubit), qubit_name=label),
                rel_time=364e-9,
            )
            schedule.add(IdlePulse(4e-9))
        if not conditional_added:
            raise ValueError("At least one label needs to be defined.")
        return schedule

    yield make_schedule


@pytest.fixture
def compiled_thresh_trig_count_cond_playback_schedule(
    mock_setup_basic_nv_qblox_hardware,
    qblox_hardware_config_nv_center,
    dual_thresholded_cond_playback_sched,
):
    def make_compiled_schedule(qubit_name: str, thresholds_and_labels: ThresholdsAndLabels):
        quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
        quantum_device.hardware_config(qblox_hardware_config_nv_center)
        config = quantum_device.generate_compilation_config()

        schedule = dual_thresholded_cond_playback_sched(qubit_name, thresholds_and_labels)

        compiler = SerialCompiler(name="compiler")
        compiled_sched = compiler.compile(
            schedule,
            config=config,
        )

        return compiled_sched

    yield make_compiled_schedule


def test_conditional_playback_trigger_count_qtm(
    compiled_thresh_trig_count_cond_playback_schedule,
    assert_equal_q1asm,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule(
        "qe1", ThresholdsAndLabels(5, 10, label_low="qe1_low")
    )

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"][
            "seq0"
        ].sequence["program"],
        """
set_mrk 2 # set markers to 2
 set_latch_en 1,4
 wait_sync 4
 upd_param 4
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 wait 4 # auto generated wait (4 ns)
 latch_rst 4 # reset trigger count
 wait 1356 # auto generated wait (1356 ns)
 set_cond 1,1,0,4 # start conditional playback
 set_awg_gain 16384,0 # setting gain for X qe1
 play 0,0,4 # play X qe1 (20 ns)
 wait 16 # auto generated wait (16 ns)
 set_cond 1,1,1,4 # else wait
 wait 16 # auto generated wait (16 ns)
 set_cond 0,0,0,0 # stop conditional playback
 loop R0,@start
 stop
""",
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].sequence["program"],
        """
 upd_thres 0,5,4
 upd_thres 1,10,4
 wait_sync 4
 upd_param 4
 move 0,R0 # Initialize acquisition bin_idx for ch0
 wait 4 # latency correction of 4 + 0 ns
 move 1,R1 # iterator for loop with label start
start:
 wait 4
 move 0,R10
 acquire_timetags 0,R0,1,R10,4 # Enable timetag acquisition of acq_channel:0, store in bin:R0
 wait 992 # auto generated wait (992 ns)
 acquire_timetags 0,R0,0,R10,4 # Disable timetag acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 wait 388 # auto generated wait (388 ns)
 loop R1,@start
 stop
""",
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].thresholded_acq_trigger_write_address_high
        == 0
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].thresholded_acq_trigger_write_address_mid
        == 0
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].thresholded_acq_trigger_write_address_low
        == 1
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].thresholded_acq_trigger_write_address_invalid
        == 0
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].thresholded_acq_trigger_write_en
        is True
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].thresholded_acq_trigger_write_invert
        is False
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq0"]
        .thresholded_acq_trigger_read_settings[1]
        .thresholded_acq_trigger_invert
        is False
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq0"]
        .thresholded_acq_trigger_read_settings[1]
        .thresholded_acq_trigger_count
        == 1
    )


def test_thresholded_trigger_count_acquisition_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qtm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(cluster_name)
    qtm = cluster._cluster_modules[qtm_name]

    counts = [10.0, 10.0, 9.0, 10.0, 5.0, 1.0, 7.0, 1.0, 4.0, 6.0]
    dummy_data = {
        "0": {
            "index": 0,
            "acquisition": {
                "bins": {
                    "count": counts,
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
                    "threshold": [2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
                    "avg_cnt": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                }
            },
        }
    }

    counts_passed = np.array(counts).astype(int)
    dataarray = xr.DataArray(
        [counts_passed],
        dims=["repetition", "acq_index_0"],
        coords={"repetition": [0], "acq_index_0": range(len(counts_passed))},
        attrs={"acq_protocol": "DualThresholdedTriggerCount"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq")
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(
        DualThresholdedTriggerCount(
            port="qe1:optical_readout",
            clock="qe1.ge0",
            duration=1e-6,
            threshold_low=3,
            threshold_high=7,
            label_low="low",
            label_mid="mid",
            label_high="high",
            label_invalid="invalid",
        )
    )

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


def test_conditional_playback_trigger_count_qtm_integration(
    make_cluster_component,
    compiled_thresh_trig_count_cond_playback_schedule,
    mocker,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule(
        "qe1", ThresholdsAndLabels(5, 10, "low")
    )

    cluster_name = "cluster0"
    ic_cluster = make_cluster_component(name=cluster_name, modules={"1": "QCM", "5": "QTM"})

    qtm = ic_cluster.instrument.module5
    mocker.patch.object(qtm["io_channel4"].parameters["thresholded_acq_trigger_en"], "set")
    mocker.patch.object(qtm["io_channel4"].parameters["thresholded_acq_trigger_address_low"], "set")
    mocker.patch.object(qtm["io_channel4"].parameters["thresholded_acq_trigger_address_mid"], "set")
    mocker.patch.object(
        qtm["io_channel4"].parameters["thresholded_acq_trigger_address_high"], "set"
    )
    mocker.patch.object(
        qtm["io_channel4"].parameters["thresholded_acq_trigger_address_invalid"], "set"
    )

    qcm = ic_cluster.instrument.module1
    mocker.patch.object(qcm["sequencer0"].parameters["trigger1_count_threshold"], "set")
    mocker.patch.object(qcm["sequencer0"].parameters["trigger1_threshold_invert"], "set")

    prog = compiled_sched["compiled_instructions"][cluster_name]

    ic_cluster.prepare(prog)
    ic_cluster.start()

    qtm["io_channel4"].parameters["thresholded_acq_trigger_en"].set.assert_called_with(True)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_low"].set.assert_called_with(1)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_mid"].set.assert_called_with(0)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_high"].set.assert_called_with(0)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_invalid"].set.assert_called_with(
        0
    )

    qcm["sequencer0"].parameters["trigger1_count_threshold"].set.assert_called_with(1)
    qcm["sequencer0"].parameters["trigger1_threshold_invert"].set.assert_called_with(False)


def test_dual_thresholded_trigger_count_all_labels(
    make_cluster_component,
    compiled_thresh_trig_count_cond_playback_schedule,
    mocker,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule(
        "qe1", ThresholdsAndLabels(5, 10, "low", "mid", "high", "invalid")
    )

    cluster_name = "cluster0"
    ic_cluster = make_cluster_component(name=cluster_name, modules={"1": "QCM", "5": "QTM"})

    qtm = ic_cluster.instrument.module5
    mocker.patch.object(qtm["io_channel4"].parameters["thresholded_acq_trigger_en"], "set")
    mocker.patch.object(qtm["io_channel4"].parameters["thresholded_acq_trigger_address_low"], "set")
    mocker.patch.object(qtm["io_channel4"].parameters["thresholded_acq_trigger_address_mid"], "set")
    mocker.patch.object(
        qtm["io_channel4"].parameters["thresholded_acq_trigger_address_high"], "set"
    )
    mocker.patch.object(
        qtm["io_channel4"].parameters["thresholded_acq_trigger_address_invalid"], "set"
    )

    qcm = ic_cluster.instrument.module1
    for i in range(1, 5):
        mocker.patch.object(qcm["sequencer0"].parameters[f"trigger{i}_count_threshold"], "set")
        mocker.patch.object(qcm["sequencer0"].parameters[f"trigger{i}_threshold_invert"], "set")

    prog = compiled_sched["compiled_instructions"][cluster_name]

    ic_cluster.prepare(prog)
    ic_cluster.start()

    qtm["io_channel4"].parameters["thresholded_acq_trigger_en"].set.assert_called_with(True)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_low"].set.assert_called_with(1)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_mid"].set.assert_called_with(2)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_high"].set.assert_called_with(3)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_invalid"].set.assert_called_with(
        4
    )

    for i in range(1, 5):
        qcm["sequencer0"].parameters[f"trigger{i}_count_threshold"].set.assert_called_with(1)
        qcm["sequencer0"].parameters[f"trigger{i}_threshold_invert"].set.assert_called_with(False)


def test_same_label_different_settings_raises(
    mock_setup_basic_nv_qblox_hardware, qblox_hardware_config_nv_center
):
    qubit = "qe1"
    port = f"{qubit}:optical_readout"
    clock = f"{qubit}.ge0"

    schedule = Schedule("test")
    schedule.add(
        DualThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=0,
            duration=1e-6,
            threshold_low=5,
            threshold_high=10,
            label_low=None,
            label_mid="mid",
            label_high=None,
            label_invalid=None,
        )
    )
    schedule.add(
        ConditionalOperation(body=X(qubit), qubit_name="mid"),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))
    schedule.add(
        DualThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=1,
            duration=1e-6,
            threshold_low=5,
            threshold_high=10,
            label_low=None,
            label_mid="mid2",
            label_high=None,
            label_invalid=None,
        )
    )
    schedule.add(
        ConditionalOperation(body=X(qubit), qubit_name="mid2"),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_nv_center)
    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Found {1, 2} as possible values for 'feedback_trigger_address mid' on the sequencer "
            "for port-clock qe1:optical_readout-qe1.ge0. 'feedback_trigger_address mid' must be "
            "unique per sequencer."
        ),
    ):
        _ = compiler.compile(
            schedule,
            config=config,
        )
