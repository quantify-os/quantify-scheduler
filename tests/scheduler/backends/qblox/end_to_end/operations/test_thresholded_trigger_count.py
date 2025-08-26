# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import numpy as np
import pytest
import xarray as xr

from quantify_scheduler import Schedule, SerialCompiler
from quantify_scheduler.enums import BinMode, TriggerCondition
from quantify_scheduler.operations import (
    ConditionalOperation,
    IdlePulse,
    MarkerPulse,
    ThresholdedTriggerCount,
    X,
)
from quantify_scheduler.schemas.examples import utils
from tests.scheduler.instrument_coordinator.components.test_qblox import (
    make_cluster_component,
)

EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER = utils.load_json_example_scheme(
    "qblox_hardware_config_nv_center.json"
)


@pytest.fixture
def thresholded_trigger_count_conditional_playback_schedule():
    def make_schedule(qubit: str, trigger_condition: TriggerCondition):
        schedule = Schedule("test")
        schedule.add(
            ThresholdedTriggerCount(
                port=f"{qubit}:optical_readout",
                clock=f"{qubit}.ge0",
                duration=1e-6,
                threshold=5,
                feedback_trigger_label=qubit,
                feedback_trigger_condition=trigger_condition,
            )
        )
        schedule.add(
            ConditionalOperation(body=X(qubit), qubit_name=qubit),
            rel_time=364e-9,
        )
        schedule.add(IdlePulse(4e-9))
        return schedule

    yield make_schedule


@pytest.fixture
def compiled_thresh_trig_count_cond_playback_schedule(
    mock_setup_basic_nv_qblox_hardware,
    qblox_hardware_config_nv_center,
    thresholded_trigger_count_conditional_playback_schedule,
):
    def make_compiled_schedule(qubit_name: str, trigger_condition: TriggerCondition):
        quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
        quantum_device.hardware_config(qblox_hardware_config_nv_center)
        config = quantum_device.generate_compilation_config()

        schedule = thresholded_trigger_count_conditional_playback_schedule(
            qubit_name, trigger_condition
        )

        compiler = SerialCompiler(name="compiler")
        compiled_sched = compiler.compile(
            schedule,
            config=config,
        )

        return compiled_sched

    yield make_compiled_schedule


@pytest.mark.parametrize(
    "trigger_condition",
    [TriggerCondition.LESS_THAN, TriggerCondition.GREATER_THAN_EQUAL_TO],
)
def test_conditional_playback_trigger_count_qrm(
    compiled_thresh_trig_count_cond_playback_schedule,
    assert_equal_q1asm,
    trigger_condition,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule("qe0", trigger_condition)

    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"][
            "seq0"
        ].sequence["program"],
        """
 set_mrk 1 # set markers to 1
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
 set_awg_gain 16384,0 # setting gain for X qe0
 play 0,0,4 # play X qe0 (20 ns)
 wait 16 # auto generated wait (16 ns)
 set_cond 1,1,1,4 # else wait
 wait 16 # auto generated wait (16 ns)
 set_cond 0,0,0,0 # stop conditional playback
 loop R0,@start
 stop
""",
    )
    assert_equal_q1asm(
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module4"]["sequencers"][
            "seq0"
        ].sequence["program"],
        """
 set_mrk 0 # set markers to 0
 wait_sync 4
 upd_param 4
 move 0,R0 # Initialize acquisition bin_idx for ch0
 wait 4 # latency correction of 4 + 0 ns
 move 1,R1 # iterator for loop with label start
start:
 reset_ph
 upd_param 4
 acquire_ttl 0,R0,1,4 # Enable TTL acquisition of acq_channel:0, store in bin:R0
 wait 992 # auto generated wait (992 ns)
 acquire_ttl 0,R0,0,4 # Disable TTL acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 wait 388 # auto generated wait (388 ns)
 loop R1,@start
 stop
""",
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module4"]["sequencers"][
            "seq0"
        ].thresholded_acq_trigger_write_address
        == 1
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module4"]["sequencers"][
            "seq0"
        ].thresholded_acq_trigger_write_en
        is True
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module4"]["sequencers"][
            "seq0"
        ].thresholded_acq_trigger_write_invert
        is False
    )

    assert compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"][
        "seq0"
    ].thresholded_acq_trigger_read_settings[1].thresholded_acq_trigger_invert == (
        trigger_condition == TriggerCondition.LESS_THAN
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq0"]
        .thresholded_acq_trigger_read_settings[1]
        .thresholded_acq_trigger_count
        == 5
    )


@pytest.mark.parametrize(
    "trigger_condition",
    [TriggerCondition.LESS_THAN, TriggerCondition.GREATER_THAN_EQUAL_TO],
)
def test_conditional_playback_trigger_count_qtm(
    compiled_thresh_trig_count_cond_playback_schedule,
    assert_equal_q1asm,
    trigger_condition,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule("qe1", trigger_condition)

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
 upd_thres 1,5,4
 wait_sync 4
 upd_param 4
 move 0,R0 # Initialize acquisition bin_idx for ch0
 wait 4 # latency correction of 4 + 0 ns
 move 1,R1 # iterator for loop with label start
start:
 wait 4
 move 0,R2
 acquire_timetags 0,R0,1,R2,4 # Enable timetag acquisition of acq_channel:0, store in bin:R0
 wait 992 # auto generated wait (992 ns)
 acquire_timetags 0,R0,0,R2,4 # Disable timetag acquisition of acq_channel:0, store in bin:R0
 add R0,1,R0 # Increment bin_idx for ch0 by 1
 wait 388 # auto generated wait (388 ns)
 loop R1,@start
 stop
""",
    )
    assert compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
        "seq4"
    ].thresholded_acq_trigger_write_address_high == (
        1 if trigger_condition == TriggerCondition.GREATER_THAN_EQUAL_TO else 0
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
            "seq4"
        ].thresholded_acq_trigger_write_address_mid
        == 0
    )
    assert compiled_sched.compiled_instructions["cluster0"]["cluster0_module5"]["sequencers"][
        "seq4"
    ].thresholded_acq_trigger_write_address_low == (
        1 if trigger_condition == TriggerCondition.LESS_THAN else 0
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


def test_thresholded_trigger_count_serializes_after_compilation(
    compiled_thresh_trig_count_cond_playback_schedule,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule(
        "qe1", TriggerCondition.LESS_THAN
    )
    _ = compiled_sched.to_json()


def test_thresholded_trigger_count_acquisition_qtm(
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

    threshold_passed = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1])
    dataarray = xr.DataArray(
        threshold_passed.reshape(-1, 1),
        dims=["repetition", "acq_index_0"],
        coords={
            "repetition": range(len(threshold_passed)),
            "acq_index_0": [0],
        },
        attrs={"acq_protocol": "ThresholdedTriggerCount"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq", repetitions=10)
    sched.add(MarkerPulse(duration=40e-9, port="qe1:switch"))
    sched.add(
        ThresholdedTriggerCount(
            duration=1e-6,
            threshold=28,
            feedback_trigger_label="qe1",
            port="qe1:optical_readout",
            clock="qe1.ge0",
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


def test_thresholded_trigger_count_acquisition_qrm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"
    qrm_name = f"{cluster_name}_module4"

    cluster = make_cluster_component(name=cluster_name, modules={"1": "QCM", "4": "QRM"})
    qrm = cluster._cluster_modules[qrm_name]

    dummy_data = {
        "0": {
            "index": 0,
            "acquisition": {
                "scope": {
                    "path0": {
                        "data": [0.0] * 2**14,
                        "out-of-range": False,
                        "avg_count": 1,
                    },
                    "path1": {
                        "data": [0.0] * 2**14,
                        "out-of-range": False,
                        "avg_count": 1,
                    },
                },
                "bins": {
                    "integration": {
                        "path0": [0.0] * 10,
                        "path1": [0.0] * 10,
                    },
                    "threshold": [0.12] * 10,
                    "avg_cnt": [
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
                },
            },
        }
    }

    threshold_passed = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1])
    dataarray = xr.DataArray(
        threshold_passed.reshape(-1, 1),
        dims=["repetition", "acq_index_0"],
        coords={
            "repetition": range(len(threshold_passed)),
            "acq_index_0": [0],
        },
        attrs={"acq_protocol": "ThresholdedTriggerCount"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq", repetitions=10)
    sched.add(
        ThresholdedTriggerCount(
            duration=1e-6,
            threshold=28,
            feedback_trigger_label="qe0",
            port="qe0:optical_readout",
            clock="qe0.ge0",
        )
    )

    mocker.patch.object(
        cluster.instrument.module4,
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

    xr.testing.assert_identical(qrm.retrieve_acquisition(), expected_dataset)


@pytest.mark.parametrize(
    "trigger_condition, threshold_invert",
    [(TriggerCondition.GREATER_THAN_EQUAL_TO, False), (TriggerCondition.LESS_THAN, True)],
)
def test_conditional_playback_trigger_count_qrm_integration(
    make_cluster_component,
    compiled_thresh_trig_count_cond_playback_schedule,
    mocker,
    trigger_condition,
    threshold_invert,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule("qe0", trigger_condition)

    cluster_name = "cluster0"
    ic_cluster = make_cluster_component(name=cluster_name, modules={"1": "QCM", "4": "QRM"})

    qrm = ic_cluster.instrument.module4
    mocker.patch.object(qrm["sequencer0"].parameters["thresholded_acq_trigger_en"], "set")
    mocker.patch.object(qrm["sequencer0"].parameters["thresholded_acq_trigger_address"], "set")
    mocker.patch.object(qrm["sequencer0"].parameters["thresholded_acq_trigger_invert"], "set")

    qcm = ic_cluster.instrument.module1
    mocker.patch.object(qcm["sequencer0"].parameters["trigger1_count_threshold"], "set")
    mocker.patch.object(qcm["sequencer0"].parameters["trigger1_threshold_invert"], "set")

    prog = compiled_sched["compiled_instructions"][cluster_name]

    ic_cluster.prepare(prog)
    ic_cluster.start()

    qrm["sequencer0"].parameters["thresholded_acq_trigger_en"].set.assert_called_with(True)
    qrm["sequencer0"].parameters["thresholded_acq_trigger_address"].set.assert_called_with(1)
    qrm["sequencer0"].parameters["thresholded_acq_trigger_invert"].set.assert_called_with(False)

    qcm["sequencer0"].parameters["trigger1_count_threshold"].set.assert_called_with(5)
    qcm["sequencer0"].parameters["trigger1_threshold_invert"].set.assert_called_with(
        threshold_invert
    )


@pytest.mark.parametrize(
    "trigger_condition, low_address, high_address",
    [(TriggerCondition.GREATER_THAN_EQUAL_TO, 0, 1), (TriggerCondition.LESS_THAN, 1, 0)],
)
def test_conditional_playback_trigger_count_qtm_integration(
    make_cluster_component,
    compiled_thresh_trig_count_cond_playback_schedule,
    mocker,
    trigger_condition,
    low_address,
    high_address,
):
    compiled_sched = compiled_thresh_trig_count_cond_playback_schedule("qe1", trigger_condition)

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
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_low"].set.assert_called_with(
        low_address
    )
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_mid"].set.assert_called_with(0)
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_high"].set.assert_called_with(
        high_address
    )
    qtm["io_channel4"].parameters["thresholded_acq_trigger_address_invalid"].set.assert_called_with(
        0
    )

    qcm["sequencer0"].parameters["trigger1_count_threshold"].set.assert_called_with(1)
    qcm["sequencer0"].parameters["trigger1_threshold_invert"].set.assert_called_with(False)


@pytest.mark.parametrize("qubit", ["qe0", "qe1"])
def test_same_label_different_settings_raises(
    mock_setup_basic_nv_qblox_hardware, qblox_hardware_config_nv_center, qubit
):
    port = f"{qubit}:optical_readout"
    clock = f"{qubit}.ge0"

    schedule = Schedule("test")
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=0,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label=qubit,
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )
    schedule.add(
        ConditionalOperation(body=X(qubit), qubit_name=qubit),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=1,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label=qubit,
            feedback_trigger_condition=TriggerCondition.GREATER_THAN_EQUAL_TO,
        )
    )
    schedule.add(
        ConditionalOperation(body=X(qubit), qubit_name=qubit),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_nv_center)
    config = quantum_device.generate_compilation_config()

    if qubit == "qe0":
        match_str = "This may happen because multiple ThresholdedTriggerCount acquisitions with "
        "conflicting threshold settings are scheduled, or ThresholdedTriggerCount acquisitions "
        "with the same feedback trigger label are scheduled on different modules."
    else:
        match_str = r"Found .*(less_than.*greater_than_equal_to|greater_than_equal_to.*less_than).*"
        f" as possible values for 'condition' on the sequencer for port-clock {port}-{clock}. "
        "'condition' must be unique per sequencer."
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match=match_str,
    ):
        _ = compiler.compile(
            schedule,
            config=config,
        )


def test_same_label_different_modules_raises(
    mock_setup_basic_nv_qblox_hardware, qblox_hardware_config_nv_center
):
    # The qe0 readout ports are connected to a QRM, while the qe1 readout ports are
    # connected to a QTM. The conditional operation plays on the same sequencer, but the
    # settings for the playing sequencer depend on the module type that does the
    # acquisition.

    schedule = Schedule("test")
    schedule.add(
        ThresholdedTriggerCount(
            port="qe0:optical_readout",
            clock="qe0.ge0",
            acq_channel=0,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label="label",
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )
    schedule.add(
        ConditionalOperation(body=X("qe0"), qubit_name="label"),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))
    schedule.add(
        ThresholdedTriggerCount(
            port="qe1:optical_readout",
            clock="qe1.ge0",
            acq_channel=1,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label="label",
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )
    schedule.add(
        ConditionalOperation(body=X("qe0"), qubit_name="label"),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_nv_center)
    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="This may happen "
        "because multiple ThresholdedTriggerCount acquisitions with conflicting threshold "
        "settings are scheduled, or ThresholdedTriggerCount acquisitions with the same "
        "feedback trigger label are scheduled on different modules.",
    ):
        _ = compiler.compile(
            schedule,
            config=config,
        )


@pytest.mark.parametrize("qubit", ["qe0", "qe1"])
def test_same_sequencer_different_labels_raises(
    mock_setup_basic_nv_qblox_hardware, qblox_hardware_config_nv_center, qubit
):
    port = f"{qubit}:optical_readout"
    clock = f"{qubit}.ge0"

    schedule = Schedule("test")
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=0,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label=f"{qubit}a",
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )
    schedule.add(
        ConditionalOperation(body=X("qe0"), qubit_name=f"{qubit}a"),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=1,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label=f"{qubit}b",
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )

    schedule.add(
        ConditionalOperation(body=X("qe1"), qubit_name=f"{qubit}b"),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_nv_center)
    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="Found {1, 2} as possible values for 'feedback_trigger_address' on the sequencer for "
        f"port-clock {port}-{clock}. 'feedback_trigger_address' must be unique per sequencer.",
    ):
        _ = compiler.compile(
            schedule,
            config=config,
        )


def test_same_cond_sequencer_different_condition_qtm_success(
    mock_setup_basic_nv_qblox_hardware, qblox_hardware_config_nv_center
):
    qblox_hardware_config_nv_center["hardware_options"]["digitization_thresholds"][
        "qe2:optical_readout-digital"
    ] = {"analog_threshold": 0.5}
    qblox_hardware_config_nv_center["connectivity"]["graph"].append(
        ["cluster0.module5.digital_input_6", "qe2:optical_readout"]
    )

    qubit = "qe1"
    port = f"{qubit}:optical_readout"
    clock = f"{qubit}.ge0"

    schedule = Schedule("test")
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=0,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label=qubit,
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )
    schedule.add(
        ConditionalOperation(body=X("qe0"), qubit_name=qubit),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    qubit = "qe2"
    port = f"{qubit}:optical_readout"
    clock = "digital"
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=1,
            duration=1e-6,
            threshold=10,
            feedback_trigger_label=qubit,
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )

    schedule.add(
        ConditionalOperation(body=X("qe0"), qubit_name=qubit),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_nv_center)
    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    # This must not error
    _ = compiler.compile(
        schedule,
        config=config,
    )


def test_same_cond_sequencer_different_condition_qrm_success(
    mock_setup_basic_nv_qblox_hardware, qblox_hardware_config_nv_center
):
    qblox_hardware_config_nv_center["hardware_options"]["sequencer_options"][
        "qe2:optical_readout-cl0.baseband"
    ] = {"ttl_acq_threshold": 0.5}
    qblox_hardware_config_nv_center["connectivity"]["graph"].append(
        ["cluster0.module4.real_input_1", "qe2:optical_readout"]
    )

    qubit = "qe0"
    port = f"{qubit}:optical_readout"
    clock = f"{qubit}.ge0"

    schedule = Schedule("test")
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=0,
            duration=1e-6,
            threshold=5,
            feedback_trigger_label=qubit,
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )
    schedule.add(
        ConditionalOperation(body=X("qe0"), qubit_name=qubit),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    qubit = "qe2"
    port = f"{qubit}:optical_readout"
    clock = "cl0.baseband"
    schedule.add(
        ThresholdedTriggerCount(
            port=port,
            clock=clock,
            acq_channel=1,
            duration=1e-6,
            threshold=10,
            feedback_trigger_label=qubit,
            feedback_trigger_condition=TriggerCondition.LESS_THAN,
        )
    )

    schedule.add(
        ConditionalOperation(body=X("qe0"), qubit_name=qubit),
        rel_time=364e-9,
    )
    schedule.add(IdlePulse(4e-9))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_nv_center)
    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    # This must not error
    _ = compiler.compile(
        schedule,
        config=config,
    )
