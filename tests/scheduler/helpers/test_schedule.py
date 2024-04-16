# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch


from __future__ import annotations

import numpy as np
from pytest import approx

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_schedule,
    get_acq_info_by_uuid,
    get_acq_uuid,
    get_operation_start,
    get_operation_end,
    get_port_timeline,
    get_pulse_info_by_uuid,
    get_pulse_uuid,
    get_total_duration,
    _extract_port_clocks_used,
)
from quantify_scheduler.helpers.collections import make_hash
from quantify_scheduler.operations.gate_library import X90, Measure, Reset
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.schedules import spectroscopy_schedules


def test_make_hash() -> None:
    my_test_dict = {"a": 5, "nested_dict": {"a": 2, "c": 4, "B": "str"}, "b": 24}

    same_test_dict_diff_order = {
        "a": 5,
        "b": 24,
        "nested_dict": {"a": 2, "c": 4, "B": "str"},
    }

    diff_test_dict = {"nested_dict": {"a": 2, "c": 4, "B": "str"}, "b": 24}

    test_hash = make_hash(my_test_dict)
    same_test_hash = make_hash(same_test_dict_diff_order)

    assert test_hash == same_test_hash

    diff_hash = make_hash(diff_test_dict)

    assert test_hash != diff_hash

    # modify dict in place, the object id won't change
    my_test_dict["efg"] = 15
    new_hash = make_hash(my_test_dict)
    assert test_hash != new_hash


def test_get_info_by_uuid_empty(empty_schedule: Schedule):
    # Act
    pulseid_pulseinfo_dict = get_pulse_info_by_uuid(empty_schedule)

    # Assert
    assert len(pulseid_pulseinfo_dict) == 0


def test_get_info_by_uuid(
    schedule_with_pulse_info: Schedule,
):
    # Arrange
    operation_id = list(schedule_with_pulse_info.schedulables.values())[0][
        "operation_id"
    ]
    pulse_info_0 = schedule_with_pulse_info.operations[operation_id]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)

    # Act
    pulseid_pulseinfo_dict = get_pulse_info_by_uuid(schedule_with_pulse_info)

    # Assert
    assert len(pulseid_pulseinfo_dict) == 1
    assert pulse_id in pulseid_pulseinfo_dict
    assert pulseid_pulseinfo_dict[pulse_id] == pulse_info_0


def test_get_info_by_uuid_are_unique(device_compile_config_basic_transmon):
    # Arrange
    schedule = Schedule("my-schedule")
    schedule.add(X90("q0"))
    schedule.add(X90("q0"))

    compiler = SerialCompiler(name="compiler")
    schedule_with_pulse_info = compiler.compile(
        schedule=schedule, config=device_compile_config_basic_transmon
    )

    operation_id = list(schedule.schedulables.values())[0]["operation_id"]
    pulse_info_0 = schedule_with_pulse_info.operations[operation_id]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)

    # Act
    pulseid_pulseinfo_dict = get_pulse_info_by_uuid(schedule_with_pulse_info)

    # Assert
    assert len(pulseid_pulseinfo_dict) == 1
    assert pulse_id in pulseid_pulseinfo_dict
    assert pulseid_pulseinfo_dict[pulse_id] == pulse_info_0


def test_get_acq_info_by_uuid(
    schedule_with_measurement: Schedule,
    device_compile_config_basic_transmon,
):
    # Arrange
    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule_with_measurement, config=device_compile_config_basic_transmon
    )

    operation_id = list(schedule.schedulables.values())[-1]["operation_id"]
    operation = schedule.operations[operation_id]
    acq_info_0 = operation["acquisition_info"][0]
    acq_pulse_infos = acq_info_0["waveforms"]

    acq_id = get_acq_uuid(acq_info_0)
    pulse_id0 = get_pulse_uuid(acq_pulse_infos[0])
    pulse_id1 = get_pulse_uuid(acq_pulse_infos[1])

    # Act
    acqid_acqinfo_dict = get_acq_info_by_uuid(schedule)

    # Assert
    assert acq_id in acqid_acqinfo_dict
    assert pulse_id0 not in acqid_acqinfo_dict
    assert pulse_id1 not in acqid_acqinfo_dict

    assert acqid_acqinfo_dict[acq_id] == acq_info_0


def test_get_port_timeline(
    schedule_with_pulse_info: Schedule,
):
    # Arrange
    operation_id = list(schedule_with_pulse_info.schedulables.values())[0][
        "operation_id"
    ]
    pulse_info_0 = schedule_with_pulse_info.operations[operation_id]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)
    port = pulse_info_0["port"]
    timeslot_index = 0

    # Act
    port_timeline_dict = get_port_timeline(schedule_with_pulse_info)

    # Assert
    assert len(port_timeline_dict) == 1
    assert port in port_timeline_dict
    assert len(port_timeline_dict[port]) == 1
    assert isinstance(port_timeline_dict[port][timeslot_index], list)
    assert port_timeline_dict[port][timeslot_index][0] == pulse_id


def test_get_port_timeline_sorted(
    device_compile_config_basic_transmon,
):
    # Arrange
    ro_acquisition_delay = -16e-9
    ro_pulse_delay = 2e-9
    schedule = spectroscopy_schedules.two_tone_spec_sched(
        spec_pulse_amp=0.6e-0,
        spec_pulse_duration=16e-9,
        spec_pulse_frequency=6.02e9,
        spec_pulse_port="q0:mw",
        spec_pulse_clock="q0.01",
        ro_pulse_amp=0.5e-3,
        ro_pulse_duration=150e-9,
        ro_pulse_delay=ro_pulse_delay,
        ro_pulse_port="q0:res",
        ro_pulse_clock="q0.ro",
        ro_pulse_frequency=7.04e9,
        ro_acquisition_delay=ro_acquisition_delay,
        ro_integration_time=500e-9,
        init_duration=1e-5,
    )

    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule, config=device_compile_config_basic_transmon
    )

    reset_operation_id = list(schedule.schedulables.values())[0]["operation_id"]
    reset_pulse_info = schedule.operations[reset_operation_id]["pulse_info"][0]
    reset_pulse_id = get_pulse_uuid(reset_pulse_info)

    qubit_operation_id = list(schedule.schedulables.values())[1]["operation_id"]
    qubit_pulse_info = schedule.operations[qubit_operation_id]["pulse_info"][0]
    qubit_pulse_id = get_pulse_uuid(qubit_pulse_info)

    ro_operation_id = list(schedule.schedulables.values())[2]["operation_id"]
    ro_pulse_info = schedule.operations[ro_operation_id]["pulse_info"][0]
    ro_pulse_id = get_pulse_uuid(ro_pulse_info)

    acq_operation_id = list(schedule.schedulables.values())[3]["operation_id"]
    acq_pulse_info = schedule.operations[acq_operation_id]["acquisition_info"][0]
    acq_id = get_acq_uuid(acq_pulse_info)

    # Act
    port_timeline_dict = get_port_timeline(schedule)

    # Assert
    assert len(port_timeline_dict) == 3
    assert [
        "None",
        "q0:mw",
        "q0:res",
    ] == list(port_timeline_dict.keys())
    assert list(port_timeline_dict["None"].items()) == [(0, [reset_pulse_id])]
    assert list(port_timeline_dict["q0:mw"].items()) == [(1, [qubit_pulse_id])]
    assert list(port_timeline_dict["q0:res"].items()) == [
        (3, [acq_id]),
        (2, [ro_pulse_id]),
    ]


def test_get_port_timeline_empty(empty_schedule: Schedule):
    # Arrange
    # Act
    port_timeline_dict = get_port_timeline(empty_schedule)

    # Assert
    assert len(port_timeline_dict) == 0


def test_get_port_timeline_are_unique(device_compile_config_basic_transmon):
    # Arrange
    schedule = Schedule("my-schedule")
    schedule.add(Reset("q0", "q1"))
    schedule.add(X90("q0"))
    schedule.add(X90("q1"))

    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule, config=device_compile_config_basic_transmon
    )

    reset_operation_id = list(schedule.schedulables.values())[0]["operation_id"]
    reset_pulse_info_q0 = schedule.operations[reset_operation_id]["pulse_info"][0]
    reset_pulse_id_q0 = get_pulse_uuid(reset_pulse_info_q0)
    reset_pulse_info_q1 = schedule.operations[reset_operation_id]["pulse_info"][1]
    reset_pulse_id_q1 = get_pulse_uuid(reset_pulse_info_q1)

    q0_operation_id = list(schedule.schedulables.values())[1]["operation_id"]
    q0_pulse_info = schedule.operations[q0_operation_id]["pulse_info"][0]
    q0_pulse_id = get_pulse_uuid(q0_pulse_info)

    q1_operation_id = list(schedule.schedulables.values())[2]["operation_id"]
    q1_pulse_info = schedule.operations[q1_operation_id]["pulse_info"][0]
    q1_pulse_id = get_pulse_uuid(q1_pulse_info)

    # Act
    port_timeline_dict = get_port_timeline(schedule)

    # Assert
    assert len(port_timeline_dict) == 3
    assert [
        "None",
        "q0:mw",
        "q1:mw",
    ] == list(port_timeline_dict.keys())
    assert port_timeline_dict["None"][0] == [reset_pulse_id_q0, reset_pulse_id_q1]
    assert port_timeline_dict["q0:mw"][1] == [q0_pulse_id]
    assert port_timeline_dict["q1:mw"][2] == [q1_pulse_id]


def test_get_port_timeline_with_duplicate_op(device_compile_config_basic_transmon):
    # Arrange
    schedule = Schedule("my-schedule")
    X90_q0 = X90("q0")
    schedule.add(X90_q0)
    schedule.add(X90_q0)

    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule, config=device_compile_config_basic_transmon
    )

    X90_q0_operation_id = list(schedule.schedulables.values())[0]["operation_id"]
    X90_q0_pulse_info = schedule.operations[X90_q0_operation_id]["pulse_info"][0]
    X90_q0_pulse_id = get_pulse_uuid(X90_q0_pulse_info)

    # Act
    port_timeline_dict = get_port_timeline(schedule)

    # Assert
    assert len(port_timeline_dict) == 1
    assert [
        "q0:mw",
    ] == list(port_timeline_dict.keys())
    assert port_timeline_dict["q0:mw"][0] == [X90_q0_pulse_id]
    assert port_timeline_dict["q0:mw"][1] == [X90_q0_pulse_id]


def test_get_port_timeline_with_acquisition(
    create_schedule_with_pulse_info,
    schedule_with_measurement: Schedule,
    device_cfg_transmon_example,
):
    # Arrange
    device_config = device_cfg_transmon_example

    schedule = create_schedule_with_pulse_info(schedule_with_measurement, device_config)

    reset_operation_id = list(schedule.schedulables.values())[0]["operation_id"]
    reset_operation = schedule.operations[reset_operation_id]
    reset_pulse_info = reset_operation["pulse_info"][0]
    reset_pulse_id = get_pulse_uuid(reset_pulse_info)

    q0_operation_id = list(schedule.schedulables.values())[1]["operation_id"]
    q0_operation = schedule.operations[q0_operation_id]
    q0_pulse_info = q0_operation["pulse_info"][0]
    q0_pulse_id = get_pulse_uuid(q0_pulse_info)

    acq_operation_id = list(schedule.schedulables.values())[2]["operation_id"]
    acq_operation = schedule.operations[acq_operation_id]

    # Acquisition consists of a reset_clock_phase instruction
    # and a readout-pulse. We select here the actual readout-pulse.
    ro_pulse_info = acq_operation["pulse_info"][1]
    ro_pulse_id = get_pulse_uuid(ro_pulse_info)

    acq_info = acq_operation["acquisition_info"][0]
    acq_id = get_acq_uuid(acq_info)

    # Act
    port_timeline_dict = get_port_timeline(schedule)

    # Assert
    assert len(port_timeline_dict) == 3
    assert [
        "None",
        "q0:mw",
        "q0:res",
    ] == list(port_timeline_dict.keys())
    assert port_timeline_dict["None"][0] == [reset_pulse_id]
    assert port_timeline_dict["q0:mw"][1] == [q0_pulse_id]
    assert port_timeline_dict["q0:res"][2] == [ro_pulse_id, acq_id]


def test_get_total_duration(
    empty_schedule: Schedule,
    schedule_with_pulse_info: Schedule,
    create_schedule_with_pulse_info,
):
    # Arrange
    spec_pulse_duration = 16e-9
    ro_acquisition_delay = 10e-9
    ro_pulse_delay = 2e-9
    ro_integration_time = 500e-9
    ro_pulse_duration = 150e-9
    init_duration = 1e-5

    # Act
    duration0: float = get_total_duration(empty_schedule)
    duration1: float = get_total_duration(schedule_with_pulse_info)
    duration2: float = get_total_duration(
        create_schedule_with_pulse_info(
            spectroscopy_schedules.two_tone_spec_sched(
                spec_pulse_amp=0.6e-0,
                spec_pulse_duration=spec_pulse_duration,
                spec_pulse_frequency=6.02e9,
                spec_pulse_port="q0:mw",
                spec_pulse_clock="q0.01",
                ro_pulse_amp=0.5e-3,
                ro_pulse_duration=ro_pulse_duration,
                ro_pulse_delay=ro_pulse_delay,
                ro_pulse_port="q0:res",
                ro_pulse_clock="q0.ro",
                ro_pulse_frequency=7.04e9,
                ro_acquisition_delay=ro_acquisition_delay,
                ro_integration_time=ro_integration_time,
                init_duration=init_duration,
            )
        )
    )

    # Assert
    assert duration0 == 0.0
    assert duration1 == 20e-9
    assert (
        duration2
        == init_duration
        + spec_pulse_duration
        + ro_pulse_delay
        + ro_integration_time
        + ro_acquisition_delay
    )


def test_get_operation_start(empty_schedule: Schedule, create_schedule_with_pulse_info):
    # Arrange
    schedule0 = Schedule("my-schedule")
    schedule0.add(X90("q0"))
    schedule0.add(Measure("q0"))
    schedule0 = create_schedule_with_pulse_info(schedule0)

    schedule1 = Schedule("my-schedule")
    schedule1.add(Measure("q0"))
    schedule1.add(X90("q0"))
    schedule1 = create_schedule_with_pulse_info(schedule1)

    # Act
    start_empty = get_operation_start(empty_schedule, timeslot_index=0)

    start0_x90 = get_operation_start(schedule0, timeslot_index=0)
    start0_measure = get_operation_start(schedule0, timeslot_index=1)

    start1_measure = get_operation_start(schedule1, timeslot_index=0)
    start1_x90 = get_operation_start(schedule1, timeslot_index=1)

    # Assert
    assert start_empty == 0.0
    assert start0_x90 == 0.0
    assert start0_measure == 20e-9

    assert start1_measure == 0.0
    assert start1_x90 == 4.2e-07


def test_get_operation_end(empty_schedule: Schedule, create_schedule_with_pulse_info):
    # Arrange
    mw_duration = 20e-9
    ro_acquisition_delay = 120e-9
    ro_integration_time = 300e-9

    schedule0 = Schedule("my-schedule")
    schedule0.add(X90("q0"))
    schedule0.add(Measure("q0"))
    schedule0 = create_schedule_with_pulse_info(schedule0)

    schedule1 = Schedule("my-schedule")
    schedule1.add(Measure("q0"))
    schedule1.add(X90("q0"))
    schedule1 = create_schedule_with_pulse_info(schedule1)

    # Act
    end_empty = get_operation_end(empty_schedule, timeslot_index=0)

    endt0_x90 = get_operation_end(schedule0, timeslot_index=0)
    end0_measure = get_operation_end(schedule0, timeslot_index=1)

    end1_measure = get_operation_end(schedule1, timeslot_index=0)
    end1_x90 = get_operation_end(schedule0, timeslot_index=1)

    # Assert
    assert end_empty == 0.0
    assert endt0_x90 == mw_duration
    assert end0_measure == approx(
        mw_duration + ro_acquisition_delay + ro_integration_time
    )
    assert end1_measure == approx(ro_acquisition_delay + ro_integration_time)
    assert end1_x90 == approx(ro_acquisition_delay + ro_integration_time + mw_duration)


def test_schedule_timing_table(mock_setup_basic_transmon_with_standard_params):
    schedule = Schedule("test_schedule_timing_table")
    schedule.add(Reset("q0"))
    schedule.add(X90("q0"))
    schedule.add(Measure("q0"))

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    q0 = mock_setup_basic_transmon_with_standard_params["q0"]
    X90_duration = q0.rxy.duration()
    measure_acq_delay = q0.measure.acq_delay()
    reset_duration = q0.reset.duration()

    expected_abs_timing = [
        0.0,
        reset_duration,
        reset_duration + X90_duration,
        reset_duration + X90_duration,
        reset_duration + X90_duration + measure_acq_delay,
    ]
    actual_abs_timing = schedule.timing_table.data.abs_time
    assert all(expected_abs_timing == actual_abs_timing)


def test_extract_acquisition_metadata_from_schedule(compiled_two_qubit_t1_schedule):
    comp_t1_sched = compiled_two_qubit_t1_schedule
    acq_metadata = extract_acquisition_metadata_from_schedule(comp_t1_sched)

    assert acq_metadata.acq_protocol == "SSBIntegrationComplex"
    assert acq_metadata.bin_mode == BinMode.AVERAGE
    assert acq_metadata.acq_return_type == complex

    # keys correspond to acquisition channels
    assert set(acq_metadata.acq_channels_metadata.keys()) == {0, 1}
    assert acq_metadata.acq_channels_metadata[0].acq_channel == 0
    assert acq_metadata.acq_channels_metadata[1].acq_channel == 1
    assert acq_metadata.acq_channels_metadata[0].acq_indices == list(np.arange(20))
    assert acq_metadata.acq_channels_metadata[1].acq_indices == list(np.arange(20))


def test_extract_port_clocks_used(create_schedule_with_pulse_info):
    schedule0 = Schedule("my-schedule")
    schedule0.add(X90("q0"))
    schedule0 = create_schedule_with_pulse_info(schedule0)
    assert _extract_port_clocks_used(schedule0) == {("q0:mw", "q0.01")}

    schedule1 = Schedule("my-schedule")
    schedule1.add(
        SSBIntegrationComplex(
            duration=1e-6,
            port="q0:ro",
            clock="q0.res",
            acq_index=0,
            acq_channel=0,
        )
    )
    schedule1 = create_schedule_with_pulse_info(schedule1)
    assert _extract_port_clocks_used(schedule1) == {("q0:ro", "q0.res")}

    schedule2 = Schedule("my-schedule")
    schedule2.add(schedule0)
    schedule2.add(schedule1)
    schedule2 = create_schedule_with_pulse_info(schedule2)
    assert _extract_port_clocks_used(schedule2) == {
        ("q0:mw", "q0.01"),
        ("q0:ro", "q0.res"),
    }
