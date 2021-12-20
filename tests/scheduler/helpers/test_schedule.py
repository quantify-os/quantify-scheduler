# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
from __future__ import annotations

import numpy as np
from pytest import approx

from quantify_scheduler import Schedule
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_schedule,
    get_acq_info_by_uuid,
    get_acq_uuid,
    get_operation_end,
    get_operation_start,
    get_port_timeline,
    get_pulse_info_by_uuid,
    get_pulse_uuid,
    get_schedule_time_offset,
    get_total_duration,
)
from quantify_scheduler.operations.gate_library import X90, Measure, Reset
from quantify_scheduler.schedules import spectroscopy_schedules


def test_get_info_by_uuid_empty(empty_schedule: Schedule):
    # Act
    pulseid_pulseinfo_dict = get_pulse_info_by_uuid(empty_schedule)

    # Assert
    assert len(pulseid_pulseinfo_dict) == 0


def test_get_info_by_uuid(
    schedule_with_pulse_info: Schedule,
):
    # Arrange
    operation_repr = schedule_with_pulse_info.timing_constraints[0]["operation_repr"]
    pulse_info_0 = schedule_with_pulse_info.operations[operation_repr]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)

    # Act
    pulseid_pulseinfo_dict = get_pulse_info_by_uuid(schedule_with_pulse_info)

    # Assert
    assert len(pulseid_pulseinfo_dict) == 1
    assert pulse_id in pulseid_pulseinfo_dict
    assert pulseid_pulseinfo_dict[pulse_id] == pulse_info_0


def test_get_info_by_uuid_are_unique(
    create_schedule_with_pulse_info,
):
    # Arrange
    schedule = Schedule("my-schedule")
    schedule.add(X90("q0"))
    schedule.add(X90("q0"))
    create_schedule_with_pulse_info(schedule)

    operation_repr = schedule.timing_constraints[0]["operation_repr"]
    pulse_info_0 = schedule.operations[operation_repr]["pulse_info"][0]
    pulse_id = get_pulse_uuid(pulse_info_0)

    # Act
    pulseid_pulseinfo_dict = get_pulse_info_by_uuid(schedule)

    # Assert
    assert len(pulseid_pulseinfo_dict) == 1
    assert pulse_id in pulseid_pulseinfo_dict
    assert pulseid_pulseinfo_dict[pulse_id] == pulse_info_0


def test_get_acq_info_by_uuid(
    create_schedule_with_pulse_info,
    schedule_with_measurement: Schedule,
    load_example_transmon_config,
):
    # Arrange
    device_config = load_example_transmon_config()
    device_config["qubits"]["q0"]["params"]["acquisition"] = "SSBIntegrationComplex"

    schedule = create_schedule_with_pulse_info(schedule_with_measurement, device_config)

    operation_repr = schedule.timing_constraints[-1]["operation_repr"]
    operation = schedule.operations[operation_repr]
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
    operation_repr = schedule_with_pulse_info.timing_constraints[0]["operation_repr"]
    pulse_info_0 = schedule_with_pulse_info.operations[operation_repr]["pulse_info"][0]
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
    create_schedule_with_pulse_info,
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
    schedule = create_schedule_with_pulse_info(schedule)

    reset_operation_id = schedule.timing_constraints[0]["operation_repr"]
    reset_pulse_info = schedule.operations[reset_operation_id]["pulse_info"][0]
    reset_pulse_id = get_pulse_uuid(reset_pulse_info)

    qubit_operation_id = schedule.timing_constraints[1]["operation_repr"]
    qubit_pulse_info = schedule.operations[qubit_operation_id]["pulse_info"][0]
    qubit_pulse_id = get_pulse_uuid(qubit_pulse_info)

    ro_operation_id = schedule.timing_constraints[2]["operation_repr"]
    ro_pulse_info = schedule.operations[ro_operation_id]["pulse_info"][0]
    ro_pulse_id = get_pulse_uuid(ro_pulse_info)

    acq_operation_id = schedule.timing_constraints[3]["operation_repr"]
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


def test_get_port_timeline_are_unique(
    create_schedule_with_pulse_info,
):
    # Arrange
    schedule = Schedule("my-schedule")
    schedule.add(Reset("q0", "q1"))
    schedule.add(X90("q0"))
    schedule.add(X90("q1"))
    create_schedule_with_pulse_info(schedule)

    reset_operation_id = schedule.timing_constraints[0]["operation_repr"]
    reset_pulse_info = schedule.operations[reset_operation_id]["pulse_info"][0]
    reset_pulse_id = get_pulse_uuid(reset_pulse_info)

    q0_operation_id = schedule.timing_constraints[1]["operation_repr"]
    q0_pulse_info = schedule.operations[q0_operation_id]["pulse_info"][0]
    q0_pulse_id = get_pulse_uuid(q0_pulse_info)

    q1_operation_id = schedule.timing_constraints[2]["operation_repr"]
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
    assert port_timeline_dict["None"][0] == [reset_pulse_id]
    assert port_timeline_dict["q0:mw"][1] == [q0_pulse_id]
    assert port_timeline_dict["q1:mw"][2] == [q1_pulse_id]


def test_get_port_timeline_with_duplicate_op(
    create_schedule_with_pulse_info,
):
    # Arrange
    schedule = Schedule("my-schedule")
    X90_q0 = X90("q0")
    schedule.add(X90_q0)
    schedule.add(X90_q0)
    create_schedule_with_pulse_info(schedule)

    X90_q0_operation_id = schedule.timing_constraints[0]["operation_repr"]
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
    load_example_transmon_config,
):
    # Arrange
    device_config = load_example_transmon_config()
    device_config["qubits"]["q0"]["params"]["acquisition"] = "SSBIntegrationComplex"

    schedule = create_schedule_with_pulse_info(schedule_with_measurement, device_config)

    reset_operation_id = schedule.timing_constraints[0]["operation_repr"]
    reset_operation = schedule.operations[reset_operation_id]
    reset_pulse_info = reset_operation["pulse_info"][0]
    reset_pulse_id = get_pulse_uuid(reset_pulse_info)

    q0_operation_id = schedule.timing_constraints[1]["operation_repr"]
    q0_operation = schedule.operations[q0_operation_id]
    q0_pulse_info = q0_operation["pulse_info"][0]
    q0_pulse_id = get_pulse_uuid(q0_pulse_info)

    acq_operation_id = schedule.timing_constraints[2]["operation_repr"]
    acq_operation = schedule.operations[acq_operation_id]

    ro_pulse_info = acq_operation["pulse_info"][0]
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
    start1_x90 = get_operation_start(schedule0, timeslot_index=1)

    # Assert
    assert start_empty == 0.0
    assert start0_x90 == 0.0
    assert start0_measure == 20e-9
    assert start1_measure == 0.0
    assert start1_x90 == 20e-9


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


def test_get_schedule_time_offset(
    empty_schedule: Schedule,
    basic_schedule: Schedule,
    schedule_with_measurement: Schedule,
    create_schedule_with_pulse_info,
):
    # Arrange
    _basic_schedule = create_schedule_with_pulse_info(basic_schedule)
    _schedule_with_measurement = create_schedule_with_pulse_info(
        schedule_with_measurement
    )
    init_duration = 200e-6

    # Act
    offset0 = get_schedule_time_offset(
        empty_schedule, get_port_timeline(empty_schedule)
    )
    offset1 = get_schedule_time_offset(
        _basic_schedule, get_port_timeline(_basic_schedule)
    )
    offset2 = get_schedule_time_offset(
        _schedule_with_measurement, get_port_timeline(_schedule_with_measurement)
    )

    # Assert
    assert offset0 == 0.0
    assert offset1 == 0.0
    assert offset2 == init_duration


def test_extract_acquisition_metadata_from_schedule(compiled_two_qubit_t1_schedule):
    comp_t1_sched = compiled_two_qubit_t1_schedule
    acq_metadata = extract_acquisition_metadata_from_schedule(comp_t1_sched)

    assert acq_metadata.acq_protocol == "ssb_integration_complex"
    assert acq_metadata.bin_mode == BinMode.AVERAGE
    assert acq_metadata.acq_return_type == complex

    # keys correspond to acquisition channels
    assert set(acq_metadata.acq_indices.keys()) == {0, 1}
    assert acq_metadata.acq_indices[0] == acq_metadata.acq_indices[1]
    assert acq_metadata.acq_indices[0] == list(np.arange(20))
