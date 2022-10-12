# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

from copy import deepcopy

import numpy as np
import pytest

from quantify_scheduler import Operation, Schedule
from quantify_scheduler.compilation import (
    determine_absolute_timing,
    device_compile,
    qcompile,
    validate_config,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.gate_library import CNOT, CZ, Measure, Reset, Rxy
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import BasebandClockResource, ClockResource, Resource
from quantify_scheduler.backends.circuit_to_device import ConfigKeyError


def test_determine_absolute_timing_ideal_clock():
    sched = Schedule("Test experiment")

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")

    ref_label_1 = "my_label"

    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(operation=CNOT(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")

    assert len(sched.data["operation_dict"]) == 4
    assert len(sched.data["schedulables"]) == 5

    for schedulable in sched.data["schedulables"].values():
        assert "abs_time" not in schedulable.keys()
        assert schedulable["timing_constraints"][0]["rel_time"] == 0

    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4]

    # add a pulse and schedule simultaneous with the second pulse
    sched.add(Rxy(90, 0, qubit=q1), ref_pt="start", ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="start", ref_op="M0")
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="end", ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="center", ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2, 1.5]

    bad_sched = Schedule("no good")
    bad_sched.add(Rxy(180, 0, qubit=q1))
    bad_sched.add(Rxy(90, 0, qubit=q1), ref_pt="bad")
    with pytest.raises(NotImplementedError):
        determine_absolute_timing(bad_sched)


def test_missing_ref_op():
    sched = Schedule("test")
    q0, q1 = ("q0", "q1")
    ref_label_1 = "test_label"
    with pytest.raises(ValueError):
        sched.add(operation=CNOT(qC=q0, qT=q1), ref_op=ref_label_1)


def test_compile_transmon_program(load_example_transmon_config):
    sched = Schedule("Test schedule")

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")
    sched = qcompile(sched, device_cfg=load_example_transmon_config)


def test_missing_edge(load_example_transmon_config):
    sched = Schedule("Bad edge")
    bad_cfg = load_example_transmon_config
    bad_cfg.edges = {}

    q0, q1 = ("q0", "q1")
    sched.add(operation=CZ(qC=q0, qT=q1))
    with pytest.raises(
        ConfigKeyError,
        match=('edge "q0_q1" is not present in the configuration file'),
    ):
        qcompile(sched, device_cfg=bad_cfg)


def test_empty_sched():
    sched = Schedule("empty")
    with pytest.raises(ValueError, match="schedule 'empty' contains no schedulables"):
        determine_absolute_timing(sched)


def test_bad_gate(load_example_transmon_config):
    class NotAGate(Operation):
        def __init__(self, q):
            plot_func = "quantify_scheduler.visualization.circuit_diagram.cnot"
            data = {
                "gate_info": {
                    "unitary": np.array(
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ),
                    "tex": r"bad",
                    "plot_func": plot_func,
                    "qubits": [q],
                    "operation_type": "bad",
                }
            }
            super().__init__("bad ({})".format(q))
            self.data.update(data)

    sched = Schedule("Bell experiment")
    sched.add(Reset("q0"))
    sched.add(NotAGate("q0"))
    with pytest.raises(
        ConfigKeyError,
        match='\'operation "bad" is not present in the configuration file.*',
    ):
        qcompile(sched, load_example_transmon_config)


def test_pulse_and_clock(load_example_transmon_config):
    sched = Schedule("pulse_no_clock")
    mystery_clock = "BigBen"
    op_label = sched.add(SquarePulse(0.5, 20e-9, "q0:mw_ch", clock=mystery_clock))
    op_hash = next(
        op for op in sched.schedulables.values() if op["label"] == str(op_label)
    )["operation_repr"]
    with pytest.raises(ValueError) as execinfo:
        qcompile(sched, device_cfg=load_example_transmon_config)

    assert str(execinfo.value) == (
        "Operation '{}' contains an unknown clock '{}'; ensure this resource has "
        "been added to the schedule.".format(op_hash, mystery_clock)
    )

    sched.add_resources([ClockResource(mystery_clock, 6e9)])
    qcompile(sched, device_cfg=load_example_transmon_config)


def test_resource_resolution(load_example_transmon_config):
    sched = Schedule("resource_resolution")
    qcm0_s0 = Resource("qcm0.s0")
    qcm0_s0["type"] = "qcm"
    qrm0_s0 = Resource("qrm0.s0")
    qrm0_s0["type"] = "qrm"

    sched.add(Rxy(90, 0, "q0"))
    sched.add(SquarePulse(0.6, 20e-9, "q0:mw_ch", clock=BasebandClockResource.IDENTITY))
    sched.add(SquarePulse(0.4, 20e-9, "q0:ro_ch", clock=BasebandClockResource.IDENTITY))

    sched.add_resources([qcm0_s0, qrm0_s0])
    sched = qcompile(sched, load_example_transmon_config)


def test_schedule_modified(load_example_transmon_config):
    q0, q1 = ("q0", "q1")

    ref_label_1 = "my_label"
    sched = Schedule("Test experiment")
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")

    copy_of_sched = deepcopy(sched)
    # to verify equality of schedule object works
    assert copy_of_sched == sched

    _ = qcompile(sched, load_example_transmon_config)

    # Fails if schedule is modified
    assert copy_of_sched == sched


def test_measurement_specification_of_binmode(load_example_transmon_config):

    qubit = "q0"

    ####################################################################################
    # Append selected
    ####################################################################################

    schedule = Schedule("binmode-test", 1)
    schedule.add(Reset(qubit), label=f"Reset {0}")
    schedule.add(
        Measure(qubit, acq_index=0, bin_mode=BinMode.APPEND), label=f"Measurement {0}"
    )

    comp_sched = qcompile(schedule, device_cfg=load_example_transmon_config)

    for key, value in comp_sched.data["operation_dict"].items():
        if "Measure" in key:
            assert value.data["acquisition_info"][0]["bin_mode"] == BinMode.APPEND

    ####################################################################################
    # AVERAGE selected
    ####################################################################################

    schedule = Schedule("binmode-test", 1)
    schedule.add(Reset(qubit), label=f"Reset {0}")
    schedule.add(
        Measure(qubit, acq_index=0, bin_mode=BinMode.AVERAGE), label=f"Measurement {0}"
    )

    comp_sched = qcompile(schedule, device_cfg=load_example_transmon_config)

    for key, value in comp_sched.data["operation_dict"].items():
        if "Measure" in key:
            assert value.data["acquisition_info"][0]["bin_mode"] == BinMode.AVERAGE

    ####################################################################################
    # Not specified uses default average mode
    ####################################################################################

    schedule = Schedule("binmode-test", 1)
    schedule.add(Reset(qubit), label=f"Reset {0}")
    schedule.add(Measure(qubit, acq_index=0), label=f"Measurement {0}")

    comp_sched = qcompile(schedule, device_cfg=load_example_transmon_config)

    for key, value in comp_sched.data["operation_dict"].items():
        if "Measure" in key:
            assert value.data["acquisition_info"][0]["bin_mode"] == BinMode.AVERAGE


def test_compile_trace_acquisition(load_example_transmon_config):
    sched = Schedule("Test schedule")
    q0 = "q0"
    sched.add(Reset(q0))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(Measure(q0, acq_protocol="Trace"), label="M0")

    sched = device_compile(sched, device_cfg=load_example_transmon_config)

    measure_repr = list(sched.schedulables.values())[-1]["operation_repr"]
    assert sched.operations[measure_repr]["acquisition_info"][0]["protocol"] == "trace"


def test_qcompile_no_device_cfg_determine_absolute_timing(mocker):
    sched = Schedule("One pulse schedule")
    sched.add_resources([ClockResource("q0.01", 3.1e9)])
    sched.add(SquarePulse(amp=1 / 4, duration=12e-9, port="q0:mw", clock="q0.01"))

    mock = mocker.patch("quantify_scheduler.compilation.determine_absolute_timing")
    qcompile(schedule=sched, device_cfg=None)

    mock.assert_called()
