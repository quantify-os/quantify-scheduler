import os
import inspect
import pytest
import numpy as np
import json
import quantify.scheduler.schemas.examples as examples
from quantify.scheduler.resources import Resource, ClockResource, BasebandClockResource
from quantify.scheduler.types import Operation
from quantify.scheduler import Schedule
from quantify.scheduler.gate_library import Reset, Measure, CNOT, Rxy, CZ
from quantify.scheduler.pulse_library import SquarePulse
from quantify.scheduler.compilation import (
    determine_absolute_timing,
    validate_config,
    add_pulse_information_transmon,
    qcompile,
)


esp = inspect.getfile(examples)

cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)


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
    assert len(sched.data["timing_constraints"]) == 5

    for constr in sched.data["timing_constraints"]:
        assert "abs_time" not in constr.keys()
        assert constr["rel_time"] == 0

    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["timing_constraints"]
    ]
    assert abs_times == [0, 1, 2, 3, 4]

    # add a pulse and schedule simultaneous with the second pulse
    sched.add(Rxy(90, 0, qubit=q1), ref_pt="start", ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["timing_constraints"]
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="start", ref_op="M0")
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["timing_constraints"]
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="end", ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["timing_constraints"]
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="center", ref_op=ref_label_1)
    timed_sched = determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["timing_constraints"]
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


def test_config_spec():
    validate_config(DEVICE_CFG, scheme_fn="transmon_cfg.json")


def test_compile_transmon_program():
    sched = Schedule("Test schedule")

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")
    # pulse information is added
    sched = add_pulse_information_transmon(sched, device_cfg=DEVICE_CFG)
    sched = determine_absolute_timing(sched, time_unit="physical")


def test_missing_edge():
    sched = Schedule("Bad edge")
    bad_cfg = DEVICE_CFG.copy()
    del bad_cfg["edges"]["q0-q1"]

    q0, q1 = ("q0", "q1")
    sched.add(operation=CZ(qC=q0, qT=q1))
    with pytest.raises(
        ValueError,
        match="Attempting operation 'CZ' on qubits q1 and q0 which lack a connective edge.",
    ):
        add_pulse_information_transmon(sched, device_cfg=bad_cfg)


def test_empty_sched():
    sched = Schedule("empty")
    with pytest.raises(ValueError, match="schedule 'empty' contains no operations"):
        determine_absolute_timing(sched)


def test_bad_gate():
    class NotAGate(Operation):
        def __init__(self, q):
            data = {
                "gate_info": {
                    "unitary": np.array(
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ),
                    "tex": r"bad",
                    "plot_func": "quantify.scheduler.visualization.circuit_diagram.cnot",
                    "qubits": q,
                    "operation_type": "bad",
                }
            }
            super().__init__("bad ({})".format(q), data=data)

    sched = Schedule("Bell experiment")
    sched.add(Reset("q0"))
    sched.add(NotAGate("q0"))
    with pytest.raises(
        NotImplementedError, match='Operation type "bad" not supported by backend'
    ):
        add_pulse_information_transmon(sched, DEVICE_CFG)


def test_pulse_and_clock():
    sched = Schedule("pulse_no_clock")
    mystery_clock = "BigBen"
    op_label = sched.add(SquarePulse(0.5, 20e-9, "q0:mw_ch", clock=mystery_clock))
    op_hash = next(op for op in sched.timing_constraints if op["label"] == op_label)[
        "operation_hash"
    ]
    with pytest.raises(
        ValueError,
        match="Operation '{}' contains an unknown clock '{}'; ensure this resource has "
        "been added to the schedule.".format(op_hash, mystery_clock),
    ):
        add_pulse_information_transmon(sched, device_cfg=DEVICE_CFG)
    sched.add_resources([ClockResource(mystery_clock, 6e9)])
    add_pulse_information_transmon(sched, device_cfg=DEVICE_CFG)


def test_resource_resolution():
    sched = Schedule("resource_resolution")
    qcm0_s0 = Resource({"name": "qcm0.s0", "type": "qcm"})
    qrm0_s0 = Resource({"name": "qrm0.s0", "type": "qrm"})

    sched.add(Rxy(90, 0, "q0"))
    sched.add(SquarePulse(0.6, 20e-9, "q0:mw_ch", clock=BasebandClockResource.IDENTITY))
    sched.add(SquarePulse(0.4, 20e-9, "q0:ro_ch", clock=BasebandClockResource.IDENTITY))

    sched.add_resources([qcm0_s0, qrm0_s0])
    sched = qcompile(sched, DEVICE_CFG)
