import json
import re
from unittest import TestCase

import numpy as np
import pytest

from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.operations.gate_library import ConditionalReset
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.json_utils import SchedulerJSONDecoder, SchedulerJSONEncoder
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    LoopOperation,
)
from quantify_scheduler.operations.gate_library import (
    CNOT,
    CZ,
    X90,
    Y90,
    Z90,
    H,
    Measure,
    Reset,
    Rxy,
    Rz,
    X,
    Y,
    Z,
)
from quantify_scheduler.operations.nv_native_library import ChargeReset
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.operations.shared_native_library import SpectroscopyOperation
from quantify_scheduler.schedules.schedule import Schedulable, Schedule
from quantify_scheduler.schemas.examples import utils


def test_schedule_add_schedulables() -> None:
    sched = Schedule("my exp")
    test_lab = "test label"
    x90_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"), label=test_lab)["label"]
    assert x90_label == test_lab

    with pytest.raises(ValueError):
        x90_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"), label=test_lab)["label"]

    uuid_label = sched.add(Rxy(theta=90, phi=0, qubit="q0"))["label"]
    assert uuid_label != x90_label

    # not specifying a label should work
    sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op=None)

    # specifying existing label should work
    sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op=x90_label)

    # specifying non-existing label should raise an error
    with pytest.raises(ValueError):
        sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op="non-existing-operation")

    # specifying a Schedulable that is not in the Schedule should raise an error
    with pytest.raises(ValueError):
        new_sched = Schedule("redundant")
        new_schedulable = new_sched.add(Rxy(theta=15.4, phi=42.6, qubit="q0"))
        sched.add(Rxy(theta=90, phi=0, qubit="q0"), ref_op=new_schedulable)

    # All schedulables should be valid
    for schedulable in sched.schedulables.values():
        assert Schedulable.is_valid(schedulable)

    assert Schedule.is_valid(sched)


def test_rxy_angle_modulo() -> None:
    """asserts that theta angles fall in the domain -180 to 180"""
    rxy_270 = Rxy(theta=270, phi=23.9, qubit="q5")
    rxy_m90 = Rxy(theta=-90, phi=23.9, qubit="q5")
    assert rxy_270 == rxy_m90

    assert rxy_270.data["gate_info"]["theta"] == -90.0

    rxy_360 = Rxy(theta=360, phi=23.9, qubit="q5")
    assert rxy_360.data["gate_info"]["theta"] == 0


@pytest.mark.parametrize(
    "operation",
    [
        Rxy(theta=124, phi=23.9, qubit="q5"),
        X90("q1"),
        X("q0"),
        Y90("q1"),
        Y("q1"),
        Rz(theta=124, qubit="q5"),
        Z("q0"),
        Z90("q1"),
        Reset("q0"),
        Reset("q0", "q1"),
        CZ("q0", "q1"),
        CNOT("q0", "q6"),
        H("q0", "q1"),
        Measure("q0", "q6"),
        Measure("q0"),
        Measure("q0", "q6", acq_index=92),
        SpectroscopyOperation("q0"),
        ChargeReset("q0"),
    ],
)
class TestGateLevelOperation:
    def test_gate_is_valid(self, operation: Operation) -> None:
        assert Operation.is_valid(operation)

    def test__repr__(self, operation: Operation) -> None:
        """
        Asserts that evaluating the representation
        of an operation is identical to the operation
        itself.
        """
        # Arrange
        operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

        # Act
        obj = json.loads(operation_state, cls=SchedulerJSONDecoder)
        assert obj == operation

    def test__str__(self, operation: Operation) -> None:
        """
        Asserts that the evaluation of the string representation
        is an instance of the the operation type.
        """
        assert isinstance(eval(str(operation)), type(operation))  # nosec B307

    def test_deserialize(self, operation: Operation) -> None:
        # Arrange
        operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

        # Act
        obj = json.loads(operation_state, cls=SchedulerJSONDecoder)

        # Assert
        if (
            "unitary" in operation.data["gate_info"]
            and operation.data["gate_info"]["unitary"] is not None
        ):
            assert isinstance(obj.data["gate_info"]["unitary"], (np.generic, np.ndarray))
            np.testing.assert_array_almost_equal(
                obj.data["gate_info"]["unitary"],
                operation.data["gate_info"]["unitary"],
                decimal=9,
            )

            # TestCase().assertDictEqual cannot compare numpy arrays for equality
            # therefore "unitary" is removed
            del obj.data["gate_info"]["unitary"]
            del operation.data["gate_info"]["unitary"]

        TestCase().assertDictEqual(obj.data, operation.data)

    def test__repr__modify_not_equal(self, operation: Operation) -> None:
        # Arrange
        operation_state: str = json.dumps(operation, cls=SchedulerJSONEncoder)

        # Act
        obj = json.loads(operation_state, cls=SchedulerJSONDecoder)
        assert obj == operation

        # Act
        obj.data["pulse_info"].append({"clock": "q0.01"})

        # Assert
        assert obj != operation


def test_rotation_unitaries() -> None:
    # Set the tolerance in terms of machine precision, one machine epsilon by default
    # Could be increased to allow for less pretty computations with more round-off
    # error.

    atol = 1 * np.finfo(np.complex128).eps
    # Test Rxy for all angles:
    # The tests are written in form: target, desired
    np.testing.assert_allclose(
        Rxy(theta=0, phi=0, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) * np.array([[1, 0], [0, 1]]),
        atol=atol,
    )
    np.testing.assert_allclose(
        Rxy(theta=90, phi=0, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, -1j], [-1j, 1]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Rxy(theta=-90, phi=90, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, 1], [-1, 1]]),
        atol=atol,
    )

    # Test for the X180, X90, Y180 and Y90 gates which are derived from Rxy
    np.testing.assert_allclose(
        X(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) * np.array([[0, -1j], [-1j, 0]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        X90(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, -1j], [-1j, 1]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Y(qubit=None).data["gate_info"]["unitary"],
        -(1.0 + 0.0j) * np.array([[0, 1], [-1, 0]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Y90(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1, -1], [1, 1]]),
        atol=atol,
    )

    # Test Rz for all angles:
    # The tests are written in form: target, desired
    np.testing.assert_allclose(
        Rz(theta=0, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) * np.array([[1, 0], [0, 1]]),
        atol=atol,
    )
    np.testing.assert_allclose(
        Rz(theta=90, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1 - 1j, 0], [0, 1 + 1j]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Rz(theta=-90, qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1 + 1j, 0], [0, 1 - 1j]]),
        atol=atol,
    )

    # Test for the Z180, Z90 gates which are derived from Rz
    np.testing.assert_allclose(
        Z(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) * np.array([[-1j, 0], [0, 1j]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        Z90(qubit=None).data["gate_info"]["unitary"],
        (1.0 + 0.0j) / np.sqrt(2) * np.array([[1 - 1j, 0], [0, 1 + 1j]]),
        atol=atol,
    )

    np.testing.assert_allclose(
        H("q0").data["gate_info"]["unitary"],
        -1j / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex),
        atol=atol,
    )


def test_conditional_reset_inside_loop(mock_setup_basic_transmon_with_standard_params):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    hardware_config = utils.load_json_example_scheme("qblox_hardware_config_transmon.json")
    quantum_device.hardware_config(hardware_config)
    config = quantum_device.generate_compilation_config()

    schedule = Schedule("")
    inner = Schedule("test")
    inner.add(ConditionalReset("q0"))
    schedule.add(LoopOperation(body=inner, repetitions=1024))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule,
        config=config,
    )

    qcm_program = (
        compiled_schedule.compiled_instructions.get("cluster0")
        .get("cluster0_module2")
        .get("sequencers")
        .get("seq0")
        .get("sequence")
        .get("program")
    )

    # Check that there is no `loop` instruction between the two `set_cond`
    # instructions.
    pattern = r"^\s*set_cond.*?(?!^\s*loop).*^\s*set_cond"
    match = re.search(pattern, qcm_program, re.DOTALL | re.MULTILINE)
    assert match is not None, "`loop` and `set_cond` are not exited correctly."


def test_conditional_reset_single_qubit(
    mock_setup_basic_transmon_with_standard_params,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    hardware_config = utils.load_json_example_scheme("qblox_hardware_config_transmon.json")
    quantum_device.hardware_config(hardware_config)
    config = quantum_device.generate_compilation_config()

    schedule = Schedule("test")
    schedule.add(X("q0"))
    schedule.add(ConditionalReset("q0"))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule,
        config=config,
    )

    seq_settings = (
        compiled_schedule.compiled_instructions.get("cluster0")
        .get("cluster0_module4")
        .get("sequencers")
        .get("seq0")
    )
    assert (expected_address := seq_settings["thresholded_acq_trigger_address"]) is not None
    assert seq_settings["thresholded_acq_trigger_en"] is True
    assert seq_settings["thresholded_acq_trigger_invert"] is False

    qcm_program = (
        compiled_schedule.compiled_instructions.get("cluster0")
        .get("cluster0_module2")
        .get("sequencers")
        .get("seq0")
        .get("sequence")
        .get("program")
    )

    qrm_program = (
        compiled_schedule.compiled_instructions.get("cluster0")
        .get("cluster0_module4")
        .get("sequencers")
        .get("seq0")
        .get("sequence")
        .get("program")
    )

    pattern = r"^\s*latch_rst.*$"
    match = re.search(pattern, qrm_program, re.MULTILINE)
    assert match is None

    pattern = r"^\s*set_latch_en\s*(\d).*$"
    match = re.search(pattern, qcm_program, re.MULTILINE)
    assert match is not None

    latch_en_arg = int(match.group(1))
    assert latch_en_arg == 1

    # The (?P<enable>\d) syntax below assigns a name to the capturing group.
    pattern = r"""
        set_cond\s*1,(?P<mask>\d+),0,4.*
        set_awg_gain.*
        play.*
        wait\s*16.*
        set_cond\s*1,1,1,4.*
        wait\s*16.*
        set_cond\s*0,0,0,0.*
    """

    compiled_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)
    match = compiled_pattern.search(qcm_program)
    assert match is not None

    mask = match.group("mask")
    expected_mask = str(2**expected_address - 1)
    assert mask == expected_mask


def test_conditional_reset_with_overlapping_pulse_for_acq(
    mock_setup_basic_transmon_with_standard_params,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    hardware_config = utils.load_json_example_scheme("qblox_hardware_config_transmon.json")
    quantum_device.hardware_config(hardware_config)
    config = quantum_device.generate_compilation_config()

    schedule = Schedule("test")
    schedule.add(
        Measure(
            "q0",
            acq_protocol="ThresholdedAcquisition",
            feedback_trigger_label="q0",
        )
    )
    schedule.add(
        ConditionalOperation(body=X("q0"), qubit_name="q0"),
        rel_time=364e-9,
    )
    schedule.add(
        SquarePulse(amp=0.1, duration=16e-9, port="q0:res", clock="q0.ro"),
        ref_pt="start",
        rel_time=96e-9,
    )

    compiler = SerialCompiler(name="compiler")
    compiler.compile(
        schedule,
        config=config,
    )


def test_conditional_acquire_without_control_flow_raises(
    mock_setup_basic_transmon_with_standard_params,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    hardware_config = utils.load_json_example_scheme("qblox_hardware_config_transmon.json")
    quantum_device.hardware_config(hardware_config)
    config = quantum_device.generate_compilation_config()

    schedule = Schedule("test")
    schedule.add(
        Measure(
            "q0",
            feedback_trigger_label="q0",
            acq_index=0,
            acq_protocol="ThresholdedAcquisition",
        )
    )
    schedule.add(
        Measure(
            "q0",
            feedback_trigger_label="q0",
            acq_index=1,
            acq_protocol="ThresholdedAcquisition",
        )
    )

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        RuntimeError,
        match="Two subsequent conditional acquisitions found, "
        "without a conditional control flow operation in between",
    ):
        _ = compiler.compile(
            schedule,
            config=config,
        )


@pytest.mark.parametrize(
    ["num_reset", "expected_exception"],
    [
        (constants.MAX_FEEDBACK_TRIGGER_ADDRESS - 1, None),
        (constants.MAX_FEEDBACK_TRIGGER_ADDRESS, None),
        (constants.MAX_FEEDBACK_TRIGGER_ADDRESS + 1, ValueError),
    ],
)
def test_max_conditional_resets(num_reset, expected_exception):
    q = {}
    quantum_device = QuantumDevice("o")

    for i in range(num_reset):
        q[i] = BasicTransmonElement(f"q{i}")
        q[i].rxy.amp180(0.9)
        q[i].rxy.motzoi(0.9)
        q[i].rxy.duration(200e-9)
        q[i].clock_freqs.f01(1.2e9)
        q[i].clock_freqs.f12(1.2e9)
        q[i].clock_freqs.readout(1.2e9)
        q[i].measure.acq_delay(100e-9)
        q[i].measure.integration_time(200e-9)
        quantum_device.add_element(q[i])

    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "QAE_cluster": {
                "instrument_type": "Cluster",
                "modules": {},
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [],
        },
    }

    for i in range(1, 16):  # From 1 to 15
        hw_config["hardware_description"]["QAE_cluster"]["modules"][str(i)] = {
            "instrument_type": "QRM"
        }

    for i in range(15):
        module_index = i + 1
        module_reference = f"QAE_cluster.module{module_index}.complex_output_0"

        hw_config["connectivity"]["graph"].append([module_reference, f"q{i}:mw"])
        hw_config["connectivity"]["graph"].append([module_reference, f"q{i}:res"])
    quantum_device.hardware_config(hw_config)

    schedule = Schedule("test")
    for qubit in q.values():
        schedule.add(ConditionalReset(qubit.name))

    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler("")

    if expected_exception is None:
        compiler.compile(schedule, config)
    else:
        with pytest.raises(expected_exception, match="Maximum number"):
            compiler.compile(schedule, config)


def test_conditional_reset_multi_qubits(
    mock_setup_basic_transmon_with_standard_params,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    hardware_config = utils.load_json_example_scheme("qblox_hardware_config_transmon.json")
    quantum_device.hardware_config(hardware_config)
    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")

    schedule = Schedule("")
    schedule.add(ConditionalReset("q0"))
    schedule.add(ConditionalReset("q4"))

    compiled_schedule = compiler.compile(
        schedule,
        config=config,
    )

    # Assert readout module sequencer settings.
    trigger_address_q0 = compiled_schedule.compiled_instructions["cluster0"]["cluster0_module4"][
        "sequencers"
    ]["seq0"]["thresholded_acq_trigger_address"]
    thresholded_acq_trigger_en_q0 = compiled_schedule.compiled_instructions["cluster0"][
        "cluster0_module4"
    ]["sequencers"]["seq0"]["thresholded_acq_trigger_en"]
    thresholded_acq_trigger_invert_q0 = compiled_schedule.compiled_instructions["cluster0"][
        "cluster0_module4"
    ]["sequencers"]["seq0"]["thresholded_acq_trigger_invert"]

    assert trigger_address_q0 == 1
    assert thresholded_acq_trigger_en_q0 is True
    assert thresholded_acq_trigger_invert_q0 is False

    # Assert readout module sequencer settings for "q4".
    trigger_address_q4 = compiled_schedule.compiled_instructions["cluster0"]["cluster0_module3"][
        "sequencers"
    ]["seq0"]["thresholded_acq_trigger_address"]
    thresholded_acq_trigger_en_q4 = compiled_schedule.compiled_instructions["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]["thresholded_acq_trigger_en"]
    thresholded_acq_trigger_invert_q4 = compiled_schedule.compiled_instructions["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]["thresholded_acq_trigger_invert"]

    assert trigger_address_q4 == 2
    assert thresholded_acq_trigger_en_q4 is True
    assert thresholded_acq_trigger_invert_q4 is False
