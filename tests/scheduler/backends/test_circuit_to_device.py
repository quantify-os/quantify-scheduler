# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from typing import List

import pytest

from quantify_scheduler import Operation

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.circuit_to_device import (
    compile_circuit_to_device,
    set_pulse_and_acquisition_clock,
    ConfigKeyError,
    DeviceCompilationConfig,
    OperationCompilationConfig,
)

from quantify_scheduler.operations.pulse_library import IdlePulse
from quantify_scheduler.operations.gate_library import (
    CNOT,
    CZ,
    Measure,
    Reset,
    Rxy,
    X,
    Y,
    Y90,
)
from quantify_scheduler.operations.pulse_factories import rxy_drag_pulse
from quantify_scheduler.resources import ClockResource

from quantify_scheduler.schemas.examples.circuit_to_device_example_cfgs import (
    example_transmon_cfg,
)


def test_compile_transmon_example_program():
    """
    Test if compilation using the new backend reproduces behavior of the old backend.
    """

    sched = Schedule("Test schedule")

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(Rxy(45, 0, qubit=q0))
    sched.add(Rxy(12, 0, qubit=q0))
    sched.add(Rxy(12, 0, qubit=q0))
    sched.add(X(qubit=q0))
    sched.add(Y(qubit=q0))
    sched.add(Y90(qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M_q0_q1")

    # test that all these operations compile correctly.
    _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_compile_basic_transmon_example_program(mock_setup_basic_transmon):
    """
    Test if compilation using the BasicTransmonElement reproduces old behaviour.
    """

    sched = Schedule("Test schedule")

    # define the resources
    q2, q3 = ("q2", "q3")
    sched.add(Reset(q2, q3))
    sched.add(Rxy(90, 0, qubit=q2))
    sched.add(Rxy(45, 0, qubit=q2))
    sched.add(Rxy(12, 0, qubit=q2))
    sched.add(Rxy(12, 0, qubit=q2))
    sched.add(X(qubit=q2))
    sched.add(Y(qubit=q2))
    sched.add(Y90(qubit=q2))
    sched.add(operation=CZ(qC=q2, qT=q3))
    sched.add(Rxy(theta=90, phi=0, qubit=q2))
    sched.add(Measure(q2, q3), label="M_q2_q3")

    # test that all these operations compile correctly.
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    _ = compile_circuit_to_device(
        sched, device_cfg=quantum_device.generate_device_config()
    )


def test_compile_asymmetric_gate(mock_setup_basic_transmon):
    """
    Test if compilation fails when performing an asymmetric operation and the
    correct edge defining the parent-child device element connection is missing from
    the device config.
    """
    sched = Schedule("Test schedule")

    # define the resources
    q2, q3 = ("q2", "q3")

    # Deliberately define an asymmetric CZ
    asymmetric_cz = CZ(qC=q3, qT=q2)
    asymmetric_cz.data["gate_info"]["symmetric"] = False

    sched.add(Reset(q2, q3))
    sched.add(operation=asymmetric_cz)
    sched.add(Measure(q2, q3), label="M_q2_q3")

    # test that all these operations compile correctly.
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(
            sched, device_cfg=quantum_device.generate_device_config()
        )


def test_rxy_operations_compile():
    sched = Schedule("Test schedule")
    sched.add(Rxy(90, 0, qubit="q0"))
    sched.add(Rxy(180, 45, qubit="q0"))
    _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_measurement_compile():
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", "q1"))  # acq_index should be 0 for both.
    sched.add(Measure("q0", acq_index=1))
    sched.add(Measure("q1", acq_index=2))  # acq_channel should be 1
    sched.add(Measure("q0", "q1", acq_index=2))
    new_dev_sched = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)

    operation_keys_list = list(new_dev_sched.operations.keys())

    m0_acq = new_dev_sched.operations[operation_keys_list[0]]["acquisition_info"]
    assert len(m0_acq) == 2  # both q0 and q1
    assert m0_acq[0]["acq_channel"] == 0
    assert m0_acq[1]["acq_channel"] == 1
    assert m0_acq[0]["acq_index"] == 0
    assert m0_acq[1]["acq_index"] == 0

    m1_acq = new_dev_sched.operations[operation_keys_list[1]]["acquisition_info"]
    assert len(m1_acq) == 1
    assert m1_acq[0]["acq_channel"] == 0
    assert m1_acq[0]["acq_index"] == 1

    m2_acq = new_dev_sched.operations[operation_keys_list[2]]["acquisition_info"]
    assert len(m2_acq) == 1
    assert m2_acq[0]["acq_channel"] == 1
    assert m2_acq[0]["acq_index"] == 2

    m3_acq = new_dev_sched.operations[operation_keys_list[3]]["acquisition_info"]
    assert len(m3_acq) == 2
    assert m3_acq[0]["acq_channel"] == 0
    assert m3_acq[1]["acq_channel"] == 1
    assert m3_acq[0]["acq_index"] == 2
    assert m3_acq[1]["acq_index"] == 2


@pytest.mark.parametrize(
    "operations, clocks_used",
    [
        ([], ["cl0.baseband"]),
        ([X(qubit="q0")], ["cl0.baseband", "q0.01"]),
        ([Measure("q0", "q1")], ["cl0.baseband", "q0.ro", "q1.ro"]),
        (
            [X(qubit="q0"), X(qubit="q1"), Measure("q0", "q1")],
            ["cl0.baseband", "q0.01", "q1.01", "q0.ro", "q1.ro"],
        ),
    ],
)
def test_only_add_clocks_used(operations: List[Operation], clocks_used: List[str]):
    sched = Schedule("Test schedule")
    for operation in operations:
        sched.add(operation)
    dev_sched = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)
    checked_dev_sched = set_pulse_and_acquisition_clock(
        dev_sched, device_cfg=example_transmon_cfg
    )

    assert set(checked_dev_sched.resources.keys()) == set(clocks_used)


def test_set_gate_clock_raises(mock_setup_basic_transmon_with_standard_params):
    sched = Schedule("Test schedule")
    operation = X("q0")
    sched.add(operation)

    device_cfg = mock_setup_basic_transmon_with_standard_params[
        "quantum_device"
    ].generate_device_config()

    with pytest.raises(RuntimeError) as error:
        _ = set_pulse_and_acquisition_clock(sched, device_cfg=device_cfg)

    assert (
        error.value.args[0]
        == f"Operation '{operation}' is a gate-level operation and must be "
        f"compiled from circuit to device; ensure compilation "
        f"is made in the correct order."
    )


def test_multiply_defined_clock_freq_raises(
    mock_setup_basic_transmon_with_standard_params,
):
    clock = "q0.01"
    clock_freq_schedule = 5e9

    device_cfg = mock_setup_basic_transmon_with_standard_params[
        "quantum_device"
    ].generate_device_config()
    clock_freq_device_cfg = device_cfg.clocks[clock]

    sched = Schedule("Test schedule")
    sched.add_resource(ClockResource(name="q0.01", freq=clock_freq_schedule))
    operation = X("q0")
    sched.add(operation)
    dev_sched = compile_circuit_to_device(schedule=sched, device_cfg=device_cfg)

    with pytest.warns(RuntimeWarning) as warning:
        compiled_sched = set_pulse_and_acquisition_clock(
            schedule=dev_sched, device_cfg=device_cfg
        )
    assert (
        warning[0].message.args[0]
        == f"Clock '{clock}' has conflicting frequency definitions: "
        f"{clock_freq_schedule} Hz in the schedule and "
        f"{clock_freq_device_cfg} Hz in the device config. "
        f"The clock is set to '{clock_freq_schedule}'. "
        f"Ensure the schedule clock resource matches the "
        f"device config clock frequency or set the "
        f"clock frequency in the device config to np.NaN "
        f"to omit this warning."
    )
    assert clock_freq_schedule != clock_freq_device_cfg
    assert compiled_sched.resources[clock]["freq"] == clock_freq_schedule


def test_set_device_cfg_clock(
    mock_setup_basic_transmon_with_standard_params,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    q0 = quantum_device.get_element("q0")
    clock = "q0.01"

    sched = Schedule("Test schedule")
    sched.add(X("q0"))

    compiler = SerialCompiler("test")
    compiled_sched = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )
    assert compiled_sched.resources[clock]["freq"] == q0.clock_freqs.f01()


def test_clock_not_defined_raises():
    simple_config = DeviceCompilationConfig(
        backend="quantify_scheduler.backends.circuit_to_device"
        + ".compile_circuit_to_device",
        clocks={
            "q0.01": 6020000000.0,
        },
        elements={
            "q0": {
                "measure": {
                    "factory_func": "quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement",
                    "gate_info_factory_kwargs": [
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SquarePulse",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_channel": 0,
                    },
                },
            },
            "q1": {},
            "q2": {},
        },
        edges={},
    )

    clock = "q0.ro"
    sched = Schedule("Test schedule")
    operation = Measure("q0", acq_protocol="Trace")
    sched.add(operation)
    dev_sched = compile_circuit_to_device(sched, device_cfg=simple_config)
    with pytest.raises(ValueError) as error:
        _ = set_pulse_and_acquisition_clock(dev_sched, device_cfg=simple_config)

    assert (
        error.value.args[0]
        == f"Operation '{operation}' contains an unknown clock '{clock}'; "
        f"ensure this resource has been added to the schedule "
        f"or to the device config."
    )


def test_reset_operations_compile():
    sched = Schedule("Test schedule")
    sched.add(Reset("q0"))
    sched.add(Reset("q0", "q1"))
    _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_qubit_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(Rxy(90, 0, qubit="q20"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)

    sched = Schedule("Test schedule")
    sched.add(Reset("q2", "q5", "q3"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)

    sched = Schedule("Test schedule")
    sched.add(Reset("q0", "q5"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_edge_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(CZ("q0", "q3"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_operation_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(CNOT("q0", "q1"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


def test_compile_schedule_with_trace_acq_protocol():
    simple_config = DeviceCompilationConfig(
        backend="quantify_scheduler.backends.circuit_to_device"
        + ".compile_circuit_to_device",
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
        },
        elements={
            "q0": {
                "measure": {
                    "factory_func": "quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement",
                    "gate_info_factory_kwargs": [
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SquarePulse",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_channel": 0,
                    },
                },
            },
            "q1": {},
            "q2": {},
        },
        edges={},
    )
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", acq_protocol="Trace"))
    _ = compile_circuit_to_device(sched, device_cfg=simple_config)


def test_compile_schedule_with_invalid_pulse_type_raises():
    simple_config = DeviceCompilationConfig(
        backend="quantify_scheduler.backends.circuit_to_device"
        + ".compile_circuit_to_device",
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
        },
        elements={
            "q0": {
                "measure": {
                    "factory_func": "quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement",
                    "gate_info_factory_kwargs": [
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SoftSquare",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_channel": 0,
                    },
                },
            },
            "q1": {},
            "q2": {},
        },
        edges={},
    )
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", acq_protocol="Trace"))
    with pytest.raises(NotImplementedError):
        _ = compile_circuit_to_device(sched, device_cfg=simple_config)


def test_operation_not_in_config_raises_custom():
    simple_config = DeviceCompilationConfig(
        backend="quantify_scheduler.backends.circuit_to_device"
        + ".compile_circuit_to_device",
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
            "q1.01": 5020000000.0,
            "q1.ro": 6900000000.0,
        },
        elements={"q0": {}, "q1": {}, "q2": {}},
        edges={},
    )

    sched = Schedule("Test missing single q op")
    sched.add(Reset("q0"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=simple_config)

    sched = Schedule("Test schedule mux missing op")
    sched.add(Reset("q0", "q1", "q2"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=simple_config)

    sched = Schedule("Test missing 2Q op")
    sched.add(CZ("q0", "q1"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device(sched, device_cfg=simple_config)


def test_config_with_callables():
    simple_config = DeviceCompilationConfig(
        backend="quantify_scheduler.backends.circuit_to_device"
        + ".compile_circuit_to_device",
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
            "q1.01": 5020000000.0,
            "q1.ro": 6900000000.0,
        },
        elements={
            "q0": {
                "Rxy": OperationCompilationConfig(
                    factory_func=rxy_drag_pulse,
                    gate_info_factory_kwargs=["theta", "phi"],
                    factory_kwargs={
                        "amp180": 0.32,
                        "motzoi": 0.45,
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "duration": 2e-08,
                    },
                ),
                "reset": OperationCompilationConfig(
                    factory_func=IdlePulse,
                    factory_kwargs={"duration": 0.0002},
                ),
            },
            "q1": {
                "reset": OperationCompilationConfig(
                    factory_func=IdlePulse,
                    factory_kwargs={"duration": 0.0002},
                ),
            },
            "q2": {},
        },
        edges={},
    )

    sched = Schedule("Test callable op")
    sched.add(Reset("q0", "q1"))
    _ = compile_circuit_to_device(sched, device_cfg=simple_config)


def test_config_validation():
    # Raises no error
    _ = OperationCompilationConfig(
        factory_func=rxy_drag_pulse,
        gate_info_factory_kwargs=["theta", "phi"],
        factory_kwargs={
            "amp180": 0.32,
            "motzoi": 0.45,
            "port": "q0:mw",
            "clock": "q0.01",
            "duration": 2e-08,
        },
    )
