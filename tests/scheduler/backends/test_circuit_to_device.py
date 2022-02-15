# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends.circuit_to_device import (
    compile_circuit_to_device,
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
    sched.add(Measure(q0, q1), label="M0")

    # test that all these operations compile correctly.
    _ = compile_circuit_to_device(sched, device_cfg=example_transmon_cfg)


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


def test_compile_schedule_with_manually_added_clock():
    sched = Schedule("Test schedule")
    sched.add_resources([ClockResource(name="q0.01", freq=5e9)])
    sched.add(X("q0"))
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
                    "gate_info_factory_kwargs": ["acq_index", "bin_mode"],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SquarePulse",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_protocol": "Trace",
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
    sched.add(Measure("q0"))
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
                    "gate_info_factory_kwargs": ["acq_index", "bin_mode"],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SoftSquare",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_protocol": "Trace",
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
    sched.add(Measure("q0"))
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
