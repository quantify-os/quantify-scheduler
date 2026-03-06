"""Tests for acquisition delay handling in multiplexed readout."""

import pytest

from quantify_scheduler import Schedule, SerialCompiler
from quantify_scheduler.operations import CZ, Measure, X


def test_mux_ro_apply_acquisition_delay_false(mock_setup_basic_transmon_with_standard_params):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    schedule = Schedule("Test MUX RO")
    schedule.add(Measure("q0", "q1", apply_acquisition_delay=False))
    schedule.add(X("q0"))

    compiled_schedule = SerialCompiler().compile(
        schedule, config=quantum_device.generate_compilation_config()
    )

    schedulables = list(compiled_schedule.schedulables.values())
    assert len(schedulables) == 2  # Measure and X

    measure_schedulable = schedulables[0]
    x_schedulable = schedulables[1]

    measure_schedule = compiled_schedule.operations[measure_schedulable["operation_id"]]
    assert isinstance(measure_schedule, Schedule), (
        f"Expected Schedule, got {type(measure_schedule)}"
    )

    assert "next_operation_delay" in measure_schedule.data

    x_timing_constraint = x_schedulable["timing_constraints"][0]

    assert x_timing_constraint.rel_time < 0, (
        f"Expected negative rel_time for X gate, got {x_timing_constraint.rel_time}"
    )


def test_mux_ro_different_acq_delays(two_qubit_device_different_acq_delays):
    quantum_device = two_qubit_device_different_acq_delays

    schedule = Schedule("Test MUX RO different delays")
    schedule.add(Measure("q0", "q1", apply_acquisition_delay=False))
    schedule.add(X("q0"))

    compiled_schedule = SerialCompiler().compile(
        schedule, config=quantum_device.generate_compilation_config()
    )

    schedulables = list(compiled_schedule.schedulables.values())

    x_schedulable = schedulables[1]
    x_timing_constraint = x_schedulable["timing_constraints"][0]

    assert x_timing_constraint.rel_time == -700e-9, (
        "Expected rel_time of -700ns (based on longest acq_delay), "
        and f"got {x_timing_constraint.rel_time}"
    )


def test_mux_ro_timing_table_verification(two_qubit_device_different_acq_delays):
    quantum_device = two_qubit_device_different_acq_delays

    schedule_false = Schedule("Test MUX RO timing False")
    schedule_false.add(Measure("q0", "q1", apply_acquisition_delay=False))
    schedule_false.add(X("q0"))

    compiled_false = SerialCompiler().compile(
        schedule_false, config=quantum_device.generate_compilation_config()
    )

    schedule_true = Schedule("Test MUX RO timing True")
    schedule_true.add(Measure("q0", "q1", apply_acquisition_delay=True))
    schedule_true.add(X("q0"))

    compiled_true = SerialCompiler().compile(
        schedule_true, config=quantum_device.generate_compilation_config()
    )

    timing_df_false = compiled_false.timing_table.data
    timing_df_true = compiled_true.timing_table.data

    x_rows_false = timing_df_false[
        (timing_df_false["port"] == "q0:mw") & (timing_df_false["is_acquisition"] == False)  # noqa: E712
    ]
    x_rows_true = timing_df_true[
        (timing_df_true["port"] == "q0:mw") & (timing_df_true["is_acquisition"] == False)  # noqa: E712
    ]

    x_start_false = x_rows_false.iloc[0]["abs_time"]
    x_start_true = x_rows_true.iloc[0]["abs_time"]

    timing_difference = x_start_true - x_start_false
    expected_difference = 700e-9  # max acq_delay (q1's acq_delay)

    assert timing_difference == pytest.approx(expected_difference, abs=1e-12), (
        f"X gate should start {expected_difference * 1e9}ns "
        "earlier with apply_acquisition_delay=False, "
        f"but the difference is {timing_difference * 1e9}ns"
    )


def test_mux_ro_timing_with_acq_delay_true(two_qubit_device_different_acq_delays):
    quantum_device = two_qubit_device_different_acq_delays

    schedule = Schedule("Test MUX RO timing with acq delay")
    schedule.add(Measure("q0", "q1", apply_acquisition_delay=True))
    schedule.add(X("q0"))

    compiled_schedule = SerialCompiler().compile(
        schedule, config=quantum_device.generate_compilation_config()
    )

    timing_df = compiled_schedule.timing_table.data

    x_gate_rows = timing_df[
        (timing_df["port"] == "q0:mw") & (timing_df["is_acquisition"] == False)  # noqa: E712
    ]

    assert len(x_gate_rows) > 0, "X gate not found in timing table"

    x_gate_start = x_gate_rows.iloc[0]["abs_time"]
    expected_start = 1700e-9

    assert x_gate_start == pytest.approx(expected_start, abs=1e-12), (
        f"X gate should start at {expected_start * 1e9}ns, but starts at {x_gate_start * 1e9}ns"
    )


def test_two_qubit_gate_after_measurement(two_qubit_device_with_cz):
    quantum_device = two_qubit_device_with_cz

    schedule = Schedule("Test CZ after measurement")
    schedule.add(Measure("q1", apply_acquisition_delay=False))
    schedule.add(X("q1"))
    schedule.add(CZ("q0", "q1"))
    schedule.add(X("q1"))

    compiled_schedule = SerialCompiler().compile(
        schedule, config=quantum_device.generate_compilation_config()
    )

    timing_df = compiled_schedule.timing_table.data

    q1_mw_rows = timing_df[
        (timing_df["port"] == "q1:mw") & (timing_df["is_acquisition"] == False)  # noqa: E712
    ]

    first_x_start = q1_mw_rows.iloc[0]["abs_time"]
    expected_first_x_start = 1000e-9  # integration_time

    assert first_x_start == pytest.approx(expected_first_x_start, abs=1e-12), (
        f"First X gate should start at {expected_first_x_start * 1e9}ns, "
        f"but starts at {first_x_start * 1e9}ns"
    )


@pytest.fixture
def two_qubit_device_with_cz():
    from quantify_scheduler import BasicTransmonElement, CompositeSquareEdge, QuantumDevice

    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QRM"}, "2": {"instrument_type": "QCM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:res"],
                ["cluster0.module1.complex_output_0", "q0:mw"],
                ["cluster0.module1.complex_input_0", "q0:res"],
                ["cluster0.module1.complex_output_0", "q1:res"],
                ["cluster0.module1.complex_output_0", "q1:mw"],
                ["cluster0.module1.complex_input_0", "q1:res"],
                ["cluster0.module2.real_output_0", "q0:fl"],
                ["cluster0.module2.real_output_1", "q1:fl"],
            ]
        },
    }

    q0 = BasicTransmonElement("q0")
    q1 = BasicTransmonElement("q1")
    edge = CompositeSquareEdge("q0", "q1")

    q0.clock_freqs.readout(100e9)
    q0.clock_freqs.f01(100e9)
    q0.measure.acq_delay(500e-9)
    q0.measure.integration_time(1000e-9)
    q0.rxy.motzoi(0.1)
    q0.rxy.duration(100e-9)
    q0.rxy.amp180(0.5)

    q1.clock_freqs.readout(100e9)
    q1.clock_freqs.f01(100e9)
    q1.measure.acq_delay(700e-9)
    q1.measure.integration_time(1000e-9)
    q1.rxy.motzoi(0.1)
    q1.rxy.duration(100e-9)
    q1.rxy.amp180(0.5)

    edge.cz.square_amp(0.5)
    edge.cz.square_duration(300e-9)

    device = QuantumDevice("test_device_cz")
    device.hardware_config(hw_config)
    device.add_element(q0)
    device.add_element(q1)
    device.add_edge(edge)

    yield device

    device.close()


@pytest.fixture
def two_qubit_device_different_acq_delays():
    from quantify_scheduler import BasicTransmonElement, QuantumDevice

    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QRM"}, "2": {"instrument_type": "QCM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:res"],
                ["cluster0.module1.complex_output_0", "q0:mw"],
                ["cluster0.module1.complex_input_0", "q0:res"],
                ["cluster0.module1.complex_output_0", "q1:res"],
                ["cluster0.module1.complex_output_0", "q1:mw"],
                ["cluster0.module1.complex_input_0", "q1:res"],
                ["cluster0.module2.real_output_0", "q0:fl"],
                ["cluster0.module2.real_output_1", "q1:fl"],
            ]
        },
    }

    q0 = BasicTransmonElement("q0")
    q1 = BasicTransmonElement("q1")

    q0.clock_freqs.readout(100e9)
    q0.clock_freqs.f01(100e9)
    q0.measure.acq_delay(500e-9)
    q0.measure.integration_time(1000e-9)
    q0.rxy.motzoi(0.1)
    q0.rxy.duration(100e-9)
    q0.rxy.amp180(0.5)

    q1.clock_freqs.readout(100e9)
    q1.clock_freqs.f01(100e9)
    q1.measure.acq_delay(700e-9)
    q1.measure.integration_time(1000e-9)
    q1.rxy.motzoi(0.1)
    q1.rxy.duration(100e-9)
    q1.rxy.amp180(0.5)

    device = QuantumDevice("test_device")
    device.hardware_config(hw_config)
    device.add_element(q0)
    device.add_element(q1)

    yield device

    device.close()
