# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pytest fixtures for quantify-scheduler."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Generator, Dict, List, Optional

import numpy as np
import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.circuit_to_device import (
    DeviceCompilationConfig,
    compile_circuit_to_device_with_config_validation,
)
from quantify_scheduler.backends.graph_compilation import SerialCompilationConfig
from quantify_scheduler.compilation import _determine_absolute_timing, flatten_schedule
from quantify_scheduler.operations.gate_library import CZ, Measure, Reset, X, X90
from quantify_scheduler.schemas.examples import utils
from quantify_scheduler.schemas.examples.device_example_cfgs import (
    example_transmon_cfg,
)


ZHINST_HARDWARE_COMPILATION_CONFIG = utils.load_json_example_scheme(
    "zhinst_hardware_compilation_config.json"
)


@pytest.fixture
def device_cfg_transmon_example() -> Generator[DeviceCompilationConfig, None, None]:
    """
    Circuit to device level compilation for the circuit_to_device
    compilation backend.
    """
    yield DeviceCompilationConfig.model_validate(example_transmon_cfg)


@pytest.fixture
def hardware_compilation_config_zhinst_example() -> (
    Generator[Dict[str, Any], None, None]
):
    yield dict(ZHINST_HARDWARE_COMPILATION_CONFIG)


@pytest.fixture
def create_schedule_with_pulse_info(
    device_cfg_transmon_example, basic_schedule: Schedule
):
    def _create_schedule_with_pulse_info(
        schedule: Optional[Schedule] = None, device_config: Optional[dict] = None
    ) -> Schedule:
        _schedule = schedule if schedule is not None else deepcopy(basic_schedule)
        _device_config = (
            device_config if device_config is not None else device_cfg_transmon_example
        )
        _schedule = compile_circuit_to_device_with_config_validation(
            schedule=_schedule,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=_device_config
            ),
        )
        _schedule = _determine_absolute_timing(schedule=_schedule, time_unit="physical")
        _schedule = flatten_schedule(schedule=_schedule)
        return _schedule

    yield _create_schedule_with_pulse_info


@pytest.fixture
def empty_schedule() -> Schedule:
    return Schedule("Empty Experiment")


@pytest.fixture
def basic_schedule(make_basic_schedule) -> Schedule:
    return make_basic_schedule("q0")


@pytest.fixture
def make_basic_schedule() -> Callable[[str], Schedule]:
    def _make_basic_schedule(qubit: str) -> Schedule:
        schedule = Schedule(f'Basic schedule{" "+qubit if qubit != "q0" else ""}')
        schedule.add(X90(qubit))
        return schedule

    return _make_basic_schedule


@pytest.fixture
def make_basic_multi_qubit_schedule() -> Callable[[List[str]], Schedule]:
    def _make_basic_schedule(qubits: List[str]) -> Schedule:
        schedule = Schedule(f"Basic schedule {qubits}")
        for qubit in qubits:
            schedule.add(X90(qubit))
        return schedule

    return _make_basic_schedule


@pytest.fixture
def schedule_with_measurement(make_schedule_with_measurement) -> Schedule:
    """
    Simple schedule with gate and measurement on qubit 0.
    """
    return make_schedule_with_measurement("q0")


@pytest.fixture
def schedule_with_measurement_q2(make_schedule_with_measurement) -> Schedule:
    """
    Simple schedule with gate and measurement on qubit 2.
    """
    return make_schedule_with_measurement("q2")


@pytest.fixture
def make_schedule_with_measurement() -> Callable[[str], Schedule]:
    """
    Simple schedule with gate and measurement on single qubit.
    """

    def _make_schedule_with_measurement(qubit: str):
        schedule = Schedule(f"Schedule with measurement {qubit}")
        schedule.add(Reset(qubit))
        schedule.add(X90(qubit))
        schedule.add(Measure(qubit))
        return schedule

    return _make_schedule_with_measurement


@pytest.fixture
def two_qubit_gate_schedule():
    sched = Schedule("two_qubit_gate_schedule")
    sched.add(Reset("q2", "q3"))
    sched.add(CZ(qC="q2", qT="q3"))
    return sched


@pytest.fixture
def schedule_with_pulse_info(create_schedule_with_pulse_info) -> Schedule:
    return create_schedule_with_pulse_info()


@pytest.fixture
def compiled_two_qubit_t1_schedule(mock_setup_basic_transmon_with_standard_params):
    """
    a schedule performing T1 on two-qubits simultaneously
    """
    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["q0"].measure.acq_channel(0)
    mock_setup["q1"].measure.acq_channel(1)

    q0, q1 = ("q0", "q1")
    repetitions = 1024
    schedule = Schedule("Multi-qubit T1", repetitions)

    times = np.arange(0, 60e-6, 3e-6)
    for i, tau in enumerate(times):
        schedule.add(Reset(q0, q1), label=f"Reset {i}")
        schedule.add(X(q0), label=f"pi {i} {q0}")
        schedule.add(X(q1), label=f"pi {i} {q1}", ref_pt="start")

        schedule.add(
            Measure(q0, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q0}{i}",
        )
        schedule.add(
            Measure(q1, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q1}{i}",
        )

    compiler = SerialCompiler(name="compiler")
    comp_t1_sched = compiler.compile(
        schedule, config=mock_setup["quantum_device"].generate_compilation_config()
    )
    return comp_t1_sched
