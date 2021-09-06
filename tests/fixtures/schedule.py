# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Pytest fixtures for quantify-scheduler."""
from __future__ import annotations


import inspect
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
from quantify_scheduler.schemas import examples
from quantify_scheduler.compilation import (
    add_pulse_information_transmon,
    determine_absolute_timing,
)
from quantify_scheduler.gate_library import X, X90, Measure, Reset
from quantify_scheduler.types import Schedule
from quantify_scheduler.compilation import qcompile


@pytest.fixture
def load_example_config() -> Dict[str, Any]:
    def _load_example_config(filename: str = "transmon_test_config.json"):
        examples_path = inspect.getfile(examples)

        file_path = os.path.abspath(os.path.join(examples_path, "..", filename))
        json_str = Path(file_path).read_text()
        return json.loads(json_str)

    yield _load_example_config


@pytest.fixture
def load_example_hardware_config() -> Dict[str, Any]:
    def _load_example_hardware_config(filename: str = "qblox_test_mapping.json"):
        examples_path = inspect.getfile(examples)

        file_path = os.path.abspath(os.path.join(examples_path, "..", filename))
        json_str = Path(file_path).read_text()
        return json.loads(json_str)

    yield _load_example_hardware_config


@pytest.fixture
def create_schedule_with_pulse_info(load_example_config, basic_schedule: Schedule):
    def _create_schedule_with_pulse_info(
        schedule: Optional[Schedule] = None, device_config: Optional[dict] = None
    ) -> Schedule:
        _schedule = schedule if schedule is not None else deepcopy(basic_schedule)
        _device_config = (
            device_config if device_config is not None else load_example_config()
        )
        add_pulse_information_transmon(_schedule, _device_config)
        determine_absolute_timing(_schedule)
        return _schedule

    yield _create_schedule_with_pulse_info


@pytest.fixture
def empty_schedule() -> Schedule:
    return Schedule("Empty Experiment")


@pytest.fixture
def basic_schedule() -> Schedule:
    schedule = Schedule("Basic schedule")
    schedule.add(X90("q0"))
    return schedule


@pytest.fixture
def schedule_with_measurement() -> Schedule:
    """
    Simple schedule with gate an measurement on qubit 0.
    """
    schedule = Schedule("Basic schedule")
    schedule.add(Reset("q0"))
    schedule.add(X90("q0"))
    schedule.add(Measure("q0"))
    return schedule


@pytest.fixture
def schedule_with_measurement_q2() -> Schedule:
    """
    Simple schedule with gate an measurement on qubit 2.
    """
    schedule = Schedule("Basic schedule")
    schedule.add(Reset("q2"))
    schedule.add(X90("q2"))
    schedule.add(Measure("q2"))
    return schedule


@pytest.fixture
def schedule_with_pulse_info(create_schedule_with_pulse_info) -> Schedule:
    return create_schedule_with_pulse_info()


@pytest.fixture
def compiled_two_qubit_t1_schedule(load_example_config):
    """
    a schedule performing T1 on two-qubits simultaneously
    """
    device_config = load_example_config()

    qubits = ["q0", "q1"]
    repetitions = 1024
    schedule = Schedule("Multi-qubit T1", repetitions)

    times = np.arange(0, 60e-6, 3e-6)

    for i, tau in enumerate(times):
        schedule.add(Reset(qubits[0], qubits[1]), label=f"Reset {i}")
        schedule.add(X(qubits[0]), label=f"pi {i} {qubits[0]}")
        schedule.add(X(qubits[1]), label=f"pi {i} {qubits[1]}", ref_pt="start")

        schedule.add(
            Measure(qubits[0], acq_index=i, acq_channel=0),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {qubits[0]}{i}",
        )
        schedule.add(
            Measure(qubits[1], acq_index=i, acq_channel=1),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {qubits[1]}{i}",
        )

    comp_t1_sched = qcompile(schedule, device_config)
    return comp_t1_sched
