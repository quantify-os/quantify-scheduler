# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Pytest fixtures for quantify-scheduler."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.compilation import (
    add_pulse_information_transmon,
    determine_absolute_timing,
    qcompile,
)
from quantify_scheduler.operations.gate_library import X90, Measure, Reset, X
from quantify_scheduler.schemas.examples import utils

# load here to avoid loading every time a fixture is used
DEVICE_CONFIG = utils.load_json_example_scheme("transmon_test_config.json")
QBLOX_HARDWARE_MAPPING = utils.load_json_example_scheme("qblox_test_mapping.json")
ZHINST_HARDWARE_MAPPING = utils.load_json_example_scheme("zhinst_test_mapping.json")


@pytest.fixture
def load_example_transmon_config() -> Dict[str, Any]:
    def _load_example_transmon_config():
        return dict(DEVICE_CONFIG)

    yield _load_example_transmon_config


@pytest.fixture
def load_example_qblox_hardware_config() -> Dict[str, Any]:
    def _load_example_qblox_hardware_config():
        return dict(QBLOX_HARDWARE_MAPPING)

    yield _load_example_qblox_hardware_config


@pytest.fixture
def load_example_zhinst_hardware_config() -> Dict[str, Any]:
    def _load_example_zhinst_hardware_config():
        return dict(ZHINST_HARDWARE_MAPPING)

    yield _load_example_zhinst_hardware_config


@pytest.fixture
def create_schedule_with_pulse_info(
    load_example_transmon_config, basic_schedule: Schedule
):
    def _create_schedule_with_pulse_info(
        schedule: Optional[Schedule] = None, device_config: Optional[dict] = None
    ) -> Schedule:
        _schedule = schedule if schedule is not None else deepcopy(basic_schedule)
        _device_config = (
            device_config
            if device_config is not None
            else load_example_transmon_config()
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
def compiled_two_qubit_t1_schedule(load_example_transmon_config):
    """
    a schedule performing T1 on two-qubits simultaneously
    """
    device_config = load_example_transmon_config()

    q0, q1 = ("q0", "q1")
    repetitions = 1024
    schedule = Schedule("Multi-qubit T1", repetitions)

    times = np.arange(0, 60e-6, 3e-6)

    for i, tau in enumerate(times):
        schedule.add(Reset(q0, q1), label=f"Reset {i}")
        schedule.add(X(q0), label=f"pi {i} {q0}")
        schedule.add(X(q1), label=f"pi {i} {q1}", ref_pt="start")

        schedule.add(
            Measure(q0, acq_index=i, acq_channel=0),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q0}{i}",
        )
        schedule.add(
            Measure(q1, acq_index=i, acq_channel=1),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q1}{i}",
        )

    comp_t1_sched = qcompile(schedule, device_config)
    return comp_t1_sched
