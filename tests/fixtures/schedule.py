# -----------------------------------------------------------------------------
# Description:    Pytest fixtures for quantify scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import pytest

import quantify.scheduler.schemas.examples as examples
from quantify.scheduler.compilation import (
    add_pulse_information_transmon,
    determine_absolute_timing,
)
from quantify.scheduler.gate_library import CZ, X90, Measure, Reset, Rxy, X
from quantify.scheduler.pulse_library import SquarePulse
from quantify.scheduler.resources import ClockResource
from quantify.scheduler.types import Schedule


@pytest.fixture
def load_example_config() -> Dict[str, Any]:
    def _load_example_config(filename: str = "transmon_test_config.json"):
        examples_path = inspect.getfile(examples)

        file_path = os.path.abspath(os.path.join(examples_path, "..", filename))
        json_str = Path(file_path).read_text()
        return json.loads(json_str)

    yield _load_example_config


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
    schedule = Schedule("Basic schedule")
    schedule.add(Reset("q0"))
    schedule.add(X90("q0"))
    schedule.add(Measure("q0"))
    return schedule


@pytest.fixture
def schedule_with_pulse_info(create_schedule_with_pulse_info) -> Schedule:
    return create_schedule_with_pulse_info()


@pytest.fixture
def chevron() -> Schedule:
    schedule = Schedule("Chevron Experiment")
    for duration in np.linspace(20e-9, 40e-9, 5):
        for amp in np.linspace(0.1, 1.0, 10):
            begin = schedule.add(Reset("q0", "q1"))
            schedule.add(X("q0"), ref_op=begin, ref_pt="start")
            # NB we specify a clock for tutorial purposes,
            # Chevron experiments do not necessarily use modulated square pulses
            square = schedule.add(SquarePulse(amp, duration, "q0:mw", clock="q0.01"))
            schedule.add(X90("q0"), ref_op=square)
            schedule.add(X90("q1"), ref_op=square)
            schedule.add(Measure("q0", "q1"))

    schedule.add_resources(
        [ClockResource("q0.01", 6.02e9)]
    )  # manually add the pulse clock

    return schedule


@pytest.fixture
def T1_experiment() -> Schedule:
    schedule = Schedule("T1 Experiment")
    times = np.arange(0, 100e-6, 3e-6)
    for tau in times:
        schedule.add(Reset("q0"))
        schedule.add(X("q0"), ref_pt="start")
        schedule.add(Measure("q0"), rel_time=tau)

    return schedule


@pytest.fixture
def bell() -> Schedule:
    schedule = Schedule("Bell Experiment")
    for theta in np.linspace(0, 360, 21):
        schedule.add(Reset("q0", "q1"))
        schedule.add(X90("q0"))
        schedule.add(X90("q1"), ref_pt="start")  # this ensures pulses are aligned
        schedule.add(CZ("q0", "q1"))
        schedule.add(Rxy(theta=theta, phi=0, qubit="q0"))
        schedule.add(Measure("q0", "q1"), label=f"M {theta:.2f} deg")

    return schedule
