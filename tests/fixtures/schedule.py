# -----------------------------------------------------------------------------
# Description:    Pytest fixtures for quantify scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations

import inspect
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import quantify.scheduler.schemas.examples as examples
from quantify.scheduler.compilation import (
    add_pulse_information_transmon,
    determine_absolute_timing,
)
from quantify.scheduler.gate_library import X90, Measure, Reset
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
