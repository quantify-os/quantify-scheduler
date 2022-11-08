# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring
# pylint: disable=no-self-use

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the helpers module."""

import pytest

from quantify_scheduler.backends.qblox import helpers
from quantify_scheduler.backends.qblox import constants


@pytest.mark.parametrize(
    "phase_expected_steps",
    [
        (0.0, 0),
        (360.0, 0),
        (10.0, 27777778),
        (11.11, 30861111),
        (123.123, 342008333),
        (90.0, 250000000),
        (-90.0, 750000000),
        (480.2, 333888889),
    ],
)
def test_get_nco_phase_arguments(phase_expected_steps):
    phase, expected_steps = phase_expected_steps
    steps = helpers.get_nco_phase_arguments(phase)
    assert steps == expected_steps


@pytest.mark.parametrize(
    "frequency_expected_steps", [(-500e6, -2000000000), (-200e3, -800000), (0.0, 0), (200e3, 800000), (500e6, 2000000000)]
)
def test_get_nco_set_frequency(frequency_expected_steps: tuple):
    frequency, expected_steps = frequency_expected_steps
    steps = helpers.get_nco_set_frequency_arguments(frequency)

    assert steps == expected_steps


@pytest.mark.parametrize(
    "frequency", [-500e6 - 1, 500e6 + 1]
)
def test_invalid_get_nco_set_frequency(frequency: float):
    with pytest.raises(ValueError):
        helpers.get_nco_set_frequency_arguments(frequency)

