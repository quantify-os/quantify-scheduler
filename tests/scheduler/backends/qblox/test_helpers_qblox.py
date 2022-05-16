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
    "phase", [0.0, 360.0, 10.0, 11.11, 123.123, 90.0, -90.0, 480.2]
)
def test_get_nco_phase_arguments(phase: float):
    phase_coarse, phase_fine, phase_ultra_fine = helpers.get_nco_phase_arguments(phase)

    phase_shift_returned = (
        phase_coarse * constants.NCO_PHASE_DEG_STEP_COARSE
        + phase_fine * constants.NCO_PHASE_DEG_STEP_FINE
        + phase_ultra_fine * constants.NCO_PHASE_DEG_STEP_U_FINE
    )

    assert phase_coarse <= constants.NCO_PHASE_NUM_STEP_COARSE
    assert phase_fine <= constants.NCO_PHASE_NUM_STEP_FINE
    assert phase_ultra_fine <= constants.NCO_PHASE_NUM_STEP_U_FINE

    expected_phase = phase % 360

    # approx due to floating point rounding errors
    assert phase_shift_returned == pytest.approx(expected_phase)
