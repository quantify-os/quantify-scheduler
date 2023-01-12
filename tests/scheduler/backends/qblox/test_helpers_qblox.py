# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring
# pylint: disable=no-self-use

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the helpers module."""

import pytest
from typing import Union

from quantify_scheduler.backends.qblox import constants, helpers


@pytest.mark.parametrize(
    "phase, expected_steps",
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
def test_get_nco_phase_arguments(phase, expected_steps):
    assert helpers.get_nco_phase_arguments(phase) == expected_steps


@pytest.mark.parametrize(
    "clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo, expected_freqs",
    [
        (
            clock_freq,
            lo_freq,
            interm_freq,
            downconverter_freq,
            mix_lo,
            helpers.Frequencies(
                clock=clock_freq
                if downconverter_freq is None or downconverter_freq == 0
                else downconverter_freq - clock_freq,
                LO=clock_freq
                if (
                    mix_lo is False
                    and (downconverter_freq is None or downconverter_freq == 0)
                )
                else downconverter_freq - clock_freq
                if mix_lo is False
                else None
                if interm_freq is None
                else clock_freq - interm_freq
                if downconverter_freq is None or downconverter_freq == 0
                else downconverter_freq - clock_freq - interm_freq,
                IF=None
                if mix_lo is False or lo_freq is None
                else clock_freq - lo_freq
                if downconverter_freq is None or downconverter_freq == 0
                else downconverter_freq - clock_freq - lo_freq,
            ),
        )
        for clock_freq in [100]
        for lo_freq in [None, 20]
        for interm_freq in [None, 3]
        for downconverter_freq in [None, 0, 400]
        for mix_lo in [False, True]
    ],
)
def test_determine_clock_lo_interm_freqs(
    clock_freq: float,
    lo_freq: Union[float, None],
    interm_freq: Union[float, None],
    downconverter_freq: Union[float, None],
    mix_lo: bool,
    expected_freqs: helpers.Frequencies,
):
    assert (
        helpers.determine_clock_lo_interm_freqs(
            clock_freq=clock_freq,
            lo_freq=lo_freq,
            interm_freq=interm_freq,
            downconverter_freq=downconverter_freq,
            mix_lo=mix_lo,
        )
        == expected_freqs
    )
