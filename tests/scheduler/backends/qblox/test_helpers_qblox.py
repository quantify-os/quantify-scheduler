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


def __get_frequencies(
    clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo
) -> helpers.Frequencies:
    freqs = helpers.Frequencies()
    if downconverter_freq is None or downconverter_freq == 0:
        freqs.clock = clock_freq
    else:
        freqs.clock = downconverter_freq - clock_freq

    freqs.LO = lo_freq
    if mix_lo is False:
        freqs.LO = freqs.clock
    elif interm_freq is not None:
        freqs.LO = freqs.clock - interm_freq

    freqs.IF = interm_freq
    if mix_lo is False:
        freqs.IF = None
    elif lo_freq is not None:
        freqs.IF = freqs.clock - lo_freq

    return freqs


@pytest.mark.parametrize(
    "frequency, expected_steps",
    [
        (-500e6, -2000000000),
        (-200e3, -800000),
        (0.0, 0),
        (200e3, 800000),
        (500e6, 2000000000),
    ],
)
def test_get_nco_set_frequency_arguments(frequency: float, expected_steps: int):
    assert helpers.get_nco_set_frequency_arguments(frequency) == expected_steps


@pytest.mark.parametrize("frequency", [-500e6 - 1, 500e6 + 1])
def test_invalid_get_nco_set_frequency(frequency: float):
    with pytest.raises(ValueError):
        helpers.get_nco_set_frequency_arguments(frequency)


def __get_frequencies(
    clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo
) -> helpers.Frequencies:
    freqs = helpers.Frequencies()
    if downconverter_freq is None or downconverter_freq == 0:
        freqs.clock = clock_freq
    else:
        freqs.clock = downconverter_freq - clock_freq

    freqs.LO = lo_freq
    if mix_lo is False:
        freqs.LO = freqs.clock
    elif interm_freq is not None:
        freqs.LO = freqs.clock - interm_freq

    freqs.IF = interm_freq
    if mix_lo is False:
        freqs.IF = None
    elif lo_freq is not None:
        freqs.IF = freqs.clock - lo_freq

    return freqs


@pytest.mark.parametrize(
    "clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo, expected_freqs",
    [
        (
            clock_freq,
            lo_freq,
            interm_freq,
            downconverter_freq,
            mix_lo,
            __get_frequencies(
                clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo
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
