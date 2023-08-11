# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring
# pylint: disable=no-self-use

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the helpers module."""

import math
from contextlib import nullcontext
from typing import Union

import pytest

from quantify_scheduler.backends.qblox import helpers
from quantify_scheduler.backends.qblox.enums import IoMode
from quantify_scheduler.backends.qblox.instrument_compilers import (
    QcmModule,
    QcmRfModule,
    QrmModule,
    QrmRfModule,
)


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
def test_invalid_get_nco_set_frequency_arguments(frequency: float):
    with pytest.raises(ValueError):
        helpers.get_nco_set_frequency_arguments(frequency)


def __get_frequencies(
    clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo
) -> Union[helpers.Frequencies, str]:
    if downconverter_freq is None or downconverter_freq == 0:
        freqs = helpers.Frequencies(clock=clock_freq)
    else:
        freqs = helpers.Frequencies(clock=downconverter_freq - clock_freq)

    if mix_lo is False:
        freqs.LO = freqs.clock
        freqs.IF = interm_freq
    else:
        if lo_freq is None and interm_freq is None:
            return "underconstrained"
        elif lo_freq is None and interm_freq is not None:
            freqs.IF = interm_freq
            freqs.LO = freqs.clock - interm_freq
        elif lo_freq is not None and interm_freq is None:
            freqs.IF = freqs.clock - lo_freq
            freqs.LO = lo_freq
        elif lo_freq is not None and interm_freq is not None:
            if math.isclose(freqs.clock, lo_freq + interm_freq):
                freqs.IF = interm_freq
                freqs.LO = lo_freq
            else:
                return "overconstrained"
    return freqs


@pytest.mark.parametrize(
    "clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo, expected_freqs",
    [  # General test cases with positive frequencies
        (
            clock_freq := 100,
            lo_freq,
            interm_freq,
            downconverter_freq,
            mix_lo,
            __get_frequencies(
                clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo
            ),
        )
        for lo_freq in [None, 20]
        for interm_freq in [None, 3]
        for downconverter_freq in [None, 0, 400]
        for mix_lo in [False, True]
    ]
    + [  # Test cases with negative frequencies
        (
            clock_freq,
            lo_freq := -200,
            interm_freq := -30,
            downconverter_freq := 400,
            mix_lo,
            __get_frequencies(
                clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo
            ),
        )
        for clock_freq in [-100, 100]
        for mix_lo in [False, True]
    ]
    + [  # Test cases for downconverter_freq
        (
            clock_freq := 100,
            lo_freq := None,
            interm_freq := None,
            downconverter_freq,
            mix_lo := True,
            __get_frequencies(
                clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo
            ),
        )
        for downconverter_freq in [0, clock_freq - 1, -400]
    ]
    + [  # Test cases for float("nan")
        (
            clock_freq := 100,
            lo_freq := float("nan"),
            interm_freq := 5,
            downconverter_freq := None,
            mix_lo := True,
            helpers.Frequencies(clock=100, LO=95, IF=5),
        )
    ],
)
def test_determine_clock_lo_interm_freqs(
    clock_freq: float,
    lo_freq: Union[float, None],
    interm_freq: Union[float, None],
    downconverter_freq: Union[float, None],
    mix_lo: bool,
    expected_freqs: Union[helpers.Frequencies, str],
):
    freqs = helpers.Frequencies(clock=clock_freq, LO=lo_freq, IF=interm_freq)
    context_mngr = nullcontext()
    if (
        downconverter_freq is not None
        and (downconverter_freq < 0 or downconverter_freq < clock_freq)
        or expected_freqs in ("overconstrained", "underconstrained")
    ):
        context_mngr = pytest.raises(ValueError)

    with context_mngr as error:
        assert (
            helpers.determine_clock_lo_interm_freqs(
                freqs=freqs,
                downconverter_freq=downconverter_freq,
                mix_lo=mix_lo,
            )
            == expected_freqs
        )
    if error is not None:
        possible_errors = []
        if expected_freqs == "underconstrained":
            possible_errors.append(
                f"Frequency settings underconstrained for {freqs.clock=}."
                f" Neither LO nor IF supplied ({freqs.LO=}, {freqs.IF=})."
            )
        elif expected_freqs == "overconstrained":
            possible_errors.append(
                f"Frequency settings overconstrained."
                f" {freqs.clock=} must be equal to {freqs.LO=}+{freqs.IF=} if both are supplied."
            )
        if downconverter_freq is not None:
            if downconverter_freq < 0:
                possible_errors.append(
                    f"Downconverter frequency must be positive "
                    f"({downconverter_freq=:e})"
                )
            elif downconverter_freq < clock_freq:
                possible_errors.append(
                    "Downconverter frequency must be greater than clock frequency "
                    f"({downconverter_freq=:e}, {clock_freq=:e})"
                )
        assert str(error.value) in possible_errors


def test_Frequencies():
    freq = helpers.Frequencies(clock=100, LO=float("nan"), IF=float("nan"))
    freq.validate()
    assert freq.LO is None
    assert freq.IF is None

    invalid_freqs = [
        helpers.Frequencies(clock=100, LO=None, IF=None),
        helpers.Frequencies(clock=100, LO=None, IF=None),
        helpers.Frequencies(clock=None, LO=None, IF=None),
    ]
    invalid_freqs[0].LO = float("nan")
    invalid_freqs[1].IF = float("nan")

    for freq in invalid_freqs:
        with pytest.raises(ValueError):
            freq.validate()


@pytest.mark.parametrize(
    "module",
    [
        QcmModule,
        QrmModule,
        QcmRfModule,
        QrmRfModule,
    ],
)
def test_validate_io_indices(module):
    def _validate_io_indices(io_name: str, io_indices: tuple) -> tuple:
        assert (
            len(io_indices) > 0
        ), "No inputs or output indices were selected for this sequencer."

        assert (
            len(io_indices) <= 2
        ), f"Too many ios specified for this channel. Given: {io_indices}"

        if len(io_indices) == 2:
            assert (
                "complex" in io_name
            ), f"Two io indices specified for {io_name}, but it must have one."
            assert sorted(io_indices) in (
                [0, 1],
                [2, 3],
            ), "Attempting to use two paths belonging to different sequencers."

        elif len(io_indices) == 1:
            assert (
                "complex" not in io_name
            ), f"Only one io index specified for {io_name}, but it must have two."

        return

    for io_name in module.static_hw_properties.valid_ios:
        io_indices = (
            helpers.output_name_to_output_indices(io_name)
            if "output" in io_name
            else helpers.input_name_to_input_indices(io_name)
        )
        _validate_io_indices(io_name, io_indices)


@pytest.mark.parametrize(
    "io_name, sequencer_mode",
    [
        ("complex_output_0", IoMode.COMPLEX),
        ("real_output_0", IoMode.REAL),
        ("real_output_1", IoMode.IMAG),
        ("digital_output_0", IoMode.DIGITAL),
    ],
)
def test_validate_sequencer_mode(io_name, sequencer_mode):
    assert sequencer_mode == helpers.get_io_info(io_name)[0]
