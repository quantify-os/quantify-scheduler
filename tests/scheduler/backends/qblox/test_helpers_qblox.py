# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the helpers module."""
from __future__ import annotations

import math
from contextlib import nullcontext

import pytest

from quantify_scheduler.backends.qblox import helpers
from quantify_scheduler.backends.qblox.hardware_config_nv_center_old_style import (
    hardware_config as hardware_config_nv_center_old_style,
)
from quantify_scheduler.backends.qblox.hardware_config_transmon_old_style import (
    hardware_config as hardware_config_transmon_old_style,
)
from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig
from quantify_scheduler.schemas.examples import utils

QBLOX_HARDWARE_CONFIG_TRANSMON = utils.load_json_example_scheme(
    "qblox_hardware_config_transmon.json"
)
QBLOX_HARDWARE_CONFIG_NV_CENTER = utils.load_json_example_scheme(
    "qblox_hardware_config_nv_center.json"
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
) -> helpers.ValidatedFrequencies | str:
    if downconverter_freq is None or downconverter_freq == 0:
        freqs = helpers.Frequencies(clock=clock_freq)
    else:
        freqs = helpers.Frequencies(clock=downconverter_freq - clock_freq)

    if mix_lo is False:
        freqs.LO = freqs.clock
        if interm_freq is None:
            return "underconstrained"
        freqs.IF = interm_freq
    elif lo_freq is None and interm_freq is None:
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
    return helpers.ValidatedFrequencies(clock=freqs.clock, IF=freqs.IF, LO=freqs.LO)


@pytest.mark.filterwarnings(r"ignore:Overriding freqs.LO.*")
@pytest.mark.filterwarnings(r"ignore:Downconverter frequency 0 supplied*")
@pytest.mark.parametrize(
    "clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo, expected_freqs",
    [  # General test cases with positive frequencies
        (
            clock_freq := 100,
            lo_freq,
            interm_freq,
            downconverter_freq,
            mix_lo,
            __get_frequencies(clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo),
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
            __get_frequencies(clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo),
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
            __get_frequencies(clock_freq, lo_freq, interm_freq, downconverter_freq, mix_lo),
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
            helpers.ValidatedFrequencies(clock=100, LO=95, IF=5),
        )
    ],
)
def test_determine_clock_lo_interm_freqs(
    clock_freq: float,
    lo_freq: float | None,
    interm_freq: float | None,
    downconverter_freq: float | None,
    mix_lo: bool,
    expected_freqs: helpers.Frequencies | str,
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
            if mix_lo:
                possible_errors.append(
                    f"Frequency settings underconstrained for {freqs.clock=}."
                    f" Neither LO nor IF supplied ({freqs.LO=}, {freqs.IF=})."
                )
            else:
                possible_errors.append(
                    f"Frequency settings underconstrained for {freqs.clock=}. "
                    "If mix_lo=False is specified, the IF must also be supplied "
                    f"({freqs.IF=})."
                )
        elif expected_freqs == "overconstrained":
            possible_errors.append(
                f"Frequency settings overconstrained."
                f" {freqs.clock=} must be equal to {freqs.LO=}+{freqs.IF=} when both are supplied."
            )
        if downconverter_freq is not None:
            if downconverter_freq < 0:
                possible_errors.append(
                    f"Downconverter frequency must be positive " f"({downconverter_freq=:e})"
                )
            elif downconverter_freq < clock_freq:
                possible_errors.append(
                    "Downconverter frequency must be greater than clock frequency "
                    f"({downconverter_freq=:e}, {clock_freq=:e})"
                )
        assert str(error.value) in possible_errors


def test_frequencies():
    freq = helpers.Frequencies(clock=100, LO=float("nan"), IF=float("nan"))
    assert freq.LO is None
    assert freq.IF is None

    with pytest.raises(ValueError):
        helpers.Frequencies(clock=None, LO=None, IF=None)


# Using the old-style / legacy hardware config dict is deprecated
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
@pytest.mark.parametrize(
    "new_style_config, old_style_config",
    [
        (QBLOX_HARDWARE_CONFIG_TRANSMON, hardware_config_transmon_old_style),
        (QBLOX_HARDWARE_CONFIG_NV_CENTER, hardware_config_nv_center_old_style),
    ],
)
def test_generate_new_style_hardware_config(new_style_config, old_style_config):
    parsed_new_style_config = QbloxHardwareCompilationConfig.model_validate(new_style_config)

    converted_new_style_hw_cfg = QbloxHardwareCompilationConfig.model_validate(old_style_config)

    # For nv-center config, transmon config still in version "0.1"
    if parsed_new_style_config.version == "0.2":
        converted_new_style_hw_cfg.version = "0.2"

    # drop cross talk compensation as old style config does not support it
    parsed_new_style_config.hardware_options.crosstalk = None

    # Partial checks
    # HardwareDescription
    assert (
        converted_new_style_hw_cfg.model_dump()["hardware_description"]
        == parsed_new_style_config.model_dump()["hardware_description"]
    )
    # Connectivity
    assert list(converted_new_style_hw_cfg.connectivity.graph.edges) == list(
        parsed_new_style_config.connectivity.graph.edges
    )
    # HardwareOptions
    assert (
        converted_new_style_hw_cfg.model_dump()["hardware_options"]
        == parsed_new_style_config.model_dump()["hardware_options"]
    )

    # Write to dict to check equality of full config contents:
    assert converted_new_style_hw_cfg.model_dump() == parsed_new_style_config.model_dump()
