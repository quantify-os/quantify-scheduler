# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the helpers module."""

import math
from contextlib import nullcontext
from typing import Union

import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.qblox import helpers
from quantify_scheduler.backends.qblox.instrument_compilers import (
    QRMCompiler,
)
from quantify_scheduler.backends.qblox.qblox_hardware_config_old_style import (
    hardware_config as qblox_hardware_config_old_style,
)
from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig
from quantify_scheduler.backends.types.qblox import (
    BasebandModuleSettings,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.helpers.collections import find_all_port_clock_combinations
from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.schedules.schedule import Schedule


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


def test_generate_legacy_hardware_config(hardware_compilation_config_qblox_example):
    sched = Schedule("All portclocks schedule")
    quantum_device = QuantumDevice("All_portclocks_device")
    quantum_device.hardware_config(hardware_compilation_config_qblox_example)

    qubits = {}
    sched = Schedule("All portclocks schedule")
    for port, clock in find_all_port_clock_combinations(
        qblox_hardware_config_old_style
    ):
        sched.add(SquarePulse(port=port, clock=clock, amp=0.25, duration=12e-9))
        if (qubit_name := port.split(":")[0]) not in quantum_device.elements():
            qubits[qubit_name] = BasicTransmonElement(qubit_name)
            quantum_device.add_element(qubits[qubit_name])

    generated_hw_config = helpers._generate_legacy_hardware_config(
        schedule=sched, compilation_config=quantum_device.generate_compilation_config()
    )

    assert generated_hw_config == qblox_hardware_config_old_style


def test_preprocess_legacy_hardware_config(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module3": {
                "instrument_type": "QRM",
                "complex_output_0": {
                    "lo_name": "lo",
                    "portclock_configs": [
                        {
                            "port": "q0:res",
                            "clock": "q0.ro",
                            "init_offset_awg_path_0": 0.1,
                            "init_offset_awg_path_1": -0.1,
                            "init_gain_awg_path_0": 0.55,
                            "init_gain_awg_path_1": 0.66,
                        },
                        {
                            "port": "q1:res",
                            "clock": "q1.ro",
                        },
                    ],
                },
            },
        },
        "lo": {
            "instrument_type": "LocalOscillator",
            "frequency": 7.2e9,
            "power": 1,
        },
    }

    mock_setup = mock_setup_basic_transmon_with_standard_params

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_config)

    schedule = Schedule("Thresholded acquisition")
    schedule.add(Measure("q0", "q1", acq_protocol="ThresholdedAcquisition"))

    compiler = SerialCompiler("compiler", quantum_device=quantum_device)
    with pytest.warns(FutureWarning, match="init_"):
        compiled_schedule = compiler.compile(schedule)

    for key in [
        "init_offset_awg_path_I",
        "init_offset_awg_path_Q",
        "init_gain_awg_path_I",
        "init_gain_awg_path_Q",
    ]:
        assert (
            key
            in compiled_schedule.compiled_instructions["cluster0"]["cluster0_module3"][
                "sequencers"
            ]["seq0"]
        )


def test_configure_input_gains_overwrite_gain():
    # Partial test of overwriting gain setting. Note: In using the new
    # QbloxHardwareOptions collisions like these are no longer possible,
    # so after migrating to the new-style hardware compilation config
    # this test can be removed

    instrument_cfg = {
        "instrument_type": "QRM",
        "real_output_1": {
            "input_gain_1": 10,
            "portclock_configs": [
                {"port": "q0:res", "clock": "q0.ro"},
            ],
        },
    }

    test_module = QRMCompiler(
        parent=None,
        name="tester",
        total_play_time=1,
        instrument_cfg=instrument_cfg,
    )

    test_module._settings = BasebandModuleSettings.extract_settings_from_mapping(
        instrument_cfg
    )
    test_module._settings.in1_gain = 5

    with pytest.raises(ValueError) as error:
        test_module._configure_input_gains()

    assert (
        str(error.value)
        == "Overwriting gain of real_output_1 of module tester to in1_gain: 10."
        "\nIt was previously set to in1_gain: 5."
    )


def test_find_channel_names(hardware_cfg_rf_legacy):
    assert helpers.find_channel_names(
        hardware_cfg_rf_legacy["cluster0"]["cluster0_module2"]
    ) == [
        "complex_output_0",
        "complex_output_1",
    ]


def test_generate_new_style_hardware_compilation_config(
    hardware_compilation_config_qblox_example,
):
    parsed_new_style_config = QbloxHardwareCompilationConfig.model_validate(
        hardware_compilation_config_qblox_example
    )

    converted_new_style_hw_cfg = QbloxHardwareCompilationConfig.model_validate(
        qblox_hardware_config_old_style
    )

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
    assert (
        converted_new_style_hw_cfg.model_dump() == parsed_new_style_config.model_dump()
    )
