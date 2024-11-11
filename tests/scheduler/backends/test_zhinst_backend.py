from __future__ import annotations

import json
from copy import deepcopy
from textwrap import dedent
from typing import Any, Generator
from unittest.mock import ANY, call

import numpy as np
import pytest
from networkx import relabel_nodes
from pydantic import ValidationError

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler, corrections, zhinst_backend
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.types import common, zhinst
from quantify_scheduler.backends.zhinst import settings
from quantify_scheduler.backends.zhinst.zhinst_hardware_config_old_style import (
    hardware_config as zhinst_hardware_config_old_style,
)
from quantify_scheduler.backends.zhinst_backend import flatten_schedule
from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.helpers import waveforms as waveform_helpers
from quantify_scheduler.helpers.collections import find_all_port_clock_combinations
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.gate_library import X90, Measure, Reset
from quantify_scheduler.operations.pulse_library import SetClockFrequency, SquarePulse
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules import spectroscopy_schedules, trace_schedules
from quantify_scheduler.schedules.verification import acquisition_staircase_sched
from quantify_scheduler.schemas.examples import utils

ARRAY_DECIMAL_PRECISION = 16

ZHINST_HARDWARE_COMPILATION_CONFIG = utils.load_json_example_scheme(
    "zhinst_hardware_compilation_config.json"
)


@pytest.fixture
def hardware_compilation_config_zhinst_example() -> Generator[dict[str, Any], None, None]:
    yield dict(ZHINST_HARDWARE_COMPILATION_CONFIG)


@pytest.fixture
def zhinst_hw_config_invalid_latency_corrections(
    hardware_compilation_config_zhinst_example,
):
    hw_config = deepcopy(hardware_compilation_config_zhinst_example["connectivity"])
    hw_config["latency_corrections"] = {"q0:mw-q0.01": 2e-8, "q1:mw-q1.01": None}

    yield hw_config


def test__determine_measurement_fixpoint_correction():
    for i in range(16):
        (
            time_corr,
            sample_correction,
        ) = zhinst_backend._determine_measurement_fixpoint_correction(
            measurement_start_sample=i, common_frequency=600e6
        )
        required_sample_corr = (-i) % 8

        assert sample_correction % 8 == required_sample_corr
        assert sample_correction / 1.8e9 == time_corr


@pytest.fixture
def make_schedule(create_schedule_with_pulse_info):
    def _make_schedule() -> Schedule:
        q0 = "q0"
        schedule = Schedule("test")
        schedule.add(Reset(q0))
        schedule.add(X90(q0))
        schedule.add(Measure(q0))
        return create_schedule_with_pulse_info(schedule)

    yield _make_schedule


@pytest.fixture
def create_typical_timing_table(make_schedule, compile_config_basic_transmon_zhinst_hardware):
    def _create_test_compile_datastructure():
        schedule = make_schedule()
        schedule = zhinst_backend.flatten_schedule(
            schedule, config=compile_config_basic_transmon_zhinst_hardware
        )
        hardware_config = zhinst_backend._generate_legacy_hardware_config(
            schedule=schedule,
            compilation_config=compile_config_basic_transmon_zhinst_hardware,
        )
        timing_table = schedule.timing_table.data

        # information is added on what output channel is used for every pulse and acq.
        port_clock_channelmapping = zhinst_backend._extract_port_clock_channelmapping(
            hardware_config
        )
        timing_table = zhinst_backend._add_channel_information(
            timing_table=timing_table,
            port_clock_channelmapping=port_clock_channelmapping,
        )

        # the timing of all pulses and acquisitions is corrected
        # based on the latency corr.
        latency_dict = corrections.determine_relative_latency_corrections(hardware_config)
        timing_table = zhinst_backend._apply_latency_corrections(
            timing_table=timing_table, latency_dict=latency_dict
        )

        # ensure that operations are still sorted by time after
        # applying the latency corr.
        timing_table.sort_values("abs_time", inplace=True)

        # add the sequencer clock cycle start and sampling start for the operations.
        timing_table = zhinst_backend._add_clock_sample_starts(timing_table=timing_table)

        # After adjusting for the latencies, the fix-point correction can be applied.
        # the fix-point correction has the goal to ensure that all measurement
        # operations will always start at a multiple of *all* relevant clock domains.
        # this is achieved by shifting all instructions between different measurements
        # by the same amount of samples.
        timing_table = zhinst_backend._apply_measurement_fixpoint_correction(
            timing_table=timing_table, common_frequency=600e6
        )

        # because of the shifting in time on a sub-clock delay,
        # up to 8 distinct waveforms may be required to realize the identical pulse.
        # Pre-modulation adds another variant depending on the starting phase of the
        # operation.
        timing_table = zhinst_backend._add_waveform_ids(timing_table=timing_table)

        zhinst_backend.ensure_no_operations_overlap(timing_table)

        # Parse the hardware configuration file, zhinst.Device is a dataclass containing
        # device descriptions (name, type, channels etc. )
        devices = zhinst_backend._parse_devices(hardware_config["devices"])

        local_oscillators = zhinst_backend._parse_local_oscillators(
            hardware_config["local_oscillators"]
        )

        ################################################
        # Constructing the waveform table
        ################################################

        device_dict = {}
        for dev in devices:
            device_dict[dev.name] = dev

        operations_dict_with_repr_keys = {str(op): op for op in schedule.operations.values()}
        numerical_wf_dict = zhinst_backend.construct_waveform_table(
            timing_table,
            operations_dict=operations_dict_with_repr_keys,
            device_dict=device_dict,
        )

        return {
            "schedule": schedule,
            "hardware_cfg": hardware_config,
            "devices": devices,
            "local_oscillators": local_oscillators,
            "timing_table": timing_table,
            "numerical_wf_dict": numerical_wf_dict,
        }

    yield _create_test_compile_datastructure


@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
@pytest.mark.parametrize(
    "unsupported_device_type", [(zhinst.DeviceType.UHFLI), (zhinst.DeviceType.MFLI)]
)
def test_compile_backend_unsupported_devices(
    unsupported_device_type: zhinst.DeviceType, create_schedule_with_pulse_info
):
    zhinst_hardware_cfg = {
        "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
        "local_oscillators": [{"name": "lo0", "frequency": 4.8e9}],
        "devices": [
            {
                "name": f"{unsupported_device_type}_1234",
                "type": f"{unsupported_device_type.value}",
                "ref": "none",
                "channel_0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "mode": "real",
                    "modulation": {"type": "none", "interm_freq": -50e6},
                    "local_oscillator": "lo0",
                },
            }
        ],
    }

    # Act
    with pytest.raises(NotImplementedError) as execinfo:
        zhinst_backend.compile_backend(
            schedule=create_schedule_with_pulse_info(), hardware_cfg=zhinst_hardware_cfg
        )

    # Assert
    assert unsupported_device_type.value in str(execinfo.value)


@pytest.fixture(scope="function", autouse=False)
def deprecated_zhinst_hardware_config_example():
    return {
        "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
        "mode": "calibration",
        "latency_corrections": {
            "q0:mw-q0.01": 95e-9,
            "q1:mw-q1.01": 95e-9,
            "q0:res-q0.ro": -95e-9,
            "q1:res-q1.ro": -95e-9,
        },
        "local_oscillators": [
            {
                "unique_name": "lo0_ch1",
                "instrument_name": "lo0",
                "frequency": {"ch_1.frequency": None},
                "power": {"power": 13},
            },
            {
                "unique_name": "lo0_ch2",
                "instrument_name": "lo0",
                "frequency": {"ch_2.frequency": None},
                "power": {"ch_2.power": 10},
            },
            {
                "unique_name": "lo1",
                "instrument_name": "lo1",
                "frequency": {"frequency": None},
                "power": {"power": 16},
            },
        ],
        "devices": [
            {
                "name": "ic_hdawg0",
                "type": "HDAWG8",
                "clock_select": 0,
                "ref": "int",
                "channelgrouping": 0,
                "channel_0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "mode": "complex",
                    "modulation": {"type": "premod", "interm_freq": -100000000.0},
                    "local_oscillator": "lo0_ch1",
                    "clock_frequency": 6000000000.0,
                    "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                    "gain1": 1,
                    "gain2": 1,
                    "mixer_corrections": {
                        "amp_ratio": 0.95,
                        "phase_error": 0.07,
                        "dc_offset_i": -0.0542,
                        "dc_offset_q": -0.0328,
                    },
                },
                "channel_1": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "mode": "complex",
                    "modulation": {"type": "premod", "interm_freq": -100000000.0},
                    "local_oscillator": "lo0_ch2",
                    "clock_frequency": 6000000000.0,
                    "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                    "gain1": 1,
                    "gain2": 1,
                    "mixer_corrections": {
                        "amp_ratio": 0.95,
                        "phase_error": 0.07,
                        "dc_offset_i": 0.042,
                        "dc_offset_q": 0.028,
                    },
                },
                "channel_2": {
                    "port": "q2:mw",
                    "clock": "q2.01",
                    "mode": "complex",
                    "modulation": {"type": "premod", "interm_freq": -100000000.0},
                    "local_oscillator": "lo0_ch2",
                    "clock_frequency": 6000000000.0,
                    "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                    "gain1": 1,
                    "gain2": 1,
                    "mixer_corrections": {
                        "amp_ratio": 0.95,
                        "phase_error": 0.07,
                        "dc_offset_i": 0.042,
                        "dc_offset_q": 0.028,
                    },
                },
                "channel_3": {
                    "port": "q3:mw",
                    "clock": "q3.01",
                    "mode": "complex",
                    "modulation": {"type": "premod", "interm_freq": -100000000.0},
                    "local_oscillator": "lo0_ch2",
                    "clock_frequency": 6000000000.0,
                    "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                    "gain1": 1,
                    "gain2": 1,
                    "mixer_corrections": {
                        "amp_ratio": 0.95,
                        "phase_error": 0.07,
                        "dc_offset_i": 0.042,
                        "dc_offset_q": 0.028,
                    },
                },
            },
            {
                "name": "ic_uhfqa0",
                "type": "UHFQA",
                "ref": "ext",
                "channel_0": {
                    "port": "q0:res",
                    "clock": "q0.ro",
                    "mode": "real",
                    "modulation": {"type": "premod", "interm_freq": 200000000.0},
                    "local_oscillator": "lo1",
                    "clock_frequency": 6000000000.0,
                    "trigger": 2,
                },
            },
        ],
    }


def test_compile_hardware_hdawg4_successfully_deprecated_hardware_config(
    create_schedule_with_pulse_info,
    deprecated_zhinst_hardware_config_example: dict[str, Any],
) -> None:
    hdawg_hardware_cfg = deprecated_zhinst_hardware_config_example
    # Arrange
    (q0, q1) = ("q0", "q1")
    schedule = Schedule("test")
    schedule.add(Reset(q0, q1))
    schedule.add(X90(q0))
    schedule.add(X90(q1))
    schedule.add(Measure(q0))
    schedule = create_schedule_with_pulse_info(schedule)

    # modulate_wave_spy = mocker.patch.object(
    #     waveform_helpers, "modulate_waveform", wraps=waveform_helpers.modulate_waveform
    # )
    # settings_builder = mocker.Mock(wraps=settings.ZISettingsBuilder())
    # mocker.patch.object(settings, "ZISettingsBuilder", return_value=settings_builder)

    expected_settings = {
        "sigouts/*/on": 0,
        "awgs/*/single": 1,
        "system/awg/channelgrouping": 0,
        "awgs/0/time": 0,
        "awgs/1/time": 0,
        "sigouts/0/on": 1,
        "sigouts/1/on": 1,
        "sigouts/0/offset": -0.0542,
        "sigouts/1/offset": -0.0328,
        "sigouts/2/on": 1,
        "sigouts/3/on": 1,
        "awgs/0/outputs/0/gains/0": 1,
        "awgs/0/outputs/1/gains/1": 1,
        "awgs/1/outputs/0/gains/0": 1,
        "awgs/1/outputs/1/gains/1": 1,
        "sigouts/2/offset": 0.042,
        "sigouts/3/offset": 0.028,
        "awgs/0/commandtable/data": ANY,
        "awgs/0/waveform/waves/0": ANY,
        "awgs/1/commandtable/data": ANY,
        "awgs/1/waveform/waves/0": ANY,
        "compiler/sourcestring": ANY,
    }

    # Act
    with pytest.warns(FutureWarning, match="0.19.0"):
        comp_sched = zhinst_backend.flatten_schedule(schedule, hdawg_hardware_cfg)
        comp_sched = zhinst_backend.compile_backend(comp_sched, hdawg_hardware_cfg)
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_hdawg0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_hdawg0"]
    zi_settings = device_config.settings_builder.build()
    zi_settings_dict = zi_settings.as_dict()

    for key, expected_value in expected_settings.items():
        assert key in zi_settings_dict
        if isinstance(expected_value, type(ANY)):
            continue
        assert (
            zi_settings_dict[key] == expected_value
        ), f"Expected {key} {zi_settings_dict[key]} to equal {expected_value}"


def test_compile_hardware_hdawg4_successfully(
    create_schedule_with_pulse_info,
    compile_config_basic_transmon_zhinst_hardware,
) -> None:
    # Arrange
    (q0, q1) = ("q0", "q1")
    schedule = Schedule("test")
    schedule.add(Reset(q0, q1))
    schedule.add(X90(q0))
    schedule.add(X90(q1))
    schedule.add(Measure(q0))

    hardware = compile_config_basic_transmon_zhinst_hardware

    q0_mw_rf = hardware.device_compilation_config.clocks["q0.01"]
    q0_mw_if = hardware.hardware_compilation_config.hardware_options.modulation_frequencies[
        "q0:mw-q0.01"
    ].interm_freq

    # modulate_wave_spy = mocker.patch.object(
    #     waveform_helpers, "modulate_waveform", wraps=waveform_helpers.modulate_waveform
    # )
    # settings_builder = mocker.Mock(wraps=settings.ZISettingsBuilder())
    # mocker.patch.object(settings, "ZISettingsBuilder", return_value=settings_builder)

    expected_settings = {
        "sigouts/*/on": 0,
        "awgs/*/single": 1,
        "system/awg/channelgrouping": 0,
        "awgs/0/time": 0,
        "awgs/1/time": 0,
        "sigouts/0/on": 1,
        "sigouts/1/on": 1,
        "sigouts/0/offset": -0.0542,
        "sigouts/1/offset": -0.0328,
        "sigouts/2/on": 1,
        "sigouts/3/on": 1,
        "awgs/0/outputs/0/gains/0": 1,
        "awgs/0/outputs/1/gains/1": 1,
        "awgs/1/outputs/0/gains/0": 1,
        "awgs/1/outputs/1/gains/1": 1,
        "sigouts/2/offset": 0.042,
        "sigouts/3/offset": 0.028,
        "awgs/0/commandtable/data": ANY,
        "awgs/0/waveform/waves/0": ANY,
        "awgs/1/commandtable/data": ANY,
        "awgs/1/waveform/waves/0": ANY,
        "compiler/sourcestring": ANY,
    }
    # Act
    compiler = SerialCompiler("compiler")
    comp_sched = compiler.compile(schedule, compile_config_basic_transmon_zhinst_hardware)
    # comp_sched = zhinst_backend.compile_backend(
    #     schedule, compile_config_basic_transmon_zhinst_hardware
    # )
    device_configs = comp_sched.compiled_instructions

    # Assert
    assert "ic_hdawg0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_hdawg0"]
    zi_settings = device_config.settings_builder.build()
    zi_settings_dict = zi_settings.as_dict()

    for key, expected_value in expected_settings.items():
        assert key in zi_settings_dict
        if isinstance(expected_value, type(ANY)):
            continue
        assert (
            zi_settings_dict[key] == expected_value
        ), f"Expected {key} {zi_settings_dict[key]} to equal {expected_value}"

    assert "generic" in device_configs

    assert device_configs["generic"]["lo0.ch1.frequency"] == q0_mw_rf - q0_mw_if


@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_compile_hardware_uhfqa_successfully_deprecated_hardware_config(
    mocker,
    make_schedule,
    deprecated_zhinst_hardware_config_example: dict[str, Any],
) -> None:
    uhfqa_hardware_cfg = deprecated_zhinst_hardware_config_example
    # Arrange
    schedule = make_schedule()
    settings_builder = mocker.Mock(wraps=settings.ZISettingsBuilder())
    mocker.patch.object(settings, "ZISettingsBuilder", return_value=settings_builder)

    expected_settings = {
        "awgs/0/single": 1,
        "qas/0/rotations/*": (1 + 1j),
        "sigouts/0/on": 1,
        "sigouts/1/on": 1,
        "awgs/0/time": 0,
        "qas/0/integration/weights/0/real": ANY,
        "qas/0/integration/weights/0/imag": ANY,
        "qas/0/integration/weights/1/real": ANY,
        "qas/0/integration/weights/1/imag": ANY,
        "qas/0/integration/weights/2/real": ANY,
        "qas/0/integration/weights/2/imag": ANY,
        "qas/0/integration/weights/3/real": ANY,
        "qas/0/integration/weights/3/imag": ANY,
        "qas/0/integration/weights/4/real": ANY,
        "qas/0/integration/weights/4/imag": ANY,
        "qas/0/integration/weights/5/real": ANY,
        "qas/0/integration/weights/5/imag": ANY,
        "qas/0/integration/weights/6/real": ANY,
        "qas/0/integration/weights/6/imag": ANY,
        "qas/0/integration/weights/7/real": ANY,
        "qas/0/integration/weights/7/imag": ANY,
        "qas/0/integration/weights/8/real": ANY,
        "qas/0/integration/weights/8/imag": ANY,
        "qas/0/integration/weights/9/real": ANY,
        "qas/0/integration/weights/9/imag": ANY,
        "sigouts/0/offset": 0.0,
        "sigouts/1/offset": 0.0,
        "awgs/0/waveform/waves/0": ANY,
        "qas/0/integration/mode": 0,
        "qas/0/integration/length": 540,
        "qas/0/result/enable": 1,
        "qas/0/monitor/enable": 0,
        "qas/0/delay": 0,
        "qas/0/result/mode": 0,
        "qas/0/result/source": 7,
        "qas/0/result/length": 1,
        "qas/0/result/averages": 1,
        "compiler/sourcestring": ANY,
    }

    # Act
    comp_sched = zhinst_backend.flatten_schedule(schedule, uhfqa_hardware_cfg)
    comp_sched = zhinst_backend.compile_backend(comp_sched, uhfqa_hardware_cfg)
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_uhfqa0"]
    zi_settings = device_config.settings_builder.build()
    compiled_settings = zi_settings.as_dict()

    for key, expected_value in expected_settings.items():
        assert key in compiled_settings
        if isinstance(expected_value, type(ANY)):
            continue
        assert compiled_settings[key] == expected_value


def test_compile_hardware_uhfqa_successfully(
    mocker,
    make_schedule,
    compile_config_basic_transmon_zhinst_hardware,
) -> None:
    # Arrange
    schedule = make_schedule()
    hardware = compile_config_basic_transmon_zhinst_hardware

    q0_ro_rf = hardware.device_compilation_config.clocks["q0.ro"]
    q0_ro_if = hardware.hardware_compilation_config.hardware_options.modulation_frequencies[
        "q0:res-q0.ro"
    ].interm_freq

    expected_settings = {
        "awgs/0/single": 1,
        "qas/0/rotations/*": (1 + 1j),
        "sigouts/0/on": 1,
        "sigouts/1/on": 1,
        "awgs/0/time": 0,
        "qas/0/integration/weights/0/real": ANY,
        "qas/0/integration/weights/0/imag": ANY,
        "qas/0/integration/weights/1/real": ANY,
        "qas/0/integration/weights/1/imag": ANY,
        "qas/0/integration/weights/2/real": ANY,
        "qas/0/integration/weights/2/imag": ANY,
        "qas/0/integration/weights/3/real": ANY,
        "qas/0/integration/weights/3/imag": ANY,
        "qas/0/integration/weights/4/real": ANY,
        "qas/0/integration/weights/4/imag": ANY,
        "qas/0/integration/weights/5/real": ANY,
        "qas/0/integration/weights/5/imag": ANY,
        "qas/0/integration/weights/6/real": ANY,
        "qas/0/integration/weights/6/imag": ANY,
        "qas/0/integration/weights/7/real": ANY,
        "qas/0/integration/weights/7/imag": ANY,
        "qas/0/integration/weights/8/real": ANY,
        "qas/0/integration/weights/8/imag": ANY,
        "qas/0/integration/weights/9/real": ANY,
        "qas/0/integration/weights/9/imag": ANY,
        "sigouts/0/offset": 0.0,
        "sigouts/1/offset": 0.0,
        "awgs/0/waveform/waves/0": ANY,
        "qas/0/integration/mode": 0,
        "qas/0/integration/length": 540,
        "qas/0/result/enable": 1,
        "qas/0/monitor/enable": 0,
        "qas/0/delay": 0,
        "qas/0/result/mode": 0,
        "qas/0/result/source": 7,
        "qas/0/result/length": 1,
        "qas/0/result/averages": 1,
        "compiler/sourcestring": ANY,
    }

    # Act
    compiler = SerialCompiler("compiler")
    comp_sched = compiler.compile(schedule, compile_config_basic_transmon_zhinst_hardware)
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_uhfqa0"]
    zi_settings = device_config.settings_builder.build()
    compiled_settings = zi_settings.as_dict()

    for key, expected_value in expected_settings.items():
        assert key in compiled_settings
        if isinstance(expected_value, type(ANY)):
            continue
        assert compiled_settings[key] == expected_value

    assert "generic" in device_configs

    assert device_configs["generic"]["lo1.frequency"] == q0_ro_rf - q0_ro_if


@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_compile_invalid_latency_corrections_hardware_config_raises(
    make_schedule,
    zhinst_hw_config_invalid_latency_corrections,
) -> None:
    hardware_cfg = zhinst_hw_config_invalid_latency_corrections
    # Arrange
    schedule = make_schedule()

    # should raise a pydantic validation error
    with pytest.raises(ValidationError):
        comp_sched = zhinst_backend.flatten_schedule(schedule, hardware_cfg)
        _ = zhinst_backend.compile_backend(comp_sched, hardware_cfg)


def test_compile_with_third_party_instrument(
    make_schedule, compile_config_basic_transmon_zhinst_hardware
):
    def _third_party_compilation_node(schedule: Schedule, config: CompilationConfig) -> Schedule:
        schedule["compiled_instructions"]["third_party_instrument"] = {"setting": "test"}
        return schedule

    config = deepcopy(compile_config_basic_transmon_zhinst_hardware)
    config.hardware_compilation_config.hardware_description["third_party_instrument"] = (
        common.HardwareDescription(instrument_type="ThirdPartyInstrument")
    )
    config.hardware_compilation_config.connectivity.graph.add_edge(
        "third_party_instrument.output", "some_qubit:some_port"
    )
    config.hardware_compilation_config.compilation_passes.insert(
        -1,
        SimpleNodeConfig(
            name="third_party_instrument_compilation",
            compilation_func=_third_party_compilation_node,
        ),
    )

    compiler = SerialCompiler(name="compiler")
    comp_sched = compiler.compile(
        make_schedule(),
        config=config,
    )

    assert comp_sched["compiled_instructions"]["third_party_instrument"]["setting"] == "test"


def test_external_lo_not_present_raises(
    make_schedule, compile_config_basic_transmon_zhinst_hardware
):
    sched = make_schedule()

    compile_config = deepcopy(compile_config_basic_transmon_zhinst_hardware)

    # Change to non-existent LO:
    relabel_nodes(
        compile_config.hardware_compilation_config.connectivity.graph,
        {"lo0_ch1.output": "non_existent_lo.output"},
        copy=False,
    )

    with pytest.raises(
        RuntimeError,
        match="External local oscillator 'non_existent_lo' set to "
        "be used for port='q0:mw' and clock='q0.01' not found! Make "
        "sure it is present in the hardware configuration.",
    ):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(sched, config=compile_config)


def test_hdawg4_sequence(
    compile_config_basic_transmon_zhinst_hardware,
    make_schedule,
) -> None:
    # Arrange
    awg_index = 0
    schedule = make_schedule()

    expected_seqc = dedent(
        """\
        // Generated by quantify-scheduler.
        // Variables
        var __repetitions__ = 1;
        wave w0 = placeholder(48);

        // Operations
        assignWaveIndex(w0, w0, 0);
        setTrigger(0);\t//  n_instr=1
        repeat(__repetitions__)
        {
          setTrigger(AWG_MARKER1 + AWG_MARKER2);\t//  n_instr=2
          wait(60054);\t\t// clock=2\t n_instr=3
          executeTableEntry(0);\t// clock=60059 pulse=0 n_instr=0
          setTrigger(0);\t// clock=60059 n_instr=1
          wait(70);\t\t// clock=60060, dead time to ensure total schedule duration\t n_instr=3
        }
        """
    ).lstrip("\n")

    # Act
    compiler = SerialCompiler("compiler")
    comp_sched = compiler.compile(
        schedule=schedule, config=compile_config_basic_transmon_zhinst_hardware
    )
    compiled_instructions = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_hdawg0" in compiled_instructions
    zi_device_config: zhinst_backend.ZIDeviceConfig = compiled_instructions["ic_hdawg0"]

    awg_settings = zi_device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[awg_index]  # for awg index 0

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


@pytest.mark.parametrize("channelgrouping,enabled_channels", [(0, [0]), (1, [0])])
def test__program_hdawg4_channelgrouping(
    mocker,
    create_typical_timing_table,
    channelgrouping: int,
    enabled_channels: list[int],
):
    test_config_dict = create_typical_timing_table()
    schedule = test_config_dict["schedule"]
    timing_table = test_config_dict["timing_table"]
    devices = test_config_dict["devices"]
    numerical_wf_dict = test_config_dict["numerical_wf_dict"]

    hdawg_device = [device for device in devices if device.name == "ic_hdawg0"][0]
    hdawg_device.channelgrouping = channelgrouping
    hdawg_device.sample_rate = int(2.4e9)

    mocker.patch.object(zhinst_backend, "_add_wave_nodes")
    with_sigouts = mocker.patch.object(settings.ZISettingsBuilder, "with_sigouts")
    with_system_channelgrouping = mocker.patch.object(
        settings.ZISettingsBuilder, "with_system_channelgrouping"
    )

    # Act
    zhinst_backend._compile_for_hdawg(
        device=hdawg_device,
        timing_table=timing_table,
        numerical_wf_dict=numerical_wf_dict,
        repetitions=schedule.repetitions,
    )

    # Assert
    with_system_channelgrouping.assert_called_with(channelgrouping)
    calls = list(map(lambda i: call(i, (1, 1)), enabled_channels))
    assert with_sigouts.call_args_list == calls


def test_validate_schedule(
    empty_schedule: Schedule,
    basic_schedule: Schedule,
    schedule_with_pulse_info: Schedule,
):
    with pytest.raises(ValueError) as execinfo:
        zhinst_backend._validate_schedule(empty_schedule)

    assert str(execinfo.value) == "Undefined schedulables for schedule 'Empty Experiment'!"

    with pytest.raises(ValueError) as execinfo:
        zhinst_backend._validate_schedule(basic_schedule)

    assert (
        str(execinfo.value)
        == "Absolute timing has not been determined for the schedule 'Basic schedule'!"
    )

    zhinst_backend._validate_schedule(schedule_with_pulse_info)


@pytest.mark.parametrize(
    "is_pulse,modulation_type,expected_modulated",
    [
        (True, zhinst.ModulationModeType.PREMODULATE, True),
        (False, zhinst.ModulationModeType.PREMODULATE, True),
        (False, zhinst.ModulationModeType.NONE, False),
    ],
)
def test_apply_waveform_corrections(
    mocker,
    is_pulse: bool,
    modulation_type: zhinst.ModulationModeType,
    expected_modulated: bool,
):
    # Arrange
    wave = np.ones(48)

    modulate_wave = mocker.patch.object(waveform_helpers, "modulate_waveform", return_value=wave)
    shift_waveform = mocker.patch.object(waveform_helpers, "shift_waveform", return_value=(0, wave))
    resize_waveform = mocker.patch.object(waveform_helpers, "resize_waveform", return_value=wave)

    channel = zhinst.Output(
        port="port",
        clock="clock",
        mode=zhinst.SignalModeType.COMPLEX,
        modulation=zhinst.Modulation(type=modulation_type.value),
        local_oscillator="lo0",
    )
    instrument_info = zhinst.InstrumentInfo(
        sample_rate=2.4e9, num_samples_per_clock=8, granularity=16
    )

    # Act
    result = zhinst_backend.apply_waveform_corrections(
        output=channel,
        waveform=wave,
        start_and_duration_in_seconds=(0, 16e-9),
        instrument_info=instrument_info,
        is_pulse=is_pulse,
    )

    # Assert
    assert (0, 48, wave) == result
    if expected_modulated:
        modulate_wave.assert_called()
    else:
        modulate_wave.assert_not_called()
    shift_waveform.assert_called()
    resize_waveform.assert_called()


@pytest.mark.parametrize("is_pulse", [True, False])
def test_apply_waveform_corrections_throw_modulation_error(is_pulse):
    # Arrange
    wave = np.ones(48)

    modulation_type = zhinst.ModulationModeType.MODULATE

    channel = zhinst.Output(
        port="port",
        clock="clock",
        mode=zhinst.SignalModeType.COMPLEX,
        modulation=zhinst.Modulation(type=modulation_type.value),
        local_oscillator="lo0",
    )
    instrument_info = zhinst.InstrumentInfo(
        sample_rate=2.4e9, num_samples_per_clock=8, granularity=16
    )

    # Act
    with pytest.raises(
        NotImplementedError, match="Hardware real-time modulation is not available yet!"
    ):
        zhinst_backend.apply_waveform_corrections(
            output=channel,
            waveform=wave,
            start_and_duration_in_seconds=(0, 16e-9),
            instrument_info=instrument_info,
            is_pulse=is_pulse,
        )


def test__get_instruction_list(create_typical_timing_table):
    # Arrange
    test_config_dict = create_typical_timing_table()
    timing_table = test_config_dict["timing_table"]

    hdawg0awg0_expected_list = [
        zhinst.Wave(
            waveform_id=timing_table.waveform_id[1],
            abs_time=timing_table.abs_time[1],
            duration=timing_table.duration[1],
            clock_cycle_start=timing_table.clock_cycle_start[1],
        )
    ]
    uhfqa0awg0_expected_list = [
        zhinst.Wave(
            waveform_id=timing_table.waveform_id[3],
            abs_time=timing_table.abs_time[3],
            duration=timing_table.duration[3],
            clock_cycle_start=timing_table.clock_cycle_start[3],
        ),
        zhinst.Acquisition(
            waveform_id=timing_table.waveform_id[4],
            abs_time=timing_table.abs_time[4],
            duration=timing_table.duration[4],
            clock_cycle_start=timing_table.clock_cycle_start[4],
        ),
    ]

    expected_instructions_list = {
        "ic_hdawg0.awg0": hdawg0awg0_expected_list,
        "ic_uhfqa0.awg0": uhfqa0awg0_expected_list,
    }

    for hardware_channel in expected_instructions_list:
        # select only the instructions relevant for the output channel.
        output_timing_table = timing_table[timing_table["hardware_channel"] == hardware_channel]

        # Act
        instructions = zhinst_backend._get_instruction_list(output_timing_table)

        # Assert
        expected_instructions = expected_instructions_list[hardware_channel]
        for expected_instruction, instruction in zip(expected_instructions, instructions):
            assert instruction == expected_instruction


def test_uhfqa_sequence1(
    make_schedule,
    compile_config_basic_transmon_zhinst_hardware,
) -> None:
    # Arrange
    awg_index = 0
    schedule = make_schedule()

    expected_seqc = dedent(
        """\
        // Generated by quantify-scheduler.
        // Variables
        var __repetitions__ = 1;
        wave w0 = "ic_uhfqa0_awg0_wave0";
        wave w1 = "ic_uhfqa0_awg0_wave1";

        // Operations
        repeat(__repetitions__)
        {
          waitDigTrigger(2, 1);\t// \t// clock=0
          wait(45006);\t\t// clock=0\t n_instr=45006
          playWave(w0);\t// \t// clock=45006\t n_instr=0
          wait(27);\t\t// clock=45006\t n_instr=27
          startQA(QA_INT_ALL, true);\t// clock=45033 n_instr=7
        }
        """
    ).lstrip("\n")

    # Act
    comp_sched = zhinst_backend.flatten_schedule(
        schedule, compile_config_basic_transmon_zhinst_hardware
    )
    comp_sched = zhinst_backend.compile_backend(
        comp_sched, compile_config_basic_transmon_zhinst_hardware
    )
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_uhfqa0"]

    awg_settings = device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[0]

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


def test_uhfqa_sequence2_trace_acquisition(
    create_schedule_with_pulse_info,
    compile_config_basic_transmon_zhinst_hardware,
):
    # Arrange
    awg_index = 0
    schedule = trace_schedules.trace_schedule(
        pulse_amp=1,
        pulse_duration=16e-9,
        pulse_delay=0,
        frequency=7.04e9,
        acquisition_delay=0,
        integration_time=1e-6,
        port="q0:res",
        clock="q0.ro",
        init_duration=1e-5,
    )
    schedule = create_schedule_with_pulse_info(schedule)

    expected_seqc = dedent(
        """\
        // Generated by quantify-scheduler.
        // Variables
        var __repetitions__ = 1;
        wave w0 = "ic_uhfqa0_awg0_wave0";
        wave w1 = "ic_uhfqa0_awg0_wave1";

        // Operations
        repeat(__repetitions__)
        {
          waitDigTrigger(2, 1);\t// \t// clock=0
          wait(2250);\t\t// clock=0\t n_instr=2250
          playWave(w0);\t// \t// clock=2250\t n_instr=0
          wait(0);\t\t// clock=2250\t n_instr=0
          startQA(QA_INT_ALL, true);\t// clock=2250 n_instr=7
        }
        """
    ).lstrip("\n")

    # Act
    comp_sched = zhinst_backend.compile_backend(
        schedule, compile_config_basic_transmon_zhinst_hardware
    )
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_uhfqa0"]

    awg_settings = device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[0]

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


def test_uhfqa_sequence3_spectroscopy(
    create_schedule_with_pulse_info,
    compile_config_basic_transmon_zhinst_hardware,
) -> None:
    # Arrange
    awg_index = 0
    ro_acquisition_delay = -40e-9
    ro_pulse_delay = 20e-9
    schedule = spectroscopy_schedules.two_tone_spec_sched(
        spec_pulse_amp=0.6e-0,
        spec_pulse_duration=200e-9,
        spec_pulse_frequency=6.02e9,
        spec_pulse_port="q0:mw",
        spec_pulse_clock="q0.01",
        ro_pulse_amp=0.5e-3,
        ro_pulse_duration=150e-9,
        ro_pulse_delay=ro_pulse_delay,
        ro_pulse_port="q0:res",
        ro_pulse_clock="q0.ro",
        ro_pulse_frequency=7.04e9,
        ro_acquisition_delay=ro_acquisition_delay,
        ro_integration_time=500e-9,
        init_duration=1e-5,
    )
    schedule = create_schedule_with_pulse_info(schedule)

    expected_seqc = dedent(
        """\
        // Generated by quantify-scheduler.
        // Variables
        var __repetitions__ = 1;
        wave w0 = "ic_uhfqa0_awg0_wave0";
        wave w1 = "ic_uhfqa0_awg0_wave1";

        // Operations
        repeat(__repetitions__)
        {
          waitDigTrigger(2, 1);\t// \t// clock=0
          wait(2292);\t\t// clock=0\t n_instr=2292
          startQA(QA_INT_ALL, true);\t// clock=2292 n_instr=7
          wait(2);\t\t// clock=2299\t n_instr=2
          playWave(w1);\t// \t// clock=2301\t n_instr=0
        }
        """
    ).lstrip("\n")

    # Act
    comp_sched = zhinst_backend.compile_backend(
        schedule, compile_config_basic_transmon_zhinst_hardware
    )
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_uhfqa0"]

    awg_settings = device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[0]

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


def test__extract_port_clock_channelmapping_hdawg() -> None:
    hardware_config = zhinst_hardware_config_old_style

    expected_dict = {
        "q0:mw-q0.01": "ic_hdawg0.awg0",
        "q1:mw-q1.01": "ic_hdawg0.awg1",
        "q2:mw-q2.01": "ic_hdawg0.awg2",
        "q3:mw-q3.01": "ic_hdawg0.awg3",
        "q0:res-q0.ro": "ic_uhfqa0.awg0",
    }
    generated_dict = zhinst_backend._extract_port_clock_channelmapping(hardware_cfg=hardware_config)
    assert generated_dict == expected_dict


def test_determine_relative_latency_corrections() -> None:
    hardware_config = zhinst_hardware_config_old_style

    expected_latency_dict = {
        "q0:mw-q0.01": 190e-9,
        "q0:res-q0.ro": 0.0,
        "q1:mw-q1.01": 190e-9,
        "q2:mw-q2.01": 9.5e-08,
        "q3:mw-q3.01": 9.5e-08,
    }
    generated_dict = corrections.determine_relative_latency_corrections(
        hardware_cfg=hardware_config
    )

    assert generated_dict == expected_latency_dict


def test_compile_latency_corrections(make_schedule, compile_config_basic_transmon_zhinst_hardware):
    """
    Tests if the compiled latency corrections are as expected from the
    settings in the hardware options.
    """
    expected_compiled_ro_latency = 0
    expected_compiled_mw_latency = 190e-9

    sched = make_schedule()

    compiler = SerialCompiler(name="compiler")
    comp_sched = compiler.compile(
        sched,
        config=compile_config_basic_transmon_zhinst_hardware,
    )

    # Extract timings before latency corrections
    timing_table = comp_sched.timing_table.data
    ro_pulse_time_before_corr = timing_table[
        timing_table["operation"].str.startswith("SquarePulse")
    ]["abs_time"].values[0]
    mw_pulse_time_before_corr = timing_table[timing_table["operation"].str.startswith("X90")][
        "abs_time"
    ].values[0]

    # Extract timings after latency corrections
    hw_timing_table = comp_sched.hardware_timing_table.data
    ro_pulse_time_after_corr = hw_timing_table[
        hw_timing_table["operation"].str.startswith("SquarePulse")
    ]["abs_time"].values[0]
    mw_pulse_time_after_corr = hw_timing_table[hw_timing_table["operation"].str.startswith("X90")][
        "abs_time"
    ].values[0]

    # Calculate compiled latencies (subtract fixpoint correction)
    compiled_ro_latency = ro_pulse_time_after_corr - ro_pulse_time_before_corr - 20e-9 / 3
    compiled_mw_latency = mw_pulse_time_after_corr - mw_pulse_time_before_corr - 20e-9 / 3

    assert np.isclose(compiled_ro_latency, expected_compiled_ro_latency, atol=1e-9)
    assert np.isclose(compiled_mw_latency, expected_compiled_mw_latency, atol=1e-9)


@pytest.mark.parametrize("device_type", [(zhinst.DeviceType.HDAWG), (zhinst.DeviceType.UHFQA)])
def test__add_wave_nodes_with_vector(create_typical_timing_table, device_type: zhinst.DeviceType):
    # Arrange
    test_waveform = np.vectorize(complex)(np.zeros(1024), np.ones(1024))
    test_config_dict = create_typical_timing_table()
    timing_table = test_config_dict["timing_table"]
    devices = test_config_dict["devices"]
    numerical_wf_dict = test_config_dict["numerical_wf_dict"]

    # Change the numerical waveforms in the numerical_wf_dict
    for wf_id in numerical_wf_dict:
        numerical_wf_dict[wf_id] = test_waveform

    # Create the expected data
    _data = np.zeros((2, 1024))
    _data[0] = np.real(test_waveform)
    _data[1] = np.imag(test_waveform)
    expected_data = (_data.reshape((-2,), order="F") * (2**15 - 1)).astype("int16")

    awg_index: int = 0
    settings_builder = settings.ZISettingsBuilder()

    for device in devices:
        if device.device_type == device_type:
            # select only the instructions relevant for the output channel.
            output_timing_table = timing_table[
                timing_table["hardware_channel"] == f"{device.name}.awg{awg_index}"
            ]
            # enumerate the waveform_ids used in this particular output channel
            unique_wf_ids = output_timing_table.drop_duplicates(subset="waveform_id")["waveform_id"]
            # this table maps waveform ids to indices in the seqc command table.
            wf_id_mapping = {}
            for i, wf_id in enumerate(unique_wf_ids):
                wf_id_mapping[wf_id] = i

            # Act
            zhinst_backend._add_wave_nodes(
                device_type=device.device_type,
                awg_index=awg_index,
                wf_id_mapping=wf_id_mapping,
                numerical_wf_dict=numerical_wf_dict,
                settings_builder=settings_builder,
            )

    # Assert
    settings_builder_dict = settings_builder.build().as_dict()

    for node, generated_waveform in settings_builder_dict.items():
        _ = node  # unused variable
        np.testing.assert_array_equal(generated_waveform, expected_data)


def test_compile_backend_with_undefined_local_oscillator(
    make_schedule, mock_setup_basic_transmon_with_standard_params
):
    # Arrange
    q0 = "q0"
    schedule = Schedule("test")
    schedule.add(Reset(q0))
    schedule.add(X90(q0))
    # no measure to keep it simple

    hardware_cfg_str = """
    {
        "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
        "local_oscillators": [{
            "unique_name": "lo0",
            "instrument_name": "lo0",
            "frequency":
                {
                    "frequency": 4.8e9
                }
        }],
        "devices": [
            {
                "name": "hdawg_1234",
                "type": "HDAWG4",
                "ref": "none",
                "channel_0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "mode": "real",
                    "modulation": {
                      "type": "none",
                      "interm_freq": -50e6
                    },
                    "local_oscillator": "lo_unknown"
                }
            }
        ]
    }
    """
    zhinst_hardware_cfg = json.loads(hardware_cfg_str)
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(zhinst_hardware_cfg)

    # Act
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(KeyError, match='Missing configuration for LocalOscillator "lo_unknown"'):
        compiler.compile(schedule, config=quantum_device.generate_compilation_config())


def test_compile_backend_with_duplicate_local_oscillator(
    make_schedule, mock_setup_basic_transmon_with_standard_params
):
    # Arrange
    q0 = "q0"
    schedule = Schedule("test")
    schedule.add(Reset(q0))
    schedule.add(X90(q0))

    hardware_cfg_str = """
    {
      "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
      "local_oscillators": [
        {
          "unique_name": "lo0",
          "instrument_name": "lo_rs_sgs100a",
          "frequency":
            {
                "frequency": 4.7e9
            }
        },
        {
          "unique_name": "lo0",
          "instrument_name": "lo_rs_sgs100a",
          "frequency":
            {
                "frequency": 4.8e9
            }
        }
      ],
      "devices": [
        {
          "name": "hdawg_1234",
          "type": "HDAWG4",
          "ref": "none",
          "channel_0": {
            "port": "q0:mw",
            "clock": "q0.01",
            "mode": "real",
            "modulation": {
              "type": "none",
              "interm_freq": -50e6
            },
            "local_oscillator": "lo0"
          }
        }
      ]
    }
    """
    zhinst_hardware_cfg = json.loads(hardware_cfg_str)
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(zhinst_hardware_cfg)

    # Act
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(RuntimeError) as execinfo:
        compiler.compile(schedule, config=quantum_device.generate_compilation_config())

    # Assert
    assert (
        str(execinfo.value) == "Duplicate entry LocalOscillators 'lo0' in hardware configuration!"
    )


def test_acquisition_staircase_unique_acquisitions(
    compile_config_basic_transmon_zhinst_hardware,
):
    schedule = acquisition_staircase_sched(
        np.linspace(0, 1, 20),
        readout_pulse_duration=1e-6,
        readout_frequency=8e9,
        acquisition_delay=100e-9,
        integration_time=2e-6,
        port="q0:res",
        clock="q0.ro",
        repetitions=1024,
    )

    # Act
    compiler = SerialCompiler(name="compiler")
    comp_sched = compiler.compile(schedule, config=compile_config_basic_transmon_zhinst_hardware)

    # Assert
    uhfqa_setts = comp_sched.compiled_instructions["ic_uhfqa0"]
    assert uhfqa_setts.acq_config.n_acquisitions == 20
    assert len(uhfqa_setts.acq_config.resolvers) == 1
    assert (
        uhfqa_setts.acq_config.resolvers[0].keywords["result_nodes"][0]
        == "qas/0/result/data/0/wave"
    )
    assert (
        uhfqa_setts.acq_config.resolvers[0].keywords["result_nodes"][1]
        == "qas/0/result/data/1/wave"
    )

    settings_dict = uhfqa_setts.settings_builder.build().as_dict()

    expected_arrays = np.array(
        [
            [1.0, 1.4088320528055174, 1.1584559306791387],
            [1.0, 0.1232568334324389, -0.8111595753452774],
        ]
    )

    # Only testing for the first 3 numerical elements in the arrays set as it is
    # untenable to save all 4096 elements for 4 different arrays.
    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/0/real"][:3],
        expected_arrays[0],
        decimal=ARRAY_DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/0/imag"][:3],
        expected_arrays[1],
        decimal=ARRAY_DECIMAL_PRECISION,
    )

    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/1/real"][:3],
        expected_arrays[1],
        decimal=ARRAY_DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/1/imag"][:3],
        -1 * expected_arrays[0],
        decimal=ARRAY_DECIMAL_PRECISION,
    )

    expected_zeros_array = np.zeros(4096)
    for i in [2, 3, 4, 5, 6, 7, 8, 9]:
        np.testing.assert_array_almost_equal(
            settings_dict[f"qas/0/integration/weights/{i}/real"],
            expected_zeros_array,
            decimal=ARRAY_DECIMAL_PRECISION,
        )
        np.testing.assert_array_almost_equal(
            settings_dict[f"qas/0/integration/weights/{i}/imag"],
            expected_zeros_array,
            decimal=ARRAY_DECIMAL_PRECISION,
        )


def test_acquisition_staircase_right_acq_channel(
    compile_config_basic_transmon_zhinst_hardware,
):
    acq_channel = 2
    schedule = acquisition_staircase_sched(
        np.linspace(0, 1, 20),
        readout_pulse_duration=1e-6,
        readout_frequency=8e9,
        acquisition_delay=100e-9,
        integration_time=2e-6,
        port="q0:res",
        clock="q0.ro",
        repetitions=1024,
        acq_channel=acq_channel,
    )

    # Act
    compiler = SerialCompiler(name="compiler")
    comp_sched = compiler.compile(schedule, config=compile_config_basic_transmon_zhinst_hardware)

    # Assert
    uhfqa_setts = comp_sched.compiled_instructions["ic_uhfqa0"]
    assert uhfqa_setts.acq_config.n_acquisitions == 20
    assert len(uhfqa_setts.acq_config.resolvers) == 1
    assert (
        uhfqa_setts.acq_config.resolvers[acq_channel].keywords["result_nodes"][0]
        == f"qas/0/result/data/{2*acq_channel}/wave"
    )
    assert (
        uhfqa_setts.acq_config.resolvers[acq_channel].keywords["result_nodes"][1]
        == f"qas/0/result/data/{2*acq_channel+1}/wave"
    )

    settings_dict = uhfqa_setts.settings_builder.build().as_dict()

    expected_arrays = np.array(
        [
            [1.0, 1.4088320528055174, 1.1584559306791387],
            [1.0, 0.1232568334324389, -0.8111595753452774],
        ]
    )

    # Only testing for the first 3 numerical elements in the arrays set as it is
    # untenable to save all 4096 elements for 4 different arrays.

    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/4/real"][:3],
        expected_arrays[0],
        decimal=ARRAY_DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/4/imag"][:3],
        expected_arrays[1],
        decimal=ARRAY_DECIMAL_PRECISION,
    )

    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/5/real"][:3],
        expected_arrays[1],
        decimal=ARRAY_DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        settings_dict["qas/0/integration/weights/5/imag"][:3],
        -1 * expected_arrays[0],
        decimal=ARRAY_DECIMAL_PRECISION,
    )

    expected_zeros_array = np.zeros(4096)

    for i in [0, 1, 2, 3, 6, 7, 8, 9]:
        np.testing.assert_array_almost_equal(
            settings_dict[f"qas/0/integration/weights/{i}/real"],
            expected_zeros_array,
            decimal=ARRAY_DECIMAL_PRECISION,
        )
        np.testing.assert_array_almost_equal(
            settings_dict[f"qas/0/integration/weights/{i}/imag"],
            expected_zeros_array,
            decimal=ARRAY_DECIMAL_PRECISION,
        )


def test_too_long_acquisition_raises_readable_exception(
    compile_config_basic_transmon_zhinst_hardware,
):
    sched = Schedule(name="Too long acquisition schedule", repetitions=1024)

    # these are kind of magic names that are known to exist in the default config.
    port = "q0:res"
    clock = "q0.ro"

    sched.add(
        SSBIntegrationComplex(
            duration=2.4e-6,  # this is longer than the allowed 4096 samples.
            port=port,
            clock=clock,
            acq_index=0,
            acq_channel=0,
        ),
    )

    # Act
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(ValueError) as exc_info:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_zhinst_hardware)

    # assert that the name of the offending operation is in the exception message.
    assert "SSBIntegrationComplex(" in str(exc_info.value)

    # assert that the number of samples we are trying to set is in the exception message
    assert "4320 samples" in str(exc_info.value)


def test_generate_legacy_hardware_config(hardware_compilation_config_zhinst_example):
    sched = Schedule("All portclocks schedule")
    quantum_device = QuantumDevice("All_portclocks_device")
    quantum_device.hardware_config(hardware_compilation_config_zhinst_example)

    qubits = {}
    for port, clock in find_all_port_clock_combinations(zhinst_hardware_config_old_style):
        sched.add(SquarePulse(port=port, clock=clock, amp=0.25, duration=12e-9))
        if clock not in sched.resources:
            clock_resource = ClockResource(name=clock, freq=7e9)
            sched.add_resource(clock_resource)
        qubit_name = port.split(":")[0]
        if qubit_name not in quantum_device.elements():
            qubit = BasicTransmonElement(qubit_name)
            quantum_device.add_element(qubit)
            qubits[qubit_name] = qubit

    generated_hw_config = zhinst_backend._generate_legacy_hardware_config(
        schedule=sched, compilation_config=quantum_device.generate_compilation_config()
    )

    assert generated_hw_config == zhinst_hardware_config_old_style


def test_generate_new_style_hardware_compilation_config(
    hardware_compilation_config_zhinst_example,
):
    parsed_new_style_config = zhinst_backend.ZIHardwareCompilationConfig.model_validate(
        hardware_compilation_config_zhinst_example
    )

    converted_new_style_hw_cfg = zhinst_backend.ZIHardwareCompilationConfig.model_validate(
        zhinst_hardware_config_old_style
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
    assert converted_new_style_hw_cfg.model_dump() == parsed_new_style_config.model_dump()


def test_flatten_schedule():
    inner = Schedule("inner")
    inner.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.501e9))

    inner2 = Schedule("inner2")
    inner2.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.502e9))

    inner.add(inner2)

    outer = Schedule("outer")
    outer.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.5e9))

    outer.add(inner)
    outer.add(inner2)
    timed_sched = _determine_absolute_timing(outer, time_unit="ideal")
    flat = flatten_schedule(timed_sched)
    assert len(flat.data["schedulables"]) == 4
