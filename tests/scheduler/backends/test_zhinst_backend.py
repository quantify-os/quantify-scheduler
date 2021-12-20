# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-lines
from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List
from unittest.mock import ANY, call

import numpy as np
import pytest
from quantify_core.data.handling import set_datadir
from zhinst.qcodes import hdawg, mfli, uhfli, uhfqa
from zhinst.toolkit.control import drivers

from quantify_scheduler import Schedule, enums
from quantify_scheduler.backends import zhinst_backend
from quantify_scheduler.backends.types import common, zhinst
from quantify_scheduler.backends.zhinst import settings
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.helpers import schedule as schedule_helpers
from quantify_scheduler.helpers import waveforms as waveform_helpers
from quantify_scheduler.operations.gate_library import X90, Measure, Reset
from quantify_scheduler.schedules import spectroscopy_schedules, trace_schedules
from quantify_scheduler.schedules.verification import acquisition_staircase_sched
from quantify_scheduler.schemas.examples.utils import load_json_example_scheme

ARRAY_DECIMAL_PRECISION = 16


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
def create_device():
    def _create_device(hardware_cfg: Dict[str, Any], index: int = 0) -> zhinst.Device:
        return zhinst.Device.schema().load(hardware_cfg["devices"][index])

    yield _create_device


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
def create_typical_timing_table(make_schedule, load_example_zhinst_hardware_config):
    def _create_test_compile_datastructure():
        schedule = make_schedule()
        hardware_config = load_example_zhinst_hardware_config()
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
        latency_dict = zhinst_backend._extract_channel_latencies(hardware_config)
        timing_table = zhinst_backend._apply_latency_corrections(
            timing_table=timing_table, latency_dict=latency_dict
        )

        # ensure that operations are still sorted by time after
        # applying the latency corr.
        timing_table.sort_values("abs_time", inplace=True)

        # add the sequencer clock cycle start and sampling start for the operations.
        timing_table = zhinst_backend._add_clock_sample_starts(
            timing_table=timing_table
        )

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

        numerical_wf_dict = zhinst_backend.construct_waveform_table(
            timing_table, operations_dict=schedule.operations, device_dict=device_dict
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


def test_compile_hardware_hdawg4_successfully(
    mocker,
    create_schedule_with_pulse_info,
    load_example_zhinst_hardware_config: Dict[str, Any],
) -> None:

    hdawg_hardware_cfg = load_example_zhinst_hardware_config()
    # Arrange
    (q0, q1) = ("q0", "q1")
    schedule = Schedule("test")
    schedule.add(Reset(q0, q1))
    schedule.add(X90(q0))
    schedule.add(X90(q1))
    schedule = create_schedule_with_pulse_info(schedule)

    modulate_wave_spy = mocker.patch.object(
        waveform_helpers, "modulate_waveform", wraps=waveform_helpers.modulate_waveform
    )
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
    comp_sched = zhinst_backend.compile_backend(schedule, hdawg_hardware_cfg)
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

    # FIXME add test for generic instrument coordinator here # pylint: disable=fixme
    # assert "lo0" in device_configs

    # freq_qubit = 6.02e9  # from the example transmon config, this is the RF frequency
    # intermodulation_frequency = -50e6
    # assert device_configs["lo0"] == freq_qubit - intermodulation_frequency


def test_compile_hardware_uhfqa_successfully(
    mocker,
    make_schedule,
    load_example_zhinst_hardware_config: Dict[str, Any],
) -> None:
    uhfqa_hardware_cfg = load_example_zhinst_hardware_config()
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
    comp_sched = zhinst_backend.compile_backend(schedule, uhfqa_hardware_cfg)
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_uhfqa0"]
    zi_settings = device_config.settings_builder.build()
    compiled_settings = zi_settings.as_dict()

    for key, expected_value in expected_settings.items():
        assert key in compiled_settings.keys()
        if isinstance(expected_value, type(ANY)):
            continue
        assert compiled_settings[key] == expected_value

    # assert "lo0" in device_configs
    # ro_freq = 7.04e9
    # intermodulation_frequency = 150e6

    # assert device_configs["lo0"] == ro_freq - intermodulation_frequency


def test_hdawg4_sequence(
    load_example_zhinst_hardware_config,
    make_schedule,
) -> None:
    hdawg_hardware_cfg = load_example_zhinst_hardware_config()
    # Arrange
    awg_index = 0
    schedule = make_schedule()

    # pylint: disable=line-too-long
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
          wait(60000);\t\t// clock=2\t n_instr=3
          executeTableEntry(0);\t// clock=60005 pulse=0 n_instr=0
          setTrigger(0);\t// clock=60005 n_instr=1
          wait(124);\t\t// clock=60006, dead time to ensure total schedule duration\t n_instr=3
        }
        """
    ).lstrip("\n")
    # pylint: enable=line-too-long

    # Act
    comp_sched = zhinst_backend.compile_backend(schedule, hdawg_hardware_cfg)
    compiled_instructions = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_hdawg0" in compiled_instructions
    zi_device_config: zhinst_backend.ZIDeviceConfig = compiled_instructions["ic_hdawg0"]

    awg_settings = zi_device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[awg_index]  # for awg index 0

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


# pylint: disable=too-many-arguments
@pytest.mark.parametrize("channelgrouping,enabled_channels", [(0, [0, 1]), (1, [0])])
def test__program_hdawg4_channelgrouping(
    mocker,
    create_typical_timing_table,
    channelgrouping: int,
    enabled_channels: List[int],
):

    test_config_dict = create_typical_timing_table()
    schedule = test_config_dict["schedule"]
    timing_table = test_config_dict["timing_table"]
    devices = test_config_dict["devices"]
    numerical_wf_dict = test_config_dict["numerical_wf_dict"]

    hdawg_device = devices[0]
    hdawg_device.channelgrouping = channelgrouping
    hdawg_device.clock_rate = int(2.4e9)

    settings_builder = settings.ZISettingsBuilder()

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

    assert (
        str(execinfo.value)
        == "Undefined timing constraints for schedule 'Empty Experiment'!"
    )

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
        (True, enums.ModulationModeType.PREMODULATE, True),
        (False, enums.ModulationModeType.PREMODULATE, True),
        (False, enums.ModulationModeType.NONE, False),
    ],
)
def test_apply_waveform_corrections(
    mocker,
    is_pulse: bool,
    modulation_type: enums.ModulationModeType,
    expected_modulated: bool,
):
    # Arrange
    wave = np.ones(48)

    modulate_wave = mocker.patch.object(
        waveform_helpers, "modulate_waveform", return_value=wave
    )
    shift_waveform = mocker.patch.object(
        waveform_helpers, "shift_waveform", return_value=(0, wave)
    )
    resize_waveform = mocker.patch.object(
        waveform_helpers, "resize_waveform", return_value=wave
    )

    channel = zhinst.Output(
        "port",
        "clock",
        enums.SignalModeType.COMPLEX,
        common.Modulation(modulation_type),
        common.LocalOscillator("lo0", 6.02e9),
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

    modulation_type = enums.ModulationModeType.MODULATE

    channel = zhinst.Output(
        "port",
        "clock",
        enums.SignalModeType.COMPLEX,
        common.Modulation(modulation_type),
        common.LocalOscillator("lo0", 6.02e9),
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


def test__flatten_dict():
    # Arrange
    collection = {0: [0, 1, 2]}

    # Act
    result = zhinst_backend._flatten_dict(collection)

    # Assert
    assert list(result) == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]


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
            waveform_id=timing_table.waveform_id[2],
            abs_time=timing_table.abs_time[2],
            duration=timing_table.duration[2],
            clock_cycle_start=timing_table.clock_cycle_start[2],
        ),
        zhinst.Acquisition(
            waveform_id=timing_table.waveform_id[3],
            abs_time=timing_table.abs_time[3],
            duration=timing_table.duration[3],
            clock_cycle_start=timing_table.clock_cycle_start[3],
        ),
    ]

    expected_instructions_list = {
        "ic_hdawg0.awg0": hdawg0awg0_expected_list,
        "ic_uhfqa0.awg0": uhfqa0awg0_expected_list,
    }

    for hardware_channel in expected_instructions_list:

        # select only the instructions relevant for the output channel.
        output_timing_table = timing_table[
            timing_table["hardware_channel"] == hardware_channel
        ]

        # Act
        instructions = zhinst_backend._get_instruction_list(output_timing_table)

        # Assert
        expected_instructions = expected_instructions_list[hardware_channel]
        for expected_instruction, instruction in zip(
            expected_instructions, instructions
        ):
            assert instruction == expected_instruction


def test_uhfqa_sequence1(
    make_schedule,
    load_example_zhinst_hardware_config,
) -> None:
    uhfqa_hardware_cfg = load_example_zhinst_hardware_config()
    # Arrange
    awg_index = 0
    schedule = make_schedule()

    # pylint: disable=line-too-long
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
          startQA(QA_INT_ALL, true);\t// clock=45033 n_instr=6
        }
        """
    ).lstrip("\n")
    # pylint: enable=line-too-long

    # Act
    comp_sched = zhinst_backend.compile_backend(schedule, uhfqa_hardware_cfg)
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
    load_example_zhinst_hardware_config,
):
    uhfqa_hardware_cfg = load_example_zhinst_hardware_config()
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

    # pylint: disable=line-too-long
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
          startQA(QA_INT_ALL, true);\t// clock=2250 n_instr=6
        }
        """
    ).lstrip("\n")
    # pylint: enable=line-too-long

    # Act
    comp_sched = zhinst_backend.compile_backend(schedule, uhfqa_hardware_cfg)
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
    load_example_zhinst_hardware_config,
) -> None:
    uhfqa_hardware_cfg = load_example_zhinst_hardware_config()
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

    # pylint: disable=line-too-long

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
          startQA(QA_INT_ALL, true);\t// clock=2292 n_instr=6
          wait(3);\t\t// clock=2298\t n_instr=3
          playWave(w1);\t// \t// clock=2301\t n_instr=0
        }
        """
    ).lstrip("\n")
    # pylint: enable=line-too-long

    # Act
    comp_sched = zhinst_backend.compile_backend(schedule, uhfqa_hardware_cfg)
    device_configs = comp_sched["compiled_instructions"]

    # Assert
    assert "ic_uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["ic_uhfqa0"]

    awg_settings = device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[0]

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


def test__extract_port_clock_channelmapping_hdawg(
    load_example_zhinst_hardware_config,
) -> None:
    hardware_config = load_example_zhinst_hardware_config()

    expected_dict = {
        "q0:mw-q0.01": "ic_hdawg0.awg0",
        "q1:mw-q1.01": "ic_hdawg0.awg1",
        "q0:res-q0.ro": "ic_uhfqa0.awg0",
    }
    generated_dict = zhinst_backend._extract_port_clock_channelmapping(
        hardware_cfg=hardware_config
    )
    assert generated_dict == expected_dict


def test__extract_channel_latencies_line_trigger_delay(
    load_example_zhinst_hardware_config,
) -> None:
    hardware_config = load_example_zhinst_hardware_config()

    expected_latency_dict = {
        "ic_hdawg0.awg0": 10e-9,
        "ic_hdawg0.awg0.trigger": hardware_config.get("devices")[0]
        .get("channel_0")
        .get("line_trigger_delay"),
        "ic_hdawg0.awg1": 10e-9,
        "ic_hdawg0.awg1.trigger": hardware_config.get("devices")[0]
        .get("channel_1")
        .get("line_trigger_delay"),
        "ic_uhfqa0.awg0": 0,
        "ic_uhfqa0.awg0.acquisition": 0,
    }
    generated_dict = zhinst_backend._extract_channel_latencies(
        hardware_cfg=hardware_config
    )

    assert generated_dict == expected_latency_dict


@pytest.mark.parametrize(
    "device_type", [(zhinst.DeviceType.HDAWG), (zhinst.DeviceType.UHFQA)]
)
def test__add_wave_nodes_with_vector(
    create_typical_timing_table, device_type: zhinst.DeviceType
):
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
    expected_data = (_data.reshape((-2,), order="F") * (2 ** 15 - 1)).astype("int16")

    awg_index: int = 0
    settings_builder = settings.ZISettingsBuilder()

    for device in devices:
        if device.device_type == device_type:
            # select only the instructions relevant for the output channel.
            output_timing_table = timing_table[
                timing_table["hardware_channel"] == f"{device.name}.awg{awg_index}"
            ]
            # enumerate the waveform_ids used in this particular output channel
            unique_wf_ids = output_timing_table.drop_duplicates(subset="waveform_id")[
                "waveform_id"
            ]
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
    make_schedule,
):
    # Arrange
    q0 = "q0"
    schedule = Schedule("test")
    schedule.add(Reset(q0))
    schedule.add(X90(q0))
    # no measure to keep it simple

    device_cfg = load_json_example_scheme("transmon_test_config.json")

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

    # Act
    with pytest.raises(
        KeyError, match='Missing configuration for LocalOscillator "lo_unknown"'
    ):
        qcompile(schedule, device_cfg, zhinst_hardware_cfg)


def test_compile_backend_with_duplicate_local_oscillator(
    make_schedule,
):
    # Arrange
    q0 = "q0"
    schedule = Schedule("test")
    schedule.add(Reset(q0))
    schedule.add(X90(q0))
    device_cfg = load_json_example_scheme("transmon_test_config.json")

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

    # Act
    with pytest.raises(RuntimeError) as execinfo:
        qcompile(schedule, device_cfg, zhinst_hardware_cfg)

    # Assert
    assert (
        str(execinfo.value)
        == "Duplicate entry LocalOscillators 'lo0' in hardware configuration!"
    )


def test_acquisition_staircase_unique_acquisitions(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)
    schedule = acquisition_staircase_sched(
        np.linspace(0, 1, 20),
        readout_pulse_duration=1e-6,
        readout_frequency=6e9,
        acquisition_delay=100e-9,
        integration_time=2e-6,
        port="q0:res",
        clock="q0.ro",
        repetitions=1024,
    )
    device_cfg = load_json_example_scheme("transmon_test_config.json")
    hw_cfg = load_json_example_scheme("zhinst_test_mapping.json")

    # Act
    comp_sched = qcompile(schedule, device_cfg=device_cfg, hardware_cfg=hw_cfg)

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


def test_acquisition_staircase_right_acq_channel(tmp_test_data_dir):
    set_datadir(tmp_test_data_dir)

    acq_channel = 2
    schedule = acquisition_staircase_sched(
        np.linspace(0, 1, 20),
        readout_pulse_duration=1e-6,
        readout_frequency=6e9,
        acquisition_delay=100e-9,
        integration_time=2e-6,
        port="q0:res",
        clock="q0.ro",
        repetitions=1024,
        acq_channel=acq_channel,
    )
    device_cfg = load_json_example_scheme("transmon_test_config.json")
    hw_cfg = load_json_example_scheme("zhinst_test_mapping.json")

    # Act
    comp_sched = qcompile(schedule, device_cfg=device_cfg, hardware_cfg=hw_cfg)

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
