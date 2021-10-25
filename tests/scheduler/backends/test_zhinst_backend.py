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

from zhinst.qcodes import hdawg, mfli, uhfli, uhfqa
from zhinst.toolkit.control import drivers


from quantify_core.data.handling import set_datadir

from quantify_scheduler.schedules.verification import acquisition_staircase_sched

from quantify_scheduler.schemas.examples.utils import load_json_example_scheme
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.backends import zhinst_backend
from quantify_scheduler import enums
from quantify_scheduler.helpers import waveforms as waveform_helpers
from quantify_scheduler.helpers import schedule as schedule_helpers

from quantify_scheduler.backends.types import zhinst
from quantify_scheduler.backends.types import common
from quantify_scheduler.backends.zhinst import settings
from quantify_scheduler.gate_library import X90, Measure, Reset
from quantify_scheduler.schedules import trace_schedules
from quantify_scheduler.schedules import spectroscopy_schedules
from quantify_scheduler import types


@pytest.fixture
def uhfqa_hardware_map() -> Dict[str, Any]:
    return json.loads(
        """
        {
          "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
          "local_oscillators": [{
            "name": "lo0",
            "frequency": null
          }],
          "devices": [
            {
              "name": "uhfqa0",
              "type": "UHFQA",
              "ref": "ext",
              "channel_0": {
                "port": "q0:res",
                "clock": "q0.ro",
                "mode": "real",
                "modulation": {
                  "type": "premod",
                  "interm_freq": 150e6
                },
                "local_oscillator": "lo0",
                "triggers": [
                  2
                ]
              }
            }
          ]
        }
        """
    )


@pytest.fixture
def hdawg_hardware_map() -> Dict[str, Any]:
    return json.loads(
        """
        {
          "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
          "local_oscillators": [{
            "name": "lo0",
            "frequency": null
          }],
          "devices": [
            {
              "name": "hdawg0",
              "type": "HDAWG4",
              "ref": "int",
              "channelgrouping": 0,
              "channel_0": {
                "port": "q0:mw",
                "clock": "q0.01",
                "mode": "complex",
                "modulation": {
                  "type": "premod",
                  "interm_freq": -50e6
                },
                "local_oscillator": "lo0",
                "markers": [
                  "AWG_MARKER1",
                  "AWG_MARKER2"
                ],
                "gain1": 1,
                "gain2": 1,
                "mixer_corrections": {
                  "amp_ratio": 0.81,
                  "phase_error": 0.07,
                  "dc_offset_I": 0.4,
                  "dc_offset_Q": 0.38
                }
              },
              "channel_1": {
                "port": "q1:mw",
                "clock": "q1.01",
                "mode": "complex",
                "modulation": {
                  "type": "premod",
                  "interm_freq": -50e6
                },
                "local_oscillator": "lo0",
                "triggers": [
                  1
                ]
              }
            }
          ]
        }
        """
    )


@pytest.fixture
def create_device():
    def _create_device(hardware_map: Dict[str, Any], index: int = 0) -> zhinst.Device:
        return zhinst.Device.schema().load(hardware_map["devices"][index])

    yield _create_device


@pytest.fixture
def create_uhfqa_mock(mocker):
    def _create_uhfqa_mock() -> uhfqa.UHFQA:
        features_mock = mocker.Mock()
        features_mock.parameters = {"devtype": mocker.Mock(return_value="UHFQA")}

        mock = mocker.Mock(spec=uhfqa.UHFQA)
        mock.configure_mock(
            **{
                "features": features_mock,
                "name": "hdawg0",
                "_serial": "dev2299",
                "result_source": mocker.Mock(),
            }
        )

        def create_uhfqa_awg(i: int) -> uhfqa.AWG:
            _sequence_params = {
                "sequence_parameters": {
                    "clock_rate": 1.8e9,
                }
            }  # GSa/s

            def get_string(value: str):
                if value == "directory":
                    return "./"
                return ""

            _module_mock = mocker.Mock()
            _module_mock.get_string = get_string
            _awg_attrs = {
                "_index": i,
                "sequence_params": _sequence_params,
                "_module": _module_mock,
            }

            awg_mock = mocker.Mock(
                spec=uhfqa.AWG,
            )
            awg_mock.configure_mock(
                **{
                    "name": "uhfqa_awg",
                    "index": i,
                    "output1": mocker.Mock(return_value=None),
                    "output2": mocker.Mock(return_value=None),
                    "_awg": mocker.Mock(spec=drivers.uhfqa.AWG, **_awg_attrs),
                }
            )

            return awg_mock

        def create_uhfqa_readoutchannel(i: int) -> uhfqa.ReadoutChannel:
            attrs = {
                "_index": i,
                "rotation": mocker.Mock(),
            }
            return mocker.MagicMock(spec=uhfqa.ReadoutChannel, **attrs)

        mock.awg = create_uhfqa_awg(0)
        mock.channels = list(map(create_uhfqa_readoutchannel, range(10)))

        return mock

    yield _create_uhfqa_mock


@pytest.fixture
def create_hdawg_mock(mocker):
    def _create_hdawg_mock(channels: int) -> hdawg.HDAWG:
        awg_count = int(channels / 2)
        features_mock = mocker.Mock()
        features_mock.parameters = {
            "devtype": mocker.Mock(return_value=f"HDAWG{channels}")
        }

        mock = mocker.Mock(
            spec=hdawg.HDAWG,
        )
        mock.configure_mock(
            **{
                "features": features_mock,
                "name": "hdawg0",
                "_serial": "dev8030",
            }
        )

        def create_hdawg_awg(i: int):
            # https://www.zhinst.com/sites/default/files/documents/2020-09/ziHDAWG_UserManual_20.07.1.pdf
            # Section: 4.14.3 Constansts and Variables (page 181)
            _sequence_params = {
                "sequence_parameters": {
                    "clock_rate": 2.4e9,
                }
            }  # GSa/s

            def get_string(value: str):
                if value == "directory":
                    return "./"
                return ""

            _module_mock = mocker.Mock()
            _module_mock.get_string = get_string
            _awg_attrs = {
                "_index": i,
                "sequence_params": _sequence_params,
                "_module": _module_mock,
            }

            awg_mock = mocker.Mock(spec=hdawg.AWG)
            awg_mock.configure_mock(
                **{
                    "name": f"hdawg_awg-{i}",
                    "index": i,
                    "output1": mocker.Mock(return_value=None),
                    "output2": mocker.Mock(return_value=None),
                    "_awg": mocker.Mock(spec=drivers.hdawg.AWG, **_awg_attrs),
                    "modulation_freq": mocker.Mock(),
                    "modulation_phase_shift": mocker.Mock(),
                    "gain1": mocker.Mock(),
                    "gain2": mocker.Mock(),
                }
            )
            return awg_mock

        mock.awgs = list(map(create_hdawg_awg, range(awg_count)))
        return mock

    yield _create_hdawg_mock


@pytest.fixture
def make_schedule(create_schedule_with_pulse_info):
    def _make_schedule() -> types.Schedule:
        q0 = "q0"
        schedule = types.Schedule("test")
        schedule.add(Reset(q0))
        schedule.add(X90(q0))
        schedule.add(Measure(q0))
        return create_schedule_with_pulse_info(schedule)

    yield _make_schedule


@pytest.mark.parametrize(
    "unsupported_device_type", [(zhinst.DeviceType.UHFLI), (zhinst.DeviceType.MFLI)]
)
def test_compile_backend_unsupported_devices(
    mocker, unsupported_device_type: zhinst.DeviceType, create_schedule_with_pulse_info
):
    # Arrange
    hardware_map_str = (
        """
    {
        "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
        "local_oscillators": [{
            "name": "lo0",
            "frequency": 4.8e9
        }],
        "devices": [
            {
                "name": "device_name",
                "type": """
        + f'"{unsupported_device_type.value}",'
        + """
                "ref": "none",
                "channel_0": {
                    "port": "q0:mw",
                    "clock": "q0.ro",
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
    )
    zhinst_hardware_map = json.loads(hardware_map_str)

    if unsupported_device_type == zhinst.DeviceType.UHFLI:
        instrument = mocker.create_autospec(uhfli.UHFLI, instance=True)
    elif unsupported_device_type == zhinst.DeviceType.MFLI:
        instrument = mocker.create_autospec(mfli.MFLI, instance=True)

    mocker.patch(
        "qcodes.instrument.base.Instrument.find_instrument",
        return_value=instrument,
    )

    # Act
    with pytest.raises(NotImplementedError) as execinfo:
        zhinst_backend.compile_backend(
            create_schedule_with_pulse_info(), zhinst_hardware_map
        )

    # Assert
    assert (
        str(execinfo.value)
        == f"Unable to create zhinst backend for '{unsupported_device_type.value}'!"
    )


def test_compile_hardware_hdawg4_successfully(
    mocker,
    create_schedule_with_pulse_info,
    hdawg_hardware_map: Dict[str, Any],
) -> None:
    # Arrange
    (q0, q1) = ("q0", "q1")
    schedule = types.Schedule("test")
    schedule.add(Reset(q0, q1))
    schedule.add(X90(q0))
    schedule.add(X90(q1))
    schedule = create_schedule_with_pulse_info(schedule)

    modulate_wave_spy = mocker.patch.object(
        waveform_helpers, "modulate_waveform", wraps=waveform_helpers.modulate_waveform
    )
    settings_builder = mocker.Mock(wraps=settings.ZISettingsBuilder())
    mocker.patch.object(settings, "ZISettingsBuilder", return_value=settings_builder)

    expected_settings = {
        "sigouts/*/on": 0,
        "awgs/*/single": 1,
        "system/awg/channelgrouping": 0,
        "awgs/0/time": 0,
        "awgs/1/time": 0,
        "sigouts/0/on": 1,
        "sigouts/1/on": 1,
        "sigouts/0/offset": 0.4,
        "sigouts/1/offset": 0.38,
        "sigouts/2/on": 1,
        "sigouts/3/on": 1,
        "awgs/0/outputs/0/gains/0": 1,
        "awgs/0/outputs/1/gains/1": 1,
        "awgs/1/outputs/0/gains/0": 0,
        "awgs/1/outputs/1/gains/1": 0,
        "sigouts/2/offset": 0.0,
        "sigouts/3/offset": 0.0,
        "awgs/0/commandtable/data": ANY,
        "awgs/0/waveform/waves/0": ANY,
        "awgs/1/commandtable/data": ANY,
        "awgs/1/waveform/waves/0": ANY,
        "compiler/sourcestring": ANY,
    }

    # Act
    device_configs = zhinst_backend.compile_backend(schedule, hdawg_hardware_map)

    # Assert
    assert "hdawg0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["hdawg0"]
    zi_settings = device_config.settings_builder.build()
    collection = zi_settings.as_dict()

    for key, expected_value in expected_settings.items():
        assert key in collection
        if isinstance(expected_value, type(ANY)):
            continue
        assert (
            collection[key] == expected_value
        ), f"Expected {key} {collection[key]} to equal {expected_value}"

    modulate_wave_spy.assert_called()

    expected_call = [call(0, 0, ANY), call(1, 0, ANY)]
    expected_lengths = [96, 128]
    assert settings_builder.with_wave_vector.call_args_list == expected_call

    # Assert waveform sizes
    for i, call_args in enumerate(settings_builder.with_wave_vector.call_args_list):
        args, _ = call_args
        waveform_data = args[2]
        assert isinstance(waveform_data, (np.ndarray, np.generic))
        assert len(waveform_data) == (expected_lengths[i])

    expected_call = [call(0, ANY), call(1, ANY)]
    assert settings_builder.with_commandtable_data.call_args_list == expected_call

    expected_call = [call(0, ANY), call(1, ANY)]
    assert settings_builder.with_compiler_sourcestring.call_args_list == expected_call

    assert "lo0" in device_configs

    freq_qubit = 6.02e9  # from the example transmon config, this is the RF frequency
    intermodulation_frequency = -50e6
    assert device_configs["lo0"] == freq_qubit - intermodulation_frequency


def test_compile_hardware_uhfqa_successfully(
    mocker,
    make_schedule,
    uhfqa_hardware_map: Dict[str, Any],
) -> None:
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
    device_configs = zhinst_backend.compile_backend(schedule, uhfqa_hardware_map)

    # Assert
    assert "uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["uhfqa0"]
    zi_settings = device_config.settings_builder.build()
    compiled_settings = zi_settings.as_dict()

    for key, expected_value in expected_settings.items():
        assert key in compiled_settings.keys()
        if isinstance(expected_value, type(ANY)):
            continue
        assert compiled_settings[key] == expected_value

    assert "lo0" in device_configs
    ro_freq = 7.04e9
    intermodulation_frequency = 150e6

    assert device_configs["lo0"] == ro_freq - intermodulation_frequency


def test_hdawg4_sequence(
    hdawg_hardware_map: Dict[str, Any],
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
    wave w0 = placeholder(48);\n
    // Operations
    // Schedule offset: 0.000200000s 60000 clocks
    // Schedule duration: 0.000000436s 131 clocks
    // Sequence start: 0.000000000s 0 clocks
    // Sequence duration: 0.000000016s 5 clocks
    // Sequence end: 0.000000016s 5 clocks
    // Line delay: -1.000000000s 0 clocks
    assignWaveIndex(w0, w0, 0);
    setTrigger(0);\t//  n_instr=1
    repeat(__repetitions__)
    {
      setTrigger(AWG_MARKER1 + AWG_MARKER2);\t//  n_instr=2
      executeTableEntry(0);\t// clock=0 pulse=0 n_instr=0
      setTrigger(0);\t// clock=0 n_instr=1
      // Dead time
      wait(60123);\t// \t// clock=1 n_instr=3
    }
    setTrigger(0);\t// \t// clock=60127 n_instr=1
    """
    ).lstrip("\n")

    # Act
    compiled_instructions = zhinst_backend.compile_backend(schedule, hdawg_hardware_map)

    # Assert
    assert "hdawg0" in compiled_instructions
    zi_device_config: zhinst_backend.ZIDeviceConfig = compiled_instructions["hdawg0"]

    awg_settings = zi_device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[awg_index]  # for awg index 0

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


# pylint: disable=too-many-arguments
@pytest.mark.parametrize("channelgrouping,enabled_channels", [(0, [0, 1]), (1, [0])])
def test__program_hdawg4_channelgrouping(
    mocker,
    create_device,
    create_schedule_with_pulse_info,
    hdawg_hardware_map,
    channelgrouping: int,
    enabled_channels: List[int],
):
    # Arrange
    (q0, q1) = ("q0", "q1")
    schedule = types.Schedule("test")
    schedule.add(Reset(q0, q1))
    schedule.add(X90(q0))
    schedule.add(X90(q1))
    schedule = create_schedule_with_pulse_info(schedule)
    schedule = schedule_helpers.CachedSchedule(schedule)

    device: zhinst.Device = create_device(hdawg_hardware_map)
    device.channelgrouping = channelgrouping
    device.clock_rate = int(2.4e9)

    settings_builder = settings.ZISettingsBuilder()

    mocker.patch.object(zhinst_backend, "_add_wave_nodes")
    with_sigouts = mocker.patch.object(settings.ZISettingsBuilder, "with_sigouts")
    with_system_channelgrouping = mocker.patch.object(
        settings.ZISettingsBuilder, "with_system_channelgrouping"
    )

    # Act
    zhinst_backend._compile_for_hdawg(device, schedule, settings_builder)

    # Assert
    with_system_channelgrouping.assert_called_with(channelgrouping)
    calls = list(map(lambda i: call(i, (1, 1)), enabled_channels))
    assert with_sigouts.call_args_list == calls


def test_validate_schedule(
    empty_schedule: types.Schedule,
    basic_schedule: types.Schedule,
    schedule_with_pulse_info: types.Schedule,
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
    instrument_info = zhinst.InstrumentInfo(2.4e9, 8, 16)

    # Act
    result = zhinst_backend.apply_waveform_corrections(
        channel, wave, (0, 16e-9), instrument_info, is_pulse
    )

    # Assert
    assert (0, 48, wave) == result
    if expected_modulated:
        modulate_wave.assert_called()
    else:
        modulate_wave.assert_not_called()
    shift_waveform.assert_called()
    resize_waveform.assert_called()


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


def test_get_wave_instruction(mocker, create_schedule_with_pulse_info):
    # Arrange
    q0 = "q0"
    schedule = types.Schedule("test")
    schedule.add(X90(q0))
    schedule = create_schedule_with_pulse_info(schedule)
    schedule = schedule_helpers.CachedSchedule(schedule)

    uuid = next(iter(schedule.pulseid_pulseinfo_dict.keys()))
    output = mocker.create_autospec(zhinst.Output, instance=True)
    instrument_info = zhinst.InstrumentInfo(2.4e9, 8, 16)

    wave = np.concatenate([np.zeros(9), np.ones(39)])
    mocker.patch.object(
        zhinst_backend, "apply_waveform_corrections", return_value=(1, 9, wave)
    )

    get_pulse_uuid = mocker.patch.object(
        schedule_helpers, "get_pulse_uuid", return_value="new_pulse"
    )

    # Act
    instruction = zhinst_backend.get_wave_instruction(
        uuid, 0, output, schedule, instrument_info
    )

    # Assert
    get_pulse_uuid.assert_called()
    assert isinstance(instruction, zhinst.Wave)
    assert instruction.uuid == "new_pulse"
    assert instruction.n_samples_scaled == 9


def test_get_wave_instruction_mode_is_calibrating(
    mocker, create_schedule_with_pulse_info
):
    # Arrange
    q0 = "q0"
    schedule = types.Schedule("test")
    schedule.add(X90(q0))
    schedule = create_schedule_with_pulse_info(schedule)
    schedule = schedule_helpers.CachedSchedule(schedule)

    uuid = next(iter(schedule.pulseid_pulseinfo_dict.keys()))
    output = mocker.create_autospec(zhinst.Output, instance=True)
    instrument_info = zhinst.InstrumentInfo(
        2.4e9, 8, 16, enums.InstrumentOperationMode.CALIBRATING
    )

    def apply_waveform_corrections(  # pylint: disable=unused-argument
        output,
        waveform,
        start_and_duration_in_seconds,
        instrument_info,
        is_pulse,
    ):
        return (start_and_duration_in_seconds[0], len(waveform), waveform)

    mocker.patch.object(
        zhinst_backend,
        "apply_waveform_corrections",
        side_effect=apply_waveform_corrections,
    )

    # Act
    instruction = zhinst_backend.get_wave_instruction(
        uuid, 0, output, schedule, instrument_info
    )

    # Assert
    assert isinstance(instruction, zhinst.Wave)
    np.testing.assert_array_equal(
        instruction.waveform, np.ones(len(instruction.waveform))
    )


def test_get_measure_instruction(mocker, create_schedule_with_pulse_info):
    # Arrange
    q0 = "q0"
    schedule = types.Schedule("test")
    schedule.add(Measure(q0))
    schedule = create_schedule_with_pulse_info(schedule)
    schedule = schedule_helpers.CachedSchedule(schedule)

    uuid = next(iter(schedule.acqid_acqinfo_dict.keys()))
    output = mocker.create_autospec(zhinst.Output, instance=True)
    instrument_info = zhinst.InstrumentInfo(1.8e9, 8, 16)

    wave = np.concatenate([np.zeros(9), np.ones(39)])
    mocker.patch.object(
        zhinst_backend, "apply_waveform_corrections", return_value=(1, 9, wave)
    )

    # Act
    instruction = zhinst_backend.get_measure_instruction(
        uuid, 0, output, schedule, instrument_info
    )

    # Assert
    assert isinstance(instruction, zhinst.Measure)


def test_uhfqa_sequence1(
    make_schedule,
    uhfqa_hardware_map: Dict[str, Any],
) -> None:
    # Arrange
    awg_index = 0
    schedule = make_schedule()

    # pylint: disable=line-too-long
    expected_seqc = dedent(
        """\
    // Generated by quantify-scheduler.
    // Variables
    var __repetitions__ = 1;
    wave w0 = "uhfqa0_awg0_wave0";

    // Operations
    // Schedule offset: 0.000200000s 45000 clocks
    // Schedule duration: 0.000000436s 98 clocks
    // Sequence start: 0.000000016s 4 clocks
    // Sequence duration: 0.000000420s 94 clocks
    // Sequence end: 0.000000436s 98 clocks
    // Line delay: -1.000000000s 0 clocks
    repeat(__repetitions__)
    {
      waitDigTrigger(2, 1);\t// clock=0
      wait(4);\t// clock=0 n_instr=4
      playWave(w0);\t// clock=4 n_instr=0
      wait(25);\t// \t// clock=4 n_instr=25
      startQAMonitor();
      startQAResult();

      wait(2000);\t// \t// clock=29 n_instr=2000
    }
    setTrigger(0);\t// Reset triggers n_instr=1
    """
    ).lstrip("\n")
    # pylint: enable=line-too-long

    # Act
    device_configs = zhinst_backend.compile_backend(schedule, uhfqa_hardware_map)

    # Assert
    assert "uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["uhfqa0"]

    awg_settings = device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[0]

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


def test_uhfqa_sequence2(
    create_schedule_with_pulse_info,
    uhfqa_hardware_map: Dict[str, Any],
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

    # pylint: disable=line-too-long
    expected_seqc = dedent(
        """\
    // Generated by quantify-scheduler.
    // Variables
    var __repetitions__ = 1;
    wave w0 = "uhfqa0_awg0_wave0";

    // Operations
    // Schedule offset: 0.000010000s 2250 clocks
    // Schedule duration: 0.000001000s 225 clocks
    // Sequence start: 0.000000000s 0 clocks
    // Sequence duration: 0.000001000s 225 clocks
    // Sequence end: 0.000001000s 225 clocks
    // Line delay: -1.000000000s 0 clocks
    repeat(__repetitions__)
    {
      waitDigTrigger(2, 1);\t// clock=0
      playWave(w0);\t// clock=0 n_instr=0
      startQAMonitor();
      startQAResult();

      wait(2000);\t// \t// clock=0 n_instr=2000
    }
    setTrigger(0);\t// Reset triggers n_instr=1
    """
    ).lstrip("\n")
    # pylint: enable=line-too-long

    # Act
    device_configs = zhinst_backend.compile_backend(schedule, uhfqa_hardware_map)

    # Assert
    assert "uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["uhfqa0"]

    awg_settings = device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[0]

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


def test_uhfqa_sequence3(
    create_schedule_with_pulse_info,
    uhfqa_hardware_map: Dict[str, Any],
) -> None:
    # Arrange
    awg_index = 0
    ro_acquisition_delay = -16e-9
    ro_pulse_delay = 2e-9
    schedule = spectroscopy_schedules.two_tone_spec_sched(
        spec_pulse_amp=0.6e-0,
        spec_pulse_duration=16e-9,
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
    wave w0 = "uhfqa0_awg0_wave0";

    // Operations
    // Schedule offset: 0.000010000s 2250 clocks
    // Schedule duration: 0.000000502s 113 clocks
    // Sequence start: 0.000000002s 0 clocks
    // Sequence duration: 0.000000166s 38 clocks
    // Sequence end: 0.000000168s 38 clocks
    // Line delay: -1.000000000s 0 clocks
    repeat(__repetitions__)
    {
      waitDigTrigger(2, 1);\t// clock=0
      startQAMonitor();
      startQAResult();

      wait(4);\t// \t// clock=0 n_instr=4
      playWave(w0);\t// clock=4 n_instr=0
    }
    setTrigger(0);\t// Reset triggers n_instr=1
    """
    ).lstrip("\n")
    # pylint: enable=line-too-long

    # Act
    device_configs = zhinst_backend.compile_backend(schedule, uhfqa_hardware_map)

    # Assert
    assert "uhfqa0" in device_configs
    device_config: zhinst_backend.ZIDeviceConfig = device_configs["uhfqa0"]

    awg_settings = device_config.settings_builder._awg_settings
    (node, zi_setting) = awg_settings[0]

    assert node == "compiler/sourcestring"
    assert zi_setting[0] == awg_index
    assert zi_setting[1].value == expected_seqc


@pytest.mark.parametrize(
    "device_type", [(zhinst.DeviceType.HDAWG), (zhinst.DeviceType.UHFQA)]
)
def test__add_wave_nodes_with_vector(mocker, device_type: zhinst.DeviceType):
    # Arrange
    waveform = np.vectorize(complex)(np.zeros(1024), np.ones(1024))
    waveforms_dict = {0: waveform}
    waveform_table = {0: 0}

    awg_index: int = 0
    settings_builder = mocker.Mock(spec=settings.ZISettingsBuilder)
    device = mocker.create_autospec(zhinst.Device, instance=True)
    device.device_type = device_type

    _data = np.zeros((2, 1024))
    _data[0] = np.real(waveform)
    _data[1] = np.imag(waveform)
    expected_data = (_data.reshape((-2,), order="F") * (2 ** 15 - 1)).astype("int16")

    # Act
    zhinst_backend._add_wave_nodes(
        device, awg_index, waveforms_dict, waveform_table, settings_builder
    )

    # Assert
    expected_call = [call(awg_index, 0, ANY)]
    if device_type == zhinst.DeviceType.HDAWG:
        assert settings_builder.with_wave_vector.call_args_list == expected_call
        args, _ = settings_builder.with_wave_vector.call_args
    elif device_type == zhinst.DeviceType.UHFQA:
        assert settings_builder.with_csv_wave_vector.call_args_list == expected_call
        args, _ = settings_builder.with_csv_wave_vector.call_args

    np.testing.assert_array_equal(args[2], expected_data)


def test_compile_backend_with_undefined_local_oscillator(
    make_schedule,
):
    # Arrange
    schedule = make_schedule()
    hardware_map_str = """
    {
        "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
        "local_oscillators": [{
            "name": "lo0",
            "frequency": 4.8e9
        }],
        "devices": [
            {
                "name": "device_name",
                "type": "HDAWG4",
                "ref": "none",
                "channel_0": {
                    "port": "q0:mw",
                    "clock": "q0.ro",
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
    zhinst_hardware_map = json.loads(hardware_map_str)

    # Act
    with pytest.raises(
        KeyError, match='Missing configuration for LocalOscillator "lo_unknown"'
    ):
        zhinst_backend.compile_backend(schedule, zhinst_hardware_map)


def test_compile_backend_with_duplicate_local_oscillator(
    make_schedule,
):
    # Arrange
    schedule = make_schedule()
    hardware_map_str = """
    {
      "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
      "local_oscillators": [
        {
          "name": "lo0",
          "frequency": 4.7e9
        },
        {
          "name": "lo0",
          "frequency": 4.8e9
        }
      ],
      "devices": [
        {
          "name": "device_name",
          "type": "HDAWG4",
          "ref": "none",
          "channel_0": {
            "port": "q0:mw",
            "clock": "q0.ro",
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
    zhinst_hardware_map = json.loads(hardware_map_str)

    # Act
    with pytest.raises(RuntimeError) as execinfo:
        zhinst_backend.compile_backend(schedule, zhinst_hardware_map)

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
    comp_sched = qcompile(schedule, device_cfg=device_cfg, hardware_mapping=hw_cfg)

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

    two_us_array = np.zeros(4096)
    two_us_array[:3600] = 1
    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/0/real"],
        two_us_array,
    )
    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/0/imag"],
        two_us_array,
    )

    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/1/real"],
        two_us_array,
    )
    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/1/imag"],
        -1 * two_us_array,
    )

    for i in [2, 3, 4, 5, 6, 7, 8, 9]:
        np.testing.assert_array_equal(
            settings_dict[f"qas/0/integration/weights/{i}/real"],
            np.zeros(4096),
        )
        np.testing.assert_array_equal(
            settings_dict[f"qas/0/integration/weights/{i}/imag"],
            np.zeros(4096),
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
    comp_sched = qcompile(schedule, device_cfg=device_cfg, hardware_mapping=hw_cfg)

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

    two_us_array = np.zeros(4096)
    two_us_array[:3600] = 1
    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/4/real"],
        two_us_array,
    )
    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/4/imag"],
        two_us_array,
    )

    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/5/real"],
        two_us_array,
    )
    np.testing.assert_array_equal(
        settings_dict["qas/0/integration/weights/5/imag"],
        -1 * two_us_array,
    )

    for i in [0, 1, 2, 3, 6, 7, 8, 9]:
        np.testing.assert_array_equal(
            settings_dict[f"qas/0/integration/weights/{i}/real"],
            np.zeros(4096),
        )
        np.testing.assert_array_equal(
            settings_dict[f"qas/0/integration/weights/{i}/imag"],
            np.zeros(4096),
        )
