# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY, call

import numpy as np
import pytest
from zhinst.qcodes import base

from quantify_scheduler import waveforms
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.types import zhinst as zi_types
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst import settings
from quantify_scheduler.schedules.verification import awg_staircase_sched


def make_ufhqa(mocker) -> base.ZIBaseInstrument:
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument.name = "uhfqa0"
    instrument._serial = "dev1234"
    instrument._type = "uhfqa"
    return instrument


def test_zi_settings_equality(
    mock_setup_basic_transmon, load_example_zhinst_hardware_config
):
    sched_kwargs = {
        "pulse_amps": np.linspace(0, 0.5, 11),
        "pulse_duration": 1e-6,
        "readout_frequency": 5e9,
        "acquisition_delay": 0,
        "integration_time": 2e-6,
        "mw_port": "q0:mw",
        "ro_port": "q0:res",
        "mw_clock": "q0.01",
        "ro_clock": "q0.ro",
        "init_duration": 10e-6,
        "repetitions": 10,
    }
    sched = awg_staircase_sched(**sched_kwargs)
    hw_cfg = load_example_zhinst_hardware_config
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    compiler = SerialCompiler(name="compiler")

    hw_cfg["devices"][1]["channel_0"]["modulation"]["interm_freq"] = 10e6
    quantum_device.hardware_config(hw_cfg)
    config = quantum_device.generate_compilation_config()
    comp_sched_a = compiler.compile(sched, config=config)

    hw_cfg["devices"][1]["channel_0"]["modulation"]["interm_freq"] = -100e6
    quantum_device.hardware_config(hw_cfg)
    config = quantum_device.generate_compilation_config()
    comp_sched_b = compiler.compile(sched, config=config)
    comp_sched_c = compiler.compile(sched, config=config)

    # Act
    sett_a = comp_sched_a.compiled_instructions["ic_uhfqa0"].settings_builder.build()
    sett_b = comp_sched_b.compiled_instructions["ic_uhfqa0"].settings_builder.build()
    sett_c = comp_sched_c.compiled_instructions["ic_uhfqa0"].settings_builder.build()

    # Assert
    # pylint: disable=comparison-with-itself
    assert sett_a == sett_a
    assert sett_a != sett_b
    assert sett_a != sett_c

    assert sett_b != sett_a
    assert sett_b == sett_b
    assert sett_b == sett_c

    assert sett_c != sett_a
    assert sett_c == sett_b
    assert sett_c == sett_c


def test_zi_setting_apply(mocker):
    # Arrange
    apply_fn = mocker.Mock()
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    node = "foo/bar"
    value = 1

    # Act
    setting = settings.ZISetting(node, value, apply_fn)
    collection = setting.as_dict()
    setting.apply(instrument)

    # Assert
    assert collection == {"foo/bar": 1}
    apply_fn.assert_called_with(instrument=instrument, node=node, value=value)


def test_zi_settings_apply(mocker):
    # Arrange
    instrument = make_ufhqa(mocker)
    apply_fn = mocker.Mock()
    daq_settings = [settings.ZISetting("daq/foo/bar", 0, apply_fn)]
    awg_settings = {0: settings.ZISetting("awg/foo/bar", 1, apply_fn)}

    # Act
    zi_settings = settings.ZISettings(daq_settings, awg_settings)
    zi_settings.apply(instrument)

    # Assert
    calls = [
        call(instrument=instrument, node="awg/foo/bar", value=1),
        call(instrument=instrument, node="/dev1234/daq/foo/bar", value=0),
    ]
    assert apply_fn.call_args_list == calls


def test_zi_settings_apply_bulk(mocker):
    # Arrange
    instrument = make_ufhqa(mocker)
    mocker.patch.object(zi_helpers, "set_vector")
    set_values = mocker.patch.object(zi_helpers, "set_values")

    zi_settings = (
        settings.ZISettingsBuilder()
        .with_wave_vector(0, 0, np.ones(64))
        .with_qas_delay(0)
        .with_qas_result_enable(0)
        .build()
    )

    # Act
    zi_settings.apply(instrument)

    # Assert
    set_values.assert_called_with(
        instrument,
        [
            ("/dev1234/qas/0/delay", 0),
            ("/dev1234/qas/0/result/enable", 0),
        ],
    )


def test_zi_settings_as_dict(mocker):
    # Arrange
    apply_fn = mocker.Mock()
    daq_settings = [settings.ZISetting("daq/foo/bar", 0, apply_fn)]
    awg_settings = {0: settings.ZISetting("awg/foo/bar", 1, apply_fn)}

    # Act
    zi_settings = settings.ZISettings(daq_settings, awg_settings)
    collection = zi_settings.as_dict()

    # Assert
    assert collection == {"daq/foo/bar": 0, "awg/foo/bar": {0: 1}}


def test_zi_settings_serialize_wave(mocker):
    # Arrange
    instrument = make_ufhqa(mocker)
    wave = np.ones(48)
    daq_settings = [settings.ZISetting("awgs/0/waveform/waves/0", wave, mocker.Mock())]
    awg_settings = {
        0: settings.ZISetting(
            "compiler/sourcestring",
            "wave w0 = gauss(128, 64, 32);",
            mocker.Mock(),
        ),
    }

    root = Path(".")
    touch = mocker.patch.object(Path, "touch")
    write_text = mocker.patch.object(Path, "write_text")
    np_savetext = mocker.patch.object(np, "savetxt")

    # Act
    zi_settings = settings.ZISettings(daq_settings, awg_settings)
    zi_settings.serialize(
        root,
        settings.ZISerializeSettings(
            instrument.name, instrument._serial, instrument._type
        ),
    )

    # Assert
    calls = [call(root / "uhfqa0_awg0_wave0.csv", ANY, delimiter=";")]
    assert np_savetext.call_args_list == calls

    args, _ = np_savetext.call_args
    csv_bug_scale_factor = 2**15 - 1  # See issue quantify-scheduler#175
    np.testing.assert_array_equal(
        args[1], np.reshape(wave / csv_bug_scale_factor, (24, -1))
    )

    touch.assert_called()

    write_text.assert_called_with(
        json.dumps(
            {
                "name": "uhfqa0",
                "serial": "dev1234",
                "type": "uhfqa",
                "awgs/0/waveform/waves/0": "uhfqa0_awg0_wave0.csv",
                "compiler/sourcestring": {0: "uhfqa0_awg0.seqc"},
            }
        )
    )


def test_zi_settings_serialize_command_table(mocker):
    # Arrange
    instrument = make_ufhqa(mocker)
    daq_settings = [
        settings.ZISetting("awgs/0/commandtable/data", {"key": 0}, mocker.Mock())
    ]

    root = Path(".")
    touch = mocker.patch.object(Path, "touch")
    write_text = mocker.patch.object(Path, "write_text")

    # Act
    zi_settings = settings.ZISettings(daq_settings, {})
    zi_settings.serialize(root, instrument)

    # Assert
    touch.assert_called()
    calls = [
        call('{"key": 0}'),
        call(
            json.dumps(
                {
                    "name": "uhfqa0",
                    "serial": "dev1234",
                    "type": "uhfqa",
                    "awgs/0/commandtable/data": "uhfqa0_awg0.json",
                }
            )
        ),
    ]
    assert write_text.call_args_list == calls


def test_zi_settings_serialize_compiler_source(mocker):
    # Arrange
    instrument = make_ufhqa(mocker)
    awg_settings = {
        0: settings.ZISetting(
            "compiler/sourcestring",
            "wave w0 = gauss(128, 64, 32);",
            mocker.Mock(),
        ),
        2: settings.ZISetting(
            "compiler/sourcestring",
            "wave w0 = gauss(128, 64, 32);",
            mocker.Mock(),
        ),
    }

    root = Path(".")
    touch = mocker.patch.object(Path, "touch")
    write_text = mocker.patch.object(Path, "write_text")

    # Act
    zi_settings = settings.ZISettings([], awg_settings)
    zi_settings.serialize(root, instrument)

    # Assert
    touch.assert_called()
    calls = [
        call("wave w0 = gauss(128, 64, 32);"),
        call("wave w0 = gauss(128, 64, 32);"),
        call(
            json.dumps(
                {
                    "name": "uhfqa0",
                    "serial": "dev1234",
                    "type": "uhfqa",
                    "compiler/sourcestring": {
                        "0": "uhfqa0_awg0.seqc",
                        "2": "uhfqa0_awg2.seqc",
                    },
                }
            )
        ),
    ]
    assert write_text.call_args_list == calls


def test_zi_settings_weights_raises():
    settings.ZISettingsBuilder().with_qas_integration_weights_real(9, np.ones(20))
    with pytest.raises(ValueError):
        settings.ZISettingsBuilder().with_qas_integration_weights_real(10, np.ones(20))
    settings.ZISettingsBuilder().with_qas_integration_weights_imag(9, np.ones(20))
    with pytest.raises(ValueError):
        settings.ZISettingsBuilder().with_qas_integration_weights_imag(10, np.ones(20))


def test_zi_settings_serialize_integration_weights(mocker):
    # Arrange
    instrument = make_ufhqa(mocker)
    time = np.arange(0, 10, 1)
    weights = np.sin(time)
    daq_settings = [
        settings.ZISetting("qas/0/integration/weights/0/real", weights, mocker.Mock())
    ]

    root = Path(".")
    touch = mocker.patch.object(Path, "touch")
    write_text = mocker.patch.object(Path, "write_text")

    # Act
    zi_settings = settings.ZISettings(daq_settings, {})
    zi_settings.serialize(root, instrument)

    # Assert
    touch.assert_called()

    write_text.assert_called_with(
        json.dumps(
            {
                "name": "uhfqa0",
                "serial": "dev1234",
                "type": "uhfqa",
                "qas/0/integration/weights/0/real": weights.tolist(),
            }
        )
    )


def test_zi_settings_builder_build():
    # Arrange
    builder = settings.ZISettingsBuilder()
    wave = np.ones(48)
    weights = np.zeros(4096)

    expected = {
        "sigouts/*/on": 0,
        "awgs/0/waveform/waves/0": wave,
        "awgs/0/commandtable/data": '{"0": 0}',
        "qas/0/delay": 0,
        "qas/0/result/enable": 0,
        "qas/0/result/length": 1024,
        "qas/0/result/averages": 1,
        "qas/0/result/mode": 0,
        "qas/0/result/source": 7,
        "qas/0/integration/length": 1024,
        "qas/0/integration/mode": 0,
        "qas/0/integration/weights/0/real": weights,
        "qas/0/integration/weights/1/real": weights,
        "qas/0/integration/weights/2/real": weights,
        "qas/0/integration/weights/3/real": weights,
        "qas/0/integration/weights/4/real": weights,
        "qas/0/integration/weights/5/real": weights,
        "qas/0/integration/weights/6/real": weights,
        "qas/0/integration/weights/7/real": weights,
        "qas/0/integration/weights/8/real": weights,
        "qas/0/integration/weights/9/real": weights,
        "qas/0/integration/weights/0/imag": weights,
        "qas/0/integration/weights/1/imag": weights,
        "qas/0/integration/weights/2/imag": weights,
        "qas/0/integration/weights/3/imag": weights,
        "qas/0/integration/weights/4/imag": weights,
        "qas/0/integration/weights/5/imag": weights,
        "qas/0/integration/weights/6/imag": weights,
        "qas/0/integration/weights/7/imag": weights,
        "qas/0/integration/weights/8/imag": weights,
        "qas/0/integration/weights/9/imag": weights,
        "qas/0/monitor/enable": 1,
        "qas/0/monitor/length": 1024,
        "qas/0/monitor/averages": 1,
        "qas/0/rotations/0": (1 + 0j),
        "qas/0/rotations/1": (1 + 0j),
        "qas/0/rotations/2": (1 + 0j),
        "qas/0/rotations/3": (1 + 0j),
        "qas/0/rotations/4": (1 + 0j),
        "qas/0/rotations/5": (1 + 0j),
        "qas/0/rotations/6": (1 + 0j),
        "qas/0/rotations/7": (1 + 0j),
        "qas/0/rotations/8": (1 + 0j),
        "qas/0/rotations/9": (1 + 0j),
        "system/awg/channelgrouping": 0,
        "awgs/0/time": 0,
        "sigouts/0/on": 1,
        "sigouts/1/on": 1,
        "sigouts/0/offset": 50.0,
        "awgs/0/outputs/0/gains/0": 0.9,
        "awgs/0/outputs/1/gains/1": 0.8,
        "compiler/sourcestring": {0: "wave w0 = gauss(128, 64, 32);"},
    }

    # Act
    zi_settings = (
        builder.with_defaults([("sigouts/*/on", 0)])
        .with_wave_vector(0, 0, wave)
        .with_commandtable_data(0, {0: 0})
        .with_qas_delay(0)
        .with_qas_result_enable(0)
        .with_qas_result_length(1024)
        .with_qas_result_averages(1)
        .with_qas_result_mode(zi_types.QasResultMode.CYCLIC)
        .with_qas_result_source(zi_types.QasResultSource.INTEGRATION)
        .with_qas_integration_length(1024)
        .with_qas_integration_mode(zi_types.QasIntegrationMode.NORMAL)
        .with_qas_integration_weights_real(range(10), weights)
        .with_qas_integration_weights_imag(range(10), weights)
        .with_qas_monitor_enable(True)
        .with_qas_monitor_length(1024)
        .with_qas_monitor_averages(1)
        .with_qas_rotations(range(10), 0)
        .with_system_channelgrouping(0)
        .with_awg_time(0, 0)
        .with_sigouts(0, (1, 1))
        .with_sigout_offset(0, 50.0)
        .with_gain(0, (0.9, 0.8))
        .with_compiler_sourcestring(0, "wave w0 = gauss(128, 64, 32);")
        .build()
    )

    # Assert
    np.testing.assert_equal(zi_settings.as_dict(), expected)


def test_awg_indexes(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    awg_settings = {
        0: settings.ZISetting(
            "compiler/sourcestring",
            "wave w0 = gauss(128, 64, 32);",
            mocker.Mock(),
        ),
        1: settings.ZISetting(
            "compiler/sourcestring",
            "wave w0 = gauss(128, 64, 32);",
            mocker.Mock(),
        ),
    }

    # Act
    zi_settings = settings.ZISettings([], awg_settings)

    # Assert
    assert [0, 1] == zi_settings.awg_indexes


def test_deserialize(mocker):
    # Arrange
    settings_json_data = json.dumps(
        {
            "type": "uhfqa",
            "name": "uhfqa0",
            "serial": "dev1234",
            "awgs/0/waveform/waves/0": "uhfqa0_awg0_wave0.csv",
            "awgs/0/commandtable/data": "uhfqa0_awg0.json",
            "qas/0/integration/weights/0/real": np.ones(4096).tolist(),
            "qas/0/integration/weights/0/imag": np.zeros(4096).tolist(),
            "qas/0/rotations/0": str((1 + 0j)).replace(" ", ""),
            "qas/0/delay": 0,
            "compiler/sourcestring": {"0": "uhfqa0_awg0.seqc"},
        }
    )
    waveform = waveforms.square(np.arange(50), 2.44)
    commandtable_data = '{"header": {"version": "0.2", "partial": false}, "table": []}'
    seqc_data = "wave w0 = gauss(128, 64, 32);"

    load_txt = mocker.patch.object(np, "loadtxt", return_value=waveform)
    mocker.patch.object(
        Path,
        "read_text",
        side_effect=[settings_json_data, commandtable_data, seqc_data],
    )

    # Act
    settings_path = Path("./data/uhfqa0_settings.json")
    builder = settings.ZISettings.deserialize(settings_path)
    zi_settings = builder.build()

    # Assert
    assert zi_settings.awg_indexes == [0]

    load_txt.assert_called_with("uhfqa0_awg0_wave0.csv", delimiter=";")

    np.testing.assert_equal(
        zi_settings.as_dict(),
        {
            "awgs/0/waveform/waves/0": waveform,
            "awgs/0/commandtable/data": commandtable_data,
            "qas/0/integration/weights/0/real": np.ones(4096),
            "qas/0/integration/weights/0/imag": np.zeros(4096),
            "qas/0/rotations/0": (1 + 0j),
            "qas/0/delay": 0,
            "compiler/sourcestring": {0: seqc_data},
        },
    )


def test_deserialize_with_invalid_filename():
    # Arrange
    filepath = Path("foo")

    # Act
    with pytest.raises(ValueError) as execinfo:
        settings.ZISettings.deserialize(filepath)

    # Assert
    assert str(execinfo.value) == (
        "Invalid value for param 'settings_path' provide path to "
        "'{instrument}_settings.json'"
    )
