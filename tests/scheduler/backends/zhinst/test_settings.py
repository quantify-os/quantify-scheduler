# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
from __future__ import annotations
import json

from pathlib import Path
from unittest.mock import ANY, call

import numpy as np
from zhinst.qcodes import base
from quantify.scheduler.backends.types import zhinst as zi_types
from quantify.scheduler.backends.zhinst import settings


def test_zi_setting(mocker):
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


def test_zi_settings(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    apply_fn = mocker.Mock()
    daq_settings = [settings.ZISetting("daq/foo/bar", 0, apply_fn)]
    awg_settings = [(0, settings.ZISetting("awg/foo/bar", 1, apply_fn))]

    # Act
    zi_settings = settings.ZISettings(instrument, daq_settings, awg_settings)
    zi_settings.apply()

    # Assert
    calls = [
        call(instrument=instrument, node="/dev1234/daq/foo/bar", value=0),
        call(instrument=instrument, node="awg/foo/bar", value=1),
    ]
    assert apply_fn.call_args_list == calls


def test_zi_settings_as_dict(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    apply_fn = mocker.Mock()
    daq_settings = [settings.ZISetting("daq/foo/bar", 0, apply_fn)]
    awg_settings = [(0, settings.ZISetting("awg/foo/bar", 1, apply_fn))]

    # Act
    zi_settings = settings.ZISettings(instrument, daq_settings, awg_settings)
    collection = zi_settings.as_dict()

    # Assert
    assert collection == {
        "/dev1234/daq/foo/bar": 0,
        "awg/foo/bar": 1,
    }


def test_zi_settings_serialize_wave(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    wave = np.ones(48)
    daq_settings = [settings.ZISetting("awgs/0/waveform/waves/0", wave, mocker.Mock())]
    awg_settings = [
        (
            0,
            settings.ZISetting(
                "compiler/sourcestring",
                "wave w0 = gauss(128, 64, 32);",
                mocker.Mock(),
            ),
        )
    ]

    root = Path(".")
    touch = mocker.patch.object(Path, "touch")
    write_text = mocker.patch.object(Path, "write_text")
    np_savetext = mocker.patch.object(np, "savetxt")

    # Act
    zi_settings = settings.ZISettings(instrument, daq_settings, awg_settings)
    zi_settings.serialize(root)

    # Assert
    calls = [call(root / "dev1234_awg0_wave0.csv", ANY, delimiter=";")]
    assert np_savetext.call_args_list == calls

    args, _ = np_savetext.call_args
    np.testing.assert_array_equal(args[1], np.reshape(wave, (24, -1)))

    touch.assert_called()

    write_text.assert_called_with(
        json.dumps(
            {
                "/dev1234/awgs/0/waveform/waves/0": "dev1234_awg0_wave0.csv",
                "compiler/sourcestring": ["dev1234_awg0.seqc"],
            }
        )
    )


def test_zi_settings_serialize_command_table(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    daq_settings = [
        settings.ZISetting("awgs/0/commandtable/data", {"key": 0}, mocker.Mock())
    ]

    root = Path(".")
    touch = mocker.patch.object(Path, "touch")
    write_text = mocker.patch.object(Path, "write_text")

    # Act
    zi_settings = settings.ZISettings(instrument, daq_settings, [])
    zi_settings.serialize(root)

    # Assert
    touch.assert_called()
    calls = [
        call('{"key": 0}'),
        call('{"/dev1234/awgs/0/commandtable/data": "dev1234_awg0.json"}'),
    ]
    assert write_text.call_args_list == calls


def test_zi_settings_serialize_compiler_source(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    awg_settings = [
        (
            0,
            settings.ZISetting(
                "compiler/sourcestring",
                "wave w0 = gauss(128, 64, 32);",
                mocker.Mock(),
            ),
        ),
        (
            2,
            settings.ZISetting(
                "compiler/sourcestring",
                "wave w0 = gauss(128, 64, 32);",
                mocker.Mock(),
            ),
        ),
    ]

    root = Path(".")
    touch = mocker.patch.object(Path, "touch")
    write_text = mocker.patch.object(Path, "write_text")

    # Act
    zi_settings = settings.ZISettings(instrument, [], awg_settings)
    zi_settings.serialize(root)

    # Assert
    touch.assert_called()
    calls = [
        call("wave w0 = gauss(128, 64, 32);"),
        call("wave w0 = gauss(128, 64, 32);"),
        call('{"compiler/sourcestring": ["dev1234_awg0.seqc", "dev1234_awg2.seqc"]}'),
    ]
    assert write_text.call_args_list == calls


def test_zi_settings_builder_build(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    builder = settings.ZISettingsBuilder()
    wave = np.ones(48)
    weights = np.zeros(4096)

    expected = {
        "/dev1234/sigouts/*/on": 0,
        "/dev1234/awgs/0/waveform/waves/0": wave,
        "/dev1234/awgs/0/commandtable/data": '{"0": 0}',
        "/dev1234/qas/0/delay": 0,
        "/dev1234/qas/0/result/enable": 0,
        "/dev1234/qas/0/result/length": 1024,
        "/dev1234/qas/0/result/averages": 1,
        "/dev1234/qas/0/result/mode": 0,
        "/dev1234/qas/0/result/source": 7,
        "/dev1234/qas/0/integration/length": 1024,
        "/dev1234/qas/0/integration/mode": 0,
        "/dev1234/qas/0/integration/weights/0/real": weights,
        "/dev1234/qas/0/integration/weights/0/imag": weights,
        "/dev1234/qas/0/integration/weights/1/real": weights,
        "/dev1234/qas/0/integration/weights/1/imag": weights,
        "/dev1234/qas/0/integration/weights/2/real": weights,
        "/dev1234/qas/0/integration/weights/2/imag": weights,
        "/dev1234/qas/0/integration/weights/3/real": weights,
        "/dev1234/qas/0/integration/weights/3/imag": weights,
        "/dev1234/qas/0/integration/weights/4/real": weights,
        "/dev1234/qas/0/integration/weights/4/imag": weights,
        "/dev1234/qas/0/integration/weights/5/real": weights,
        "/dev1234/qas/0/integration/weights/5/imag": weights,
        "/dev1234/qas/0/integration/weights/6/real": weights,
        "/dev1234/qas/0/integration/weights/6/imag": weights,
        "/dev1234/qas/0/integration/weights/7/real": weights,
        "/dev1234/qas/0/integration/weights/7/imag": weights,
        "/dev1234/qas/0/integration/weights/8/real": weights,
        "/dev1234/qas/0/integration/weights/8/imag": weights,
        "/dev1234/qas/0/integration/weights/9/real": weights,
        "/dev1234/qas/0/integration/weights/9/imag": weights,
        "/dev1234/qas/0/monitor/enable": 1,
        "/dev1234/qas/0/monitor/length": 1024,
        "/dev1234/qas/0/monitor/averages": 1,
        "/dev1234/qas/0/rotations/0": (1 + 0j),
        "/dev1234/qas/0/rotations/1": (1 + 0j),
        "/dev1234/qas/0/rotations/2": (1 + 0j),
        "/dev1234/qas/0/rotations/3": (1 + 0j),
        "/dev1234/qas/0/rotations/4": (1 + 0j),
        "/dev1234/qas/0/rotations/5": (1 + 0j),
        "/dev1234/qas/0/rotations/6": (1 + 0j),
        "/dev1234/qas/0/rotations/7": (1 + 0j),
        "/dev1234/qas/0/rotations/8": (1 + 0j),
        "/dev1234/qas/0/rotations/9": (1 + 0j),
        "/dev1234/system/awg/channelgrouping": 0,
        "/dev1234/awgs/0/time": 0,
        "/dev1234/sigouts/0/on": 1,
        "/dev1234/sigouts/1/on": 1,
        "compiler/sourcestring": "wave w0 = gauss(128, 64, 32);",
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
        .with_qas_integration_weights(range(10), weights, weights)
        .with_qas_monitor_enable(True)
        .with_qas_monitor_length(1024)
        .with_qas_monitor_averages(1)
        .with_qas_rotations(range(10), 0)
        .with_system_channelgrouping(0)
        .with_awg_time(0, 0)
        .with_sigouts(0, (1, 1))
        .with_compiler_sourcestring(0, "wave w0 = gauss(128, 64, 32);")
        .build(instrument)
    )

    collection = dict()
    for setting in zi_settings._daq_settings:
        collection = {**collection, **setting.as_dict()}

    for (_, setting) in zi_settings._awg_settings:
        collection = {**collection, **setting.as_dict()}

    # Assert
    np.testing.assert_equal(expected, collection)


def test_awg_indexes(mocker):
    # Arrange
    instrument = mocker.create_autospec(base.ZIBaseInstrument, instance=True)
    instrument._serial = "dev1234"
    awg_settings = [
        (
            0,
            settings.ZISetting(
                "compiler/sourcestring",
                "wave w0 = gauss(128, 64, 32);",
                mocker.Mock(),
            ),
        ),
        (
            1,
            settings.ZISetting(
                "compiler/sourcestring",
                "wave w0 = gauss(128, 64, 32);",
                mocker.Mock(),
            ),
        ),
    ]

    # Act
    zi_settings = settings.ZISettings(instrument, [], awg_settings)

    # Assert
    assert [0, 1] == zi_settings.awg_indexes
