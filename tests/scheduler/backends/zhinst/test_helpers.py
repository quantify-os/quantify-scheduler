# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import ANY, call

import numpy as np
import pytest
from zhinst.qcodes.base import ZIBaseInstrument

from quantify.scheduler.backends.types.zhinst import DeviceType, QasIntegrationMode
from quantify.scheduler.backends.zhinst import helpers as zi_helpers


@pytest.mark.parametrize(
    "node",
    [
        ("awgs/0/commandtable/data"),
        ("/dev1234/awgs/0/commandtable/data"),
    ],
)
def test_get_value(mocker, node: str):
    # Arrange
    return_value = '{"header": {"version": "0.2", "partial": false}, "table": []}'
    controller = mocker.Mock(**{"_get.return_value": return_value})

    instrument = mocker.Mock(**{"_serial": "dev1234"}, spec=ZIBaseInstrument)
    instrument._controller = controller

    # Act
    value = zi_helpers.get_value(instrument, node)

    # Assert
    assert value == return_value
    controller._get.assert_called_with("/dev1234/awgs/0/commandtable/data")


@pytest.mark.parametrize(
    "node",
    [
        ("qas/0/integration/mode"),
        ("/dev1234/qas/0/integration/mode"),
    ],
)
def test_set_value(mocker, node: str):
    # Arrange
    controller = mocker.Mock()

    instrument = mocker.Mock(**{"_serial": "dev1234"}, spec=ZIBaseInstrument)
    instrument._controller = controller

    # Act
    zi_helpers.set_value(instrument, node, value=1)

    # Assert
    controller._set.assert_called_with("/dev1234/qas/0/integration/mode", 1)


def test_set_wave_vector(mocker):
    # Arrange
    set_vector = mocker.patch.object(zi_helpers, "set_vector")
    instrument = mocker.Mock(spec=ZIBaseInstrument, **{"_serial": "dev1234"})

    awg_index = 0
    wave_index = 1

    expected_node: str = f"awgs/{awg_index}/waveform/waves/{wave_index}"
    vector = np.zeros(16)

    # Act
    zi_helpers.set_wave_vector(instrument, awg_index, wave_index, vector)

    # Assert
    set_vector.assert_called_with(instrument, expected_node, vector)


@pytest.mark.parametrize(
    "node",
    [
        ("awgs/0/commandtable/data"),
        ("/dev1234/awgs/0/commandtable/data"),
    ],
)
def test_set_vector(mocker, node: str):
    # Arrange
    return_value = '{"header": {"version": "0.2", "partial": false}, "table": []}'
    controller = mocker.Mock(
        **{"_controller._connection._daq.setVector.return_value": return_value}
    )

    instrument = mocker.Mock(spec=ZIBaseInstrument, **{"_serial": "dev1234"})
    instrument._controller = controller
    expected_node: str = "/dev1234/awgs/0/commandtable/data"

    # Act
    vector = '{"foo": "bar"}'
    zi_helpers.set_vector(instrument, node, vector)

    # Assert
    controller._controller._connection._daq.setVector.assert_called_with(
        expected_node, vector
    )


@pytest.mark.parametrize(
    "n_awgs,node",
    [
        (1, "compiler/sourcestring"),
        (3, "compiler/sourcestring"),
    ],
)
def test_set_awg_value(mocker, n_awgs: int, node: str):
    # Arrange
    instrument = mocker.Mock(**{"_serial": "dev1234"}, spec=ZIBaseInstrument)
    awg = mocker.Mock()
    awg._awg._module = mocker.Mock()
    if n_awgs > 1:
        instrument.awgs = [awg]
    else:
        instrument.awg = awg

    expected_node = "compiler/sourcestring"
    awg_index = 0
    value = "foo"

    # Act
    zi_helpers.set_awg_value(instrument, awg_index, node, value)

    # Assert
    awg._awg._module.set.assert_called_with(expected_node, value)


@pytest.mark.parametrize(
    "json_data",
    [
        ('{"header": {"version": "0.2", "partial": false}, "table": []}'),
        ({"header": {"version": "0.2", "partial": False}, "table": []}),
    ],
)
def test_set_commandtable_data(mocker, json_data):
    # Arrange
    awg_index = 0

    set_vector = mocker.patch.object(zi_helpers, "set_vector")
    instrument = mocker.Mock(spec=ZIBaseInstrument)

    expected_node: str = f"awgs/{awg_index}/commandtable/data"
    expected_json_data: str = (
        json.dumps(json_data) if not isinstance(json_data, str) else json_data
    )

    # Act
    zi_helpers.set_commandtable_data(instrument, awg_index, json_data)

    # Assert
    set_vector.assert_called_with(instrument, expected_node, expected_json_data)


def test_get_directory(mocker):
    # Arrange
    awg_core = mocker.Mock(**{"_awg._module.get_string.return_value": "./foo/"})
    expected = Path("./foo/")

    # Act
    path: Path = zi_helpers.get_directory(awg_core)

    # Assert
    assert path == expected


def test_get_src_directory(mocker):
    # Arrange
    awg_core = mocker.Mock(**{"_awg._module.get_string.return_value": "./foo/"})
    expected = Path("./foo/awg/src")

    # Act
    path: Path = zi_helpers.get_src_directory(awg_core)

    # Assert
    assert path == expected


def test_get_waves_directory(mocker):
    # Arrange
    awg_core = mocker.Mock(**{"_awg._module.get_string.return_value": "./foo/"})
    expected = Path("./foo/awg/waves")

    # Act
    path: Path = zi_helpers.get_waves_directory(awg_core)

    # Assert
    assert path == expected


@pytest.mark.parametrize(
    "device_type, expected",
    [
        (DeviceType.HDAWG, 2400000000),
        (DeviceType.UHFQA, 1800000000),
        (DeviceType.UHFLI, 0),
        (DeviceType.MFLI, 0),
        (DeviceType.PQSC, 0),
    ],
)
def test_get_clock_rate(device_type, expected):
    # Act
    clock_rate: int = zi_helpers.get_clock_rate(device_type)

    # Assert
    assert clock_rate == expected


def test_write_seqc_file(mocker):
    # Arrange
    get_src_directory = mocker.patch.object(zi_helpers, "get_src_directory")
    get_src_directory.return_value = Path("./foo/awg/src/")
    write_text = mocker.patch.object(Path, "write_text")

    contents: str = '{ "foo": "bar" }'

    # Act
    path: Path = zi_helpers.write_seqc_file(mocker.Mock(), contents, "awg0.seqc")

    # Assert
    assert path == Path("./foo/awg/src/awg0.seqc")
    write_text.assert_called_with(contents)


@pytest.mark.parametrize(
    "pulse_ids,pulseid_pulseinfo_dict,expected",
    [
        (
            [0, 1, 2],
            {
                0: {"port": "q0:mw"},
                1: {"port": "q0:mw"},
                2: {"port": "q0:mw"},
            },
            {0: 0, 1: 1, 2: 2},
        ),
        (
            [0, 1, 2],
            {
                0: {"port": "q0:mw"},
                1: {"port": None},
                2: {"port": "q0:mw"},
            },
            {0: 0, 2: 1},
        ),
        (
            [0, 1, 1],
            {
                0: {"port": "q0:mw"},
                1: {"port": "q0:mw"},
            },
            {0: 0, 1: 1},
        ),
    ],
)
def test_get_commandtable_map(
    pulse_ids: List[int],
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]],
    expected,
):
    # Act
    commandtable_map: Dict[int, int] = zi_helpers.get_commandtable_map(
        pulse_ids, pulseid_pulseinfo_dict
    )

    # Assert
    assert commandtable_map == expected


def test_set_qas_parameters(mocker):
    # Arrange
    set_value = mocker.patch.object(zi_helpers, "set_value")
    instrument = mocker.Mock(spec=ZIBaseInstrument, **{"_serial": "dev1234"})

    expected_calls = [
        call(instrument, "qas/0/integration/length", 1024),
        call(instrument, "qas/0/integration/mode", 0),
        call(instrument, "qas/0/delay", 0),
    ]

    # Act
    zi_helpers.set_qas_parameters(instrument, 1024, QasIntegrationMode.NORMAL, 0)

    # Assert
    assert set_value.mock_calls == expected_calls


def test_set_integration_weights(mocker):
    # Arrange
    set_value = mocker.patch.object(zi_helpers, "set_vector")
    instrument = mocker.Mock(spec=ZIBaseInstrument, **{"_serial": "dev1234"})

    channel_index = 0
    weights_i = np.ones(4096)
    weights_q = np.zeros(4096)

    expected_calls = [
        call(instrument, f"qas/0/integration/weights/{channel_index}/real", ANY),
        call(instrument, f"qas/0/integration/weights/{channel_index}/imag", ANY),
    ]

    # Act
    zi_helpers.set_integration_weights(instrument, channel_index, weights_i, weights_q)

    # Assert
    assert set_value.call_args_list == expected_calls
    for i, call_args in enumerate(set_value.call_args_list):
        args, _ = call_args

        assert isinstance(args[2], (np.ndarray, np.generic))

        if i % 2 == 0:
            assert args[2].tolist() == weights_i.tolist()
        else:
            assert args[2].tolist() == weights_q.tolist()


@pytest.mark.parametrize(
    "readout_channels_count,expected",
    [
        (0, "0b0000000000"),
        (1, "0b0000000001"),
        (6, "0b0000111111"),
        (10, "0b1111111111"),
    ],
)
def test_get_readout_channel_bitmask(readout_channels_count: int, expected: str):
    # Act
    bitmask = zi_helpers.get_readout_channel_bitmask(readout_channels_count)

    # Assert
    assert bitmask == expected
