# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from zhinst.qcodes.base import ZIBaseInstrument

from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.helpers import time


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


def test_set_values(mocker):
    # Arrange
    controller = mocker.Mock()

    instrument = mocker.Mock(**{"_serial": "dev1234"}, spec=ZIBaseInstrument)
    instrument._controller = controller
    values = [("/dev2299/qas/0/integration/mode", 1), ("/dev2299/sigouts/1/offset", 0)]

    # Act
    zi_helpers.set_values(instrument, values)

    # Assert
    controller._set.assert_called_with(values)


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
    update_value_awg_index_dict = {"index": awg_index}
    awg._awg._module.update.assert_called_with(**update_value_awg_index_dict)
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
    commandtable_map: Dict[int, int] = zi_helpers.get_waveform_table(
        pulse_ids, pulseid_pulseinfo_dict
    )

    # Assert
    assert commandtable_map == expected


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


@pytest.mark.parametrize(
    "base_sampling_rate,expected",
    [
        (
            2.4e9,
            {
                0: 2400000000,  # 2.4 GHz
                1: 1200000000,  # 1.2 GHz
                2: 600000000,  # 600 MHz
                3: 300000000,  # 300 MHz
                4: 150000000,  # 150 MHz
                5: 75000000,  # 75 MHz
                6: 37500000,  # 37.50 MHz
                7: 18750000,  # 18.75 MHz
                8: 9375000,  # 9.38 MHz
                9: 4687500,  # 4.69 MHz
                10: 2343750,  # 2.34 MHz
                11: 1171875,  # 1.17 MHz
                12: 585937,  # 585.94 kHz
                13: 292968,  # 292.97 kHz
            },
        ),
        (
            1.8e9,
            {
                0: 1800000000,  # 1.80 GHz
                1: 900000000,  # 900 MHz
                2: 450000000,  # 450 MHz
                3: 225000000,  # 225 MHz
                4: 112500000,  # 112.5 MHz
                5: 56250000,  # 56.25 MHz
                6: 28125000,  # 28.12 MHz
                7: 14062500,  # 14.06 MHz
                8: 7031250,  # 7.03 MHz
                9: 3515625,  # 3.52 MHz
                10: 1757812,  # 1.76 MHz
                11: 878906,  # 878.91 kHz
                12: 439453,  # 439.45 kHz
                13: 219726,  # 219.73 kHz
            },
        ),
    ],
)
def test_get_sampling_rates(base_sampling_rate: float, expected: Dict[int, int]):
    # Act
    values = zi_helpers.get_sampling_rates(base_sampling_rate)

    # Assert
    assert values == expected


def test_set_and_compile_awg_seqc_successfully(mocker):
    # Arrange
    awg_module = mocker.Mock()
    awg_module.get_int.side_effect = [0, 1, 1]
    awg_module.get_double.side_effect = [1.0]
    awg = mocker.Mock()
    awg._awg._module = awg_module
    instrument = mocker.create_autospec(ZIBaseInstrument, instance=True)
    instrument.awg = awg

    mocker.patch.object(time, "sleep")
    set_awg_value = mocker.patch.object(zi_helpers, "set_awg_value")
    mocker.patch.object(zi_helpers, "get_value", return_value="")

    awg_index = 0
    node: str = "compiler/sourcestring"
    value: str = "abc"

    # Act
    zi_helpers.set_and_compile_awg_seqc(instrument, awg_index, node, value)

    # Assert
    set_awg_value.assert_called_with(instrument, awg_index, node, value)


def test_set_and_compile_awg_seqc_skip_compilation(mocker):
    """
    FIXME: We remove this test as we are always triggering a recompilation.
    Once we have found a solution to compare the waveforms, please re-enable
    and add more tests.

    # Arrange
    awg_module = mocker.Mock()
    awg_module.get_int.side_effect = [0, 1, 1]
    awg_module.get_double.side_effect = [1.0]
    awg = mocker.Mock()
    awg._awg._module = awg_module
    instrument = mocker.create_autospec(ZIBaseInstrument, instance=True)
    instrument.awg = awg

    awg_index = 0
    node: str = "compiler/sourcestring"
    value: str = "abc"

    mocker.patch.object(time, "sleep")
    set_awg_value = mocker.patch.object(zi_helpers, "set_awg_value")
    mocker.patch.object(zi_helpers, "get_value", return_value=value)

    # Act
    zi_helpers.set_and_compile_awg_seqc(instrument, awg_index, node, value)

    # Assert
    set_awg_value.assert_not_called()
    """


def test_set_and_compile_awg_seqc_upload_failed(mocker):
    # Arrange
    awg_module = mocker.Mock()
    awg_module.get_int.side_effect = [1]
    awg_module.get_string.side_effect = ["Some error occured"]
    awg = mocker.Mock()
    awg._awg._module = awg_module
    instrument = mocker.create_autospec(ZIBaseInstrument, instance=True)
    instrument.awg = awg

    mocker.patch.object(zi_helpers, "get_value", return_value="")
    mocker.patch.object(time, "sleep")

    awg_index = 0
    node: str = "compiler/sourcestring"
    value: str = "abc"

    # Act
    with pytest.raises(Exception) as execinfo:
        zi_helpers.set_and_compile_awg_seqc(instrument, awg_index, node, value)

    # Assert
    assert str(execinfo.value) == "Upload failed: \nSome error occured"


def test_set_and_compile_awg_seqc_upload_timeout(mocker):
    # Arrange
    awg_module = mocker.Mock()
    awg_module.get_int.side_effect = [0, 0, 1]
    awg_module.get_double.side_effect = [0.9]
    awg = mocker.Mock()
    awg._awg._module = awg_module
    instrument = mocker.create_autospec(ZIBaseInstrument, instance=True)
    instrument.awg = awg

    mocker.patch.object(zi_helpers, "get_value", return_value="")
    mocker.patch.object(time, "sleep")

    awg_index = 0
    node: str = "compiler/sourcestring"
    value: str = "abc"

    # Act
    with mocker.patch.object(
        time, "get_time", side_effect=[0, 0 + 1000]
    ), pytest.raises(Exception) as execinfo:
        zi_helpers.set_and_compile_awg_seqc(instrument, awg_index, node, value)

    # Assert
    assert str(execinfo.value) == "Program upload timed out!"
