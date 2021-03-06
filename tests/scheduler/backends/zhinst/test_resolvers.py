# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from unittest.mock import call

import numpy as np

from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst.resolvers import (
    monitor_acquisition_resolver,
    result_acquisition_resolver,
)


def test_monitor_acquisition_resolver(mocker):
    # Arrange
    real_data = [1, 2, 3]
    imag_data = [4, 5, 6]
    get_mock = mocker.patch.object(
        zi_helpers,
        "get_value",
        side_effect=[
            real_data,
            imag_data,
        ],
    )
    instrument = mocker.Mock()
    node_channel_0 = "qas/0/monitor/inputs/0/wave"
    node_channel_1 = "qas/0/monitor/inputs/1/wave"
    monitor_nodes = (node_channel_0, node_channel_1)
    expected_calls = [
        call(instrument, node_channel_0),
        call(instrument, node_channel_1),
    ]

    # Act
    complex_result = monitor_acquisition_resolver(instrument, monitor_nodes)

    # Assert
    assert get_mock.mock_calls == expected_calls
    assert complex_result.real.tolist() == real_data
    assert complex_result.imag.tolist() == imag_data


def test_result_acquisition_resolver(mocker):
    # Arrange
    real_data = [1, 2, 3]
    imag_data = [4, 5, 6]
    get_mock = mocker.patch.object(
        zi_helpers,
        "get_value",
        return_value=np.vectorize(complex)(real_data, imag_data),
    )
    instrument = mocker.Mock()
    result_node = "qas/0/result/data/0/wave"

    # Act
    complex_result = result_acquisition_resolver(instrument, result_node)

    # Assert
    get_mock.assert_called_with(instrument, result_node)
    assert complex_result.real.tolist() == real_data
    assert complex_result.imag.tolist() == imag_data
