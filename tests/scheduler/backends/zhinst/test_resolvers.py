from unittest.mock import call

import numpy as np
from pytest_mock.plugin import MockerFixture

from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst.resolvers import (
    monitor_acquisition_resolver,
    result_acquisition_resolver,
)


def test_monitor_acquisition_resolver(mocker: MockerFixture) -> None:
    # Arrange
    real_data = np.array([1, 2, 3])
    imag_data = np.array([4, 5, 6])
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
    np.testing.assert_array_almost_equal(complex_result.real, real_data)
    np.testing.assert_array_almost_equal(complex_result.imag, imag_data)


def test_result_acquisition_resolver(mocker: MockerFixture) -> None:
    # Arrange
    real_data = np.array([1, 2, 3])
    imag_data = np.array([4, 5, 6])
    get_mock = mocker.patch.object(
        zi_helpers,
        "get_value",
        return_value=np.vectorize(complex)(real_data, imag_data),
    )
    instrument = mocker.Mock()
    result_nodes = ["qas/0/result/data/0/wave", "qas/0/result/data/1/wave"]

    # Act
    complex_result = result_acquisition_resolver(instrument, result_nodes)

    # Assert
    get_mock.assert_called_with(instrument, result_nodes[1])  # can only check the last call
    np.testing.assert_array_equal(complex_result.real, real_data + imag_data)
    np.testing.assert_array_equal(complex_result.imag, real_data + imag_data)
