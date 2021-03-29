# -----------------------------------------------------------------------------
# Description:    Tests for Math utility functions module.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
import pytest
import quantify.scheduler.math as math


@pytest.mark.parametrize(
    "n,m,expected", [(2, 4, 4), (2, 16, 16), (39, 16, 48), (127, 16, 128)]
)
def test_closest_number_ceil(n: int, m: int, expected: int):
    # Act
    result = math.closest_number_ceil(n, m)

    # Assert
    assert result == expected
