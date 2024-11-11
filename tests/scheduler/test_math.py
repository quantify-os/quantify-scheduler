# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import pytest

from quantify_scheduler import math


@pytest.mark.parametrize("n,m,expected", [(2, 4, 4), (2, 16, 16), (39, 16, 48), (127, 16, 128)])
def test_closest_number_ceil(n: int, m: int, expected: int) -> None:
    """
    Testing closest number ceiling
    """
    # Act
    result = math.closest_number_ceil(n, m)

    # Assert
    assert result == expected
