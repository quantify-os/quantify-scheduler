# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Math utility functions module."""

from typing import Union


def closest_number_ceil(number: Union[int, float], multiple: Union[int, float]) -> int:
    """
    Returns the closest next value that is a multiple of M.

    Parameters
    ----------
    number :
        The number.
    multiple :
        The modulo or granularity level.

    Returns
    -------
    int
        The closest next number.
    """
    if not number > 0:
        raise ValueError(f"number must be positive. Got {number}.")
    if not multiple > 0:
        raise ValueError(f"multiple must be positive. Got {multiple}.")

    # Find the quotient
    quotient = int(number / multiple)

    # Find next possible closest number
    closest_number = multiple * (quotient + 1)

    return int(closest_number)
