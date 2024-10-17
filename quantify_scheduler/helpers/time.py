# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Python time wrapper functions.

These function help to make time dependent modules testable.
"""

import time


def get_time() -> float:
    """
    Return the time in seconds since the epoch as a floating point number.

    Acts as a wrapper around :func:`time.time` in order to make it testable.
    Mocking time.time() can conflicts with the internal python ticker thread.

    Returns
    -------
    :
        Time since epoch

    """
    return time.time()


def sleep(seconds: float) -> None:
    """
    Delay execution for a given number of seconds.

    The argument may be a floating point
    number for subsecond precision.

    Parameters
    ----------
    seconds :
        The amount of time to wait.

    """
    time.sleep(seconds)
