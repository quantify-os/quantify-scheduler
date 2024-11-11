from __future__ import annotations

import re
from typing import Iterator

import pytest


@pytest.fixture
def assert_equal_q1asm():
    """
    Fixture containing a function that compares Q1ASM programs while ignoring whitespace
    and comments.

    Empty lines or lines containing only comments are also ignored. For example, the
    following two programs are considered equal:

    ```
        reset_ph
        set_awg_gain 8192,0  # a comment
        play         0,0,4
        wait 96
    ```

    and

    ```
        reset_ph
        set_awg_gain 8192,0
        play         0,0,4
        # a line with only a comment
        wait 96
    ```

    Note that _all_ whitespace is ignored, meaning that e.g. 'wait 4' is turned into
    'wait4' for comparison.
    """

    def inner(program_1: str, program_2: str):
        for (stripped_1, original_1), (stripped_2, original_2) in zip(
            _strip_and_skip_empty_lines(program_1),
            _strip_and_skip_empty_lines(program_2),
        ):
            # pytest's detailed string diff does not work in fixtures, so we display the
            # original strings upon AssertionError.
            assert stripped_1 == stripped_2, f"Original lines:\n{original_1}\n{original_2}"

    return inner


def _strip_and_skip_empty_lines(program: str) -> Iterator[tuple[str, str]]:
    """
    Strip each line of the program of whitespace and comments.

    If a line is only whitespace and/or comments, it is skipped.

    Returns an iterator of tuples, containing the stripped and original line (in that
    order).
    """
    for line in program.splitlines():
        stripped_line = re.sub(r"\s+|#.*$", "", line)
        if stripped_line == "":
            continue
        yield stripped_line, line
