from __future__ import annotations

import re
from collections.abc import Iterator

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

    def inner(program_1: str | list[str], program_2: str | list[str]):
        if isinstance(program_1, str):
            program_1 = program_1.splitlines()
        if isinstance(program_2, str):
            program_2 = program_2.splitlines()
        stripped_program_1 = _strip_and_skip_empty_lines(program_1)
        stripped_program_2 = _strip_and_skip_empty_lines(program_2)
        assert len(stripped_program_1) == len(stripped_program_2), (
            "Programs have differing amount of lines"
            f"{len(stripped_program_1)} vs {len(stripped_program_2)}"
        )
        for i in range(len(stripped_program_1)):
            (stripped_1, original_1) = stripped_program_1[i]
            (stripped_2, original_2) = stripped_program_2[i]
            # pytest's detailed string diff does not work in fixtures, so we display the
            # original strings upon AssertionError.
            assert stripped_1 == stripped_2, (
                f"Line {i} not the same. Original lines:\n{original_1}\n{original_2}"
            )

    return inner


def _strip_and_skip_empty_lines(program: list[str]) -> list[tuple[str, str]]:
    """
    Strip each line of the program of whitespace and comments.

    If a line is only whitespace and/or comments, it is skipped.

    Returns an iterator of tuples, containing the stripped and original line (in that
    order).
    """
    program_list = []
    for line in program:
        stripped_line = re.sub(r"\s+|#.*$", "", line)
        if stripped_line == "":
            continue
        program_list.append((stripped_line, line))
    return program_list
