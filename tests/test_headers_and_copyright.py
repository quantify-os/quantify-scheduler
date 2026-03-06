import datetime
import os
import pprint
import re
from pathlib import Path

import pytest


def test_header() -> None:
    skipfiles = {
        "__init__.py",
        "conftest.py",
        "setup.py",
        "_version.py",
        "_static_version.py",
    }
    skipdirs = {
        "docs",
        ".",
        "tests",
        "__pycache__",
        "venv",
        "env",
        "lib",
        "lib64",
        "site-packages",
        ".git",
    }
    failures = []
    quantify_scheduler_path = Path(__file__).resolve().parent.parent.resolve()
    header_lines = [
        "# Repository: https://gitlab.com/quantify-os/quantify-scheduler",
        "# Licensed according to the LICENCE file on the main branch",
    ]
    for root, _, files in os.walk(quantify_scheduler_path):
        root_path = Path(root)
        parts = root_path.parts
        if any(part.startswith(name) for part in parts for name in skipdirs):
            continue
        if "site-packages" in parts or ".git" in parts or "python=" in str(root_path):
            continue
        for file_name in files:
            if file_name[-3:] == ".py" and file_name not in skipfiles:
                file_path = root_path / file_name
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError as err:
                    failures.append(f"{file_path!s} (not UTF-8: {err})")
                    continue
                lines = [line.strip() for line in content.splitlines()]
                line_matches = [
                    expected_line == line for expected_line, line in zip(header_lines, lines)
                ]
                if not all(line_matches):
                    failures.append(str(file_path))
    if failures:
        pytest.fail(f"Bad headers:\n{pprint.pformat(failures)}")


def test_docs_copyright() -> None:
    quantify_scheduler_path = Path(__file__).resolve().parent.parent.resolve()
    conf_file = quantify_scheduler_path / "docs" / "source" / "conf.py"
    current_year = str(datetime.datetime.now().year)
    cr_match = 'copyright = "2020-20.*Qblox & Orange Quantum Systems'
    with open(conf_file, encoding="utf-8") as file:
        for line in file:
            if re.match(cr_match, line):
                if current_year in line:
                    pass
                break
