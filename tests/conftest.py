from pathlib import Path

import pytest

from tests.fixtures.generic import *
from tests.fixtures.mock_setup import *
from tests.fixtures.schedule import *
from tests.scheduler.backends.qblox.fixtures.assembly import *
from tests.scheduler.backends.qblox.fixtures.hardware_config import *
from tests.scheduler.backends.qblox.fixtures.mock_api import *


def pytest_collection_modifyitems(config, items):
    if not is_zhinst_available():  # noqa: F405
        skip_zhinst = pytest.mark.skip(reason="zhinst backend is not available.")
        for item in items:
            if "needs_zhinst" in item.keywords:
                item.add_marker(skip_zhinst)


def pytest_ignore_collect(collection_path: Path, config):
    return not is_zhinst_available() and "zhinst" in str(collection_path)  # noqa: F405


def pytest_addoption(parser):
    parser.addoption(
        "--skip_qblox_driver_version_check",
        action="store_true",
        default=False,
        help="Skip the Qblox driver version check",
    )


def pytest_configure(config):
    if config.getoption("skip_qblox_driver_version_check"):
        from quantify_scheduler.backends.qblox import driver_version_check

        driver_version_check.raise_on_version_mismatch = False
