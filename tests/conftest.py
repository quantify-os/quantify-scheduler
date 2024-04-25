# -*- coding: utf-8 -*-


import pytest  # noqa

from tests.fixtures.mock_setup import *  # noqa: F401
from tests.fixtures.schedule import *  # noqa: F401
from tests.fixtures.generic import *  # noqa: F401
from tests.scheduler.backends.qblox.fixtures.assembly import *  # noqa: F401
from tests.scheduler.backends.qblox.fixtures.hardware_config import *  # noqa: F401
from tests.scheduler.backends.qblox.fixtures.mock_api import *  # noqa: F401


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if not is_zhinst_available():  # noqa: F405
        skip_zhinst = pytest.mark.skip(reason="zhinst backend is not available.")
        for item in items:
            if "needs_zhinst" in item.keywords:
                item.add_marker(skip_zhinst)


def pytest_ignore_collect(path, config):  # noqa: ARG001
    if not is_zhinst_available() and "zhinst" in str(path):  # noqa: F405
        return True
    return False
