# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import pytest

from quantify_scheduler.schemas.examples import utils


@pytest.mark.parametrize(
    "filename",
    [
        "qblox_test_mapping.json",
        "transmon_test_config.json",
        "zhinst_test_mapping.json",
    ],
)
def test_load_json_example_scheme(filename: str) -> None:
    utils.load_json_example_scheme(filename)
