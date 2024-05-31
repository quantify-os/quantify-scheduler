import pytest

from quantify_scheduler.schemas.examples import utils


@pytest.mark.parametrize(
    "filename",
    [
        "qblox_hardware_config_transmon.json",
        "zhinst_hardware_compilation_config.json",
    ],
)
def test_load_json_example_scheme(filename: str) -> None:
    utils.load_json_example_scheme(filename)
