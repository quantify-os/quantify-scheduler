# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains methods to check compatibility with optional installs."""

import importlib
import sys


def check_zhinst_compatibility() -> None:
    """
    Check if the zhinst backend can be safely used.

    Raises
    ------
    RuntimeError
        When running an incompatible python version.
    ModuleNotFoundError
        When the zhinst backend is not installed.
    """
    version = sys.version_info
    major, minor = version.major, version.minor

    if version >= (3, 10):
        raise RuntimeError(
            "The zhinst backend is only compatible with Python 3.8 and Python 3.9, "
            f"but you have Python {major}.{minor}. Please install a compatible python version."
        )
    if importlib.util.find_spec("zhinst") is None:  # type: ignore[attr-defined]
        raise ModuleNotFoundError(
            "The zhinst backend could not be found. "
            "Please install the zhinst backend via `pip install quantify-scheduler[zhinst]`."
        )
