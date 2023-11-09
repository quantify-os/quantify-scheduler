# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing example schedule utility functions."""

import json
from pathlib import Path
from typing import Any, Dict

from quantify_scheduler.schemas import examples


def load_json_example_scheme(filename: str) -> Dict[str, Any]:
    """
    Returns a JSON-file by name as a python dictionary.

    Parameters
    ----------
    filename :
        The example filename to load.

    Returns
    -------
    :
        The json file as a dictionary.
    """
    path = Path(examples.__file__).parent.joinpath(filename)
    return json.loads(path.read_text())
