# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

import importlib
from typing import Any


def import_python_object_from_string(function_string: str) -> Any:
    """
    Based on https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
    """  # pylint: disable=line-too-long
    mod_name, func_name = function_string.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func
