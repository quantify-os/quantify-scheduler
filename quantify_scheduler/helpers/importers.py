# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing methods to import and export objects by string."""
import importlib
from typing import Any


def import_python_object_from_string(function_string: str) -> Any:
    """
    Import a python object from a string.

    This function does the inverse operation of
    :func:`export_python_object_to_path_string`.

    (Based on https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string)
    """
    mod_name, func_name = function_string.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def export_python_object_to_path_string(obj: Any) -> str:
    """
    Get the absolute path (dot-separated) to a python object.

    This function does the inverse operation of
    :func:`import_python_object_from_string`.

    Parameters
    ----------
    obj : Any
        Any python object.

    Returns
    -------
    str
        A string containing a dot-separated absolute path to the object.
    """
    module = obj.__module__
    name = obj.__name__
    if module == "__main__":
        path = name
    else:
        path = ".".join([module, name])
    return path
