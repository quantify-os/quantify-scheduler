# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Python inspect helper functions."""
from types import ModuleType
import inspect
import sys
from typing import Any, Dict


def get_classes(*module: ModuleType) -> Dict[str, Any]:
    """
    Returns the classes of a module.

    Parameters
    ----------
    module :
        The imported module.

    Returns
    -------
    :
        A dictionary containing the class names by class reference.
    """
    classes = list()
    for m in module:
        classes += inspect.getmembers(
            sys.modules[m.__name__],
            lambda member: inspect.isclass(member) and member.__module__ == m.__name__,
        )
    return dict(classes)
