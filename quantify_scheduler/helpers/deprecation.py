# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helper functions for code deprecation."""
from __future__ import annotations

import functools
import warnings
from typing import Any, Callable


def deprecated_arg_alias(depr_version: str, **aliases: str) -> Callable:
    """
    Decorator for deprecated function and method arguments.

    From https://stackoverflow.com/a/49802489.

    Use as follows:

    .. code-block:: python

        @deprecated_arg_alias("0.x.0", old_arg="new_arg")
        def myfunc(new_arg):
            ...

    Parameters
    ----------
    depr_version
        The quantify-scheduler version in which the parameter names will be removed.
    aliases
        Parameter name aliases provided as ``old="new"``.

    Returns
    -------
    :
        The same function or method, that raises a FutureWarning if a deprecated
        argument is passed, or a TypeError if both the new and the deprecated arguments
        are passed.
    """

    def deco(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):  # noqa: ANN202, ANN002, ANN003
            _rename_kwargs(f.__name__, depr_version, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def _rename_kwargs(
    func_name: str, depr_version: str, kwargs: dict[str, Any], aliases: dict[str, str]
) -> None:
    """Helper function for deprecating function arguments."""
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    f"{func_name} received both {alias} and {new} as arguments! "
                    f"{alias} is deprecated and will be removed in quantify-scheduler "
                    f">= {depr_version}, use {new} instead."
                )
            warnings.warn(
                message=(
                    f"{alias} is deprecated as an argument to {func_name} and will be "
                    f"removed in quantify-scheduler >= {depr_version}; use {new} "
                    "instead."
                ),
                category=FutureWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)
