# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Python inspect helper functions."""
from __future__ import annotations

import inspect
import os
import subprocess as sp  # nosec B404
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Type, Union


def get_classes(*modules: ModuleType) -> Dict[str, Type[Any]]:
    """
    Return a dictionary of class names by class types.

    .. code-block::

        from quantify_scheduler.helpers import inspect
        from my_module import foo

        class_dict: Dict[str, type] = inspect.get_classes(foo)
        print(class_dict)
        // { 'Bar': my_module.foo.Bar }

    Parameters
    ----------
    modules :
        Variable length of modules.

    Returns
    -------
    :
        A dictionary containing the class names by class reference.
    """
    classes = list()
    for module in modules:
        module_name: str = module.__name__
        classes += inspect.getmembers(
            sys.modules[module_name],
            lambda member: inspect.isclass(member) and member.__module__ == module_name,
        )
    return dict(classes)


def make_uml_diagram(
    obj_to_plot: Union[ModuleType, Type[Any]],
    options: list[str],
) -> str:
    """
    Generate a UML diagram of a given module or class.

    This function is a wrapper of `pylint.pyreverse`.

    Parameters
    ----------
    obj_to_plot
        The module or class to visualize
    options
        A string containing the plotting options for pyreverse

    Returns
    -------
    :
        The name of the generated ``png`` image
    """
    basic_options = ["--colorized", "-m", "n"]

    sp_args = {"stdout": sp.DEVNULL, "stderr": sp.STDOUT}
    sp_err = (
        f"Something went wrong in the plotting backend. "
        f"Please make sure pylint is installed and the provided options have the "
        f"correct syntax: {options}"
    )
    dot_err = "Error running 'dot': is 'graphviz' installed?"

    if inspect.ismodule(obj_to_plot):
        abs_module_path = Path(obj_to_plot.__file__).parent

        try:
            sp.run(  # nosec B603
                [
                    "pyreverse",
                    "--only-classnames",
                    *basic_options,
                    *options,
                    abs_module_path,
                ],
                check=True,
                **sp_args,
            )
        except (sp.CalledProcessError, FileNotFoundError):
            # FileNotFoundError is raised, as opposed to CalledProcessError,
            # when the executable is not found.
            print(sp_err)

        try:
            diagram_name = f"{abs_module_path.name}.png"
            sp.run(  # nosec B603
                ["dot", "-Tpng", "classes.dot", "-o", diagram_name], check=True
            )
            os.remove("classes.dot")
            os.remove("packages.dot")
        except (sp.CalledProcessError, FileNotFoundError):
            print(dot_err)
            diagram_name = None

    elif inspect.isclass(obj_to_plot):
        class_module_str = obj_to_plot.__module__
        class_name_str = obj_to_plot.__name__
        class_path_str = f"{class_module_str}.{class_name_str}"

        class_module = sys.modules[class_module_str]
        repo_path = str(Path(class_module.__file__).parent)

        try:
            sp.run(  # nosec B603
                [
                    "pyreverse",
                    *basic_options,
                    *options,
                    "-c",
                    class_path_str,
                    repo_path,
                ],
                check=True,
                **sp_args,
            )
        except (sp.CalledProcessError, FileNotFoundError):
            print(sp_err)

        try:
            diagram_name = f"{class_name_str}.png"
            sp.run(  # nosec B603
                ["dot", "-Tpng", f"{class_path_str}.dot", "-o", diagram_name],
                check=True,
            )
            os.remove(f"{class_path_str}.dot")
        except (sp.CalledProcessError, FileNotFoundError):
            print(dot_err)
            diagram_name = None

    else:
        raise TypeError("Argument must be either a module or a class")

    return diagram_name
