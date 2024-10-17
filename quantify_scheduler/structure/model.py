# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Root models for data structures used within the package."""

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from quantify_scheduler.helpers.importers import (
    import_python_object_from_string,
)


class DataStructure(BaseModel):
    """
    A parent for all data structures.

    Data attributes are generated from the class' type annotations, similarly to
    `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_. If data
    attributes are JSON-serializable, data structure can be serialized using
    ``json()`` method. This string can be deserialized using ``parse_raw()`` classmethod
    of a correspondent child class.

    If required, data fields can be validated, see examples for more information.
    It is also possible to define custom field types with advanced validation.

    This class is a pre-configured `pydantic <https://docs.pydantic.dev/>`_
    model. See its documentation for details of usage information.

    .. admonition:: Examples
        :class: dropdown

        .. include:: /examples/structure.DataStructure.rst
    """

    model_config = ConfigDict(
        extra="forbid",
        # ensures exceptions are raised when passing extra argument that are not
        # part of a model when initializing.
        validate_assignment=True,
        # run validation when assigning attributes
    )


def deserialize_function(fun: str) -> Callable[..., Any]:
    """
    Import a python function from a dotted import string (e.g.,
    "quantify_scheduler.structure.model.deserialize_function").

    Parameters
    ----------
    fun : str
        A dotted import path to a function (e.g.,
        "quantify_scheduler.waveforms.square"), or a function pointer.

    Returns
    -------
    Callable[[Any], Any]


    Raises
    ------
    ValueError
        Raised if the function cannot be imported from path in the string.

    """
    try:
        return import_python_object_from_string(fun)
    except ImportError as exc:
        raise ValueError(f"{fun} is not a valid path to a known function.") from exc


def deserialize_class(cls: str) -> type:
    """
    Import a python class from a dotted import string (e.g.,
    "quantify_scheduler.structure.model.DataStructure").

    Parameters
    ----------
    cls : str
        A dotted import path to a class (e.g.,
        "quantify_scheduler.structure.model.DataStructure"), or a class pointer.

    Returns
    -------
    :
        The type you are trying to import.

    Raises
    ------
    ValueError
        Raised if the class cannot be imported from path in the string.

    """
    try:
        return import_python_object_from_string(cls)
    except ImportError as exc:
        raise ValueError(f"{cls} is not a valid path to a known class.") from exc
