# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Root models for data structures used within the package."""

from typing import Any, Callable

import orjson
from pydantic import BaseModel, Extra


def orjson_dumps(obj: Any, *, default: Callable[[Any], Any]) -> str:
    """Dump an object to a JSON string using :mod:`orjson` library.

    Parameters
    ----------
    obj
        Object to dump
    default
        A function that is called if an object can't be serialized otherwise.
        It should return a JSON-encodable version of an object or raise a
        :class:`TypeError`.

    Returns
    -------
    str
        JSON-encoded string representation of an object

    Raises
    ------
    TypeError
        If value can't be serialized.
    """
    return orjson.dumps(obj, default=default).decode()


class DataStructure(BaseModel):  # pylint: disable=too-few-public-methods
    """A parent for all data structures.

    Data attributes are generated from the class' type annotations, similarly to
    `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_. If data
    attributes are JSON-serializable, data structure can be serialized using
    ``json()`` method. This string can be deserialized using ``parse_raw()`` classmethod
    of a correspondent child class.

    If required, data fields can be validated, see examples for more information.
    It is also possible to define custom field types with advanced validation.

    This class is a pre-configured `pydantic <https://pydantic-docs.helpmanual.io/>`_
    model. See its documentation for details of usage information.

    .. admonition:: Examples
        :class: dropdown

        .. include:: examples/structure.DataStructure.py.rst
    """

    class Config:  # pylint: disable=too-few-public-methods,missing-class-docstring
        json_loads = orjson.loads
        json_dumps = orjson_dumps

        # ensures exceptions are raised when passing extra argument that are not
        # part of a model when initializing.
        extra = Extra.forbid
