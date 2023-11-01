# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Validated and serializable data structures using :mod:`pydantic`.

In this module we provide :class:`pre-configured Pydantic model <.DataStructure>` and
:mod:`custom field types <.types>` that allow serialization of typical data objects
that we frequently use in ``quantify-scheduler``, like functions and arrays.
"""

from .model import DataStructure
from .types import Graph, NDArray

__all__ = ["DataStructure", "Graph", "NDArray"]
