# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Types that support validation in Pydantic.

Pydantic recognizes magic method ``__get_validators__`` to receive additional
validators, that can be used, i.e., for custom serialization and deserialization.
We implement several custom types here to tune behavior of our models.

See `Pydantic documentation`_ for more information about implementing new types.

.. _Pydantic documentation: https://docs.pydantic.dev/latest/usage/types/custom/
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from pydantic_core import core_schema

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class NDArray(np.ndarray):
    """
    Pydantic-compatible version of :class:`numpy.ndarray`.

    Serialization is implemented using custom methods :meth:`.to_dict` and
    :meth:`.from_dict`. Data array is encoded in Base64.
    """

    def __new__(cls: type[NDArray], array_like: ArrayLike) -> NDArray:  # noqa: D102
        return np.asarray(array_like).view(cls)

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: D105
        cls: type[NDArray],
        _source_type: Any,  # noqa: ANN401
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        def to_dict(v: NDArray) -> dict[str, Any]:
            """Convert the array to JSON-compatible dictionary."""
            return {
                "data": base64.b64encode(v.tobytes()).decode("ascii"),
                "shape": v.shape,
                "dtype": str(v.dtype),
            }

        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                to_dict, when_used="json"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the array to JSON-compatible dictionary."""
        return {
            "data": base64.b64encode(self.tobytes()).decode("ascii"),
            "shape": self.shape,
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_dict(cls: type[NDArray], serialized: Mapping[str, Any]) -> NDArray:
        """
        Construct an instance from a dictionary generated by :meth`to_dict`.

        Parameters
        ----------
        serialized
            Dictionary that has ``"data"``, ``"shape"`` and ``"dtype"`` keys.",
            where data is a base64-encoded bytes array, shape is a tuple and dtype is
            a string representation of a Numpy data type.

        """
        return (
            np.frombuffer(
                base64.b64decode(serialized["data"]), dtype=serialized["dtype"]
            )
            .reshape(serialized["shape"])
            .view(cls)
        )

    @classmethod
    def validate(cls: type[NDArray], v: Any) -> NDArray:  # noqa: ANN401
        """Validate the data and cast from all known representations."""
        if isinstance(v, Mapping):
            return cls.from_dict(v)  # type: ignore
        return cls(v)


class Graph(nx.Graph):
    """Pydantic-compatible version of :class:`networkx.Graph`."""

    # Avoid showing inherited init docstring (which leads to cross-reference issues)
    def __init__(
        self, incoming_graph_data=None, **attr  # noqa: ANN001, ANN003
    ) -> None:
        """Create a new graph instance."""
        super().__init__(incoming_graph_data, **attr)

    @classmethod
    def __get_pydantic_core_schema__(  # noqa: D105
        cls: type[Graph],
        _source_type: Any,  # noqa: ANN401
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                nx.node_link_data, when_used="always"
            ),
        )

    @classmethod
    def validate(cls: type[Graph], v: Any) -> Graph:  # noqa: ANN401
        """Validate the data and cast from all known representations."""
        if isinstance(v, dict):
            return cls(nx.node_link_graph(v))
        return cls(v)
