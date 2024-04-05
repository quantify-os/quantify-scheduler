import networkx as nx
import numpy as np

from quantify_scheduler.structure import DataStructure, Graph, NDArray


def test_ndarray():
    class Model(DataStructure):
        a_string: str
        an_array: NDArray

    instance = Model(a_string="foo", an_array=[1 + 1j, 101])  # type: ignore
    assert isinstance(instance.an_array, NDArray)
    assert instance.an_array.dtype == np.complex128

    serialized = instance.model_dump_json()
    deserialized = Model.model_validate_json(serialized)

    np.testing.assert_equal(instance.an_array, deserialized.an_array)


def test_graph():
    class Model(DataStructure):
        a_string: str
        a_graph: Graph

    instance = Model(a_string="foo", a_graph=nx.complete_graph(5))
    assert isinstance(instance.a_graph, Graph)

    serialized = instance.model_dump_json()
    deserialized = Model.model_validate_json(serialized)

    assert isinstance(deserialized.a_graph, nx.Graph)
    assert instance.a_graph.edges == deserialized.a_graph.edges
    assert instance.a_graph.nodes == deserialized.a_graph.nodes
