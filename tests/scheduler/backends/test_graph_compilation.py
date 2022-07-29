"""
This test style covers the classes and functions in the backends/graph_compilation.py
file.
"""

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from matplotlib.axes import Axes
from quantify_scheduler.backends.graph_compilation import (
    QuantifyCompiler,
    SimpleNode,
)
from quantify_scheduler import Schedule


from quantify_scheduler.operations.gate_library import (
    Reset,
)

# pylint: disable=unused-argument
def dummy_compile_add_reset_q0(schedule: Schedule, config=None) -> Schedule:
    schedule.add(Reset("q0"))
    return schedule


dummy_node_A = SimpleNode(
    name="dummy_node_A",
    compilation_func=dummy_compile_add_reset_q0,
)


dummy_node_B = SimpleNode(
    name="dummy_node_B",
    compilation_func=dummy_compile_add_reset_q0,
)

dummy_node_C = SimpleNode(
    name="dummy_node_C",
    compilation_func=dummy_compile_add_reset_q0,
)


dummy_node_D = SimpleNode(
    name="dummy_node_D",
    compilation_func=dummy_compile_add_reset_q0,
)


def test_draw_backend():
    """
    Tests if we can visualize a the graph defined by a generic backend.
    This test will only test if the draw code can be executed and a matplotlib figure
    is created. It will not test the details of how the figure looks.
    """
    test_graph = QuantifyCompiler(name="test")
    test_graph.add_node(dummy_node_A)
    test_graph.add_node(dummy_node_B)
    test_graph.add_edge(dummy_node_A, dummy_node_B)

    test_graph.add_edge(dummy_node_C, dummy_node_B)
    test_graph.add_edge(dummy_node_C, dummy_node_A)

    ax = test_graph.draw()
    assert isinstance(ax, Axes)
