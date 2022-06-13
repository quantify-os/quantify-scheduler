# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import pytest
import networkx as nx
from quantify_scheduler.backends.graph_compilation import (
    CompilationBackend,
    CompilationNode,
    CompilationError,
)
from quantify_scheduler import Schedule, CompiledSchedule

from quantify_scheduler.backends.device_compile import DeviceCompile
from quantify_scheduler.schemas.examples.circuit_to_device_example_cfgs import (
    example_transmon_cfg,
)

from quantify_scheduler.operations.gate_library import (
    Measure,
    Reset,
    Rxy,
)


def dummy_compile_add_reset_q0(schedule: Schedule, config=None) -> Schedule:
    schedule.add(Reset("q0"))
    return schedule


dummy_node_A = CompilationNode(
    name="dummy_node_A",
    compilation_func=dummy_compile_add_reset_q0,
    config_key=None,
    config_validator=None,
)


dummy_node_B = CompilationNode(
    name="dummy_node_B",
    compilation_func=dummy_compile_add_reset_q0,
    config_key=None,
    config_validator=None,
)

dummy_node_C = CompilationNode(
    name="dummy_node_C",
    compilation_func=dummy_compile_add_reset_q0,
    config_key=None,
    config_validator=None,
)


dummy_node_D = CompilationNode(
    name="dummy_node_D",
    compilation_func=dummy_compile_add_reset_q0,
    config_key=None,
    config_validator=None,
)


def test_compilation_backend_empty_graph_raises():
    """
    A graph in which the input and output are not connect should raise an exception.
    """

    empty_backend = CompilationBackend()
    empty_cfg = {}
    empty_sched = Schedule("test schedule")

    with pytest.raises(CompilationError):
        _ = empty_backend.compile(schedule=empty_sched, config=empty_cfg)


def test_compilation_backend_trivial_graph():
    """
    A trivial graph where the input is connected directly to the output should return
    the same schedule
    """

    trivial_graph = CompilationBackend()
    trivial_graph.add_edge("input", "output")
    empty_cfg = {}
    empty_sched = Schedule("test schedule")

    # issue, how do we define what "node" to compile to/where to feed in the input.
    comp_sched = trivial_graph.compile(schedule=empty_sched, config=empty_cfg)
    assert isinstance(comp_sched, Schedule)
    assert comp_sched == empty_sched


def test_device_compile_graph_timings_reset():

    sched = Schedule("Test schedule")
    sched.add(Reset("q0"))
    sched.add(Rxy(90, 0, qubit="q0"))

    config = {"device_cfg": example_transmon_cfg}

    comp_sched = DeviceCompile().compile(sched, config=config)
    assert isinstance(comp_sched, Schedule)
    assert comp_sched.timing_table.data.iloc[0].abs_time == 0
    # this is the reset duration of q0 specified in the example config
    assert comp_sched.timing_table.data.iloc[1].abs_time == 200e-6


@pytest.mark.xfail(reason="NotImplemented")
def test_compile_a_graph_without_gates():
    raise NotImplementedError


def test_dummy_nodes_add_operation():
    first_graph = CompilationBackend()
    first_graph.add_node(dummy_node_A)
    first_graph.add_edge("input", dummy_node_A)
    first_graph.add_node(dummy_node_B)
    first_graph.add_edge(dummy_node_A, dummy_node_B)
    first_graph.add_edge(dummy_node_B, "output")

    sched = Schedule("Test schedule")
    assert len(sched.schedulables) == 0
    first_graph.compile(sched, config={})
    assert len(sched.schedulables) == 2


def test_merging_graphs():
    """
    here we test that we can easily create a new backend by "composing" two graphs.
    We want the output of the first graph to be connected to the input of the second
    graph.
    """
    first_graph = CompilationBackend()
    first_graph.add_node(dummy_node_A)
    first_graph.add_edge("input", dummy_node_A)
    first_graph.add_node(dummy_node_B)
    first_graph.add_edge(dummy_node_A, dummy_node_B)
    first_graph.add_edge(dummy_node_B, "output")

    assert nx.shortest_path(first_graph, "input", "output") == [
        "input",
        dummy_node_A,
        dummy_node_B,
        "output",
    ]

    second_graph = CompilationBackend()
    second_graph.add_node(dummy_node_C)
    second_graph.add_edge("input", dummy_node_C)
    second_graph.add_node(dummy_node_D)
    second_graph.add_edge(dummy_node_C, dummy_node_D)
    second_graph.add_edge(dummy_node_D, "output")

    assert nx.shortest_path(second_graph, "input", "output") == [
        "input",
        dummy_node_C,
        dummy_node_D,
        "output",
    ]

    comp_graph = first_graph.compose(first_graph, second_graph)
    assert nx.shortest_path(comp_graph, "input", "output") == [
        "input",
        dummy_node_A,
        dummy_node_B,
        dummy_node_C,
        dummy_node_D,
        "output",
    ]
