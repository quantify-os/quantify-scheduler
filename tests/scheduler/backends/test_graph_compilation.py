# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import pytest
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


@pytest.mark.xfail(reason="NotImplemented")
def test_compilation_backend_empty_graph_raises():

    empty_backend = CompilationBackend()
    empty_cfg = {}
    empty_sched = Schedule("test schedule")

    with pytest.raises(CompilationError):
        comp_sched = empty_backend.compile(schedule=empty_sched, cfg=empty_cfg)


@pytest.mark.xfail(reason="NotImplemented")
def test_compilation_backend_trivial_graph_compiles():

    trivial_graph = CompilationBackend()
    trivial_graph.add_edge("input", "output")
    empty_cfg = {}
    empty_sched = Schedule("test schedule")

    # issue, how do we define what "node" to compile to/where to feed in the input.
    comp_sched = trivial_graph.compile(schedule=empty_sched, cfg=empty_cfg)
    assert isinstance(comp_sched, CompiledSchedule)


@pytest.mark.xfail(reason="NotImplemented")
def test_device_compile_graph_timings_reset():

    sched = Schedule("Test schedule")
    sched.add(Reset("q0"))
    sched.add(Rxy(90, 0, qubit="q0"))

    config = {"device_cfg": example_transmon_cfg}

    comp_sched = DeviceCompile.compile(sched, cfg=config)
    assert isinstance(comp_sched, CompiledSchedule)


@pytest.mark.xfail(reason="NotImplemented")
def test_compile_a_graph_without_gates():
    raise NotImplementedError


@pytest.mark.xfail(reason="NotImplemented")
def test_merging_graphs():
    # here we test that we can easily create a new backend by "composing" two graphs.
    # the example we test should compile the device compile and the hardware compile.

    raise NotImplementedError
