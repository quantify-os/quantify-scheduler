import pytest

from quantify_scheduler.operations.control_flow_library import Loop
from quantify_scheduler.operations.gate_library import Measure, Rxy
from quantify_scheduler.schedules.schedule import Schedule

from .compiles_all_backends import _CompilesAllBackends


class TestSubschedules(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        inner_schedule = Schedule("inner", repetitions=1)
        ref = inner_schedule.add(Rxy(0, 0, "q0"), label="inner0")
        inner_schedule.add(Rxy(0, 1, "q0"), rel_time=40e-9, ref_op=ref, label="inner1")

        outer_schedule = Schedule("outer", repetitions=10)
        ref = outer_schedule.add(Rxy(1, 0, "q0"), label="outer0")
        outer_schedule.add(inner_schedule, rel_time=80e-9, ref_op=ref)
        outer_schedule.add(Measure("q0"), label="measure")
        cls.uncomp_sched = outer_schedule

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == 10


class TestLoops:
    @classmethod
    def setup_class(cls):
        inner_schedule = Schedule("inner", repetitions=1)
        ref = inner_schedule.add(Rxy(0, 0, "q0"), label="inner0")
        inner_schedule.add(Rxy(0, 1, "q0"), rel_time=40e-9, ref_op=ref, label="inner1")

        outer_schedule = Schedule("outer", repetitions=1)
        ref = outer_schedule.add(Rxy(1, 0, "q0"), label="outer0")

        outer_schedule.add(
            inner_schedule,
            control_flow=Loop(repetitions=10),
            label="loop",
        )

        outer_schedule.add(Measure("q0"), label="measure")
        cls.uncomp_sched = outer_schedule

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == 1


def test_add_raises() -> None:
    schedule = Schedule("Test")
    with pytest.raises(ValueError):
        schedule.add(Loop(1))
    schedule.add(Measure("q0"))
