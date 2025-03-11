# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

"""Most Schedule tests are in the legacy test file `test_types.py`"""

from quantify_scheduler.schedules.schedule import Schedule


def test_init_schedule_defaults():
    schedule = Schedule()
    assert schedule.repetitions == 1
    assert schedule.name == "schedule"


def test_two_schedules_with_same_name():
    sched1 = Schedule()
    sched2 = Schedule("schedule")
    assert sched1 == sched2
    assert sched1 is not sched2


def test_two_schedules_with_different_names_are_not_equal():
    assert Schedule("x") != Schedule("y")
