# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals

import pytest
from quantify_scheduler.gettables import ScheduleGettableSingleChannel


@pytest.mark.xfail(reason="not implemented")
def test_ScheduleGettableSingleChannel():
    # FIXME Test not implemented
    gettable = ScheduleGettableSingleChannel

    assert False
