import re

import ipywidgets
import matplotlib.pyplot as plt
import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.operations.gate_library import (
    Measure,
    Reset,
    X,
)
from quantify_scheduler.schedules.schedule import Schedule


@pytest.fixture
def mock_widgets(mocker):
    mocker.patch("ipywidgets.Tab")
    mocker.patch("ipywidgets.Output")
    mocker.patch("matplotlib.pyplot.show")
    mocker.patch("quantify_scheduler.backends.qblox.visualization.display")


@pytest.fixture
def compilation_config(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def compiled_schedule(compile_config_basic_transmon_qblox_hardware_cluster):
    schedule = Schedule("demo compiled instructions")
    schedule.add(Reset("q0"))
    schedule.add(X("q0"))
    schedule.add(Measure("q0", coords={"index": 0}))

    compiler = SerialCompiler("compiler")
    yield compiler.compile(schedule, config=compile_config_basic_transmon_qblox_hardware_cluster)


def test_display_compiled_instructions_not_None(  # noqa: N802
    mock_widgets,
    compiled_schedule,
):
    from quantify_scheduler.backends.qblox.visualization import _display_compiled_instructions

    tab = _display_compiled_instructions(compiled_schedule.compiled_instructions)
    assert tab is not None


def test_display_compiled_instructions_calls_mpl_show(
    mock_widgets,
    compiled_schedule,
):
    from quantify_scheduler.backends.qblox.visualization import _display_compiled_instructions

    _ = _display_compiled_instructions(compiled_schedule.compiled_instructions)
    assert plt.show.call_count == 3  # type: ignore


def test_display_compiled_instructions_programs_are_displayed(
    mock_widgets,
    compiled_schedule,
):
    from quantify_scheduler.backends.qblox.visualization import _display_compiled_instructions

    _ = _display_compiled_instructions(compiled_schedule.compiled_instructions)
    tab_instance = ipywidgets.Tab.return_value  # type: ignore

    program_tab_count = sum(
        1 for call in tab_instance.set_title.call_args_list if call[0][1] == "program"
    )

    # two sequencers should be used
    expected_tab_count = 2
    assert program_tab_count == expected_tab_count


def test_display_compiled_instructions_programs_active_modules_are_displayed(
    mock_widgets,
    compiled_schedule,
):
    from quantify_scheduler.backends.qblox.visualization import _display_compiled_instructions

    _ = _display_compiled_instructions(compiled_schedule.compiled_instructions)
    tab_instance = ipywidgets.Tab.return_value  # type: ignore

    cluster_module_tabs = [
        call[0][1]
        for call in tab_instance.set_title.call_args_list
        if re.match(r"^cluster\d_module\d$", call[0][1])
    ]

    assert set(cluster_module_tabs) == {"cluster0_module1", "cluster0_module3"}


def test_compiled_instructions_displayed_in_jupyter(
    mock_widgets,
    compiled_schedule,
):
    """Test that compiled_instructions triggers display in Jupyter."""
    from quantify_scheduler.backends.qblox.visualization import display

    compiled_instructions = compiled_schedule.compiled_instructions
    compiled_instructions._ipython_display_()

    assert display.call_count >= 1  # type: ignore
