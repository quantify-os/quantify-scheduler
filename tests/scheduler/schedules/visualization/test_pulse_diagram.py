# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import numpy as np
import matplotlib.pyplot as plt
import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.gate_library import CZ, Measure, Reset, Rxy
from quantify_scheduler.operations.pulse_library import SquarePulse, WindowOperation
from quantify_scheduler.resources import BasebandClockResource
from quantify_scheduler.schedules._visualization.pulse_diagram import (
    get_window_operations,
    plot_acquisition_operations,
    plot_window_operations,
    pulse_diagram_matplotlib,
    pulse_diagram_plotly,
    sample_schedule,
)
from quantify_scheduler.visualization.pulse_diagram import (
    get_window_operations as get_window_operations_deprecated,
)
from quantify_scheduler.visualization.pulse_diagram import (
    plot_acquisition_operations as plot_acquisition_operations_deprecated,
)
from quantify_scheduler.visualization.pulse_diagram import (
    plot_window_operations as plot_window_operations_deprecated,
)
from quantify_scheduler.visualization.pulse_diagram import (
    pulse_diagram_matplotlib as pulse_diagram_matplotlib_deprecated,
)
from quantify_scheduler.visualization.pulse_diagram import (
    pulse_diagram_plotly as pulse_diagram_plotly_deprecated,
)
from quantify_scheduler.visualization.pulse_diagram import (
    sample_schedule as sample_schedule_deprecated,
)

# All test_*_deprecated can be removed when quantify_scheduler.visualization module is removed


# Proper verification of this, probably requires some horrible selenium malarkey
def test_pulse_diagram_plotly(device_compile_config_basic_transmon) -> None:
    sched = Schedule("Test schedule")

    # Define the resources
    q0, q2 = ("q0", "q2")

    sched.add(Reset(q0, q2))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q2))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q2), label="M0")

    # Pulse information is added
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched, config=device_compile_config_basic_transmon
    )

    # It should be possible to generate this visualization after compilation
    fig = pulse_diagram_plotly(compiled_sched)

    assert fig.data


def test_pulse_diagram_matplotlib() -> None:
    schedule = Schedule("test")
    schedule.add(SquarePulse(amp=0.2, duration=4e-6, port="SDP"))
    schedule.add(SquarePulse(amp=0.3, duration=6e-6, port="SDP"))
    schedule.add(
        WindowOperation(window_name="second pulse", duration=6e-6), ref_pt="start"
    )
    schedule.add(SquarePulse(amp=0.25, duration=6e-6, port="SDP"))
    determine_absolute_timing(schedule=schedule)

    plt.figure(1)
    plt.clf()
    pulse_diagram_matplotlib(schedule, sampling_rate=20e6)

    window_operations = get_window_operations(schedule)
    plot_window_operations(schedule)

    assert len(window_operations) == 1
    window = window_operations[0]
    assert window[0] == pytest.approx(4e-6)
    assert window[1] == pytest.approx(10e-6)
    assert isinstance(window[2], WindowOperation)

    plt.close(1)


def test_plot_acquisition_operations() -> None:
    schedule = Schedule("test")
    schedule.add(SquarePulse(amp=0.2, duration=4e-6, port="SDP"))
    determine_absolute_timing(schedule=schedule)
    handles = plot_acquisition_operations(schedule)
    assert len(handles) == 0

    schedule.add(SSBIntegrationComplex("P", clock="cl0.baseband", duration=2e-6))
    determine_absolute_timing(schedule=schedule)
    handles = plot_acquisition_operations(schedule)
    assert len(handles) == 1


def test_sample_schedule() -> None:
    schedule = Schedule("test")
    r = SquarePulse(amp=0.2, duration=4e-9, port="SDP")
    schedule.add(r)
    rm = SquarePulse(amp=-0.2, duration=6e-9, port="T")
    schedule.add(rm, ref_pt="start")
    r = SquarePulse(amp=0.3, duration=6e-9, port="SDP")
    schedule.add(r)
    schedule.add(r)
    determine_absolute_timing(schedule=schedule)

    timestamps, waveforms = sample_schedule(schedule, sampling_rate=0.5e9)

    np.testing.assert_array_almost_equal(
        timestamps,
        np.array(
            [
                0.0e00,
                2.0e-09,
                4.0e-09,
                6.0e-09,
                8.0e-09,
                1.0e-08,
                1.2e-08,
                1.4e-08,
                1.6e-08,
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        waveforms["SDP"], np.array([0.2, 0.2, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    )
    np.testing.assert_array_almost_equal(
        waveforms["T"], np.array([-0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_sample_modulated_waveform() -> None:
    schedule = Schedule("test")
    clock0 = BasebandClockResource("clock0")
    clock0["freq"] = 0.15e9
    schedule.add_resource(clock0)

    square_pulse_op = SquarePulse(amp=0.2, duration=3e-9, port="SDP", clock="clock0")
    schedule.add(square_pulse_op)
    square_pulse_op = SquarePulse(amp=0.2, duration=3e-9, port="T")
    schedule.add(square_pulse_op, ref_pt="start")
    determine_absolute_timing(schedule=schedule)

    _, waveforms = sample_schedule(
        schedule,
        sampling_rate=1e9,
        modulation="clock",
    )

    assert waveforms["SDP"].dtype.kind == "c"
    np.testing.assert_array_almost_equal(
        waveforms["SDP"],
        np.array([0.2 + 0.0j, 0.117557 - 0.161803j, -0.061803 - 0.190211j]),
    )
    np.testing.assert_array_almost_equal(waveforms["T"], np.array([0.2, 0.2, 0.2]))


def test_sample_custom_port_list() -> None:
    schedule = Schedule("test")
    r = SquarePulse(amp=0.2, duration=4e-9, port="SDP")
    schedule.add(r)
    determine_absolute_timing(schedule=schedule)

    _, waveforms = sample_schedule(schedule, sampling_rate=0.5e9, port_list=["SDP"])
    assert list(waveforms.keys()) == ["SDP"]


def test_sample_empty_schedule() -> None:
    schedule = Schedule("test")

    with pytest.raises(RuntimeError):
        _, _ = sample_schedule(schedule, sampling_rate=1e9)


# Proper verification of this, probably requires some horrible selenium malarkey
def test_pulse_diagram_plotly_deprecated(device_compile_config_basic_transmon) -> None:
    sched = Schedule("Test schedule")

    # Define the resources
    q0, q2 = ("q0", "q2")

    sched.add(Reset(q0, q2))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q2))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q2), label="M0")

    # Pulse information is added
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched, config=device_compile_config_basic_transmon
    )

    # It should be possible to generate this visualization after compilation
    fig = pulse_diagram_plotly_deprecated(compiled_sched)

    assert fig.data


def test_pulse_diagram_matplotlib_deprecated() -> None:
    schedule = Schedule("test")
    schedule.add(SquarePulse(amp=0.2, duration=4e-6, port="SDP"))
    schedule.add(SquarePulse(amp=0.3, duration=6e-6, port="SDP"))
    schedule.add(
        WindowOperation(window_name="second pulse", duration=6e-6), ref_pt="start"
    )
    schedule.add(SquarePulse(amp=0.25, duration=6e-6, port="SDP"))
    determine_absolute_timing(schedule=schedule)

    plt.figure(1)
    plt.clf()
    pulse_diagram_matplotlib_deprecated(schedule, sampling_rate=20e6)

    window_operations = get_window_operations_deprecated(schedule)
    plot_window_operations_deprecated(schedule)

    assert len(window_operations) == 1
    window = window_operations[0]
    assert window[0] == pytest.approx(4e-6)
    assert window[1] == pytest.approx(10e-6)
    assert isinstance(window[2], WindowOperation)

    plt.close(1)


def test_plot_acquisition_operations_deprecated() -> None:
    schedule = Schedule("test")
    schedule.add(SquarePulse(amp=0.2, duration=4e-6, port="SDP"))
    determine_absolute_timing(schedule=schedule)
    handles = plot_acquisition_operations_deprecated(schedule)
    assert len(handles) == 0

    schedule.add(SSBIntegrationComplex("P", clock="cl0.baseband", duration=2e-6))
    determine_absolute_timing(schedule=schedule)
    handles = plot_acquisition_operations_deprecated(schedule)
    assert len(handles) == 1


def test_sample_schedule_deprecated() -> None:
    schedule = Schedule("test")
    r = SquarePulse(amp=0.2, duration=4e-9, port="SDP")
    schedule.add(r)
    rm = SquarePulse(amp=-0.2, duration=6e-9, port="T")
    schedule.add(rm, ref_pt="start")
    r = SquarePulse(amp=0.3, duration=6e-9, port="SDP")
    schedule.add(r)
    schedule.add(r)
    determine_absolute_timing(schedule=schedule)

    timestamps, waveforms = sample_schedule_deprecated(schedule, sampling_rate=0.5e9)

    np.testing.assert_array_almost_equal(
        timestamps,
        np.array(
            [
                0.0e00,
                2.0e-09,
                4.0e-09,
                6.0e-09,
                8.0e-09,
                1.0e-08,
                1.2e-08,
                1.4e-08,
                1.6e-08,
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        waveforms["SDP"], np.array([0.2, 0.2, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    )
    np.testing.assert_array_almost_equal(
        waveforms["T"], np.array([-0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_sample_modulated_waveform_deprecated() -> None:
    schedule = Schedule("test")
    clock0 = BasebandClockResource("clock0")
    clock0["freq"] = 0.15e9
    schedule.add_resource(clock0)

    square_pulse_op = SquarePulse(amp=0.2, duration=3e-9, port="SDP", clock="clock0")
    schedule.add(square_pulse_op)
    square_pulse_op = SquarePulse(amp=0.2, duration=3e-9, port="T")
    schedule.add(square_pulse_op, ref_pt="start")
    determine_absolute_timing(schedule=schedule)

    _, waveforms = sample_schedule_deprecated(
        schedule,
        sampling_rate=1e9,
        modulation="clock",
    )

    assert waveforms["SDP"].dtype.kind == "c"
    np.testing.assert_array_almost_equal(
        waveforms["SDP"],
        np.array([0.2 + 0.0j, 0.117557 - 0.161803j, -0.061803 - 0.190211j]),
    )
    np.testing.assert_array_almost_equal(waveforms["T"], np.array([0.2, 0.2, 0.2]))


def test_sample_custom_port_list_deprecated() -> None:
    schedule = Schedule("test")
    r = SquarePulse(amp=0.2, duration=4e-9, port="SDP")
    schedule.add(r)
    determine_absolute_timing(schedule=schedule)

    _, waveforms = sample_schedule_deprecated(
        schedule, sampling_rate=0.5e9, port_list=["SDP"]
    )
    assert list(waveforms.keys()) == ["SDP"]


def test_sample_empty_schedule_deprecated() -> None:
    schedule = Schedule("test")

    with pytest.raises(RuntimeError):
        _, _ = sample_schedule_deprecated(schedule, sampling_rate=1e9)
