import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.gate_library import CNOT, Measure, Reset, Rxy
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_hybrid_circuit_diagram_matplotlib():
    schedule = Schedule("Test experiment")

    q0, q1 = ("q0", "q1")
    ref_label_1 = "my_label"

    schedule.add(Reset(q0, q1))
    schedule.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    schedule.add(operation=CNOT(qC=q0, qT=q1))
    schedule.add(Measure(q0, q1), label="M0")

    f, _ = circuit_diagram_matplotlib(schedule)
    return f


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_hybrid_circuit_diagram_baseband_matplotlib():
    schedule = Schedule("Test experiment")

    q0, q1 = ("q0", "q1")
    ref_label_1 = "my_label"

    schedule.add(Reset(q0, q1))
    schedule.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    schedule.add(SquarePulse(0.8, 20e-9, port="q0:mw", clock="cl0.baseband"))
    schedule.add(operation=CNOT(qC=q0, qT=q1))
    schedule.add(Measure(q0, q1), label="M0")

    f, _ = circuit_diagram_matplotlib(schedule)
    return f


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_hybrid_circuit_acquisitions_matplotlib():
    """Tests drawing acquisitions"""
    schedule = Schedule("Test experiment")

    q0, q1 = ("q0", "q1")
    ref_label_1 = "my_label"

    schedule.add(Reset(q0, q1))
    schedule.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    schedule.add(
        SSBIntegrationComplex(port="q0:mw", clock="cl0.baseband", duration=1e-6)
    )
    schedule.add(operation=CNOT(qC=q0, qT=q1))
    schedule.add(Measure(q0, q1), label="M0")

    f, _ = circuit_diagram_matplotlib(schedule)
    return f


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_hybrid_circuit_diagram_modulated_matplotlib():
    schedule = Schedule("Test experiment")

    q0, q1 = ("q0", "q1")
    ref_label_1 = "my_label"

    schedule.add_resource(ClockResource(name="q0.01", freq=6.02e9))

    schedule.add(Reset(q0, q1))
    schedule.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    schedule.add(SquarePulse(0.8, 20e-9, port="q0:mw", clock="q0.01"))
    schedule.add(operation=CNOT(qC=q0, qT=q1))
    schedule.add(Measure(q0, q1), label="M0")

    f, _ = circuit_diagram_matplotlib(schedule)
    return f


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_hybrid_circuit_diagram_unknown_port_matplotlib():
    schedule = Schedule("Test experiment")

    q0, q1 = ("q0", "q1")
    ref_label_1 = "my_label"

    schedule.add_resource(ClockResource(name="q0.01", freq=6.02e9))

    schedule.add(Reset(q0, q1))
    schedule.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    schedule.add(SquarePulse(0.8, 20e-9, port="unknown_port", clock="q0.01"))
    schedule.add(operation=CNOT(qC=q0, qT=q1))
    schedule.add(Measure(q0, q1), label="M0")

    f, _ = circuit_diagram_matplotlib(schedule)
    return f
