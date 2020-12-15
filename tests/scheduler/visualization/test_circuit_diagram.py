import pytest
from quantify.scheduler import Schedule
from quantify.scheduler.gate_library import Reset, Measure, CNOT, Rxy
from quantify.scheduler.pulse_library import SquarePulse
from quantify.scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib


@pytest.mark.mpl_image_compare(style='default', savefig_kwargs={'dpi': 300})
def test_circuit_diagram_matplotlib():
    sched = Schedule('Test experiment')

    q0, q1 = ('q0', 'q1')
    ref_label_1 = 'my_label'

    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(operation=CNOT(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label='M0')

    f, ax = circuit_diagram_matplotlib(sched)
    return f


def test_hybrid_circuit_diagram_matplotlib():
    sched = Schedule('Test experiment')

    q0, q1 = ('q0', 'q1')
    ref_label_1 = 'my_label'

    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(SquarePulse(0.8, 20e-9, port='q0:mw', clock='cl0.baseband'))
    sched.add(operation=CNOT(qC=q0, qT=q1))
    sched.add(Measure(q0, q1), label='M0')

    with pytest.raises(NotImplementedError):
        f, ax = circuit_diagram_matplotlib(sched)
