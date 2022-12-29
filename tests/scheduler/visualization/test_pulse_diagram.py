# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import quantify_scheduler.visualization.pulse_diagram as plsd
from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.operations.gate_library import CZ, Measure, Reset, Rxy


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
    fig = plsd.pulse_diagram_plotly(compiled_sched)

    assert fig.data
