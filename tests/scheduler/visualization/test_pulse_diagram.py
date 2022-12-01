# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import quantify_scheduler.visualization.pulse_diagram as plsd
from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.operations.gate_library import Measure, Reset, Rxy

# Proper verification of this, probably requires some horrible selenium malarkey
def test_pulse_diagram_plotly(mock_setup_basic_transmon_with_standard_params) -> None:
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    compilation_config = quantum_device.generate_compilation_config()

    sched = Schedule("Test schedule")

    # define the resources
    qubit_0, qubit_1 = ("q0", "q1")
    sched.add(Reset(qubit_0, qubit_1))
    sched.add(Rxy(90, 0, qubit=qubit_0))
    # sched.add(operation=CZ(qC=qubit_0, qT=qubit_1)) # not implemented in config
    sched.add(Rxy(theta=90, phi=0, qubit=qubit_0))
    sched.add(Measure(qubit_0, qubit_1), label="M0")
    # pulse information is added

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(schedule=sched, config=compilation_config)

    # It should be possible to generate this visualization after compilation
    fig = plsd.pulse_diagram_plotly(compiled_sched)

    assert fig.data
