# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import json
import inspect
from pathlib import Path

from quantify.scheduler import Schedule
from quantify.scheduler.gate_library import Reset, Measure, Rxy
from quantify.scheduler.compilation import qcompile
import quantify.scheduler.visualization.pulse_diagram as plsd
import quantify.scheduler.schemas.examples as es


esp = inspect.getfile(es)
cfg_f = Path(esp).parent / "transmon_test_config.json"
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

# Proper verification of this, probably requires some horrible selenium malarkey
def test_pulse_diagram_plotly():
    sched = Schedule("Test schedule")

    # define the resources
    qubit_0, qubit_1 = ("q0", "q1")
    sched.add(Reset(qubit_0, qubit_1))
    sched.add(Rxy(90, 0, qubit=qubit_0))
    # sched.add(operation=CZ(qC=qubit_0, qT=qubit_1)) # not implemented in config
    sched.add(Rxy(theta=90, phi=0, qubit=qubit_0))
    sched.add(Measure(qubit_0, qubit_1), label="M0")
    # pulse information is added
    sched = qcompile(sched, DEVICE_CFG, None)

    # It should be possible to generate this visualization after compilation
    fig = plsd.pulse_diagram_plotly(sched)

    assert fig.data
