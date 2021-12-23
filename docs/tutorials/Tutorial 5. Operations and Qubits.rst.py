# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python [conda env:pysweep-env]
#     language: python
#     name: conda-env-pysweep-env-py
# ---

# %% [raw]
#  .. _sec-tutorial5:
#
# Tutorial 5. Operations and Qubits
# ================================
#
#  .. jupyter-kernel::
#    :id: Tutorial 5. Operations and Qubits
#
# .. seealso::
#
#     The complete source code of this tutorial can be found in
#
#     :jupyter-download:notebook:`Tutorial 5. Operations and Qubits`
#
#     :jupyter-download:script:`Tutorial 5. Operations and Qubits`

# %%
from pprint import pprint

# %% [raw]
# Gates, measurements and qubits
# ------------------------------
#
# In the previous tutorials, operations were created on the pulse level. On the pulse level, operations are defined by their waveforms which have to be defined explicitly and are not linked directly to a qubit operation.
# In addition to the pulse level, `quantify_scheduler` allows creating operations on the circuit level.
# Instead of specifying waveforms, operations are defined by the intended qubit operation.
#
# Many of the gates used in the circuit layer description are defined in :module:`quantify_scheduler.operations.gate_library` such as `Reset`, `X90` and `Measure`.
# Operations are instantiated by providing them with the name of the qubit(s) on which they operate:

# %%
from quantify_scheduler.operations.gate_library import CZ, X90, Measure, Reset, Rxy
q0, q1 = ("q0", "q1")
X90(q0)
Measure(q1)
CZ(q0, q1)
Reset(q0)

# %% [markdown]
# Let's investigate the different components present in the circuit level description of the operation. As an example, we create an 45 degree rotation operation over the x-axis.

# %%
from pprint import pprint
pprint(Rxy(theta=45.0, phi=0.0, qubit=q0).data)

# %% [markdown]
# As we can see, the structure of a circuit level operation is similar to a pulse lever operation. However, the information is contained inside the `gate_info` entry rather than the `pulse_info` entry of the data dictionary. The schema for the `gate_info` entry can be obtained as:

# %%
from quantify_core.utilities import general
general.load_json_schema(Reset(q0).__file__,'operation.json')

# %%
from quantify_scheduler.schemas import operation

# %% [raw]
# Schedule creation from the circuit layer
# ----------------------------------------
#
# `quantify_scheduler`  allows defining schedules on the circuit level rather than the pulse level as well.
#  allowing for creating schedules on a more abstract level. We exemplify this extra layer of abstraction using `Bell violations`. can be added to the schedule directly,
#
# The high-level gate layer can be mixed with the pulse level.
# This mixed representation is useful for experiments where some pulses cannot easily be represented as qubit gates. An example of this is given by the `Chevron` experiment given in sec. 1.6.

# %% [raw]
# Bell circuit
# ~~~~~~~~~~~~
#
# As the first example, we want to perform the
# `Bell experiment <https://en.wikipedia.org/wiki/Bell%27s_theorem>`_ .
# The goal of the Bell experiment is to create a Bell state
# :math:`|\Phi ^+\rangle=\frac{1}{2}(|00\rangle+|11\rangle)` followed by a measurement.
# By rotating the measurement basis, or equivalently one of the qubits, it is possible
# to observe violations of the CSHS inequality.
#
# We create this experiment using a
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>` description.
# This allows defining the Bell schedule as:

# %%
import numpy as np

from quantify_scheduler import Schedule


sched = Schedule("Bell experiment")

# we use a regular for loop as we have to unroll the changing theta variable here
for acq_idx, theta in enumerate(np.linspace(0, 360, 21)):
    sched.add(Reset(q0, q1))
    sched.add(X90(q0))
    sched.add(X90(q1), ref_pt="start")  # this ensures pulses are aligned
    sched.add(CZ(q0, q1))
    sched.add(Rxy(theta=theta, phi=0, qubit=q0))

    sched.add(Measure(q0, acq_index=acq_idx), label="M q0 {:.2f} deg".format(theta))
    sched.add(
        Measure(q1, acq_index=acq_idx),
        label="M q1 {:.2f} deg".format(theta),
        ref_pt="start",
    )

# %% [raw]
# Visualizing the quantum circuit
# -------------------------------
#
# And we can use this to create a default visualization:

# %%
# %matplotlib inline

f, ax = sched.plot_circuit_diagram_mpl()
# all gates are plotted, but it doesn't all fit in a matplotlib figure
ax.set_xlim(-0.5, 9.5)

# %% [raw]
# Device configuration and compilation
# ------------------------------------
#
# The compilation from the gate-level to the pulse-level description is done using the
# :ref:`device configuration file<Device configuration file>`.
#
# Here we will use a configuration file for a transmon based system that is part of the
# quantify-scheduler test suite.

# %%
import inspect
import json
import os
from pathlib import Path

import quantify_scheduler.schemas.examples as es

esp = inspect.getfile(es)
cfg_f = Path(esp).parent / "transmon_test_config.json"


with open(cfg_f, "r") as f:
    transmon_test_config = json.load(f)

transmon_test_config

# %%
from quantify_scheduler.compilation import device_compile

pulse_sched = device_compile(sched, transmon_test_config)

# %%
f, ax = pulse_sched.plot_pulse_diagram_mpl()
ax.set_xlim(0.4005e-3, 0.4006e-3)

# %% [raw]
# Quantum Backends
# ----------------

# %%

# %% [raw]
# Mixing pulse and circuit layer operations
# -----------------------------------------
#
# As well as defining our schedules in terms of Gates, we can also mix the circuit layer representation with pulse level operations.
# This can be useful for experiments involving pulses not easily represented by Gates, such as the
# Chevron experiment.
# In this experiment, we want to vary the length and amplitude of a square pulse between X gates on a pair of qubits.

# %%
from quantify_scheduler.operations.gate_library import X90, Measure, Reset, X
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource

sched = Schedule("Chevron Experiment")
acq_idx = 0

for duration in np.linspace(
    20e-9, 60e-9, 6
):  # NB multiples of 4 ns need to be used due to limitations of the pulsars
    for amp in np.linspace(0.1, 1.0, 10):
        begin = sched.add(Reset("q0", "q1"))
        sched.add(X("q0"), ref_op=begin, ref_pt="end")
        # NB we specify a clock for tutorial purposes,
        # Chevron experiments do not necessarily use modulated square pulses
        square = sched.add(SquarePulse(amp, duration, "q0:mw", clock="q0.01"))
        sched.add(X90("q0"), ref_op=square)
        sched.add(X90("q1"), ref_op=square)
        sched.add(Measure(q0, acq_index=acq_idx), label=f"M q0 {acq_idx}")
        sched.add(
            Measure(q1, acq_index=acq_idx), label=f"M q1 {acq_idx}", ref_pt="start"
        )

        acq_idx += 1


sched.add_resources([ClockResource("q0.01", 6.02e9)])  # manually add the pulse clock

# %% [raw]
# Note that we add Pulses using the same interface as Gates. Pulses are Operations, and
# as such support the same timing and reference operators as Gates.
#
# .. warning::
#
#     When adding a Pulse to a schedule, the clock is not automatically added to the
#     resources of the schedule. It may be necessary to add this clock manually, as in
#     the final line of the above example
#
# We can also quickly compile using the :func:`!qcompile` function and the associated
# configuration files:

# %%
from quantify_scheduler.compilation import qcompile

cfg = qcompile(sched, transmon_test_config, qblox_test_mapping)

# %%
