# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [raw]
# .. _sec-tutorial1:
#
# Tutorial 1. Basic experiments
# =============================
#
# .. jupyter-kernel::
#   :id: Tutorial 1. Basic experiment
#
# .. seealso::
#
#     The complete source code of this tutorial can be found in
#
#     :jupyter-download:notebook:`Tutorial 1. Basic experiment`
#
#     :jupyter-download:script:`Tutorial 1. Basic experiment`
#
# .. tip::
#     Following this Tutorial requires familiarity with the **core concepts** of
#     Quantify-scheduler, we **highly recommended** to consult the (short)
#     :ref:`User guide <sec-user-guide>` before proceeding.
#
#
# The benefit of allowing the user to mix the high-level gate description of a circuit
# with the lower-level pulse description can be understood through an example.
# Below we first give an example of basic usage using `Bell violations`.
# We next show the `Chevron` experiment in which the user is required to mix gate-type
# and pulse-type information when defining the
# :class:`quantify_scheduler.schedules.schedule.Schedule`.
#
# Basics: The Bell experiment
# ---------------------------
#
# As the first example, we want to perform the
# `Bell experiment <https://en.wikipedia.org/wiki/Bell%27s_theorem>`_ .
# The goal of the Bell experiment is to create a Bell state
# :math:`|\Phi ^+\rangle=\frac{1}{2}(|00\rangle+|11\rangle)` followed by a measurement.
# By rotating the measurement basis, or equivalently one of the qubits, it is possible
# to observe violations of the CSHS inequality. If everything is done properly, one
# should observe the following oscillation:

# %%
import numpy as np
import plotly.graph_objects as go

x = np.linspace(0, 360, 361)
y = np.cos(np.deg2rad(x - 180))
yc = np.minimum(x / 90 - 1, -x / 90 + 3)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name="Quantum"))
fig.add_trace(go.Scatter(x=x, y=yc, name="Classical"))

fig.update_layout(
    title="Bell experiment",
    xaxis_title="Angle between detectors (deg)",
    yaxis_title="Correlation",
)
fig.show()

# %% [raw]
# Bell circuit
# ~~~~~~~~~~~~
#
#
# We create this experiment using a
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>` description.
#
#
# We start by initializing an empty
# :class:`quantify_scheduler.schedules.schedule.Schedule`

# %%
# Make output easier to read
from pathlib import Path

from quantify_core.data.handling import set_datadir
from rich import pretty

from quantify_scheduler import Schedule

pretty.install()

set_datadir(Path.home() / "quantify-data")
sched = Schedule("Bell experiment")
sched

# %% [raw]
# Under the hood, the :class:`quantify_scheduler.schedules.schedule.Schedule` is based
# on a dictionary that can be serialized
#

# %%
sched.data

# %% [raw]
# We also need to define the qubits.

# %%
q0, q1 = (
    "q0",
    "q1",
)  # we use strings because qubit resources have not been implemented yet.

# %% [raw]
# Creating the circuit
# ~~~~~~~~~~~~~~~~~~~~
#
# We will now add some operations to the schedule.
# Because this experiment is most conveniently described on the gate level, we use
# operations defined in the :mod:`quantify_scheduler.operations.gate_library` .
#

# %%
from quantify_scheduler.operations.gate_library import CZ, X90, Measure, Reset, Rxy

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
# Visualizing the circuit
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# And we can use this to create a default visualization:
#

# %%
# %matplotlib inline

f, ax = sched.plot_circuit_diagram_mpl()
# all gates are plotted, but it doesn't all fit in a matplotlib figure
ax.set_xlim(-0.5, 9.5)

# %% [raw]
#
# Datastructure internals
# ~~~~~~~~~~~~~~~~~~~~~~~
# Let's take a look at the internals of the
# :class:`quantify_scheduler.schedules.schedule.Schedule`.
#

# %%
sched

# %% [raw]
#
#
# We can see that the number of unique operations is 26 corresponding to 5 operations
# that occur in every loop (the measurement is considered an operation as well) and 21
# unique rotations for the different theta angles.
#

# %%
sched.data.keys()

# %% [raw]
# The schedule consists of a hash table containing all the operations.
# This allows efficient loading of pulses or gates to memory and also enables efficient
# adding of pulse type information as a compilation step.
#

# %%
from itertools import islice

# showing the first 5 elements of the operation dict
dict(islice(sched.data["operation_dict"].items(), 5))

# %% [raw]
# The timing constraints are stored as a list of pulses.
#

# %%
sched.data["timing_constraints"][:6]

# %% [raw]
#
# Similar to the schedule, :class:`quantify_scheduler.operations.operation.Operation`
# objects are also based on dicts.
#

# %%
theta = 314
rxy_theta = Rxy(theta=theta, phi=0, qubit=q0)
rxy_theta.data

# %% [raw]
# Compilation of a circuit diagram into pulses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The compilation from the gate-level to the pulse-level description is done using the
# :ref:`device configuration file<Device configuration file>`.
#
# Here we will use a configuration file for a transmon based system that is part of the
# quantify-scheduler test suite.
#

# %%
import inspect
import json
import os

import quantify_scheduler.schemas.examples as es

esp = inspect.getfile(es)
cfg_f = Path(esp).parent / "transmon_test_config.json"


with open(cfg_f, "r") as f:
    transmon_test_config = json.load(f)

transmon_test_config

# %%
from quantify_scheduler.compilation import (
    add_pulse_information_transmon,
    determine_absolute_timing,
)

add_pulse_information_transmon(sched, device_cfg=transmon_test_config)
determine_absolute_timing(schedule=sched)

# %%
from quantify_scheduler.visualization.pulse_scheme import pulse_diagram_plotly

pulse_diagram_plotly(
    sched,
    port_list=["q0:mw", "q0:res", "q0:fl", "q1:mw"],
    modulation_if=10e6,
    sampling_rate=1e9,
)

# %% [raw]
#
# Compilation of pulses onto physical hardware
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# %%
sched = Schedule("Bell experiment")
for acq_idx, theta in enumerate(np.linspace(0, 360, 21)):
    sched.add(Reset(q0, q1))
    sched.add(X90(q0))
    sched.add(X90(q1), ref_pt="start")  # this ensures pulses are aligned
    # sched.add(CZ(q0, q1)) # FIXME Commented out because of not implemented error
    sched.add(Rxy(theta=theta, phi=0, qubit=q0))

    sched.add(Measure(q0, acq_index=acq_idx), label=f"M q0 {acq_idx} {theta:.2f} deg")
    sched.add(
        Measure(q1, acq_index=acq_idx),
        label=f"M q1 {acq_idx} {theta:.2f} deg",
        ref_pt="start",
    )


add_pulse_information_transmon(sched, device_cfg=transmon_test_config)
determine_absolute_timing(schedule=sched)

# %% [raw]
# The compilation from the pulse-level description for execution on physical hardware is
# done using a backend and based on the
# :ref:`hardware configuration file <sec-hardware-config>`.
#
# Here we will use the
# :class:`~quantify_scheduler.backends.qblox_backend.hardware_compile` made for the
# Qblox pulsar series hardware.
#

# %%
cfg_f = Path(esp).parent / "qblox_test_mapping.json"

with open(cfg_f, "r") as f:
    qblox_test_mapping = json.load(f)

qblox_test_mapping

# %% [raw]
# The Pulsar QCM provides a QCoDeS based Python API. As well as interfacing with real
# hardware, it provides a mock driver we can use for testing and development, which we
# will also use for demonstration purposes as part of this tutorial:
#

# %%
from pulsar_qcm.pulsar_qcm import pulsar_qcm_dummy
from pulsar_qrm.pulsar_qrm import pulsar_qrm_dummy

qcm0 = pulsar_qcm_dummy("qcm0")
qrm0 = pulsar_qrm_dummy("qrm0")

# %%
from pulsar_qcm.pulsar_qcm import pulsar_qcm
from qcodes import Instrument

from quantify_scheduler.backends.qblox_backend import hardware_compile

config = hardware_compile(sched, qblox_test_mapping)["compiled_instructions"]

# %% [raw]
# The compiled schedule can be uploaded to the hardware using the following commands.

# %%
seq_fn = config["qrm0"]["seq0"]["seq_fn"]
qrm0.sequencer0_waveforms_and_program(seq_fn)

# %% [raw]
# At this point, the assembler on the device will load the waveforms into memory and
# verify the program can be executed. We must next arm and then start the device:
#

# %%
qcm0.arm_sequencer()
qrm0.arm_sequencer()

qcm0.start_sequencer()
qrm0.start_sequencer()

# %% [raw]
#
#
#
# Precise timing control: The Ramsey experiment
# ---------------------------------------------
#
# .. todo::
#
#     This tutorial should showcase in detail the timing options possible in the
#     schedule.
#
#
#
# A hybrid experiment: The Chevron
# --------------------------------
#
# As well as defining our schedules in terms of Gates, we can also interleave arbitrary
# Pulse shapes, or even define a schedule entirely with Pulses. This can be useful for
# experiments involving pulse sequences not easily represented by Gates, such as the
# Chevron experiment. In this experiment, we want to vary the length and amplitude of a
# square pulse between X gates on a pair of qubits.
#

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
#
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
#

# %%
from quantify_scheduler.compilation import qcompile

cfg = qcompile(sched, transmon_test_config, qblox_test_mapping)

# %%
