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
# =================================
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

# %% [raw]
# Gates, measurements and qubits
# ------------------------------
#
# In the previous tutorials, experiments were created on the quantum-device level. On this level,
# operations are defined in terms of explicit signals and locations on chip, rather than the qubit and the intended operation.
# To allow working at a greater level of abstraction, `quantify_scheduler` allows creating operations on the
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>`.
# Instead of signals, clocks, and ports, operations are defined by the the effect they have on specific qubits.
#
# Many of the gates used in the circuit layer description are defined in
# :class:`quantify_scheduler.operations.gate_library` such as `Reset`, `X90` and
# `Measure`.
# Operations are instantiated by providing them with the name of the qubit(s) on which
# they operate:

# %%
from quantify_scheduler.operations.gate_library import CZ, X, X90, Measure, Reset, Rxy

q0, q1 = ("q0", "q1")
X90(q0)
Measure(q1)
CZ(q0, q1)
Reset(q0)

# %% [raw]
# Let's investigate the different components present in the circuit level description of
# the operation. As an example, we create a 45 degree rotation operation over the
# x-axis.

# %%
from pprint import pprint

rxy45 = Rxy(theta=45.0, phi=0.0, qubit=q0)
pprint(rxy45.data)

# %% [raw]
# As we can see, the structure of a circuit level operation is similar to a pulse level
# operation. However, the information is contained inside the `gate_info` entry rather
# than the `pulse_info` entry of the data dictionary.
# Importantly, there is no device-specific information coupled to the operation such that
# it represents the abstract notion of this qubit rotation, rather than how to perform it
# on any physical qubit implementation.
#
# The entries present above are documented in the `operation` schema.
# Generally, these schemas are only important when defining custom operations, which is
# not part of this tutorial. This schema can be inspected via:

# %%
import importlib.resources
from quantify_scheduler import schemas
import json

operation_schema = json.loads(importlib.resources.read_text(schemas, "operation.json"))
pprint(operation_schema["properties"]["gate_info"]["properties"])

# %% [raw]
# Schedule creation from the circuit layer
# ----------------------------------------
#
# The circuit level operations can be used to create a `schedule` within
# `quantify_scheduler` using the same method as for the pulse level operations.
# This enables creating schedules on a more abstract level.
# We exemplify this extra layer of abstraction by creating a `schedule` for measuring
# `Bell violations`.
#
# .. note:: Within a single `schedule`, high-level circuit layer operations can be mixed with pulse level operations. This mixed representation is useful for experiments where some pulses cannot easily be represented as qubit gates. An example of this is given by the `Chevron` experiment given in :ref:`Mixing pulse and circuit layer operations <Mixing pulse and circuit layer operations>`.

# %% [raw]
# As the first example, we want to create a schedule for performing the
# `Bell experiment <https://en.wikipedia.org/wiki/Bell%27s_theorem>`_.
# The goal of the Bell experiment is to create a Bell state
# :math:`|\Phi ^+\rangle=\frac{1}{2}(|00\rangle+|11\rangle)` followed by a measurement.
# By rotating the measurement basis, or equivalently one of the qubits, it is possible
# to observe violations of the CSHS inequality.
#
# We create this experiment using the
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

sched

# %% [raw]
# By scheduling 7 operations for 21 different values for `theta` we indeed get a schedule containing 7\*21=147 operations. To minimize the size of the schedule, identical operations are stored only once. For example, all 21 `CZ` operations are stored only once, which leaves only 66 unique operations in the schedule.
# .. note :: The acquisitions are different for every iteration due to their different `acq_index`. The `Rxy` rotates over a different angle every iteration and must therefore also be different for every iteration (except for the last since $R^{360}=R^0$). Hence the number of unique operations is 3\*21-1+4=66.

# %% [raw]
# Visualizing the quantum circuit
# -------------------------------
#
# We can directly visualize the created schedule on the
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>`.
# This visualization shows every operation on a line representing the different qubits.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

_, ax = sched.plot_circuit_diagram()
# all gates are plotted, but it doesn't all fit in a matplotlib figure.
# Therefore we use `set_xlim` to limit the number of gates shown.
ax.set_xlim(-0.5, 9.5)
plt.show()

# %% [raw]
# In previous tutorials, we visualized the `schedules` on the pulse level using :meth:`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram` .
# Up until now, however, all gates have been defined on the
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>` without defining the
# corresponding pulse shapes.
# Therefore, trying to run :meth:`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram` will raise an error which
# signifies no `pulse_info` is present in the schedule:
#
# .. jupyter-execute::
#     :raises:
#
#     sched.plot_pulse_diagram()

# %% [raw]
# And similarly for the `timing_table`:
#
# .. jupyter-execute::
#     :raises:
#
#     sched.timing_table

# %% [raw]
# Device configuration and compilation
# ------------------------------------
#
# The aim of this section is to add device specific information to the schedule.
# Before adding this, the schedule is not specific to any qubit implementation.
# In this section, we compile the schedule to the quantum-device layer to enable running it on a specific qubit implementation.
# In order to generate the necessary information, some device properties must be specified which
# is done using the :ref:`device configuration file<sec-device-config>`.
#
# Using the device configuration file, the schedule can be compiled to the quantum-device layer, appending pulse
# information to every gate in the schedule. Before continuing to the compilation step,
# however, we will first unpack the configuration file.
#
# Here we will use a configuration file for a transmon based system that is used in the
# quantify-scheduler test suite.

# %%
import inspect
from pathlib import Path

import quantify_scheduler.schemas.examples as es

esp = inspect.getfile(es)
cfg_f = Path(esp).parent / "transmon_test_config.json"


with open(cfg_f, "r") as f:
    transmon_test_config = json.load(f)

pprint(list(transmon_test_config.keys()))

# %% [raw]
# Before explaning how this can be used to compile schedules, let us first investigate
# the contents of the configuration file.

# %%
transmon_test_config["backend"]

# %% [raw]
# The backend of the configuration file specifies what function will be used to add
# pulse information to the gates. In other words, it specifies how to interpret the
# qubit parameters present in the configuration file and achieve the required gates.
#
# Let us briefly investigate this function:

# %%
from quantify_core.utilities.general import import_python_object_from_string

device_compilation_backend = import_python_object_from_string(
    transmon_test_config["backend"]
)
help(device_compilation_backend)

# %% [raw]
# A more detailed description of the configuration file can be obtained from the
# specified JSON schema:

# %%
transmon_schema = json.loads(
    importlib.resources.read_text(schemas, "transmon_cfg.json")
)
pprint(transmon_schema["properties"])

# %% [raw]
# As can be seen form the JSON schema, the
# :ref:`device configuration file<sec-device-config>` also contains the
# parameters required by the `device_compilation_backend` for all qubits and edges.

# %%
pprint(list(transmon_test_config["qubits"].keys()))

# %%
# For every qubit we can investigate the contained parameters
pprint(transmon_test_config["qubits"]["q0"])

# %% [raw]
# Now that we went through the different components of the configuration file, let's use
# it to compile our previously defined schedule.
# The `device_compile` function takes care of this task and adds pulse information based
# on the configuration file, as discussed above.
# It also determines the timing of the different pulses in the schedule.

# %%
from quantify_scheduler.compilation import device_compile

pulse_sched = device_compile(sched, transmon_test_config)

# %% [raw]
# Now that the timings have been determined, we can show the first few rows of the `timing_table`:

# %%
pulse_sched.timing_table.hide_index(slice(11, None)).hide_columns("waveform_op_id")

# %% [raw]
# And since all pulse information has been determined, we can show the pulse diagram as
# well:

# %%
f, ax = pulse_sched.plot_pulse_diagram()
ax.set_xlim(0.4005e-3, 0.4006e-3)

# %% [raw]
# Quantum Devices and Elements
# ----------------------------
# The :ref:`device configuration file<sec-device-config>` contains all knowledge
# of the physical device under test (DUT). Together with the
# :ref:`hardware configuration file<Hardware configuration file>` which contains the
# different hardware connections, all information is represented.
# To generate these configuration files on the fly, `quantify_scheduler` provides the
# :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` and
# :code:`DeviceElement` classes.
# These classes contain the information necessary to generate the config files and allow
# changing their parameters on-the-fly.
# The :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` class
# represents the DUT containing different :code:`DeviceElement` s.
# Currently, `quantify_scheduler` contains the
# :class:`~quantify_scheduler.device_under_test.transmon_element.TransmonElement` class
# to represent a qubit connected to a feedline. We show their interaction below:

# %%
from quantify_scheduler.device_under_test.transmon_element import TransmonElement
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice

# first create a device under test:
dut = QuantumDevice("DUT")

# then create a transmon element
transmon = TransmonElement("transmon1")

# Finally, add the transmon element to the QuantumDevice:
dut.add_component(transmon)
dut, dut.components()

# %% [raw]
# The different transmon properties can be set through the `TransmonElement`
# e.g.

# %%
transmon.freq_01(6e9)
list(transmon.parameters.keys())

# %% [raw]
# The device configuration is now simply obtained using `dut.generate_device_config()`.
# In order for this command to provide a correct device configuration, the different
# parameters need to be specified in the `TransmonElement` and `QuantumDevice` objects.

# %%
pprint(dut.generate_device_config())

# %% [raw]
# Mixing pulse and circuit layer operations
# -----------------------------------------
#
# As well as defining our schedules in terms of gates, we can also mix the circuit layer
# representation with pulse level operations.
# This can be useful for experiments involving pulses not easily represented by Gates,
# such as the Chevron experiment.
# In this experiment, we want to vary the length and amplitude of a square pulse between
# X gates on a pair of qubits.

# %%
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource

sched = Schedule("Chevron Experiment")
acq_idx = 0

# NB multiples of 4 ns need to be used due to limitations of the pulsars
for duration in np.linspace(20e-9, 60e-9, 6):
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

# %%
fig, ax = sched.plot_circuit_diagram()
ax.set_xlim(-0.5, 9.5)
for t in ax.texts:
    if t.get_position()[0] > 9.5:
        t.set_visible(False)

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
# Rather than first using :func:`~quantify_scheduler.compilation.device_compile` and subsequently
# :func:`~quantify_scheduler.compilation.hardware_compile`, the two function calls can be combined using
# :func:`~quantify_scheduler.compilation.qcompile`.

# %%
from quantify_scheduler.compilation import qcompile

dut.close()
dut = QuantumDevice("DUT")
q0 = TransmonElement("q0")
q1 = TransmonElement("q1")
dut.add_component(q0)
dut.add_component(q1)
dut.get_component("q0").mw_amp180(0.6)
dut.get_component("q1").mw_amp180(0.6)
compiled_sched = qcompile(
    sched, dut.generate_device_config(), dut.generate_hardware_config()
)

# %% [raw]
# So, finally, we can show the timing table associated to the chevron schedule and plot
# its pulse diagram:

# %%
compiled_sched.timing_table.hide_index(slice(11, None)).hide_columns("waveform_op_id")

# %%
f, ax = compiled_sched.plot_pulse_diagram()
ax.set_xlim(200e-6, 200.4e-6)

# %%
