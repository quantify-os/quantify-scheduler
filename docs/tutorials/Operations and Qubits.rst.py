# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [raw]
# .. _sec-tutorial-ops+qubits:
#
# Tutorial: Operations and Qubits
# ===============================
#
#  .. jupyter-kernel::
#    :id: Tutorial: Operations and Qubits
#
# .. seealso::
#
#     The complete source code of this tutorial can be found in
#
#     :jupyter-download-notebook:`Operations and Qubits`
#
#     :jupyter-download-script:`Operations and Qubits`

# %% [raw]
# Gates, measurements and qubits
# ------------------------------
#
# In the previous tutorials, experiments were created on the :ref:`quantum-device level<sec-user-guide-quantum-device>`. On this level,
# operations are defined in terms of explicit signals and locations on chip, rather than the qubit and the intended operation.
# To allow working at a greater level of abstraction, `quantify_scheduler` allows creating operations on the
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>`.
# Instead of signals, clocks, and ports, operations are defined by the the effect they have on specific qubits. This representation of the schedules can be compiled to the quantum-device level to create the pulse schemes.
#
# In this tutorial we show how to define operations on the :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>`, combine them into schedules, and show their circuit-level visualization.
# We go through the configuration file needed to compile the schedule to the quantum-device level and show how these configuration files can be created automatically and dynamically.
# Finally, we showcase the hybrid nature of `quantify_scheduler`, allowing the scheduling circuit-level and device-level operations side by side in the same schedule.
#
# Many of the gates used in the circuit layer description are defined in
# :class:`~quantify_scheduler.operations.gate_library` such as :class:`~quantify_scheduler.operations.gate_library.Reset`, :class:`~quantify_scheduler.operations.gate_library.X90` and
# :class:`~quantify_scheduler.operations.gate_library.Measure`.
# Operations are instantiated by providing them with the name of the qubit(s) on which
# they operate:

# %%
from quantify_scheduler.operations.gate_library import CZ, Measure, Reset, X90

q0, q1 = ("q0", "q1")
X90(q0)
Measure(q1)
CZ(q0, q1)
Reset(q0)

# %% [raw]
# Let's investigate the different components present in the circuit-level description of
# the operation. As an example, we create a 45 degree rotation operation over the
# x-axis.

# %%
from pprint import pprint
from quantify_scheduler.operations.gate_library import Rxy

rxy45 = Rxy(theta=45.0, phi=0.0, qubit=q0)
pprint(rxy45.data)

# %% [raw]
# As we can see, the structure of a circuit-level operation is similar to a pulse-level
# operation. However, the information is contained inside the :code:`gate_info` entry rather
# than the :code:`pulse_info` entry of the data dictionary.
# Importantly, there is no device-specific information coupled to the operation such that
# it represents the abstract notion of this qubit rotation, rather than how to perform it
# on any physical qubit implementation.
#
# The entries present above are documented in the `operation` schema.
# Generally, these schemas are only important when defining custom operations, which is
# not part of this tutorial. This schema can be inspected via:

# %%
import importlib.resources
import json
from quantify_scheduler import schemas

operation_schema = json.loads(importlib.resources.read_text(schemas, "operation.json"))
pprint(operation_schema["properties"]["gate_info"]["properties"])

# %% [raw]
# Schedule creation from the circuit layer (Bell)
# -----------------------------------------------
#
# The circuit-level operations can be used to create a `schedule` within
# `quantify_scheduler` using the same method as for the pulse-level operations.
# This enables creating schedules on a more abstract level.
# We exemplify this extra layer of abstraction by creating a `schedule` for measuring
# `Bell violations`.
#
# .. note:: Within a single `schedule`, high-level circuit layer operations can be mixed with quantum-device level operations. This mixed representation is useful for experiments where some pulses cannot easily be represented as qubit gates. An example of this is given by the `Chevron` experiment given in :ref:`Mixing pulse and circuit layer operations (Chevron)`.

# %% [raw]
# As the first example, we want to create a schedule for performing the
# `Bell experiment <https://en.wikipedia.org/wiki/Bell%27s_theorem>`_.
# The goal of the Bell experiment is to create a Bell state
# :math:`|\Phi ^+\rangle=\frac{1}{2}(|00\rangle+|11\rangle)` which is a perfectly entangled state, followed by a measurement.
# By rotating the measurement basis, or equivalently one of the qubits, it is possible
# to observe violations of the CSHS inequality.
#
# We create this experiment using the
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>` description.
# This allows defining the Bell schedule as:

# %%
import numpy as np
from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import CZ, Measure, Reset, Rxy, X90

sched = Schedule("Bell experiment")

for acq_idx, theta in enumerate(np.linspace(0, 360, 21)):
    sched.add(Reset(q0, q1))
    sched.add(X90(q0))
    sched.add(X90(q1), ref_pt="start")  # This ensures pulses are aligned
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
# By scheduling 7 operations for 21 different values for :code:`theta` we indeed get a schedule containing 7\*21=147 operations. To minimize the size of the schedule, identical operations are stored only once. For example, the :class:`~quantify_scheduler.operations.gate_library.CZ` operation is stored only once but used 21 times, which leaves only 66 unique operations in the schedule.
#
# .. note :: The acquisitions are different for every iteration due to their different :code:`acq_index`. The :class:`~quantify_scheduler.operations.gate_library.Rxy`-gate rotates over a different angle every iteration and must therefore also be different for every iteration (except for the last since :math:`R^{360}=R^0`). Hence the number of unique operations is 3\*21-1+4=66.

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
# Therefore we use :code:`set_xlim` to limit the number of gates shown.
ax.set_xlim(-0.5, 9.5)
plt.show()

# %% [raw]
# In previous tutorials, we visualized the `schedules` on the pulse level using :meth:`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram` .
# Up until now, however, all gates have been defined on the
# :ref:`quantum-circuit level<sec-user-guide-quantum-circuit>` without defining the
# corresponding pulse shapes.
# Therefore, trying to run :meth:`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram` will raise an error which
# signifies no :code:`pulse_info` is present in the schedule:

# %%
rst_conf = {"indent": "    ", "jupyter_execute_options": [":raises:"]}
sched.plot_pulse_diagram()

# %% [raw]
# And similarly for the :code:`timing_table`:

# %%
rst_conf = {"indent": "    ", "jupyter_execute_options": [":raises:"]}
sched.timing_table

# %% [raw]
# Device configuration and compilation
# ------------------------------------
#
# Up until now the schedule is not specific to any qubit implementation.
# The aim of this section is to add device specific information to the schedule.
# This knowledge is contained in the :ref:`device configuration file<sec-device-config>`, which we introduce in this section.
# By compiling the schedule to the quantum-device layer, we incorporate the device configuration into the schedule (for example by adding pulse information to every gate) and thereby enable it to run on a specific qubit implementation.
#
# To start this section, we will unpack the structure of the configuration file.
# Here we will use a configuration file for a transmon based system that is used in the
# `quantify-scheduler` test suite.

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
# Before explaining how this can be used to compile schedules, let us first investigate
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
# parameters required by the :code:`device_compilation_backend` for all qubits and edges.

# %%
pprint(list(transmon_test_config["qubits"].keys()))

# %% [raw]
# In fact, this configuration file specifies more qubits than we need for our schedule. The unused qubit will be ignored. This allows writing only a single configuration file for separate experiments using a different number of qubits.

# %%
# For every qubit we can investigate the contained parameters
pprint(transmon_test_config["qubits"]["q0"])

# %% [raw]
# Now that we went through the different components of the configuration file, let's use
# it to compile our previously defined schedule.
# The :func:`~quantify_scheduler.compilation.device_compile` function takes care of this task and adds pulse information based
# on the configuration file, as discussed above.
# It also determines the timing of the different pulses in the schedule.

# %%
from quantify_scheduler.compilation import device_compile

pulse_sched = device_compile(sched, transmon_test_config)

# %% [raw]
# Now that the timings have been determined, we can show the first few rows of the :code:`timing_table`:

# %%
pulse_sched.timing_table.hide(slice(11, None), axis="index").hide(
    "waveform_op_id", axis="columns"
)

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
# of the physical device under test (DUT).
# To generate these configuration files on the fly, `quantify_scheduler` provides the
# :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` and
# :class:`~quantify_scheduler.device_under_test.device_element.DeviceElement` classes.
#
# These classes contain the information necessary to generate the config files and allow
# changing their parameters on-the-fly.
# The :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` class
# represents the DUT containing different :class:`~quantify_scheduler.device_under_test.device_element.DeviceElement` s.
# Currently, `quantify_scheduler` contains the
# :class:`~quantify_scheduler.device_under_test.transmon_element.TransmonElement` class
# to represent a transmon qubit connected to a feedline. We show their interaction below:

# %%
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import TransmonElement

# First create a device under test:
dut = QuantumDevice("DUT")

# Then create a transmon element
qubit = TransmonElement("qubit")

# Finally, add the transmon element to the QuantumDevice:
dut.add_component(qubit)
dut, dut.components()

# %% [raw]
# The different transmon properties can be set through attributes of the :class:`~quantify_scheduler.device_under_test.transmon_element.TransmonElement` class instance, e.g.:

# %%
qubit.freq_01(6e9)
list(qubit.parameters.keys())

# %% [raw]
# The device configuration is now simply obtained using :code:`dut.generate_device_config()`.
# In order for this command to provide a correct device configuration, the different
# parameters need to be specified in the :class:`~quantify_scheduler.device_under_test.transmon_element.TransmonElement` and :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` objects.

# %%
pprint(dut.generate_device_config())

# %% [raw]
# Mixing pulse and circuit layer operations (Chevron)
# ---------------------------------------------------
#
# As well as defining our schedules in terms of gates, we can also mix the circuit layer
# representation with pulse-level operations.
# This can be useful for experiments involving pulses not easily represented by Gates,
# such as the Chevron experiment.
# In this experiment, we want to vary the length and amplitude of a square pulse between
# X gates on a pair of qubits.

# %%
from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import Measure, Reset, X, X90
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource

sched = Schedule("Chevron Experiment")
acq_idx = 0

# NB multiples of 4 ns need to be used due to sampling rate of the Qblox modules
for duration in np.linspace(start=20e-9, stop=60e-9, num=6):
    for amp in np.linspace(start=0.1, stop=1.0, num=10):
        begin = sched.add(Reset("q0", "q1"))
        sched.add(X("q0"), ref_op=begin, ref_pt="end")
        # NB we specify a clock for tutorial purposes, Chevron experiments do not necessarily use modulated square pulses
        square = sched.add(SquarePulse(amp, duration, "q0:mw", clock="q0.01"))
        sched.add(X90("q0"), ref_op=square)
        sched.add(X90("q1"), ref_op=square)
        sched.add(Measure(q0, acq_index=acq_idx), label=f"M q0 {acq_idx}")
        sched.add(
            Measure(q1, acq_index=acq_idx), label=f"M q1 {acq_idx}", ref_pt="start"
        )

        acq_idx += 1

sched.add_resources([ClockResource("q0.01", 6.02e9)])  # Manually add the pulse clock

# %%
fig, ax = sched.plot_circuit_diagram()
ax.set_xlim(-0.5, 9.5)
for t in ax.texts:
    if t.get_position()[0] > 9.5:
        t.set_visible(False)

# %% [raw]
# This example shows that we add gates using the same interface as pulses. Gates are Operations, and
# as such support the same timing and reference operators as Pulses.
#
# .. warning::
#
#     When adding a Pulse to a schedule, the clock is not automatically added to the
#     resources of the schedule. It may be necessary to add this clock manually, as in
#     the final line of the example above.
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
compiled_sched = qcompile(sched, dut.generate_device_config())

# %% [raw]
# So, finally, we can show the timing table associated to the chevron schedule and plot
# its pulse diagram:

# %%
compiled_sched.timing_table.hide(slice(11, None), axis="index").hide(
    "waveform_op_id", axis="columns"
)

# %%
f, ax = compiled_sched.plot_pulse_diagram()
ax.set_xlim(200e-6, 200.4e-6)

# %%
