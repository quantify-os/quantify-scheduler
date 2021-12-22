# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [raw]
#  .. _sec-tutorial3:
#
# Tutorial 3. Compilation and hardware execution
# ==============================================
#
#  .. jupyter-kernel::
#    :id: Tutorial 3. Compilation and hardware execution
#
# .. seealso::
#
#     The complete source code of this tutorial can be found in
#
#     :jupyter-download:notebook:`Tutorial 3. Compilation and hardware execution`
#
#     :jupyter-download:script:`Tutorial 3. Compilation and hardware execution`

# %% [raw]
# Schedule definition
# -------------------
#
# Compilation allows converting the schedules introduced in Tutorial 1 into a set of instructions that the control hardware needs to execute.
#
# In this notebook we will define an example schedule, demonstrate how to compile it and run it with our hardware setup.

# %% [raw]
# We start by defining an example schedule.

# %%
from quantify_scheduler import Schedule
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource


sched = Schedule("Simple schedule")
square_pulse = sched.add(
    SquarePulse(amp=1, duration=1e-6, port="q0:res", clock="q0.ro")
)

readout_clock = ClockResource(name="q0.ro", freq=7e9)
sched.add_resource(readout_clock)

sched

# %% [raw]
# Hardware configuration
# ----------------------
#
# To compile the schedule, we will need to provide the compiler with a dictionary detailing the hardware configuration.
#
# Please check the documentation on how to properly create such a configuration for the supported backends:
# - :ref:`sec-backend-qblox`
# - :ref:`sec-backend-zhinst`
#
# Below we create an example hardware configuration dictionary, for the Qblox backend.

# %%
hw_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "ic_qcm": {
        "instrument_type": "Pulsar_QCM",
        "ref": "internal",
        "complex_output_0": {
            "line_gain_db": 0,
            "lo_name": "lo0",
            "dc_mixer_offset_I": 0.1234,
            "dc_mixer_offset_Q": 0.0546,
            "seq0": {
                "mixer_amp_ratio": 0.9998,
                "mixer_phase_error_deg": -4.1,
                "port": "q0:res",
                "clock": "q0.ro",
                "interm_freq": 50e6,
            },
        },
    },
    "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 17},
}

# %% [raw]
# Compilation
# -----------
#
# In our example setup, we will use a Qblox Pulsar QCM baseband module and an external Local Oscillator (LO).
#
# In the presented configuration, `interm_freq` (which stands for Intermediate Frequency or IF) the frequency with which the device modulates the pulses.
# Since the Pulsar QCM baseband module is not capable of outputting signals at the qubit frequency, a Local Oscillator is used in order to upconvert the signals to the desired frequency.
# In this case, the LO frequency is not specified but gets automatically calculated by the backend, such that the relation :math:`clock = LO + IF` is respected.
#
# Now we are ready to proceed to the compilation stage. The compilation procedure is constituted by two main steps:
#
# 1. Device compilation
#    - The device compilation step transforms the provided schedule (consisting of pulses and/or gates) into a schedule executable by the quantum device (/quantum chip). This involves, for example, converting the schedule's operations into operations that the quantum chip can perform, as well as timing the operations according to the chip's characteristics.
# 2. Hardware compilation
#    - This step generates:
#      1. A set of parameters for each of the control stack's instruments in order to configure them properly for the execution of the schedule at hand. These parameters typically don't change during the whole execution of the schedule.
#      2. A compiled program for each instrument (compilation target) containing instructions that dictates what the quantum device must do in order for the schedule to be executed.
#
# We can perform each of these steps independently (via :func:`~quantify_scheduler.compilation.device_compile` and :func:`~quantify_scheduler.compilation.hardware_compile` functions), or perform them directly together via :func:`~quantify_scheduler.compilation.qcompile`.
#
# Since the :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` concept will be introduced in a subsequent tutorial, for now we will pass `None` to the `device_config` argument of `qcompile`. This procedure works, since only pulses (and no gates) have been defined in the schedule.
#
# We start by setting the directory where the compilation output files will be stored, via `set_datadir <https://quantify-quantify-core.readthedocs-hosted.com/en/develop/usage.py.html#data-directory)>`.
#

# %%
from pathlib import Path
from quantify_core.data.handling import set_datadir

set_datadir(Path.home() / "quantify-data")

# %%
from quantify_scheduler.compilation import qcompile

compilation_output = qcompile(sched, device_cfg=None, hardware_cfg=hw_config)

# %% [raw]
# We can now inspect the output of the compilation procedure.

# %%
compilation_output.data["compiled_instructions"]

# %% [raw]
# Execution on the hardware
# -------------------------
#
# In the compilation output, we have all the information necessary to execute the schedule.
#
# In this specific case, only `seq0` of the QCM is necessary. The compilation output contains the filepath where the sequencer's program is stored, as well as the Qcodes parameters that need to be set in the device.
#
# Now that we have the output of the compilation, we are almost ready to execute it with our control setup.
#
# To achieve this, two steps are necessary:
# 1. Connect to our control instruments by instantiating the appropriate classes;
# 2. Instantiate a :class:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator` object and attach to it the appropriate subclass of :class:`~quantify_scheduler.instrument_coordinator.components.base.InstrumentCoordinatorComponentBase`.
#
# The `InstrumentCoordinator` object is responsible for the smooth and in-concert operation of the different instruments, so that the provided schedule is correctly executed. Essentially, it "coordinates" the control stack instruments, giving the relevant commands to the different instruments of the control stack at each point in time.
#
# We start by connecting to the control instruments below.
#
# *Note*: Currently, every instrument aside from Local Oscillators needs to be referred in the hardware configuration with an additional `ic_` prefix!
# This will be addressed in a future merge request.

# %%
from pulsar_qcm.pulsar_qcm import pulsar_qcm_dummy
from quantify_scheduler.instrument_coordinator.components.generic import (
    MockLocalOscillator,
)

qcm = pulsar_qcm_dummy("qcm")
lo0 = MockLocalOscillator("lo0")

# %% [raw]
# And we attach these instruments to the `InstrumentCoordinator` via the appropriate `InstrumentCoordinatorComponent` wrapper class.
#
# In the case of the Local Oscillator, it interfaces with the `InstrumentCoordinator` via the GenericInstrumentCoordinatorComponent

# %%
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import (
    PulsarQCMComponent,
)
from quantify_scheduler.instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)

ic = InstrumentCoordinator("ic")
ic.add_component(PulsarQCMComponent(qcm))
ic.add_component(GenericInstrumentCoordinatorComponent())

# %% [raw]
# We prepare the instruments with the appropriate settings and upload the schedule program by calling the :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.prepare` and passing the compilation output as argument.
#
# Finally, the experiment can be executed by calling :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.start`.
#
# The :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.wait_done` method is useful to wait for the experiment to finish and assure the synchronicity of the python script.
#
#

# %%
# Set the qcodes parameters and upload the schedule program
ic.prepare(compilation_output)

# Start the experiment
ic.start()

# Wait for the experiment to finish or for a timeout
ic.wait_done(timeout_sec=10)
