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
# .. _sec-tutorial3:
#
# Tutorial 3. Compilation and hardware execution
# ==============================================
#
# .. jupyter-kernel::
#   :id: Tutorial 3. Compilation and hardware execution
#
# .. seealso::
#
#     The complete source code of this tutorial can be found in
#
#     :jupyter-download:notebook:`Tutorial 3. Compilation and hardware execution`
#
#     :jupyter-download:script:`Tutorial 3. Compilation and hardware execution`

# %% [raw]
# Compilation allows converting the schedules introduced in :ref:`Tutorial 1 <sec-tutorial1>` into a set of instructions that can be executed on the control hardware.
#
# In this notebook we will define an example schedule, demonstrate how to compile it, and run it on a virtual hardware setup.

# %% [raw]
# Schedule definition
# -------------------
#
# We start by defining an example schedule.

# %%
from quantify_scheduler import Schedule
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource


sched = Schedule("Simple schedule")
square_pulse = sched.add(
    SquarePulse(amp=0.2, duration=1e-6, port="q0:res", clock="q0.ro")
)

readout_clock = ClockResource(name="q0.ro", freq=7e9)
sched.add_resource(readout_clock)

sched

# %% [raw]
# Hardware configuration
# ----------------------
#
# In our example setup, we will use a Qblox Cluster containing a QCM-RF module. To compile the schedule, we will need to provide the compiler with a dictionary detailing the hardware configuration.
#
# Please check the documentation on how to properly create such a configuration for the supported backends:
# - :ref:`sec-backend-qblox`
# - :ref:`sec-backend-zhinst`
#
# Below we create an example hardware configuration dictionary, for the Qblox backend.
# In this configuration, we include:

# - The backend that we want to use (the Qblox backend, in this case).
# - A Cluster containing a QCM-RF module (in the 2nd slot).
# - A Local Oscillator.

# In the QCM-RF output's settings, `interm_freq` (which stands for Intermediate Frequency or IF) is the frequency with which the device modulates the pulses.
# In this case, the internal LO frequency is not specified but is automatically calculated by the backend, such that the relation :math:`clock = LO + IF` is respected.

# %%
hw_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster0": {
        "ref": "internal",
        "instrument_type": "Cluster",
        "cluster0_module2": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_freq": None,
                "dc_mixer_offset_I": -0.00552,
                "dc_mixer_offset_Q": -0.00556,
                "seq0": {
                    "mixer_amp_ratio": 0.9998,
                    "mixer_phase_error_deg": -4.1,
                    "port": "q0:res",
                    "clock": "q0.ro",
                    "interm_freq": 50e6,
                },
            },
        },
    },
}

# %% [raw]
# Note that, for any experiment, all the required instruments need to be present in the hardware config.

# %% [raw]
# Compilation
# -----------
#
# Now we are ready to proceed to the compilation stage. This will be done in two steps:
#
# 1. **Determine the schedule's absolute timing**
#    - During the schedule's definition, we didn't assign absolute times to the operations. Instead, only the duration was defined. In order for the instruments to know how to execute the schedule, the absolute timing of the operations has to be calculated.
#
# 2. **Hardware compilation**
#    - This step generates:
#
#   * A set of parameters for each of the control stack's instruments in order to configure them properly for the execution of the schedule at hand. These parameters typically don't change during the whole execution of the schedule.
#   * A compiled program (for the instruments that require it) containing instructions that dictate what the instrument must do in order for the schedule to be executed.
#
# We can perform each of these steps via :func:`~quantify_scheduler.compilation.determine_absolute_timing` and :func:`~quantify_scheduler.compilation.hardware_compile`, respectively.
#
# We start by setting the directory where the compilation output files will be stored, via `set_datadir <https://quantify-quantify-core.readthedocs-hosted.com/en/develop/usage.py.html#data-directory)>`.
#

# %%
from pathlib import Path
from quantify_core.data.handling import set_datadir

set_datadir(Path.home() / "quantify-data")

# %%
from quantify_scheduler.compilation import determine_absolute_timing, hardware_compile

sched = determine_absolute_timing(sched)
compilation_output = hardware_compile(sched, hardware_cfg=hw_config)

# %% [raw]
# The cell above compiles the schedule, returning a :class:`quantify_scheduler.schedules.schedule.CompiledSchedule` object. This class differs from :class:`quantify_scheduler.schedules.schedule.Schedule` in that it is immutable and contains the :attr:`quantify_scheduler.schedules.schedule.CompiledSchedule.compiled_instructions` attribute.  We inspect these instructions below.

# %%
compilation_output.compiled_instructions

# %% [raw]
# Execution on the hardware
# -------------------------
#
# In the compilation output, we have all the information necessary to execute the schedule.
#
# In this specific case, only `seq0` of the QCM-RF is necessary. The compilation output contains the filepath where the sequencer's program is stored, as well as the Qcodes parameters that need to be set in the device.
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

# %%
from qblox_instruments import Cluster, ClusterType

cluster0 = Cluster("cluster0", dummy_cfg={"2": ClusterType.CLUSTER_QCM_RF})

# %% [raw]
# And we attach these instruments to the `InstrumentCoordinator` via the appropriate `InstrumentCoordinatorComponent` wrapper class.
#
# %%
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import (
    ClusterComponent,
)

ic = InstrumentCoordinator("ic")
ic.add_component(ClusterComponent(cluster0))

# %% [raw]
# The experiment can now be conducted using the methods of `InstrumentCoordinator`:
#
# 1. We prepare the instruments with the appropriate settings and upload the schedule program by calling the :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.prepare` method and passing the compilation output as argument.
# 2. We start the hardware execution by calling the :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.start` method.
#
# Additionally, the :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.wait_done` method is useful to wait for the experiment to finish and assure the synchronicity of the python script.
#
#

# %%
# Set the qcodes parameters and upload the schedule program
ic.prepare(compilation_output)

# Start the hardware execution
ic.start()

# Wait for the experiment to finish or for a timeout
ic.wait_done(timeout_sec=10)

# %% [raw]
# The `InstrumentCoordinator` has two more functions which were not covered in this experiment:
#
# - :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.retrieve_acquisition`
#   - In case the schedule contained acquisitions, this method retrieves the acquired data.
# - :meth:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.stop`
#   - Stops all running instruments.
#
# We conclude tutorial with the remark that the schedule used in this tutorial was defined purely in terms of pulses.
# However, quantify-scheduler also supports the usage of quantum gates in schedules. Given that gates may require different pulses when executed in different quantum devices. Consequently, when using gates, one requires an additional compilation step, called "Device Compilation", that converts these gates into pulses that can be interpreted by the backend. This use case will be covered in Tutorial 5.
