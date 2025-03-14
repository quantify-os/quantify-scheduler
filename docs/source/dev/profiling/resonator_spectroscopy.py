# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Resonator spectroscopy
#
# https://docs.qblox.com/en/main/applications/quantify/qubit_spec.html

# %% [markdown]
# ## Setup
# In this section we configure the hardware configuration which specifies the connectivity of our system.

# %% [markdown]
# ### Configuration file
#
# This is a template hardware configuration file for a 1-qubit system with a flux-control line which can be used to tune the qubit frequency.
#
# The hardware setup is as follows, by cluster slot:
# 6. **QCM-RF**
#     - Drive line for `qubit` using fixed 80 MHz IF.
# 2. **QCM**
#     - Flux line for `qubit`.
# 8. **QRM-RF**
#     - Readout line for `qubit` using a fixed LO set at 7.5 GHz.
# `sequence_to_file` is a boolean flag which dumps waveforms and program dict to JSON file if `True`.
# `"ref": "internal"` specifies the use of the shared clock reference of the cluster
#

# Note that in the hardware configuration below the mixers are uncorrected, but for high fidelity experiments this should also be done for all the modules.

# %%
from pathlib import Path

import numpy as np
import utils
from qcodes.instrument import Instrument
from qcodes.parameters import ManualParameter

import quantify_core.data.handling as dh
from quantify_scheduler import QuantumDevice
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.schedules import heterodyne_spec_sched_nco

# %%
# Setup Instruments
ip, hardware_cfg, device_path = utils.set_up_config()
quantum_device = QuantumDevice.from_json_file(str(device_path))
quantum_device.hardware_config(hardware_cfg)
qubit = quantum_device.get_element("q0")
meas_ctrl, _, cluster = utils.initialize_hardware(quantum_device, ip=ip)

cluster.reset()
# %% [markdown]
# Note that a dummy cluster will raise error flags, this is expected behavior and can be ignored.

# %%
# Importing qubit from the seriliazed version in the quantum device
qubit = quantum_device.get_element("q0")
# %%
# Enter your own dataset directory here!
dh.set_datadir(Path("example_data").resolve())

# %% [markdown]
# ### Configure external flux control
# In the case of flux-tunable transmon qubits, we need to have some way of controlling the external flux.
#
# This can be done by setting an output bias on a module of the cluster which is then connected to the flux line.
#
# ```python
#     # e.g. nullify external flux by setting current to 0 A
#     cluster.module2.out0_current(0.0)
# ```
#
# If your system is not using flux-tunable transmons, then you can skip to the next section.

# %% [markdown]
# ### Activate NCO delay compensation
# Compensate for the digital propagation delay for each qubit (i.e each sequencer)
#
#
# To avoid mismatches between modulation and demodulation, the delay between any readout frequency or phase changes and the next acquisition should be equal or greater than the total propagation delay (146ns + user defined value).

# %%

rf_mod = getattr(
    cluster,
    [
        devices[0]
        for devices in hardware_cfg["connectivity"]["graph"]
        if devices[1] == qubit.ports.readout()
    ][0].split(".")[1],
)

for i in range(6):
    getattr(rf_mod, f"sequencer{i}").nco_prop_delay_comp_en(True)
    getattr(rf_mod, f"sequencer{i}").nco_prop_delay_comp(50)


# %% [markdown]
# ## Characterization experiments
# The sweep setpoints for all experiments (e.g. `frequency_setpoints` in the spectroscopy experiments) are only examples. The sweep setpoints should be changed to match your own system.
#
# We show two sets of experiments: The first contains generic characterization experiments for transmon qubits. The second contains 2D experiments for finding the flux sweetspot, applicable for flux-tunable qubits.
#
#
# Here we consider five standard characterization experiments for single qubit tuneup. The experiments are:
# 1. Resonator spectroscopy
#     - Used to find the frequency response of the readout resonator when the qubit is in $|0\rangle$.
# %% [markdown]
# ## Resonator spectroscopy


# %%
def create_schedule(*args, **kwargs):
    return heterodyne_spec_sched_nco(*args, **kwargs)


# %%
freq = ManualParameter(name="freq", unit="Hz", label="Frequency")
freq.batched = True  # type: ignore
freq.batch_size = 100  # type: ignore

spec_sched_kwargs = dict(
    pulse_amp=1 / 6,
    pulse_duration=2e-6,
    frequencies=freq,
    acquisition_delay=196e-9,
    integration_time=2e-6,
    init_duration=10e-6,
    port=qubit.ports.readout(),
    clock=qubit.name + ".ro",
)
gettable = ScheduleGettable(
    quantum_device,
    schedule_function=create_schedule,
    schedule_kwargs=spec_sched_kwargs,
    real_imag=False,
    batched=True,
)

meas_ctrl.gettables(gettable)
meas_ctrl.verbose(False)
# show_args(spec_sched_kwargs, title="spec_sched_kwargs")

# %%
# set_readout_attenuation(quantum_device, qubit, out_att=50, in_att=0)

quantum_device.cfg_sched_repetitions(400)

center = 7.7e9
frequency_setpoints = np.linspace(center - 20e6, center + 20e6, 200)
meas_ctrl.settables(freq)
meas_ctrl.setpoints(frequency_setpoints)


def run_experiment():
    meas_ctrl.run(
        "resonator spectroscopy",
    )


# %%
def schedule_duration():
    gettable.initialize()
    return gettable.compiled_schedule.duration


# %%
def close_experiment():
    Instrument.close_all()
