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
# # Random gates

# %% [markdown]
# ## Setup configurations

# %%

import random

import utils
from qcodes.instrument import Instrument

from quantify_core.data import handling as dh
from quantify_scheduler import QuantumDevice, Schedule
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.operations.gate_library import X90, Measure, Reset, X, Y

# %%
dh.set_datadir(dh.default_datadir(verbose=False))

# %%
# ## Setup Instruments
ip, hardware_cfg, device_path = utils.set_up_config()
quantum_device = QuantumDevice.from_json_file(str(device_path))
quantum_device.hardware_config(hardware_cfg)
qubit = quantum_device.get_element("q0")
_, _, cluster = utils.initialize_hardware(quantum_device, ip=ip)
cluster.reset()
# %%
qubit1 = quantum_device.get_element("q0")
qubit1.clock_freqs.readout(7.4e9)
qubit1.clock_freqs.f01(8.1e9)
qubit1.clock_freqs.f12(8.1e9)
qubit1.rxy.amp180(0.1)
qubit1.rxy.motzoi(0.1)
qubit1.measure.pulse_duration(1e-07)
qubit1.measure.acq_delay(1.2e-07)
qubit1.measure.integration_time(2e-08)

qubit1 = quantum_device.get_element("q1")
qubit1.clock_freqs.readout(10e9)

quantum_device.cfg_sched_repetitions(1)

# %% [markdown]
# ## Setup schedule, dummy data

# %%
signal_delay = 148e-9
num_acquisitions = 1000


# %%
def create_schedule(
    repetitions: int = 1,
) -> Schedule:
    OPERATIONS = (X, X90, Y)

    sched = Schedule("Complex Sequencing")
    sched.add(Reset("q0"))
    for _ in range(4000):
        gate = random.choice(OPERATIONS)  # nosec
        sched.add(gate("q0"))

    sched.add(Measure("q0"))
    sched.repetitions = repetitions

    return sched


# %%
simple_binned_acquisition_kwargs = {}


# %% [markdown]
# ## Run schedule and profiling


def schedule_gettable():
    return ScheduleGettable(
        quantum_device,
        create_schedule,
        simple_binned_acquisition_kwargs,
        batched=True,
        real_imag=False,
        data_labels=["q0 abs", "q0 phase", "q1 abs", "q1 phase"],
        num_channels=2,
    )


# %%
def run_experiment():
    my_gettable = schedule_gettable()
    acq = my_gettable.get()
    return acq


# %%
def schedule_duration():
    my_gettable = schedule_gettable()
    my_gettable.initialize()
    return my_gettable.compiled_schedule.duration


# %%
def close_experiment():
    Instrument.close_all()
