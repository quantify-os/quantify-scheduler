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
# # Simple binned acquisition

# %% [markdown]
# ## Setup configurations

# %%
import numpy as np
import utils
from qblox_instruments import DummyBinnedAcquisitionData
from qcodes.instrument import Instrument
from qcodes.instrument.parameter import ManualParameter

from quantify_core.data import handling as dh
from quantify_scheduler import QuantumDevice, Schedule
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.pulse_library import IdlePulse, SquarePulse

# %%
dh.set_datadir(dh.default_datadir(verbose=False))

# %%
# ## Setup Instruments
ip, hardware_cfg, device_path = utils.set_up_config()
quantum_device = QuantumDevice.from_json_file(str(device_path))
quantum_device.hardware_config(hardware_cfg)
qubit = quantum_device.get_element("q0")
_, _, cluster = utils.initialize_hardware(quantum_device, ip=ip)
# %% [markdown]
# ## Setup schedule, dummy data

# %%
quantum_device.cfg_sched_repetitions(1)
signal_delay = 148e-9
num_acquisitions = 1000


# %%
def set_dummy_data(repetitions=1):
    dummy_slot_idx = 4
    cluster.delete_dummy_binned_acquisition_data(slot_idx=dummy_slot_idx)
    dummy_data_0 = [
        DummyBinnedAcquisitionData(data=(4 * i, 0), thres=0, avg_cnt=0)
        for i in range(num_acquisitions)
    ] * repetitions
    cluster.set_dummy_binned_acquisition_data(
        slot_idx=dummy_slot_idx, sequencer=0, acq_index_name="0", data=dummy_data_0
    )
    dummy_data_1 = [
        DummyBinnedAcquisitionData(data=(0, 8 * i), thres=0, avg_cnt=0)
        for i in range(num_acquisitions)
    ] * repetitions
    cluster.set_dummy_binned_acquisition_data(
        slot_idx=dummy_slot_idx, sequencer=1, acq_index_name="0", data=dummy_data_1
    )


# %%
def add_increasing_pulse_level_acquisitions(
    schedule,
    max_pulse_level,
    port,
    clock,
    acq_channel,
    bin_mode,
    num_acquisitions,
):
    pulse_duration = 120e-9
    pulse_levels = (
        [max_pulse_level]
        if num_acquisitions == 1
        else np.linspace(0, max_pulse_level, num_acquisitions)
    )
    for pulse_level in pulse_levels:
        schedule.add(
            SquarePulse(
                duration=pulse_duration,
                amp=pulse_level,
                port=port,
                clock=clock,
            ),
            rel_time=1e-6,
        )
        schedule.add(
            SSBIntegrationComplex(
                t0=-pulse_duration + signal_delay,
                duration=pulse_duration,
                port=port,
                clock=clock,
                acq_channel=acq_channel,
                bin_mode=bin_mode,
            )
        )


def create_schedule(
    repetitions: int,
    pulse_level_0: complex,
    pulse_level_1: complex,
    bin_mode=BinMode.AVERAGE,
    num_acquisitions=5,
) -> Schedule:
    schedule = Schedule("Test schedule", repetitions=repetitions)

    schedule.add(IdlePulse(duration=1e-6))

    add_increasing_pulse_level_acquisitions(
        schedule,
        pulse_level_0,
        "q0:res",
        "q0.ro",
        0,
        bin_mode,
        num_acquisitions,
    )

    add_increasing_pulse_level_acquisitions(
        schedule,
        pulse_level_1,
        "q1:res",
        "q1.ro",
        1,
        bin_mode,
        num_acquisitions,
    )

    return schedule


# %%
if not ip:
    set_dummy_data()
# %%
pulse_levels = []
for i in [0, 1]:
    pulse_level = ManualParameter(
        name=f"pulse_level_q{i}", initial_value=0.0, unit="-", label="pulse level"
    )
    pulse_level(0.125 * (i + 1))
    pulse_level.batched = False  # type: ignore
    pulse_levels.append(pulse_level)

# %%
simple_binned_acquisition_kwargs = {
    "pulse_level_0": pulse_levels[0],
    "pulse_level_1": pulse_levels[1],
    "num_acquisitions": num_acquisitions,
}

# %%
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
