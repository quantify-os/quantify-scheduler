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
from quantify_scheduler import Schedule
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from quantify_scheduler.operations.pulse_library import IdlePulse, SquarePulse
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.enums import BinMode
from qcodes.instrument.parameter import ManualParameter
from qblox_instruments import Cluster, ClusterType
from quantify_core.data import handling as dh

# %%
dh.set_datadir(dh.default_datadir(verbose=False))

# %% [markdown]
# ## Setup cluster

# %%
cluster = Cluster(
    "cluster0",
    "<ip>",
    debug=1,
    dummy_cfg={15: ClusterType.CLUSTER_QRM},
)

ic_cluster0 = ClusterComponent(cluster)
my_instr_coord = InstrumentCoordinator("my_instr_coord")
my_instr_coord.add_component(ic_cluster0)

# %%
print(ic_cluster0.name)

# %%
my_device = QuantumDevice("my_device")

# %%
transmon0 = BasicTransmonElement(f"q0")
transmon0.clock_freqs.readout(10e9)
my_device.add_element(transmon0)
transmon1 = BasicTransmonElement(f"q1")
transmon1.clock_freqs.readout(10e9)
my_device.add_element(transmon1)

# %%
my_device.instr_instrument_coordinator("my_instr_coord")

# %%
hw_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster0": {
        "ref": "internal",
        "instrument_type": "Cluster",
        "cluster0_module15": {
            "instrument_type": "QRM",
            "complex_output_0": {
                "portclock_configs": [
                    {"port": "q0:res", "clock": "q0.ro", "interm_freq": 0},
                    {"port": "q1:res", "clock": "q1.ro", "interm_freq": 0},
                ]
            },
        },
    },
}

# %%
my_device.hardware_config(hw_config)
my_device.cfg_sched_repetitions(1)

# %% [markdown]
# ## Setup schedule, dummy data

# %%
signal_delay = 148e-9
num_acquisitions = 1000

# %%
from qblox_instruments import DummyBinnedAcquisitionData


# %%
def set_dummy_data(repetitions=1):
    dummy_slot_idx = 15
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
        slot_idx=dummy_slot_idx, sequencer=1, acq_index_name="1", data=dummy_data_1
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
    for acq_index, pulse_level in enumerate(pulse_levels):
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
                acq_index=acq_index,
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
set_dummy_data()

# %%
pulse_levels = []
for i in [0, 1]:
    pulse_level = ManualParameter(
        name=f"pulse_level_q{i}", initial_value=0.0, unit="-", label="pulse level"
    )
    pulse_level(0.125 * (i + 1))
    pulse_level.batched = False
    pulse_levels.append(pulse_level)

# %%
simple_binned_acquisition_kwargs = {
    "pulse_level_0": pulse_levels[0],
    "pulse_level_1": pulse_levels[1],
    "num_acquisitions": num_acquisitions,
}


# %% [markdown]
# ## Run schedule and profiling


# %%
def run_experiment():
    my_gettable = ScheduleGettable(
        my_device,
        create_schedule,
        simple_binned_acquisition_kwargs,
        batched=True,
        real_imag=False,
        data_labels=[f"q0 abs", f"q0 phase", f"q1 abs", f"q1 phase"],
        num_channels=2,
    )
    acq = my_gettable.get()
    return acq
