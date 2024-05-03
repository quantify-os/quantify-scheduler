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
from quantify_scheduler import Schedule
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from quantify_scheduler.operations.gate_library import Measure, X, X90, Y, Reset
from qblox_instruments import Cluster, ClusterType
from quantify_core.data import handling as dh
import random

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
transmon0.clock_freqs.f01(10e9)
transmon0.clock_freqs.f12(10e9)
transmon0.rxy.amp180(0.1)
transmon0.rxy.motzoi(0.1)
# transmon0.measure.acq_duration(2e-08)
transmon0.measure.pulse_duration(1e-07)
transmon0.measure.acq_delay(1.2e-07)
transmon0.measure.integration_time(2e-08)

my_device.add_element(transmon0)
transmon1 = BasicTransmonElement(f"q1")
transmon1.clock_freqs.readout(10e9)
my_device.add_element(transmon1)

# %%
my_device.instr_instrument_coordinator("my_instr_coord")

# %%
hw_config = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "modules": {"15": {"instrument_type": "QRM"}},
            "ref": "internal",
        }
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q0:res-q0.ro": {"interm_freq": 0},
            "q1:res-q1.ro": {"interm_freq": 0},
            "q0:mw-q0.01": {"interm_freq": 0},
            "q1:mw-q1.01": {"interm_freq": 0},
        }
    },
    "connectivity": {
        "graph": [
            ["cluster0.module15.complex_output_0", "q0:res"],
            ["cluster0.module15.complex_output_0", "q1:res"],
            ["cluster0.module15.complex_output_0", "q0:mw"],
            ["cluster0.module15.complex_output_0", "q1:mw"],
        ]
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
def create_schedule(
    repetitions: int,
) -> Schedule:
    OPERATIONS = (X, X90, Y)

    sched = Schedule("Complex Sequencing")
    sched.add(Reset("q0"))
    for _ in range(5000):
        gate = random.choice(OPERATIONS)  # nosec
        sched.add(gate("q0"))

    sched.add(Measure("q0"))
    sched.repetitions = 1

    return sched


# %%
simple_binned_acquisition_kwargs = {}


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
