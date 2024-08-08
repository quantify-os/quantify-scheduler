# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import quantify_scheduler

print(quantify_scheduler.__version__)
print(quantify_scheduler.__file__)

# %%
from quantify_scheduler import Schedule
from quantify_scheduler.operations.pulse_library import (
    SquarePulse,
    IdlePulse,
    SetClockFrequency,
)
from quantify_scheduler.operations.gate_library import Reset, X, Measure
from quantify_scheduler.operations import control_flow_library
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.resources import ClockResource


def create_subschedule(acquisition_index: int):
    subschedule = Schedule(f"Subschedule {acquisition_index}", repetitions=1)

    subschedule.add(Reset("q0"))
    subschedule.add(X("q0"))
    subschedule.add(Measure("q0", acq_index=acquisition_index))

    return subschedule


def create_schedule(repetitions: int = 1):
    sched = Schedule("Schedule", repetitions=1)

    for i in range(100):
        sched.add(Reset("q0"))
        subschedule = create_subschedule(acquisition_index=i)
        sched.add(
            control_flow_library.LoopOperation(
                subschedule,
                repetitions=1000,
            ),
            rel_time=1e-6,
        )

    return sched


# %%
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from contextlib import suppress

quantum_device = QuantumDevice("DUT")
quantum_device.add_element(q0 := BasicTransmonElement("q0"))

hardware_comp_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "10": {"instrument_type": "QCM_RF"},
                "12": {"instrument_type": "QRM_RF"},
            },
        },
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q0:res-q0.ro": {"interm_freq": 50e6},
            "q0:mw-q0.01": {"interm_freq": 50e6},
        },
    },
    "connectivity": {
        "graph": [
            ("cluster0.module12.complex_output_0", "q0:res"),
            ("cluster0.module10.complex_output_0", "q0:mw"),
        ]
    },
}

quantum_device.hardware_config(hardware_comp_cfg)

# %%
qubit = quantum_device.get_element("q0")
qubit.rxy.amp180(0.5)
qubit.rxy.duration(20e-9)
qubit.measure.pulse_duration(2e-6)
qubit.clock_freqs.f01(4e9)
qubit.clock_freqs.readout(6e9)
qubit.reset.duration(200e-6)

# %%
from qblox_instruments import Cluster, ClusterType
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator

cluster = Cluster(
    "cluster0",
    "<ip>",
    debug=1,
    dummy_cfg={10: ClusterType.CLUSTER_QCM_RF, 12: ClusterType.CLUSTER_QRM_RF},
)

ic_cluster0 = ClusterComponent(cluster)
my_instr_coord = InstrumentCoordinator("my_instr_coord")
my_instr_coord.add_component(ic_cluster0)

quantum_device.instr_instrument_coordinator("my_instr_coord")

# %%
from quantify_scheduler.gettables import ScheduleGettable


def run_experiment():
    my_gettable = ScheduleGettable(
        quantum_device,
        create_schedule,
        {},
        batched=True,
        real_imag=False,
    )
    acq = my_gettable.get()
    return acq
