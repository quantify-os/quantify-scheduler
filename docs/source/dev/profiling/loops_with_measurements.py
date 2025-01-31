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
import utils
from qcodes.instrument import Instrument

import quantify_scheduler
from quantify_scheduler import QuantumDevice

print(quantify_scheduler.__version__)
print(quantify_scheduler.__file__)

from quantify_scheduler import Schedule
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.operations import control_flow_library
from quantify_scheduler.operations.gate_library import Measure, Reset, X

# %%
# ## Setup Instruments
ip, hardware_cfg, device_path = utils.set_up_config()
quantum_device = QuantumDevice.from_json_file(str(device_path))
quantum_device.hardware_config(hardware_cfg)
_, _, cluster = utils.initialize_hardware(quantum_device, ip=ip)
cluster.reset()
# %%
qubit = quantum_device.get_element("q0")
qubit.rxy.amp180(0.5)
qubit.rxy.duration(20e-9)
qubit.measure.pulse_duration(2e-6)
qubit.clock_freqs.f01(7.9e9)
qubit.clock_freqs.readout(7.4e9)
qubit.reset.duration(200e-6)
quantum_device.cfg_sched_repetitions(1)


def create_subschedule(acquisition_index: int):
    subschedule = Schedule(f"Subschedule {acquisition_index}", repetitions=1)

    subschedule.add(Reset("q0"))
    subschedule.add(X("q0"))
    subschedule.add(Measure("q0", acq_index=acquisition_index))

    return subschedule


def create_schedule(repetitions: int = 1):
    sched = Schedule("Schedule", repetitions=repetitions)

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


def schedule_gettable():
    return ScheduleGettable(
        quantum_device,
        create_schedule,
        {},
        batched=True,
        real_imag=False,
    )


def run_experiment():
    my_gettable = schedule_gettable()
    acq = my_gettable.get()
    return acq


# %%
def schedule_duration():
    gettable = schedule_gettable()
    gettable.initialize()
    return gettable.compiled_schedule.duration


# %%
def close_experiment():
    Instrument.close_all()
