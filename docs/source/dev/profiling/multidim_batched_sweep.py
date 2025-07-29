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
# # Multidimensional batched sweep

# %% [markdown]
# # Imports

# %%
import numpy as np
import utils
from qcodes.instrument import Instrument
from qcodes.parameters import ManualParameter

from quantify_core.data import handling as dh
from quantify_scheduler import QuantumDevice, Schedule
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.operations.gate_library import Measure, Reset, Rxy
from quantify_scheduler.operations.pulse_library import SoftSquarePulse

# %%
dh.set_datadir(dh.default_datadir(verbose=False))

# %% [markdown]
# ### Hardware Cfg

# %% [markdown]
# ### Device Cfg

# %%
# Instentiate instruments
ip, hardware_cfg, device_path = utils.set_up_config()
quantum_device = QuantumDevice.from_json_file(str(device_path))
quantum_device.hardware_config(hardware_cfg)
qubit = quantum_device.get_element("q0")
meas_ctrl, _, cluster = utils.initialize_hardware(quantum_device, ip=ip)

cluster.reset()
# %%

q0 = quantum_device.get_element("q0")
q1 = quantum_device.get_element("q1")

q0.clock_freqs.f01(9.1e9)
q0.rxy.amp180(0.14)
q0.rxy.duration(48e-9)

q0.measure.acq_channel(0)
q0.measure.acq_delay(40e-9)
q0.measure.pulse_amp(0.125)
q0.measure.pulse_duration(3e-6)
q0.measure.integration_time(2800e-9)
q0.clock_freqs.readout(7.45e9)

q1.clock_freqs.f01(9.2e9)
q1.rxy.amp180(0.14)
q1.rxy.duration(48e-9)

q1.measure.acq_channel(1)
q1.measure.acq_delay(40e-9)
q1.measure.pulse_amp(0.125)
q1.measure.pulse_duration(3e-6)
q1.measure.integration_time(2800e-9)
q1.clock_freqs.readout(7.48e9)

quantum_device.cfg_sched_repetitions(1)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Manually writing schedule


# %%
def coupler_conditional_oscillation_sched_phase_correction_ssro4(
    oscillation_qubit: str,
    spectator_qubit: str,
    flux_amplitude: float,
    flux_duration: float,
    rise_time: float,
    phases,
    phase_q1: float,
    qname_phase_q1: str,
    phase_q2: float,
    qname_phase_q2: str,
    spectator_rotations,
    repetition_param,
    separation_time: float = 100e-9,
    repetitions: int = 1,
) -> Schedule:
    """
    Schedule for the conditional oscillation experiment.

     Schedule sequence::

         Oscillation qubit:  ----X90-------o--- Rxy(90,phi) ---- measure
                                           |
         Spectator qubit:    --- (X180) ---o--- (X180) --------- measure

     (X180) means we either apply it or not, depending on the :code:`spectator_rotations`
     parameter.

    Parameters
    ----------
     oscillation_qubit:
         The name of the oscillating qubit e.g., :code:`"q0"`
     spectator_qubit:
         The name of the spectator (e.g. on which the oscillation is conditioned)
         qubit e.g., :code:`"q1"`
     phases:
         The phase of the Rxy gate applied after the CZ gate
     spectator_rotations:
         The rotation of the X gate on the control qubit, typically either 0 or 180 deg
     separation_time:
         Time between the end of the first single qubit operations and the start of the
         second set of single qubit operations.
     repetitions:
         The number of times the schedule will be repeated

    Returns
    -------
     :
         An experiment schedule

    """
    schedule = Schedule("Conditional oscillation", repetitions)

    phases = np.asarray(phases)
    spectator_rotations = np.asarray(spectator_rotations)

    for i, (phase, spectator_rotation) in enumerate(zip(phases, spectator_rotations)):
        schedule.add(Reset(oscillation_qubit, spectator_qubit), label=f"Reset {i}")
        schedule.add(
            Rxy(qubit=oscillation_qubit, theta=90, phi=0),
            label=f"initial oscillation qubit pulse {i}",
        )

        # An excitation is added or not added based on the spectator_rotation parameter
        schedule.add(
            Rxy(qubit=spectator_qubit, theta=spectator_rotation, phi=0),
            ref_pt="start",
            label=f"initial spectator qubit pulse {i}",
        )

        schedule.add(
            SoftSquarePulse(
                duration=flux_duration,
                amp=flux_amplitude,
                port="q1:fl",
                clock="cl0.baseband",
            )
        )

        schedule.add(
            Rxy(qubit=oscillation_qubit, theta=90, phi=phase),
            label=f"final oscillation qubit pulse {i}",
            ref_op=f"initial oscillation qubit pulse {i}",
            rel_time=separation_time,
        )
        schedule.add(
            Rxy(qubit=spectator_qubit, theta=spectator_rotation, phi=0),
            label=f"final spectator qubit pulse {i}",
            ref_pt="start",
        )
        schedule.add(
            Measure(oscillation_qubit, spectator_qubit, acq_index=i, bin_mode=BinMode.APPEND),
            label=f"Measure {i}",
        )

    return schedule


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Meas Ctrl

# %%
phase = ManualParameter(name="phase", unit="s", label="Delay")
phase.batched = True  # type: ignore

spec_rotation = ManualParameter(name="spectator_rotation", unit="s", label="Delay")
spec_rotation.batched = True  # type: ignore

repetition_param = ManualParameter(name="repetition", unit="s", label="Repetition")
repetition_param.batched = True  # type: ignore
repetitions = 16

schedule_kwargs = {
    "oscillation_qubit": q0.name,
    "spectator_qubit": q1.name,
    "flux_amplitude": 0.1,
    "flux_duration": 20e-9,
    "rise_time": 4e-9,
    "phases": phase,
    "phase_q1": np.pi,
    "qname_phase_q1": "phase_q1",
    "phase_q2": np.pi,
    "qname_phase_q2": "phase_q2",
    "spectator_rotations": spec_rotation,
    "repetition_param": repetition_param,
}

gettable = ScheduleGettable(
    quantum_device,
    schedule_function=coupler_conditional_oscillation_sched_phase_correction_ssro4,
    schedule_kwargs=schedule_kwargs,
    real_imag=False,
    batched=True,
    num_channels=2,
    max_batch_size=60,
)

quantum_device.cfg_sched_repetitions(1)

meas_ctrl.settables([repetition_param, spec_rotation, phase])
meas_ctrl.setpoints_grid(
    [
        np.arange(repetitions),
        np.linspace(1, 2 * np.pi, 2),
        np.linspace(1, 2 * np.pi, 20),
    ]
)
meas_ctrl.gettables(gettable)


# %%
def run_experiment():
    meas_ctrl.run()


# %%
def schedule_duration():
    gettable.initialize()
    return gettable.compiled_schedule.duration


# %%
def close_experiment():
    Instrument.close_all()
