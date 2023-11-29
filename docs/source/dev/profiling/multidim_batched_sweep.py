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
from quantify_core.measurement import MeasurementControl
from quantify_scheduler import Schedule
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from quantify_core.data import handling as dh
from qblox_instruments import Cluster, ClusterType

from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.enums import BinMode

from quantify_scheduler.operations.gate_library import Measure, Reset, Rxy

from quantify_scheduler.operations.pulse_library import SoftSquarePulse

# %%
dh.set_datadir(dh.default_datadir())

# %% [markdown]
# # Connect to hardware

# %%
dummy_cfg = {
    2: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    10: ClusterType.CLUSTER_QRM_RF,
    12: ClusterType.CLUSTER_QCM,
}

cluster = Cluster(name="cluster0", identifier=None, dummy_cfg=dummy_cfg)

# %%
ic = InstrumentCoordinator("ic")

# %%
qcm = cluster.module1
qrm = cluster.module2
qcm_rf = cluster.module3
qrm_rf = cluster.module4

# %%
ic.add_component(ClusterComponent(cluster))

# %%
meas_ctrl = MeasurementControl("mc")

# %%
quantum_device = QuantumDevice("my_device")

# %%
quantum_device.instr_instrument_coordinator(ic.name)
quantum_device.instr_measurement_control(meas_ctrl.name)

# %% [markdown]
# ### Hardware Cfg

# %%
hardware_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster0": {
        "sequence_to_file": False,
        "ref": "internal",
        "instrument_type": "Cluster",
        "cluster0_module2": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "dc_mixer_offset_I": -0.0024496,
                "dc_mixer_offset_Q": -0.0109159,
                "lo_freq": 4895200000.0,
                "portclock_configs": [
                    {
                        "port": "q1:mw",
                        "clock": "q1.01",
                        "mixer_amp_ratio": 0.9416,
                        "mixer_phase_error_deg": -17.36234,
                    },
                    {
                        "port": "q1:mw",
                        "clock": "q1.12",
                        "mixer_amp_ratio": 0.9416,
                        "mixer_phase_error_deg": -17.36234,
                    },
                ],
            },
        },
        "cluster0_module4": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "dc_mixer_offset_I": -0.0024927,
                "dc_mixer_offset_Q": -0.0051141,
                "lo_freq": 4802000000.0,
                "portclock_configs": [
                    {
                        "port": "q2:mw",
                        "clock": "q2.01",
                        "mixer_amp_ratio": 0.9519,
                        "mixer_phase_error_deg": -19.85496,
                    },
                    {
                        "port": "q2:mw",
                        "clock": "q2.12",
                        "mixer_amp_ratio": 0.9519,
                        "mixer_phase_error_deg": -19.85496,
                    },
                ],
                "output_att": 0,
            },
        },
        "cluster0_module10": {
            "instrument_type": "QRM_RF",
            "complex_output_0": {
                "dc_mixer_offset_I": -0.009068,
                "dc_mixer_offset_Q": -0.0082944,
                "lo_freq": 7050000000.0,
                "portclock_configs": [
                    {
                        "port": "q1:res",
                        "clock": "q1.ro",
                        "mixer_amp_ratio": 0.9973,
                        "mixer_phase_error_deg": 12.46307,
                    },
                    {
                        "port": "q2:res",
                        "clock": "q2.ro",
                        "mixer_amp_ratio": 0.9973,
                        "mixer_phase_error_deg": 12.46307,
                    },
                ],
                "output_att": 24,
            },
        },
        "cluster0_module12": {
            "instrument_type": "QCM",
            "real_output_0": {
                "portclock_configs": [{"port": "q2:fl", "clock": "cl0.baseband"}]
            },
        },
    },
}

# %%
quantum_device.hardware_config(hardware_config)

# %% [markdown]
# ### Device Cfg

# %% [markdown]
# Load device config and fill out information in the transmon element from the loaded file

# %%
q1 = BasicTransmonElement("q1")
q2 = BasicTransmonElement("q2")

# %%
quantum_device.add_element(q1)
quantum_device.add_element(q2)

# %%
q1.clock_freqs.f01(4815200000.0)
q1.rxy.amp180(0.14)
q1.rxy.duration(48e-9)

q2.clock_freqs.f01(4729610000.0)
q2.rxy.amp180(0.14)
q2.rxy.duration(48e-9)

q1.measure.acq_channel(0)
q1.measure.acq_delay(40e-9)
q1.measure.pulse_amp(0.125)
q1.measure.pulse_duration(3e-6)
q1.measure.integration_time(2800e-9)
q1.clock_freqs.readout(6995499000)

q2.measure.acq_channel(1)
q2.measure.acq_delay(40e-9)
q2.measure.pulse_amp(0.125)
q2.measure.pulse_duration(3e-6)
q2.measure.integration_time(2800e-9)
q2.clock_freqs.readout(6849880000.0)


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
                port="q2:fl",
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
            Measure(
                oscillation_qubit, spectator_qubit, acq_index=i, bin_mode=BinMode.APPEND
            ),
            label=f"Measure {i}",
        )

    return schedule


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Meas Ctrl

# %%
phase = ManualParameter(name="phase", unit="s", label="Delay")
phase.batched = True

spec_rotation = ManualParameter(name="spectator_rotation", unit="s", label="Delay")
spec_rotation.batched = True

repetition_param = ManualParameter(name="repetition", unit="s", label="Repetition")
repetition_param.batched = True
repetitions = 200

schedule_kwargs = {
    "oscillation_qubit": q1.name,
    "spectator_qubit": q2.name,
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
)

quantum_device.cfg_sched_repetitions(1)

meas_ctrl.settables([repetition_param, spec_rotation, phase])
meas_ctrl.setpoints_grid(
    [
        np.arange(repetitions),
        np.linspace(1, 2 * np.pi, 2),
        np.linspace(1, 2 * np.pi, 40),
    ]
)
meas_ctrl.gettables(gettable)


# %%
def run_experiment():
    meas_ctrl.run()
