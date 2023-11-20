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
# https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/applications/quantify/tuning_transmon_qubit.html

# %% [markdown]
# ## Setup
# In this section we configure the hardware configuration which specifies the connectivity of our system.

# %% [markdown]
# ### Configuration file
#
# This is a template hardware configuration file for a 1-qubit system with a flux-control line which can be used to tune the qubit frequency.
#
# The hardware setup is as follows, by cluster slot:
# 1. **QCM-RF**
#     - Drive line for `qubit` using fixed 80 MHz IF.
# 2. **QCM**
#     - Flux line for `qubit`.
# 6. **QRM-RF**
#     - Readout line for `qubit` using a fixed LO set at 7.5 GHz.
#
# Note that in the hardware configuration below the mixers are uncorrected, but for high fidelity experiments this should also be done for all the modules.

# %%
hardware_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster0": {
        "sequence_to_file": False,  # Boolean flag which dumps waveforms and program dict to JSON file
        "ref": "internal",  # Use shared clock reference of the cluster
        "instrument_type": "Cluster",
        # ============ DRIVE ============#
        "cluster0_module1": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "output_att": 0,
                "dc_mixer_offset_I": 0.0,
                "dc_mixer_offset_Q": 0.0,
                "portclock_configs": [
                    {
                        "port": "qubit:mw",
                        "clock": "qubit.01",
                        "interm_freq": 80e6,
                        "mixer_amp_ratio": 1.0,
                        "mixer_phase_error_deg": 0.0,
                    }
                ],
            },
        },
        # ============ FLUX ============#
        "cluster0_module2": {
            "instrument_type": "QCM",
            "real_output_0": {
                "portclock_configs": [{"port": "qubit:fl", "clock": "cl0.baseband"}]
            },
        },
        # ============ READOUT ============#
        "cluster0_module3": {
            "instrument_type": "QRM_RF",
            "complex_output_0": {
                "output_att": 0,
                "input_att": 0,
                "dc_mixer_offset_I": 0.0,
                "dc_mixer_offset_Q": 0.0,
                "lo_freq": 7.5e9,
                "portclock_configs": [
                    {
                        "port": "qubit:res",
                        "clock": "qubit.ro",
                        "mixer_amp_ratio": 1.0,
                        "mixer_phase_error_deg": 0.0,
                    }
                ],
            },
        },
    },
}

# %%
from pathlib import Path

import numpy as np
import quantify_core.data.handling as dh
from qblox_instruments import Cluster, ClusterType
from qcodes import Instrument
from qcodes.parameters import ManualParameter
from quantify_core.measurement.control import MeasurementControl
from quantify_core.visualization.pyqt_plotmon import PlotMonitor_pyqt as PlotMonitor
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from quantify_scheduler.schedules import heterodyne_spec_sched_nco


# %% [markdown]
# ### Connect to Cluster
#
# We now make a connection with the Cluster selected in the dropdown widget. We also define a function to find the modules we're interested in. We select the readout and control module we want to use.

# %%
# Close all existing QCoDeS Instrument instances
Instrument.close_all()

# Here we have the option to use a dummy device so that you can run your tests without a physical cluster
dummy_cfg = {
    1: ClusterType.CLUSTER_QCM_RF,
    2: ClusterType.CLUSTER_QCM,
    3: ClusterType.CLUSTER_QRM_RF,
}

cluster = Cluster(name="cluster0", identifier="", dummy_cfg=dummy_cfg)

# print(f"{connect.label} connected")

# %% [markdown]
# ### Reset the Cluster
#
# We reset the Cluster to enter a well-defined state. Note that resetting will clear all stored parameters and repeats startup calibration, so resetting between experiments is usually not desirable.

# %%
cluster.reset()
# print(cluster.get_system_state())

# %% [markdown]
# Note that a dummy cluster will raise error flags, this is expected behavior and can be ignored.

# %% [markdown]
# ### Quantum device settings
# Here we initialize our `QuantumDevice` and our qubit parameters,  checkout this [tutorial](https://quantify-quantify-scheduler.readthedocs-hosted.com/en/latest/tutorials/Operations%20and%20Qubits.html) for further details.
#
# In short, a `QuantumDevice` contains device elements where we save our found parameters.

# %%
qubit = BasicTransmonElement("qubit")
qubit.measure.acq_channel(0)

quantum_device = QuantumDevice("device_1q")
quantum_device.hardware_config(hardware_cfg)

quantum_device.add_element(qubit)


# %% [markdown]
# ### Configure measurement control loop
# We will use a `MeasurementControl` object for data acquisition as well as an `InstrumentCoordinator` for controlling the instruments in our setup.
#
# The `PlotMonitor` is used for live plotting.
#
# All of these are then associated with the `QuantumDevice`.


# %%
def configure_measurement_control_loop(
    device: QuantumDevice, cluster: Cluster, live_plotting: bool = False
) -> None:
    # Close QCoDeS instruments with conflicting names
    for name in [
        "PlotMonitor",
        "meas_ctrl",
        "ic",
        "ic_generic",
        f"ic_{cluster.name}",
    ] + [f"ic_{module.name}" for module in cluster.modules]:
        try:
            Instrument.find_instrument(name).close()
        except KeyError:
            pass

    meas_ctrl = MeasurementControl("meas_ctrl")
    ic = InstrumentCoordinator("ic")

    # Add cluster to instrument coordinator
    ic_cluster = ClusterComponent(cluster)
    ic.add_component(ic_cluster)

    if live_plotting:
        # Associate plot monitor with measurement controller
        plotmon = PlotMonitor("PlotMonitor")
        meas_ctrl.instr_plotmon(plotmon.name)

    # Associate measurement controller and instrument coordinator with the quantum device
    device.instr_measurement_control(meas_ctrl.name)
    device.instr_instrument_coordinator(ic.name)

    return (meas_ctrl, ic)


meas_ctrl, instrument_coordinator = configure_measurement_control_loop(
    quantum_device, cluster
)

# %% [markdown]
# ### Set data directory
# This directory is where all of the experimental data as well as all of the post processing will go.

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

# %%
flux_settable: callable = cluster.module2.out0_offset
flux_settable(0.0)

# %% [markdown]
# ### Activate NCO delay compensation
# Compensate for the digital propagation delay for each qubit (i.e each sequencer)
#
# For more info, please see: https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/sequencer.html#pulsar-qcm-sequencer-nco-prop-delay-comp-en
#
# To avoid mismatches between modulation and demodulation, the delay between any readout frequency or phase changes and the next acquisition should be equal or greater than the total propagation delay (146ns + user defined value).

# %%
for i in range(6):
    getattr(cluster.module3, f"sequencer{i}").nco_prop_delay_comp_en(True)
    getattr(cluster.module3, f"sequencer{i}").nco_prop_delay_comp(50)


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
# 2. Qubit spectroscopy (a.k.a two-tone)
#     - Used to find the $|0\rangle \rightarrow |1\rangle$ drive frequency.
# 3. Rabi oscillations
#     - Used to find precise excitation pulse required to excite the qubit to $|1\rangle$.
# 4. Ramsey oscillations
#     - Used to tune the $|0\rangle \rightarrow |1\rangle$ drive frequency more precisely.
#     - Used to measure $T_2^*$.
# 5. T1
#     - Used to find the time it takes for the qubit to decay from $|1\rangle$ to $|0\rangle$, the $T_1$ time.

# %% [markdown]
# ## Resonator spectroscopy


# %%
def create_schedule(*args, **kwargs):
    return heterodyne_spec_sched_nco(*args, **kwargs)


# %%
freq = ManualParameter(name="freq", unit="Hz", label="Frequency")
freq.batched = True
freq.batch_size = 100

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
frequency_setpoints = np.linspace(center - 20e6, center + 20e6, 300)
meas_ctrl.settables(freq)
meas_ctrl.setpoints(frequency_setpoints)


def run_experiment():
    meas_ctrl.run(
        "resonator spectroscopy",
    )
