---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-tutorial-hello-world)=

# Tutorial: Hello World

```{seealso}
The complete source code of this tutorial can be found in

{nb-download}`Hello World.ipynb`
```

# Quantify Workflow

This notebook presents a structure for setting up experiments using a combination
of {mod}`quantify-scheduler` and {mod}`quantify-core`.

{mod}`quantify-scheduler` provides a high-level interface with hardware, allowing
users to abstract hardware-specific nuances. {mod}`quantify-core` on the other hand,
serves as an experiment management tool, using {mod}`quantify-scheduler` as its
hardware interface. This allows users to manage, execute, and analyze
experiments easily. The following is a general workflow for using Quantify:

1. Initial setup
    - Set the directory for data storage for the experiment
    - Initialize the MeasurementControl and InstrumentCoordinator objects
2. Hardware setup
    - Connect to the control hardware
    - Setup the hardware configuration
3. Device setup  
    - Setup a device configuration for the device under test
4. Create a schedule
    - Create a schedule containing the timeline of operations for the experiment
    -  Compile the schedule to the hardware
    -  Visualize the schedule
4. Run and analyze the experiment
    - Setup MeasurementControl to run the experiment
    - Run the Experiment
    - Analyze the Results 


## Initial Setup
We first set up the directory in which all experimental data will be stored and managed. 

```{code-cell} ipython3
from quantify_core.data import handling as dh
dh.set_datadir()
```

Next, we need to initialize two classes:
{class}`~quantify_core.measurement.control.MeasurementControl` for managing the
experiment and  
{class}`~quantify_scheduler.instrument_coordinator.InstrumentCoordinator` for managing the hardware.

```{code-cell} ipython3
from quantify_core.measurement.control import MeasurementControl
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator

from qcodes import Instrument

Instrument.close_all()

measurement_control = MeasurementControl("measurement_control")
instrument_coordinator = InstrumentCoordinator("instrument_coordinator")
```

By initializing these classes, we can easily manage both the experiment and the
hardware. Note that we have also closed all previously opened instruments using
{meth}`~qcodes.Instrument.close_all()` to prevent any conflicts with our current
setup.

## Hardware Setup

Let us now set up the connections to the control hardware. A `cluster` is created
with a dummy configuration consisting of a readout module (QRM) in slot 1 in this case. 

```{code-cell} ipython3
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent

from qblox_instruments import Cluster, ClusterType

cluster = Cluster(
    "cluster",
    dummy_cfg={
        1: ClusterType.CLUSTER_QRM,
    },
)

ic_cluster = ClusterComponent(cluster)
instr_coord.add_component(ic_cluster)
```

Then, we set up the quantum device on which we perform the actual experiments.
This two-qubit chip is represented by `QuantumDevice` where we add a single
transmon qubit `q0` to it.

```{code-cell} ipython3
## Device Parameters
FREQ_01 = 4e9
NCO_FREQ = 100e6
READOUT_FREQ = 4e9
LO_FREQ = READOUT_FREQ - NCO_FREQ
READOUT_AMP = 0.01
```

```{code-cell} ipython3
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

quantum_device = QuantumDevice("quantum_device")
quantum_device.instr_measurement_control(measurement_control.name)
quantum_device.instr_instrument_coordinator(instrument_coordinator.name)

q0 = BasicTransmonElement("q0")
quantum_device.add_element(q0)

q0.measure.pulse_amp(READOUT_AMP)
q0.clock_freqs.readout(READOUT_FREQ)  
q0.clock_freqs.f01(FREQ_01)
q0.measure.acq_delay(100e-9)

```

The last part of setting up the hardware is to define the hardware configuration
and attach it to our quantum device. The **Hardware Configuration** is a
JSON-formatted data structure, stored either in a file or in a python
dictionary. It contains all of the information about the instruments used to run
the experiment and is used to compile the schedule to hardware. 

For more information on how to structure these JSON schemas, please go to
[hardware config!](hyperlink)

```{code-cell} ipython3
hardware_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    f"{cluster.name}": {
        "ref": "internal",
        "instrument_type": "Cluster",
        f"{cluster.module1.name}": {
            "instrument_type": "QRM",
            "complex_output_0": {
                "portclock_configs": [
                    {
                        "port": "q0:res",
                        "clock": "q0.ro",
                    }
                ],
            },
        },
    },
}
```

```{code-cell} ipython3
# Tie hardware config to device
quantum_device.hardware_config(hardware_cfg)
```

### Create a schedule

Now we must create a schedule, consisting of a set of operations that we wish to
perform on the device under test. We set up the schedule independent of the
hardware configuration and then use Quantify's ability to compile a schedule to
convert it to hardware-level commands. For this tutorial, we
will define a simple schedule that will run a T1 experiment. 

```{code-cell} ipython3
from quantify_scheduler import Schedule
import quantify_scheduler.operations.gate_library as gate_lib 
import quantify_scheduler.operations.pulse_library as pulse_lib
import quantify_scheduler.resources as resource_lib

def t1_sched(
    times,
    qubit_name,
):
    schedule = Schedule("T1")
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit_name), label=f"Reset {i}")
        schedule.add(X(qubit_name), label=f"pi {i}")
        schedule.add(
            Measure(qubit_name, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {i}",
        )
    return schedule
```

We can inspect details of the compiled schedule in three different ways: via a
timing table containing the timing and other information of each operation, or
visually through circuit and pulse diagrams. Before doing so, we need to compile
it first, which will calculate the correct timings of each operation. 

```{code-cell} ipython3
from quantify_scheduler.backends.graph_compilation import SerialCompiler

compiler = SerialCompiler(name="compiler", quantum_device = quantum_device)
compiled_schedule = compiler.compile(
    schedule=simple_sched()
)
```

Please note that further below, the compilation is handled by an instance of class:`MeasurementControl` and is done here only to enable viewing the timing table and pulse/gate visualization.

The timing table can be simply accessed via the `.timing_table` property of a schedule:

```{code-cell} ipython3
compiled_schedule.timing_table
```

whereas the gate and pulse diagram can be viewed using the methods `.plot_circuit_diagram()` and `.plot_pulse_diagram()`.

```{code-cell} ipython3
_, ax = compiled_schedule.plot_circuit_diagram()
```


```{code-cell} ipython3
compiled_schedule.plot_pulse_diagram()
```

## Setup Quantify Core to run an experiment

We can now begin to set up our measurement control to run the experiment. In
this case, we will perform a 1-dimensional sweep using a qubit element
parameter. Sweeping such a qubit parameter will physically change the hardware
output as the sweep is performed.

```{code-cell} ipython3
import numpy as np
from quantify_scheduler.gettables import ScheduleGettable
from qcodes.instrument.parameter import ManualParameter

quantum_device.cfg_sched_repetitions(1)

# Configure the settable
sample_par = ManualParameter("sample", label="Sample time", unit="s")
sample_times = np.linspace(3e-7, 6e-5, num=41)

#qcodes_par = q1.clock_freqs.readout
qcodes_par = q1.measure.pulse_amp
amps = np.linspace(0,0.03,10)

# Configure the gettable
gettable = ScheduleGettable(
    quantum_device=quantum_device,
    schedule_function=simple_sched,
    schedule_kwargs={},
)

# Configure measurement control
meas_ctrl.settables(qcodes_par)
meas_ctrl.setpoints_grid([sample_times, amps])
meas_ctrl.gettables(gettable)
```

## Run Experiment
```{code-cell} ipython3
# Run!
dataset = meas_ctrl.run("Simple schedule")
```

## Analyze Results

Now that we have run our experiment, we can perform an analysis of the results.
This can be done by a variety of classes from the `quantify_core.analysis`
package. For a T1 experiment, we use the class`T1Analysis` class which is used
to fit the data, extract relevant parameters and visualize the result.

```{code-cell} ipython3
from quantify_core.analysis import T1Analysis

tmp_test_data_dir = '/home/rsoko/qblox/quantify-core/tests/test_data'
tuid = "20210322-205253-758-6689"
T1Analysis(tuid=tuid).run().display_figs_mpl()
```

Note that for demonstration purposes, we show here an analysis of a physical T1 experiment. 