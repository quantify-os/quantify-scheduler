---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-tutorial-compiling)=

# Tutorial: Compiling to Hardware

```{seealso}
The complete source code of this tutorial can be found in

{nb-download}`Compiling to Hardware.ipynb`
```

Compilation allows converting the schedules introduced in {ref}`sec-tutorial-sched-pulse` into a set of instructions that can be executed on the control hardware.

In this notebook, we will define an example schedule, demonstrate how to compile it, and run it on a virtual hardware setup.

## Schedule definition

We start by defining an example schedule.

```{code-cell} ipython3

from quantify_scheduler import ClockResource, Schedule
from quantify_scheduler.operations import SquarePulse

sched = Schedule("Simple schedule")
sched.add(SquarePulse(amp=0.2, duration=8e-9, port="q0:res", clock="q0.ro"))
sched.add(SquarePulse(amp=0.1, duration=12e-9, port="q0:res", clock="q0.ro"))

readout_clock = ClockResource(name="q0.ro", freq=7e9)
sched.add_resource(readout_clock)

sched


```

## Hardware compilation configuration

In our example setup, we will use a Qblox Cluster containing an RF control module (QCM-RF). To compile the schedule, we will need to provide the compiler with a dictionary detailing the hardware compilation configuration.

Please check the documentation on how to properly create such a configuration for the supported backends:

- {ref}`sec-backend-qblox`
- {ref}`sec-backend-zhinst`

```{admonition} Creating an example Qblox hardware compilation configuration
:class: dropdown

Below we create an example hardware configuration dictionary, for the Qblox backend.
In this configuration, we include:

- The `"config_type"`, which points to the specific {class}`~quantify_scheduler.backends.types.common.HardwareCompilationConfig` datastructure for the backend we want to use (the Qblox backend, in this case).
- The description of the setup, including:
    - A Cluster containing a QCM-RF module (in the 2nd slot).
- The hardware options we want to use in the compilation.
- The connectivity between the control hardware and the quantum device.
```

```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
  code_prompt_show: "Example hardware configuration"
---
hardware_comp_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "2": {
                    "instrument_type": "QCM_RF"
                },
            },
        },
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q0:res-q0.ro": {"interm_freq": 50e6},
        },
        "mixer_corrections": {
            "q0:res-q0.ro": {
                "dc_offset_i": -0.00552,
                "dc_offset_q": -0.00556,
                "amp_ratio": 0.9998,
                "phase_error": -4.1
            }
        }
    },
    "connectivity": {
        "graph": [
            ("cluster0.module2.complex_output_0", "q0:res"),
        ]
    },
}
```

(sec-tutorial-compiling-to-hardware-compilation)=
## Compilation

Now we are ready to proceed to the compilation stage. For each of the control stack's instruments, the compilation generates:

- The schedule's absolute timing. During the schedule's definition, we didn't assign absolute times to the operations. Instead, only the duration was defined. For the instruments to know how to execute the schedule, the absolute timing of the operations is calculated.
- A set of parameters that are used to properly configure each instrument for the execution of the schedule. These parameters typically don't change during the execution of the schedule.
- A compiled program that contains instructions on what the instrument must do in order for the schedule to be executed.

We perform the compilation via {func}`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler.compile`.

We start by setting the directory where the compiled schedule files will be stored, via [set_datadir](https://quantify-os.org/docs/quantify-core/dev/user/concepts.html#data-directory).

```{code-cell} ipython3

from quantify_core.data import handling as dh

dh.set_datadir(dh.default_datadir())


```

Next, we create a device configuration that contains all knowledge of the physical device under test (DUT). To generate it we use the {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` class.

The schedule defined at the beginning of this tutorial consists of 2 pulse operations. As such, the hardware compilation configuration must contain the necessary information to execute the schedule. We add the hardware compilation configuration to the `QuantumDevice` object and compile the schedule using this information. We visualize the compiled schedule in a {meth}`pulse diagram <.plot_pulse_diagram>`.

```{code-cell} ipython3

from quantify_scheduler import QuantumDevice, SerialCompiler

quantum_device = QuantumDevice("DUT")

quantum_device.hardware_config(hardware_comp_cfg)
compiler = SerialCompiler(name="compiler")
compiled_sched = compiler.compile(
    schedule=sched, config=quantum_device.generate_compilation_config()
)

compiled_sched.plot_pulse_diagram(plot_backend='plotly')


```

The cell above compiles the schedule, returning a {class}`~quantify_scheduler.schedules.schedule.CompiledSchedule` object. This class differs from {class}`~quantify_scheduler.schedules.schedule.Schedule` in that it is immutable and contains the {attr}`~quantify_scheduler.schedules.schedule.CompiledSchedule.compiled_instructions` attribute.  We inspect these instructions below.

```{code-cell} ipython3

compiled_sched.compiled_instructions


```

## Execution on the hardware

In the compiled schedule, we have all the information necessary to execute the schedule.
In this specific case, only sequencer {code}`seq0` of the RF control module (QCM-RF) is needed. The compiled schedule contains the file path where the sequencer's program is stored, as well as the QCoDeS parameters that need to be set in the device.

Now that we have compiled the schedule, we are almost ready to execute it with our control setup.

We start by connecting to a dummy cluster device by passing a `dummy_cfg` argument when initializing a `Cluster`:

```{code-cell} ipython3

from qblox_instruments import Cluster, ClusterType

Cluster.close_all()  # Closes all registered instruments (not just Clusters)

cluster0 = Cluster("cluster0", dummy_cfg={"2": ClusterType.CLUSTER_QCM_RF})

```
Here, {code}`dummy_cfg={"2": ClusterType.CLUSTER_QCM_RF}` initializes a dummy cluster instrument that contains an RF control module in slot 2, as specified by the example hardware config.

We attach these instruments to the {class}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator` via the appropriate {class}`~quantify_scheduler.instrument_coordinator.components.base.InstrumentCoordinatorComponentBase` component wrapper class. More information on the scheduler execution can be found in the {ref}`Execution section of User guide <sec-user-guide-execution>`.

```{code-cell} ipython3

from quantify_scheduler import InstrumentCoordinator
from quantify_scheduler.qblox import ClusterComponent

ic = InstrumentCoordinator("ic")
ic.add_component(ClusterComponent(cluster0))


```

The {class}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator` object is responsible for the smooth and in-concert operation of the different instruments, so that the provided schedule is correctly executed.
Essentially, it "coordinates" the control stack instruments, giving the relevant commands to the different instruments of the control stack at each point in time.

The experiment can now be conducted using the methods of {class}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator`:

1. We prepare the instruments with the appropriate settings and upload the schedule program by calling the {meth}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.prepare` method and passing the compiled schedule as an argument.
2. We start the hardware execution by calling the {meth}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.start` method.

Additionally, the {meth}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.wait_done` method is useful to wait for the experiment to finish and assure the synchronicity of the python script.

```{code-cell} ipython3

# Set the qcodes parameters and upload the schedule program
ic.prepare(compiled_sched)

# Start the hardware execution
ic.start()

# Wait for the experiment to finish or for a timeout
ic.wait_done(timeout_sec=10)


```

The {class}`~~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator` has two more methods that were not covered in this experiment:

- {meth}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.retrieve_acquisition`
  \- In case the schedule contained acquisitions, this method retrieves the acquired data.
- {meth}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.stop`
  \- Stops all running instruments.

Note that the schedule used in this tutorial was defined purely in terms of pulses.
However, `quantify-scheduler` also supports the usage of quantum gates in schedules. Given that gates may require different pulses depending on the type of quantum system, an extra step of defining the quantum device configuration, i.e. the qubits, is necessary. This use case is covered in the {ref}`sec-tutorial-ops-qubits` tutorial.
