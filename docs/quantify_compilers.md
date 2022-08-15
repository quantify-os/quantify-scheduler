---
file_format: mystnb
kernelspec:
    name: python3

---


```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

# Make output easier to read
from rich import pretty
pretty.install()

```

# Compilers

In order to execute a {class}`~.Schedule` on physical hardware or a simulator one needs to compile the schedule.
This is done using a {class}`~.backends.graph_compilation.QuantifyCompiler`.
The {meth}`~.backends.graph_compilation.QuantifyCompiler.compile` method requires both the {class}`~.Schedule` to compile and a {class}`~.CompilationConfig` describing the information required to perform the compilation.

Upon the start of the compilation, the {class}`~.backends.graph_compilation.QuantifyCompiler` defines a directed acyclic graph in which individual nodes represent compilation steps.
A {class}`~.Schedule` can be compiled by traversing the graph.
The {class}`~.Schedule` class serves as the intermediate representation which is modified by the compiler passes.
For most practical purposes, a user does not need to be aware of the internal structure of the compilation backends.

## Compiling a schedule

To compile a schedule, one needs to instantiate a {class}`~.backends.graph_compilation.QuantifyCompiler` and call the {meth}`~.backends.graph_compilation.QuantifyCompiler.compile` method.
This method requires both the {class}`~.Schedule` to compile as well as a {meth}`~.QuantumDevice.generate_compilation_config()`.
This config can conveniently be generated from a {class}`~.QuantumDevice` object which describes the knowledge required for compilation.

```{note}
Here we focus on using a {class}`~.backends.graph_compilation.QuantifyCompiler` to compile a {class}`~.Schedule` in isolation.
When executing schedules, one needs to interact with and manage the parameters of an experimental setup.
For this we refer to the {ref}`section on execution <sec-user-guide-execution>` in the user guide.
```

First we set up a mock setup and create a simple schedule that we want to compile.

```{code-cell}

    import numpy as np
    from quantify_scheduler.device_under_test.mock_setup import set_up_mock_transmon_setup, set_standard_params_transmon
    from quantify_scheduler.schedules.timedomain_schedules import echo_sched

    # instantiate the instruments of the mock setup
    mock_setup = set_up_mock_transmon_setup()

    # provide some sensible values to allow compilation without errors
    set_standard_params_transmon(mock_setup)


    echo_schedule = echo_sched(times=np.arange(0, 60e-6, 1.5e-6), qubit="q0", repetitions=1024)

```

Next, we retrieve the {class}`~.CompilationConfig` from the quantum device and see for which compilation backend this is suitable.
In the current example we have a simple {class}`~.backends.graph_compilation.SerialCompiler` that is used to do different compilation passes as a linear chain.

```{code-cell}


    quantum_device = mock_setup["quantum_device"]
    config = quantum_device.generate_compilation_config()

    print(config.backend)


```

We can then instantiate the compiler and compile the program.

```{code-cell}

from quantify_scheduler.backends.graph_compilation import SerialCompiler

compiler = SerialCompiler(name="Device compile")
comp_sched = compiler.compile(schedule=echo_schedule, config=config)

comp_sched

```

## Understanding the structure of compilation

A compilation backend defines a graph of compilation steps.
This makes it really easy to visualize the different steps in the compilation process by drawing the graph.

Here we show the compilation structure for several commonly used compilers.
To do this, we will use the example configuration files of the different compilers and then use the quantum device to generate the relevant {class}`~.CompilationConfig` s.
Note that in the future we want to improve how the hardware config is managed so one does not need to set a custom dictionary to the hardware config parameter of the ``quantum_device`` object.


```{code-cell}

from quantify_scheduler.schemas.examples import utils

QBLOX_HARDWARE_MAPPING = utils.load_json_example_scheme("qblox_test_mapping.json")
ZHINST_HARDWARE_MAPPING = utils.load_json_example_scheme("zhinst_test_mapping.json")

dev_cfg = quantum_device.generate_compilation_config()

quantum_device.hardware_config(QBLOX_HARDWARE_MAPPING)
qblox_cfg = quantum_device.generate_compilation_config()

quantum_device.hardware_config(ZHINST_HARDWARE_MAPPING)
zhinst_cfg = quantum_device.generate_compilation_config()

```

```{code-cell}

from quantify_scheduler.backends import SerialCompiler



# constructing graph is normally done when at compile time as it
# requires information from the compilation config.

dev_compiler = SerialCompiler(name="Device compiler")
dev_compiler.construct_graph(dev_cfg)


qblox_compiler = SerialCompiler(name="Qblox compiler")
qblox_compiler.construct_graph(qblox_cfg)

zhinst_compiler = SerialCompiler(name="Zhinst compiler")
zhinst_compiler.construct_graph(zhinst_cfg)




import matplotlib.pyplot as plt
f, axs = plt.subplots(1,3, figsize=(16,7))

# Show the graph of the currently included compilers
dev_compiler.draw(axs[0])
axs[0].set_title("Device Backend")
qblox_compiler.draw(axs[1])
axs[1].set_title("Qblox Backend")
zhinst_compiler.draw(axs[2])
axs[2].set_title("Zhinst Backend")
f

```
