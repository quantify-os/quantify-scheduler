.. jupyter-kernel::
  :id: compilation_backends

.. jupyter-execute::
    :hide-code:

    # Make output easier to read
    from rich import pretty
    pretty.install()


====================
Compilation backends
====================

In order to execute a :class:`~.Schedule` on physical hardware or a simulator one needs to compile the schedule.
This is done using a :class:`~.backends.graph_compilation.CompilationBackend`.
The :meth:`~.backends.graph_compilation.CompilationBackend.compile` method requires both the :class:`~.Schedule` to compile and a configuration describing the information required to perform the compilation.

A :class:`~.backends.graph_compilation.CompilationBackend` defines a directed acyclic graph in which the nodes represent compiler passes.
A :class:`~.Schedule` can be compiled by traversing the graph.
The :class:`~.Schedule` class serves as the intermediate representation which is modified by the compiler passes.
For most practical purposes, a user does not need to be aware of the internal structure of the compilation backends.





Compiling a schedule
====================

To compile a schedule, one needs to instantiate a :class:`~.backends.graph_compilation.CompilationBackend` and call the :meth:`~.backends.graph_compilation.CompilationBackend.compile` method.
This method requires both the :class:`~.Schedule` to compile as well as a :attr:`~.QuantumDevice.compilation_config`.
This config can conveniently be generated from a :class:`QuantumDevice` object which describes the knowledge required for compilation.

.. note::

    Here we focus on using a :class:`~.backends.graph_compilation.CompilationBackend` to compile a :class:`~.Schedule` in isolation.
    When executing schedules, one needs to interact with and manage the parameters of an experimental setup.
    For this we refer to the :ref:`section on execution <sec-user-guide-execution>` in the user guide.

First we set up a mock setup and create a simple schedule that we want to compile.

.. jupyter-execute::

    import numpy as np
    from quantify_scheduler.device_under_test.mock_setup import set_up_mock_transmon_setup, set_standard_params
    from quantify_scheduler.schedules.timedomain_schedules import echo_sched

    # instantiate the instruments of the mock setup
    mock_setup = set_up_mock_transmon_setup()

    # provide some sensible values to allow compilation without errors
    set_standard_params(mock_setup)


    echo_schedule = echo_sched(times=np.arange(0, 60e-6, 1.5e-6), qubit="q0", repetitions=1024)


Next, we retrieve the compilation config from the quantum device and see for which compilation backend this is suitable.

.. jupyter-execute::


    quantum_device = mock_setup['quantum_device']
    config = quantum_device.compilation_config

    print(config['backend'])



We can then instantiate the backend and compile the program.


.. jupyter-execute::

    from quantify_scheduler.backends.device_compile import DeviceCompile

    backend = DeviceCompile()
    comp_sched = backend.compile(schedule=echo_schedule, config=config)

    comp_sched


Understanding the structure of compilation
==========================================

A compilation backend defines a graph of compilation steps.
This makes it really easy to visualize the different steps in the compilation process by drawing the graph.
Below we show the graphs defined by the :class:`.backends.DeviceCompile`, the :class:`.backends.QbloxBackend`, and the :class:`.backends.ZhinstBackend`.

.. jupyter-execute::

    from quantify_scheduler.backends import DeviceCompile, QbloxBackend, ZhinstBackend

    dev_backend = DeviceCompile()
    qblox_backend = QbloxBackend()
    zhinst_backend= ZhinstBackend()

    import matplotlib.pyplot as plt
    f, axs = plt.subplots(1,3, figsize=(16,7))

    # Show the graph of the currently included backends
    dev_backend.draw(axs[0])
    axs[0].set_title('DeviceBackend')
    qblox_backend.draw(axs[1])
    axs[1].set_title('QbloxBackend')
    zhinst_backend.draw(axs[2])
    axs[2].set_title('ZhinstBackend')
    f


One might notice that some nodes appear in multiple backends.
This is intentional and showcases how we are reusing certain modular compilation steps.

[Planned feature] When using a compilation backend, the graph based structure also allows us to verify the output at the end of every node. This can be particularly useful when the compilation is not producing the output expected by the user.

Creating a custom compilation backend
=====================================

(advanced user/basic developer)

Here we describe the internals for developers who want to add a custom node or parts of the config that are different.

Selection mechanism for what backend is used/instantiated to compile with.
(change this to your custom backend).
Emphasize modularity and testing on how to develop a custom backend.



Understanding the structure of compilation.
It is a graph.
Different parts of the config are used in different nodes.
Show an example of a graph.


Future ideas
============

Explain the idea of the graph.
Where does the config come from?
What steps does it take?
Showing the steps in the backend to understand what happens in the compilation.


Dynamically generate graphs based on the structure of the config.
Currently we only support static graphs, but it makes sense to dynamically generate the graph structure upon instantiation of the backend.
Figuring out how we want to support this requires further thought. My gutfeel teels me that this is related to the part of the hardware configuration that remains fixed.

How to deal with non-linear graphs (nodes in parallel) is not 100% clear yet. The meaning of parallelism is something I am getting to now, but it is not fully clear yet how to deal with input output definitions of nodes yet.




Backend internals



.. jupyter-execute::
    :hide-code:

    %reset -f


