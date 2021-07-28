.. _sec-user-guide:

Getting started
===============

.. jupyter-kernel::
  :id: Scheduler getting started

.. jupyter-execute::
    :hide-code:

    # Make output easier to read
    from rich import pretty
    pretty.install()


Introduction
------------
Quantify-scheduler is a python module for writing quantum programs featuring a hybrid gate-pulse control model with explicit timing control.
It extends the circuit model from quantum information processing by adding a pulse-level representation to operations defined at the gate-level, and the ability to specify timing constraints between operations.
Thus, a user is able to mix gate- and pulse-level operations in a quantum circuit.


In quantify-scheduler, both a quantum circuit consisting of gates and measurements and a timed sequence of control pulses are described as a :class:`~quantify_scheduler.types.Schedule` .
The :class:`~quantify_scheduler.types.Schedule` contains information on *when* operations should be performed.
When adding operations to a schedule, one does not need to specify how to represent this :class:`~quantify_scheduler.types.Operation` on all (both gate and pulse) abstraction levels.
Instead, this information can be added later during :ref:`Compilation`.
This allows the user to effortlessly mix the gate- and pulse-level descriptions as is required for many experiments.
We support a similar flexibility in the timing constraints, one can either explicitly specify the timing using :attr:`~quantify_scheduler.types.Schedule.timing_constraints`, or rely on the compilation which will use the duration of operations to schedule them back-to-back.


Creating a schedule
-------------------

The most convenient way to interact with a :class:`~quantify_scheduler.types.Schedule` is through the :mod:`quantify_scheduler` API.
In the following example, we set up an element of a `Bell experiment <https://en.wikipedia.org/wiki/Bell%27s_theorem>`_ and visualize the circuit.


.. jupyter-execute::
    :hide-output:

    # import the Schedule class and some basic operations.
    from quantify_scheduler import Schedule
    from quantify_scheduler.gate_library import Reset, Measure, CZ, Rxy, X90

    sched = Schedule("Bell experiment")

    sched.add(Reset("q0", "q1"))  # initialize the qubits
    sched.add(X90(qubit="q0"))
    # Here we use a timing constraint to explicitly schedule the second gate to start
    # simultaneously with the first gate.
    sched.add(X90(qubit="q1"), ref_pt="start", rel_time=0)
    sched.add(CZ(qC="q0", qT="q1"))
    sched.add(Rxy(theta=45.0, phi=0, qubit="q0") )  # pick an angle for maximal Bell violation
    sched.add(Measure("q0", "q1", acq_index=(0, 1)))  # denote where to store the data



.. jupyter-execute::

    # import the circuit visualizer
    from quantify_scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib

    # visualize the circuit
    f, ax = circuit_diagram_matplotlib(sched)


For more details on how to create schedules, specify timing constraints and seamlessly mix the gate- and pulse-level descriptions, see :ref:`Tutorial 1 <sec-tutorial1>`.

.. _sec-compilation:

Compilation
-----------

Different compilation steps are required to go from a high-level description of a schedule to something that can be executed on hardware.
The scheduler supports two main compilation steps, the first from the gate to the pulse level, and a second from the pulse-level to a hardware back end.
This is schematically shown in :numref:`compilation_overview`.


.. figure:: /images/compilation_overview.svg
    :name: compilation_overview
    :align: center
    :width: 900px

    A schematic overview of the different abstraction layers and the compilation process.
    Both a quantum circuit, consisting of gates and measurements of qubits, and timed sequences of control pulses are represented as a :class:`~quantify_scheduler.types.Schedule` .
    The information specified in the :ref:`device configuration<sec-device-config>` is used during compilation to add information on how to represent :class:`~quantify_scheduler.types.Operation` s specified at the quantum-circuit level as control pulses.
    The information in the :ref:`hardware configuration <sec-hardware-config>` is then used to compile the control pulses into instructions suitable for hardware execution.


In the first compilation step, pulse information is added to all operations that are not valid pulses (see :attr:`~quantify_scheduler.types.Operation.valid_pulse`) based on the information specified in the :ref:`device configuration file<sec-device-config>`.

A second compilation step takes the schedule at the pulse level and translates this for use on a hardware back end.
This compilation step is performed using a hardware dependent compiler and uses the information specified in the :ref:`hardware configuration file<sec-hardware-config>`.

Both compilation steps can be triggered by passing a :class:`~quantify_scheduler.types.Schedule` and the appropriate configuration files to :func:`~quantify_scheduler.compilation.qcompile`.

.. note::

    We use the term "**device**" to refer to the physical object(s) on the receiving end of the control pulses, e.g. a thin-film chip inside a dilution refrigerator.

    And we employ the term "**hardware**" to refer to the instruments (electronics) that are involved in the pulse generations / signal digitization.



.. _sec-device-config:

Device configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~

The device configuration file is used to compile from the quantum-circuit layer to the quantum-device layer.
The main responsibility is to contain the information required to add a pulse-representation to every operation that only has a quantum-circuit layer description.
The device configuration contains information for all qubits, and all edges.
Edges are pairs of qubits (separated by a dash :code:`-`) on which gates can act.
A configuration file can be written down manually as a JSON file or be code generated.


.. admonition:: Device configuration JSON schema for the transmon backend
    :class: dropdown

    A valid device configuration is described by the schema shown here:

    .. jsonschema:: ../quantify_scheduler/schemas/transmon_cfg.json



.. admonition:: Example device configuration file
    :class: dropdown

    .. jupyter-execute::
        :hide-code:

        from pathlib import Path
        import json
        import quantify_scheduler.schemas.examples as examples

        path = Path(examples.__file__).parent / "transmon_test_config.json"
        json_data = json.loads(path.read_text())
        json_data

.. _sec-hardware-config:

Hardware configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hardware configuration file is used to compile pulses (and acquisition protocols) along with their timing information to instructions compatible with the specific control electronics.
To do this, it contains information on what control electronics to compile to and the connectivity: which ports are connected to which hardware outputs/inputs, as well as other hardware-specific settings.
Similar to the device configuration file, the hardware configuration file can be written down manually as JSON or be code generated.


.. admonition:: Example Qblox hardware configuration file
    :class: dropdown

    .. jupyter-execute::
        :hide-code:

        from pathlib import Path
        import json
        import quantify_scheduler.schemas.examples as examples

        path = Path(examples.__file__).parent / "qblox_test_mapping.json"
        json_data = json.loads(path.read_text())
        json_data



.. admonition:: Example Zurich Instruments hardware configuration file
    :class: dropdown

    .. jupyter-execute::
        :hide-code:

        from pathlib import Path
        import json
        import quantify_scheduler.schemas.examples as examples

        path = Path(examples.__file__).parent / "zhinst_test_mapping.json"
        json_data = json.loads(path.read_text())
        json_data


Execution
---------

Different kinds of instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to execute a schedule, one needs both physical instruments to execute the compiled instructions as well as a way to manage the calibration parameters used to compile the schedule.
Although one could use manually written configuration files and send the compiled files directly to the hardware, the Quantify framework provides different kinds of :class:`~qcodes.instrument.base.Instrument`\s to control the experiments and the management of the configuration files (:numref:`instruments_overview`).


.. figure:: /images/instruments_overview.svg
    :name: instruments_overview
    :align: center
    :width: 600px

    A schematic overview of the different kinds of instruments present in an experiment.
    Physical instruments are QCoDeS drivers that are directly responsible for executing commands on the control hardware.
    On top of the physical instruments is a hardware abstraction layer, that provides a hardware agnostic interface to execute compiled schedules.
    The instruments responsible for experiment control are treated to be as stateless as possible [*]_ .
    The knowledge about the system that is required to generate the configuration files is described by the :code:`QuantumDevice` and :code:`DeviceElement`\s.
    Several utility instruments are used to control the flow of the experiments.

Physical instruments
^^^^^^^^^^^^^^^^^^^^

`QCoDeS instrument drivers <https://qcodes.github.io/Qcodes/api/generated/qcodes.instrument_drivers.html>`_ are used to represent the physical hardware.
For the purpose of quantify-scheduler, these instruments are treated as stateless, the desired configurations for an experiment being described by the compiled instructions.
Because the instruments correspond to physical hardware, there is a significant overhead in querying and configuring these parameters.
As such, the state of the instruments in the software is intended to track the state of the physical hardware to facilitate lazy configuration and logging purposes.

Hardware abstraction layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
Because different physical instruments have different interfaces, a hardware abstraction layer serves to provide a uniform interface.
This hardware abstraction layer is implemented as the :class:`~quantify_scheduler.instrument_coordinator.InstrumentCoordinator` to which individual :class:`InstrumentCoordinatorComponent <quantify_scheduler.instrument_coordinator.components.base.InstrumentCoordinatorComponentBase>`\s are added that provide the uniform interface to the individual instruments.


The quantum device and the device elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The knowledge of the system is described by the :code:`QuantumDevice` and :code:`DeviceElement`\s.
The :code:`QuantumDevice` directly represents the device under test (DUT) and contains a description of the connectivity to the control hardware as well as parameters specifying quantities like cross talk, attenuation and calibrated cable-delays.
The :code:`QuantumDevice` also contains references to individual :code:`DeviceElement`\s, representations of elements on a device (e.g, a transmon qubit) containing the (calibrated) control-pulse parameters.

Because the :code:`QuantumDevice` and the :code:`DeviceElement`\s are an :class:`~qcodes.instrument.base.Instrument`, the parameters used to generate the configuration files can be easily managed and are stored in the snapshot containing the experiment's metadata.

Experiment flow
~~~~~~~~~~~~~~~

To use schedules in an experimental setting, in which the parameters used for compilation as well as the schedules themselves routinely change, we provide a framework for performing experiments making use of the concepts of :mod:`quantify_core`.
Central in this framework are the schedule :mod:`quantify_scheduler.gettables` that can be used by the :class:`~quantify_core.measurement.MeasurementControl` and are responsible for the experiment flow.
This flow is schematically show in :numref:`experiments_control_flow`.


.. figure:: /images/experiments_control_flow.svg
    :name: experiments_control_flow
    :align: center
    :width: 800px

    A schematic overview of the experiments control flow.


Let us consider the example of an experiment used to measure the coherence time :math:`T_1`.
In this experiment a :math:`\pi` pulse is used to excite the qubit, which is left to idle for a time :math:`\tau` before it is measured.
This experiment is then repeated for different :math:`\tau` and averaged.

In terms of settables and gettables to use with the :class:`~quantify_core.measurement.MeasurementControl`, the settable in this experiment is the delay time :math:`\tau`, and the gettable is the execution of the schedule.

We represent the settable as a :class:`qcodes.instrument.parameter.ManualParameter`:

.. jupyter-execute::

    from qcodes.instrument.parameter import ManualParameter

    tau = ManualParameter("tau", label=r"Delay time", initial_value=0, unit="s")


To execute the schedule with the right parameters, the :code:`ScheduleGettable` needs to have a reference to a template function that generates the schedule, the appropriate keyword arguments for that function, and a reference to the :code:`QuantumDevice` to generate the required configuration files.

For the :math:`T_1` experiment, quantify-scheduler provides a schedule generating function as part of the :mod:`quantify_scheduler.schedules.timedomain_schedules`: the :func:`quantify_scheduler.schedules.timedomain_schedules.t1_sched`.

.. jupyter-execute::

    from quantify_scheduler.schedules.timedomain_schedules import t1_sched
    schedule_function = t1_sched


Inspecting the :func:`quantify_scheduler.schedules.timedomain_schedules.t1_sched`, we find that we need to provide the times :math:`\tau`, the name of the qubit, and the number of times we want to repeat the schedule.
Rather than specifying the values of the delay times, we pass the parameter :code:`tau`.

.. jupyter-execute::

    qubit_name = "q0"
    sched_kwargs = {
        "times": tau,
        "qubit": qubit_name,
        "repetitions": 1024 # could also be a parameter
    }

The :code:`ScheduleGettable` is set up to evaluate the value of these parameter on every call of :code:`ScheduleGettable.get`.
This flexibility allows the user to create template schedules that can then be measured by varying any of it's input parameters using the :class:`~quantify_core.measurement.MeasurementControl`.

Similar to how the schedule keyword arguments are evaluated for every call to :code:`ScheduleGettable.get`, the device config and hardware config files are re-generated from the :code:`QuantumDevice` for every iteration.
This ensures that if a calibration parameter is changed on the :code:`QuantumDevice`, the compilation will be affected as expected.

.. warning::

    :code:`QuantumDevice` class is not implemented yet.

.. jupyter-execute::

    # device = QuantumDevice(name="quantum_sample")
    device = None # placeholder value

These ingredients can then be combined to perform the experiment:

.. jupyter-execute::

    from quantify_core.measurement import MeasurementControl
    meas_ctrl = MeasurementControl("meas_ctrl")

.. warning::

    :code:`ScheduleGettable` class is not implemented yet.

.. code-block:: python

    t1_gettable = ScheduleGettable(
        device=device,
        schedule_function=schedule_function,
        schedule_kwargs=sched_kwargs
    )

    meas_ctrl.settables(tau)
    meas_ctrl.setpoints(times)
    meas_ctrl.gettables(t1_gettable)
    label = f"T1 experiment {qubit_name}"
    dataset = meas_ctrl.run(label)


and the resulting dataset can be analyzed using

.. jupyter-execute::

    from quantify_core.analysis.t1_analysis import T1Analysis
    # analysis = T1Analysis(label=label).run()

.. tip::
    For a more technical overview of the concepts and terminology, we recommend to consult the section on :ref:`concepts and terminology <sec-concepts-terminology>`.



.. rubric:: Footnotes

.. [*] Quantify-scheduler threats physical instruments as stateless in the sense that the compiled instructions contain all information that specify the executing of a schedule. However, for performance reasons, it is important to not reconfigure all parameters of all instruments whenever a new schedule is executed. The parameters (state) of the instruments are used to track the state of physical instruments to allow lazy configuration as well as ensuring metadata containing the current settings is stored correctly.

