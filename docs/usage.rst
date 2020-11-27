User guide
================================

.. jupyter-kernel::
  :id: Tutorial 1. Scheduler concepts



Introduction
----------------
Quantify-scheduler is a module for writing quantum programs.
It is designed for experimentalists to easily define complex experiments, and produces synchronized pulse schedules to be distributed to control hardware.

The :mod:`quantify.scheduler` can be used to schedule operations on the control hardware.
The :mod:`quantify.scheduler` is designed to provide access to hardware functionality at a high-level (hardware agnostic) interface.


To understand how the scheduler works it is important to understand the basic concepts.
In the quantify scheduler, multiple :class:`~quantify.scheduler.types.Operation` s acting on :class:`~quantify.scheduler.Resource` s are added to a :class:`~quantify.scheduler.Schedule` which is then compiled for execution on a backend.
The :class:`~quantify.scheduler.Schedule` is responsible for scheduling *when* operations are applied, the :class:`~quantify.scheduler.types.Operation` is responsible for specifying *what* is applied, while the :class:`~quantify.scheduler.Resource` is responsible for *where* it is applied.

To schedule operations on control hardware different compilation steps take the schedule and compile it for a hardware backend.
These compilation steps depend on configuration files that describe how to translate gates into pulses, and pulses onto the control hardware. This is described in detail in :ref:`the section on compilation<Compilation>`.


.. list-table:: Overview of concepts and their representation at different levels of abstraction
   :widths: 25 25 25 25
   :header-rows: 0

   * -
     - Concept
     - Gate-level description
     - Pulse-level description
   * - When
     - :class:`~quantify.scheduler.Schedule`
     - --
     - --
   * - What
     - :class:`~quantify.scheduler.types.Operation`
     - unitaries and POVMs
     - parameterized waveforms
   * - Where
     - :class:`~quantify.scheduler.Resource`
     - qubits
     - ports & clocks


Schedules and Operations
--------------------------------

The :class:`~quantify.scheduler.Schedule` is a data structure that is at the core of the Quantify-scheduler.
The :class:`~quantify.scheduler.Schedule` contains information on *when* *what* operations should be performed.

The :class:`~quantify.scheduler.types.Operation` object is a datastructure that describes the operation that should be performed, it also contains information on *where* it should be applied.
An operation can be represented at different levels of abstraction such as the (quantum) :ref:`Gate-level description` and the :ref:`Pulse-level description`.
The :mod:`quantify.scheduler` comes with the  :mod:`quantify.scheduler.gate_library` and the :mod:`quantify.scheduler.pulse_library` , both containing common operations.


The :class:`~quantify.scheduler.Schedule` contains information on the :attr:`~quantify.scheduler.Schedule.operations`  and :attr:`~quantify.scheduler.Schedule.timing_constraints`.
:attr:`~quantify.scheduler.Schedule.operations` is a dictionary of all unique operations used in the schedule and contain the information on *what* operation to apply *where*.
:attr:`~quantify.scheduler.Schedule.timing_constraints`


When adding an :class:`~quantify.scheduler.types.Operation` to a :class:`~quantify.scheduler.Schedule` using the :meth:`~quantify.scheduler.Schedule.add` method, it is possible to specify precisely *when* to perform this operation.
However, at this point it is not required to specify how to respresent this :class:`~quantify.scheduler.types.Operation` on all (both gate and pulse) abstraction levels.
Instead, this information can be added later during :ref:`Compilation`.
This allows the user to effortlessly mix the gate- and pulse-level descriptions as is required for many calibration experiments.
An example of such an experiment is shown in :ref:`Tutorial 1. basic experiment`.


Gate- and Pulse-level description
-----------------------------------
A core feature of the :mod:`quantify.scheduler` is that it is possible to use both operations that are described at the gate level and operations that are described at the pulse level.
This is possible because the schedule describes *when* operations should be performed while the operations describe *what* should be done using what resources. The resources describe *where* the opartion is applied.
The description of *what* and *where* is different for the gate- and pulse-level descriptions.


Gate-level description
~~~~~~~~~~~~~~~~~~~~~~~~~
The (quantum) gate-level description is an idealized mathematical description of the operations.
In this describtion operations are `quantum gates <https://en.wikipedia.org/wiki/Quantum_logic_gate>`_  that act on idealized qubits as part of a `quantum circuit <https://en.wikipedia.org/wiki/Quantum_circuit>`_.
Operations can be represented by (idealized) unitaries acting on qubits which are represented here as strings (e.g., "q0", "q1", "qubit_left", etc.).
Qubits are a valid :class:`~quantify.scheduler.Resource`.
The :mod:`~quantify.scheduler.gate_library` contains common gates (including the measurement operation).


..
  TODO: qubit should be a valid resource. this needs to be correct and have an associated test.

.. note::
  Stricly speaking a measurement is not a gate as it cannot be described by a unitary. However, as it is a fundamental building block of circuit diagrams, we include it as this level of abstraction.

..
  This explanation is correct and very common, but possibly very confusing to the non-expert. Please help me make this a clearer explanation. - MAR



A :class:`~quantify.scheduler.Schedule` containing operations can be visualized using as a circuitdiagram using :func:`quantify.scheduler.visualization.circuit_diagram.circuit_diagram_matplotlib`.
An example of such a visualization is shown below.

.. jupyter-execute::
  :hide-code:

  from quantify.scheduler import Schedule
  from quantify.scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib
  from quantify.scheduler.gate_library import Reset, Measure, CZ, Rxy, X90

  sched = Schedule('Bell experiment')
  sched
  q0, q1 = ('q0', 'q1')

  sched.add(Reset(q0, q1))
  sched.add(Rxy(theta=90, phi=0, qubit=q0))
  sched.add(CZ(qC=q0, qT=q1))
  sched.add(Rxy(theta=23, phi=0, qubit=q0))
  sched.add(Measure(q0, q1))
  f, ax = circuit_diagram_matplotlib(sched)

To summarize:

- Gates are described by unitaries.
- Gates are applied to qubits.
- Qubit resources are represented by strings.



Pulse-level description
~~~~~~~~~~~~~~~~~~~~~~~~~
The pulse-level description describes waveforms applied to a sample.
These waveforms can be used to implement the unitaries of the gate-level description, in which case there is a one-to-one correspondence, but this is not required.
The pulse-level description typically contain parameterisation information, such as amplitudes, durations and so forth required to synthesise the waveform on control hardware.
The :mod:`~quantify.scheduler.pulse_library` contains a collection of commonly used pulses.
To specify *where* an operation is applied, the pulse-level description needs to specify both the location in physical space as well as in frequency space.
The location on chip is denoted by a *port* while the frequency is set using a *clock*, both are represented as strings.
These resources are described in detail in :ref:`the next section<Resources: Qubits, Ports and Clocks>`.

To summarize:

- Pulses are described as parameterized waveforms.
- Pulses are applied to *ports* at a frequency specified by a *clock*.
- Ports and clocks are represented by strings.


Resources: Qubits, Ports and Clocks
--------------------------------------

:mod:`quantify.scheduler.resources` denote where an opartion should be applied.
Here we explain these concept using a simple cQED device shown in :numref:`resources_fig` .
However, these concepts should be easy to generalize to other devices and systems.

At the gate-level description, operations are applied to (abstract) qubits.
For many systems, it is possible to associate a qubit with an element or location on a device.
However, qubits typically have many different ports that signals can be applied to.
A :class:`~quantify.scheduler.resources.PortResource` is used to indicate a location on a device that a signal can be applied to.
It can be associated with a qubit by including the qubit name in the name of the port.
This information can be used when visualizing a schedule and can be convenient to keep configuration files readable.
Besides the physical location on a device, a pulse is typically applied at a certain frequency.
A :class:`~quantify.scheduler.resources.ClockResource` can be used to track the phase of a certain transition or simply to ensure the signal ends up at the right frequency.
If the frequency of a clock is set to 0, the pulse is applied at baseband and is assumed to be real-valued.


.. list-table:: Operations and resources on different levels of abstraction
   :widths: 25 25 50
   :header-rows: 1

   * -
     - Gate-level description
     - Pulse-level description
   * - What
     - Unitaries and POVMs
     - Waveforms
   * - Where (space)
     - Qubits
     - Ports
   * - Where (frequency)
     - (implied)
     - Clocks


.. figure:: /images/Device_ports_clocks.svg
  :width: 800
  :name: resources_fig

  Resources are used to indicate *where* operations are applied.
  (a) Ports (purple) indicate a location on a device.
  By prefixing the name of a qubit in a port name a port can be associated with a qubit, but this is not required.
  (b) Clocks (blue) denote the location in frequency space and can be set to track the phase of a known transition. This can correspond to a qubit, but is not required.
  Device image from `Dickel (2018) <https://doi.org/10.4233/uuid:78155c28-3204-4130-a645-a47e89c46bc5>`_ .




Compilation
-------------

.. blockdiag::

    blockdiag scheduler {
      orientation = portrait

      qf_input [label="quantify API"];
      ext_input [label="Q A S M-like\nformats", stacked];
      hw_bck [label="Hardware\nbackends", stacked];
      gt_lvl [label="Gate-level"];

      ext_input -> qf_input;
      qf_input -> gt_lvl;
      qf_input -> Pulse-level;
      gt_lvl -> Pulse-level [label="Config"];
      Pulse-level -> hw_bck [label="Mapping"];
      group {
        label= "Input formats";
        qf_input
        ext_input
        color="#90EE90"
        }

      group {

        gt_lvl
        Pulse-level
        color=cyan
        label="Schedule"
        }

      group {
        label = "";
        color = orange;
        hw_bck
        }
    }




Device configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to compile from the idealized gate-level to the device specific pulse-level description.

* Resources vs params
* Amplitudes represent amplitudes at port!

Add example config file here and discuss it.


Hardware mapping file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to compile from the device-specific pulse-level description to control-hardware-specific instructions.

* JSON files that contains all instruments that can be handled by the scheduler backend
* Contains instrument settings
* Contains gain between port and instrument output
* Multiple “ports” can be linked to a single (complex) output.

Add example mapping file here and discuss it.


OLD OUTDATED STUFF
------------------------



A compilation step is a transformation of the :class:`~quantify.scheduler.Schedule` and results in a new :class:`~quantify.scheduler.Schedule`.
A compilation step can be used to e.g., add pulse information to operations containing only a gate-level representation or to determine the absolute timing based on timing constraints.
A final compilation step translates the :class:`~quantify.scheduler.Schedule` into a format compatible with the desired backend.

The following diagram provides an overview of how to interact with the :class:`~quantify.scheduler.Schedule` class.
The user can create a new schedule using the quantify API, or load a schedule based on one of the supported :mod:`~quantify.scheduler.frontends` for QASM-like formats such as qiskit QASM or OpenQL cQASM (todo).
One or multiple compilation steps modify the :class:`~quantify.scheduler.Schedule` until it contains the information required for the :mod:`~quantify.scheduler.visualization` used for visualization, simulation or compilation onto the hardware or back into a common QASM-like format.

.. blockdiag::

    blockdiag scheduler {

      qf_input [label="quantify API"];
      ext_input [label="Q A S M-like\nformats", stacked];
      vis_bck [label="Visualization \nbackends", stacked];
      hw_bck [label="Hardware\nbackends", stacked];
      sim_bck [label="Simulator\nbackends", stacked];
      ext_fmts [label="Q A S M-like\n formats", stacked];

      qf_input, ext_input -> Schedule;
      Schedule -> Schedule [label="Compile"];
      Schedule -> vis_bck;
      Schedule -> hw_bck;
      Schedule -> sim_bck ;
      Schedule -> ext_fmts;

      group {
        label= "Input formats";
        qf_input
        ext_input
        color="#90EE90"
        }

      group {

        Schedule
        color=red
        label="Compilation"
        }

      group {
        label = "Backends";
        color = orange;
        vis_bck, hw_bck, sim_bck, ext_fmts
        }
    }

