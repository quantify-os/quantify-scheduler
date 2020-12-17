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
     - qubits (str)
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
Valid qubits are strings that appear in the device_config.json file.
Qubits are a valid :class:`~quantify.scheduler.Resource`.
The :mod:`~quantify.scheduler.gate_library` contains common gates (including the measurement operation).


.. note::
  Stricly speaking a measurement is not a gate as it cannot be described by a unitary. However, as it is a fundamental building block of circuit diagrams, we include it at this level of abstraction.

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
  sched.add(Rxy(theta=90, phi=0, qubit=q1), ref_pt='start')
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

A :class:`~quantify.scheduler.Schedule` containing operations can be visualized using as a pulsediagram using :func:`quantify.scheduler.visualization.circuit_diagram.pulse_diagram_plotly`.
An example of such a visualization is shown below:


.. jupyter-execute::
  :hide-code:


  import json
  import pprint
  import os, inspect
  from quantify.scheduler.compilation import add_pulse_information_transmon, determine_absolute_timing
  from quantify.scheduler.visualization.pulse_scheme import pulse_diagram_plotly


  import quantify.scheduler.schemas.examples as es

  esp = inspect.getfile(es)
  cfg_f = os.path.abspath(os.path.join(esp, '..', 'transmon_test_config.json'))


  with open(cfg_f, 'r') as f:
      transmon_test_config = json.load(f)


  add_pulse_information_transmon(sched, device_cfg=transmon_test_config)
  determine_absolute_timing(schedule=sched)
  pulse_diagram_plotly(sched, port_list=["q0:mw", "q1:mw", "q0:fl", "q1:fl", "q0:res" ], modulation_if = 10e6, sampling_rate = 1e9)

In this visualization, the different rows correspond to different ports to which the pulses are applied, the clocks are used to modulate the respective signals, and time is shown on the x-axis.


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
Associating a port with a single qubit is not required so as not to complicate matters when ports are associated with multiple qubits or with non-qubit elements such as tunable couplers.

Besides the physical location on a device, a pulse is typically applied at a certain frequency.
A :class:`~quantify.scheduler.resources.ClockResource` can be used to track the phase of a certain transition or simply to ensure the signal ends up at the right frequency.
Similar to ports, clocks can be associated with qubits by including it in the name, but this is not required to account for non-qubit elements.
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

Different compilation steps are required to go from a high-level description of a schedule to something that can be executed on physical hardware. The scheduler currently supports two main compilation steps, the first from the gate to the pulse level, and a second from the pulse-level to a hardware backend.

In the first compilation step, pulse information is added to all operations that are not valid pulses (:meth:`~quantify.scheduler.types.Operation.valid_pulse` ) based on the information specified in the :ref:`configuration file<Device configuration file>`.

A second compilation step takes the schedule at the pulse level and translates this for use on a hardware backend.
This compilation step is performed using a hardware dependent compiler and uses the information specified in the :ref:`mapping file<Hardware mapping file>`.

The block diagram below shows an overview of the different compilation steps.
The :mod:`quantify.scheduler.compilation` contains the main compilation functions.

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

The device configuration file is used to compile from the idealized gate-level to the device specific pulse-level description.

Here we show an example of such a device configuration file.

.. todo::

  Add schema for schedule here.
  Describe what functionality the backend needs to provide and what fields must be there.


.. jupyter-execute::

  :hide-code:

  import json
  import pprint
  import os, inspect
  import quantify.scheduler.schemas.examples as es

  esp = inspect.getfile(es)
  cfg_f = os.path.abspath(os.path.join(esp, '..', 'transmon_test_config.json'))


  with open(cfg_f, 'r') as f:
      transmon_test_config = json.load(f)

  pprint.pprint(transmon_test_config)

.. todo::

  import code snippet from the test device config used in the
  examples + add test.

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


.. todo::

  import code snippet from the test mapping file used in the
  examples + add test.
