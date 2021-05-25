.. _sec-user-guide:

User guide
==========

.. jupyter-kernel::
  :id: Scheduler user guide


Introduction
------------
Quantify-scheduler is a python module for writing (hybrid) quantum programs.
It extends the circuit model from quantum information processing by adding a pulse-level representation to operations defined at the gate-level, and the ability to specify timing constraints between operations. Thus, a user is able to mix gate- and pulse-level operations in a quantum circuit.

This module is then designed to fit both the needs of the quantum algorithm designer as well as experimentalists.
For the algorithm designer, this module allows better control over the pulse shapes sent to the QPU when performing a noisy variational algorithms, while for an experimentalist this combination of fine pulse-level control and the more abstract gate-level control allows them to easily define complex experiments.

Quantify-scheduler can be understood by understanding the following concepts.

- :ref:`Schedules <sec-schedule>`: describe when an operation needs to be applied.
- :ref:`Operations <sec-operation>`: describe what needs to be done.
- :ref:`Resources <sec-resources>`: describe where an operation should be applied.
- :ref:`Compilation <sec-compilation>`: between different abstraction layers and onto a hardware backend.

The following table shows an overview of the different concepts and how these are represented at the gate- and pulse-level abstraction.


.. list-table:: Overview of concepts and their representation at different levels of abstraction
   :widths: 25 25 25 25
   :header-rows: 0

   * -
     - Concept
     - Gate-level description
     - Pulse-level description
   * - When
     - :class:`~quantify.scheduler.types.Schedule`
     - --
     - --
   * - What
     - :class:`~quantify.scheduler.types.Operation`
     - unitaries and `POVMs <https://en.wikipedia.org/wiki/POVM>`_
     - parameterized waveforms
   * - Where
     - :class:`~quantify.scheduler.Resource`
     - qubits (:code:`str`)
     - ports (:code:`str`) & clocks  (:class:`~quantify.scheduler.resources.ClockResource`)



To schedule operations on control hardware different compilation steps take the schedule and compile it for a hardware backend.
The following block diagram gives an overview of the different compilation steps.
A schedule can be created using the quantify API (shown in :ref:`Tutorial 1 <sec-tutorial1>`).

.. blockdiag::
    :scale: 150

    blockdiag scheduler {
      orientation = portrait

      qf_input [label="quantify API"];
      hw_bck [label="Hardware\nbackends", stacked];
      gt_lvl [label="Gate-level"];

      qf_input -> gt_lvl;
      qf_input -> Pulse-level;
      gt_lvl -> Pulse-level [label="d. config", fontsize=8];
      Pulse-level -> hw_bck [label="h. config", fontsize=8];
      group {
        label= "Input formats";
        qf_input
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

A first :ref:`compilation <sec-compilation>` step uses the :ref:`device configuration (file)<sec-device-config>` to add a pulse representation to operations with a gate representation.
A second compilation step uses the :ref:`hardware configuration (file)<sec-hardware-config>` to compile the pulses onto a hardware backend.


.. note::

    We use the term "**device**" to refer to the physical object(s) on the receiving end of the control pulses, e.g. a thin-film chip inside a dilution refrigerator.

    And we employ the term "**hardware**" to refer to the instruments (electronics) that are involved in the pulse generations / signal digitization.





.. _sec-schedule:

Schedule
--------

The :class:`~quantify.scheduler.types.Schedule` is a data structure that is at the core of the Quantify-scheduler.
The :class:`~quantify.scheduler.types.Schedule` contains information on *when* operations should be performed.

When adding an :class:`~quantify.scheduler.types.Operation` to a :class:`~quantify.scheduler.types.Schedule` using the :meth:`~quantify.scheduler.types.Schedule.add` method, it is possible to specify precisely *when* to perform this operation using timing constraints.
However, at this point it is not required to specify how to represent this :class:`~quantify.scheduler.types.Operation` on all (both gate and pulse) abstraction levels.
Instead, this information can be added later during :ref:`Compilation`.
This allows the user to effortlessly mix the gate- and pulse-level descriptions as is required for many (calibration) experiments.
An example of such an experiment is shown in :ref:`Tutorial 1 <sec-tutorial1>`.


The :class:`~quantify.scheduler.types.Schedule` contains information on the :attr:`~quantify.scheduler.types.Schedule.operations` and :attr:`~quantify.scheduler.types.Schedule.timing_constraints`.
The :attr:`~quantify.scheduler.types.Schedule.operations` is a dictionary of all unique operations used in the schedule and contain the information on *what* operation to apply *where*.
The :attr:`~quantify.scheduler.types.Schedule.timing_constraints` is a list of dictionaries describing timing constraints between operations, i.e. when to apply an operation.


.. _sec-operation:

Operation
---------


The :class:`~quantify.scheduler.types.Operation` object is a datastructure that describes the operation that should be performed, it also contains information on *where* it should be applied.
An operation can be represented at different levels of abstraction such as the (quantum) :ref:`Gate-level description` and the :ref:`Pulse-level description`.
The :mod:`quantify.scheduler` comes with the  :mod:`quantify.scheduler.gate_library` and the :mod:`quantify.scheduler.pulse_library` , both containing common operations.


Gate-level description
~~~~~~~~~~~~~~~~~~~~~~
The (quantum) gate-level description is an idealized mathematical description of the operations.
In this description operations are `quantum gates <https://en.wikipedia.org/wiki/Quantum_logic_gate>`_  that act on idealized qubits as part of a `quantum circuit <https://en.wikipedia.org/wiki/Quantum_circuit>`_.
Operations can be represented by (idealized) unitaries acting on qubits which are represented here as strings (e.g., :code:`"q0"`, :code:`"q1"`, :code:`"qubit_left"`, etc.).
Valid qubits are strings that appear in the :ref:`device configuration file<Device configuration file>` used when compiling the schedule.
The :mod:`~quantify.scheduler.gate_library` contains common gates (including the measurement operation).


.. note::
  Strictly speaking a measurement is not a gate as it cannot be described by a unitary. However, it is a fundamental building block of circuit diagrams and therefore included at this level of abstraction.


A :class:`~quantify.scheduler.types.Schedule` containing operations can be visualized using as a circuit diagram using :func:`quantify.scheduler.visualization.circuit_diagram.circuit_diagram_matplotlib`.
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
  sched.add(Measure(q0, q1, acq_index=(0, 1)))
  f, ax = circuit_diagram_matplotlib(sched)

To summarize:

- Gates are described by unitaries.
- Gates are applied to qubits.
- Qubit resources are represented by strings.



Pulse-level description
~~~~~~~~~~~~~~~~~~~~~~~

The pulse-level description describes waveforms applied to a sample.
These waveforms can be used to implement the unitaries of the gate-level description, in which case there is a one-to-one correspondence, but this is not required.
The pulse-level description typically contain parameterization information, such as amplitudes, durations and so forth required to synthesize the waveform on control hardware.
The :mod:`~quantify.scheduler.pulse_library` contains a collection of commonly used pulses.
To specify *where* an operation is applied, the pulse-level description needs to specify both the location in physical space as well as in frequency space.
The location on chip is denoted by a *port* while the frequency is set using a *clock*, both are represented as strings.
These resources are described in detail in :ref:`the next section<sec-resources>`.

A :class:`~quantify.scheduler.types.Schedule` containing operations can be visualized using as a pulse diagram using :func:`quantify.scheduler.visualization.pulse_scheme.pulse_diagram_plotly`.
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

.. _sec-resources:

Resources
---------

Resources denote where an operation should be applied.
Here we explain these concept using a simple cQED device shown in :numref:`resources_fig`.
These concepts should be easy to generalize to other devices and systems.

.. figure:: /images/Device_ports_clocks.svg
  :width: 800
  :name: resources_fig

  Resources are used to indicate *where* operations are applied.
  (a) Ports (purple) indicate a location on a device.
  By prefixing the name of a qubit in a port name (separated by a colon :code:`:`) a port can be associated with a qubit (red), but this is not required.
  (b) Clocks (blue) denote the location in frequency space and can be set to track the phase of a known transition.
  By prefixing the name of a qubit in a clock name (separated by a colon :code:`:`) a clock can be associated with a qubit (red), but this is not required.
  Device image from `Dickel (2018) <https://doi.org/10.4233/uuid:78155c28-3204-4130-a645-a47e89c46bc5>`_ .


Qubits
~~~~~~

At the gate-level description, operations are applied to (abstract) qubits.
Qubits are represented by strings corresponding to the name of a qubit (e.g., :code:`q0`, :code:`q1`, :code:`A1`, :code:`QL`, :code:`qubit_1`, etc.).
Valid qubit names are those that appear in the :ref:`device config<sec-device-config>` used for compilation.

Ports
~~~~~

For many systems, it is possible to associate a qubit with an element or location on a device that a signal can be applied to.
We call such a location on a device a port.
Like qubits, ports are represented as strings (e.g., :code:`P0`, :code:`feedline_in`, :code:`q0:mw_drive`, etc.).
A port can be associated with a qubit by including the qubit name in the name of the port (separated by a colon :code:`:`).
Valid qubit names are those that appear in the :ref:`hardware configuration file<sec-hardware-config>` used for compilation.


Associating a qubit can be useful when visualizing a schedule and or to keep configuration files readable.
Associating a port with a single qubit is not required so as not to complicate matters when ports are associated with multiple qubits or with non-qubit elements such as tunable couplers.

Clocks
~~~~~~

Besides the physical location on a device, a pulse is typically applied at a certain frequency.
A :class:`~quantify.scheduler.resources.ClockResource` can be used to track the phase of a certain transition or simply to ensure the signal ends up at the right frequency.
Similar to ports, clocks can be associated with qubits by including it in the name, but this is not required to account for non-qubit elements.
If the frequency of a clock is set to 0 (zero), the pulse is applied at baseband and is assumed to be real-valued.

.. _sec-compilation:

Compilation
-----------

Different compilation steps are required to go from a high-level description of a schedule to something that can be executed on hardware.
The scheduler supports two main compilation steps, the first from the gate to the pulse level, and a second from the pulse-level to a hardware backend.

In the first compilation step, pulse information is added to all operations that are not valid pulses (:meth:`~quantify.scheduler.types.Operation.valid_pulse` ) based on the information specified in the :ref:`device configuration file<sec-device-config>`.

A second compilation step takes the schedule at the pulse level and translates this for use on a hardware backend.
This compilation step is performed using a hardware dependent compiler and uses the information specified in the :ref:`hardware configuration file<sec-hardware-config>`.

Both compilation steps can be triggered by passing a :class:`~quantify.scheduler.types.Schedule` and the appropriate configuration files to :func:`~quantify.scheduler.compilation.qcompile`.


.. _sec-device-config:

Device configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~

The device configuration file is used to compile from the  gate-level to the device specific pulse-level description.
The main responsibility is to add a pulse-representation to every operation that has a gate-level description.
To do this, it contains information for all qubits, and all edges.
Edges are pairs of qubits (separated by a dash :code:`-`) on which gates can act.
The specified "backend" determines how the data for each qubit is used to create pulses.


A valid device configuration is described by the schema shown here:

.. jsonschema:: ../quantify/scheduler/schemas/transmon_cfg.json


Example device configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here we show an example of such a device configuration file:

.. jupyter-execute::
  :hide-code:

  from pathlib import Path
  import json
  import quantify.scheduler.schemas.examples as examples

  path = Path(examples.__file__).parent.joinpath('transmon_test_config.json')
  json_data = json.loads(path.read_text())
  print(json.dumps(json_data, indent=4, sort_keys=True))

.. _sec-hardware-config:

Hardware configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hardware configuration file is used to compile pulses to specific control electronics.
To do this, it contains information on what ports are connected to what hardware outputs/inputs, as well as other hardware-specific settings.
The backend key of the hardware configuration specifies what backend is used to compile onto the control electronics.
Here we show an example of such a device configuration file:

Example Qblox hardware configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::
  :hide-code:

  from pathlib import Path
  import json
  import quantify.scheduler.schemas.examples as examples

  path = Path(examples.__file__).parent.joinpath('qblox_test_mapping.json')
  json_data = json.loads(path.read_text())
  print(json.dumps(json_data, indent=4, sort_keys=True))


Example Zurich Instruments hardware configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::
  :hide-code:

  from pathlib import Path
  import json
  import quantify.scheduler.schemas.examples as examples

  path = Path(examples.__file__).parent.joinpath('zhinst_test_mapping.json')
  json_data = json.loads(path.read_text())
  print(json.dumps(json_data, indent=4, sort_keys=True))

