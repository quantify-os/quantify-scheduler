.. _user_guide:



User guide (new)
================

.. jupyter-kernel::
  :id: Scheduler user guide new

.. jupyter-execute::
    :hide-code:

    # Make output easier to read
    from rich import pretty
    pretty.install()


.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Scheduler user guide`

    :jupyter-download:script:`Scheduler user guide`


Introduction
------------
Quantify-scheduler is a python module for writing (hybrid) quantum programs featuring a hybrid gate-pulse control model.
It extends the circuit model from quantum information processing by adding a pulse-level representation to operations defined at the gate-level, and the ability to specify timing constraints between operations.
Thus, a user is able to mix gate- and pulse-level operations in a quantum circuit.

.. todo::

    add paragraph on concepts and what a schedule is here.





Compilation
-----------

Different compilation steps are required to go from a high-level description of a schedule to something that can be executed on hardware.
The scheduler supports two main compilation steps, the first from the gate to the pulse level, and a second from the pulse-level to a hardware backend.
This is schematically shown in :numref:`compilation_overview`.

In the first compilation step, pulse information is added to all operations that are not valid pulses (:meth:`~quantify_scheduler.types.Operation.valid_pulse` ) based on the information specified in the :ref:`device configuration file<sec-device-config>`.

A second compilation step takes the schedule at the pulse level and translates this for use on a hardware backend.
This compilation step is performed using a hardware dependent compiler and uses the information specified in the :ref:`hardware configuration file<sec-hardware-config>`.

Both compilation steps can be triggered by passing a :class:`~quantify_scheduler.types.Schedule` and the appropriate configuration files to :func:`~quantify_scheduler.compilation.qcompile`.


.. figure:: /images/compilation_overview.svg
    :width: 800
    :name: compilation_overview

    A schematic overview of the different abstraction layers and the compilation process.


Execution
---------

In order to execute a schedule, one needs configuration files describing the system to compile the schedule, and physical instruments to execute the compiled instructions.
Within the Quantify framework, we use different kinds of :class:`~qcodes.instrument.base.Instrument`s to control the experiments and the management of the configuration files.
The different kinds of instruments and their responsibility in this framework is schematically shown in :numref:`instruments_overview`.


Different kinds of instruments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /images/instruments_overview.svg
    :width: 600
    :name: instruments_overview

    A schematic overview of the different kinds of instruments present in an experiment.
    Add links to objects in API ref.
    Add explanation of asterix.




The init script
~~~~~~~~~~~~~~~

Why single process.
How possible to
Basic import statements
Connecting to instruments
Loading settings




Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~



