Tutorials Zurich Instruments Backend
====================================


Introduction
^^^^^^^^^^^^

:mod:`quantify_scheduler` provides a stateless module: :mod:`~quantify_scheduler.backends.zhinst_backend`,
that abstracts the complexity of setting up experiments using `Zurich Instruments <https://www.zhinst.com>`_ hardware.
:mod:`quantify_scheduler` uses information on the quantum device
and instrument properties to compile a :class:`quantify_scheduler.schedules.schedule.Schedule` into waveforms and sequencing instructions suitable for execution on Zurich Instruments hardware.
More information about `compilation` can be found in the :ref:`User Guide <sec-user-guide>`.

Using existing programming interfaces provided via :doc:`zhinst-qcodes <zhinst-qcodes:index>` and :doc:`zhinst-toolkit <zhinst-toolkit:index>`, :mod:`quantify_scheduler` prepares the instruments that are present in the :ref:`sec-hardware configuration file <Hardware configuration file>`.

Finally, after configuring and running :func:`~quantify_scheduler.backends.zhinst_backend.compile_backend`
successfully the instruments are prepared for execution.

The Zurich Instruments backend provides:

- Synchronized execution of a program in a UHFQA and an HDAWG through the use of Triggers and Markers.
- Automatic generation of the SeqC Sequencer instructions.
- Waveform generation and modulation.
- Memory-efficient Sequencing with the CommandTable.
- Configuration of the relevant settings on all instruments.
- Configuration for Triggers and Markers.

Supported Instruments
^^^^^^^^^^^^^^^^^^^^^

The Zurich Instruments backend currently supports the **HDAWG** and the **UHFQA** instruments.
In addition to the Zurich Instruments devices, the hardware backend supports using microwave sources such as the R&S SGS100A.

Basic paradigm and inner workings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Zurich Instruments back end threats a :class:`~.Schedule` as a linear timeline of operations executed on the quantum device and translates this into pulses and acquisitions to be executed on the different input and output channels of an HDAWG and a UHFQA.
In this context a single HDAWG is treated as the master device that sends out a single marker pulse for synchronization at the start of each iteration of the schedule that is used to trigger all other devices.
After the synchronization trigger is given, all devices execute a compiled program for which the timings of all instructions have been calculated.

The compilation from operations at the quantum-device layer to instructions that the hardware can execute is done in several steps.
The zhinst :func:`~quantify_scheduler.backends.zhinst_backend.compile_backend` starts from the :attr:`.ScheduleBase.timing_table` and maps the operations to channels on the hardware using information specified in the
:ref:`sec-hardware configuration file <hardware configuration file>`.
Corrections for channel latency, as well as moving operations around to ensure all measurements start at the first sample (0) of a clock cycle are also done at this stage.

Once the starting time and sample of each operation is known, the numerical waveforms that have to be uploaded to the hardware can be generated.
The numerical waveforms differ from the idealized waveforms of the device description in that the they include corrections for effects such as mixer-skewness and linear-dynamical distortions (not implemented yet), and intermediate frequency modulation if required.

Both the :attr:`.CompiledSchedule.hardware_timing_table` and :attr:`.CompiledSchedule.hardware_waveform_dict` are available as properties of the :class:`.CompiledSchedule`.

In the next step, the clock-accurate Seqc instructions for each awg of each device are determined, as well as the settings (nodes) to be configured including the linking of the numerical waveforms to these nodes.
All of this information is combined in :class:`~.backends.zhinst.settings.ZISettingsBuilder`\s and added to the compiled instructions of the :class:`.CompiledSchedule`.


Limitations
^^^^^^^^^^^
There are several limitation to the paradigm and to the current implementation.
Some of these are relatively easy to address while others are more fundamental to the paradigm.
Here we give an overview of the known limitations.
Note that some of these can be quite specific.

Inherent limitations
~~~~~~~~~~~~~~~~~~~~
There are some inherent limitations to the paradigm of describing the program as a single linear timeline that is started using a single synchronization trigger.
These limitations cannot easily be addressed so should be taken into account when thinking about experiments.


- Because the **synchronization of the HDAWG and the UFHQA relies on a trigger on two devices operating at different clock frequencies** one cannot guarantee at what sample within the clock domain the slave device gets triggered. The consequence is that although the triggering is stable within an experiment, the exact time difference (in number of samples) between the different devices varies between different experiments. This problem is inherent to the triggering scheme and cannot be easily resolved.
- The paradigm of a single fixed timeline with a single synchronizing trigger is **not compatible with control loop affecting feedback**.

Limitations with the current implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are also some practical limitations to the implementation.
Keep these in mind when operating the hardware.

- **Real-time modulation is currently not supported**, relying on pre-modulated waveforms, it is important to start waveforms at a multiple of the modulation frequency. Sticking to a 10 ns grid, it is recommended to use a modulation frequency of 100 MHz.
- **All operations need to start at an integer number of samples**. Because of the choice of sampling rates of 2.4 GSps (~0.416 ns) and 1.8 GSps (~0.555 ns) it is useful to stick to a 10 ns grid for HDAWG (microwave and flux) pulses and a 40 ns grid for UHFQA (readout) pulses.
- **Different instructions on the same "awg" cannot start in the same clock cycle.** This implies that the readout acquisition delay cannot be 0 (but it can be 40 ns or - 40ns).
- **All measurements are triggered simultaneously** using the `StartQA(QA_INT_ALL, true)` instruction. This implies it is not possible to read out only a specific qubit/channel.
- **All measurements have to start at the same sample within a clock cycle** because one can only define a single integration weight per channel. To gaurantee this, all operations are shifted around a bit (the measurement fixpoint correction). As a consequence, the reset/initialization operation can sometimes be a bit longer than specified in the schedule.
- Because the **timing between two devices needs to align over longer schedules**, it is important that the clock-rates are accurate. To ensure phase stability, use a 10 MHz shared reference and operate the hardware in external reference mode.

- Only a single HDAWG supported as the primary device, other HDAWGs need to be configured as secondary devices.
- **Multiplexed readout is currently not supported**. One can only read out a single channel. (#191)

Hardware configuration
^^^^^^^^^^^^^^^^^^^^^^

.. note::

    This section will move to the documentation of the hardware configuration file themselves. See issue #222.

The :mod:`~quantify_scheduler.backends.zhinst_backend` allows Zurich Instruments to be
configured individually or collectively by enabling master/slave configurations via
Triggers and Markers.

Instruments can be configured by adding them to the :ref:`hardware configuration file<user-guide-example-zhinst-config>`.
The configuration file contains parameters about the Instruments and properties required
to map :class:`quantify_scheduler.operations.operation.Operation`\s, which act on
qubits, onto physical properties of the instrument.


The Zurich Instruments hardware configuration file is divided in two main sections.

1. The `backend` property defines the python method which will be executed by
:func:`~quantify_scheduler.compilation.qcompile` in order to compile the backend.

2. The `devices` property is an array of :class:`~quantify_scheduler.backends.types.zhinst.Device`.
A Device describes the type of Zurich Instruments and the physical setup.


.. code-block:: json
    :linenos:

    {
      "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
      "devices": [

      ]
    }


The entries in the `devices` section of the configuration file are strictly mapped
according to the :class:`~quantify_scheduler.backends.types.zhinst.Device` and
:class:`~quantify_scheduler.backends.types.zhinst.Output` domain models.

* In order for the backend to find the QCodes Instrument it is required that the
  :class:`~quantify_scheduler.backends.types.zhinst.Device`'s `name` must be equal to
  the name given to the QCodes Instrument during instantiation with an `ic` prepend.

    * Example: If the hdawg QCodes Instrument name is "hdawg_dev8831" then the :class:`~quantify_scheduler.backends.types.zhinst.Device`'s `name` is "ic_hdawg_dev8831"

* The `type` property defines the instrument's model. The :class:`~quantify_scheduler.backends.types.zhinst.DeviceType`
  is parsed from the string as well as the number of channels.

    * Example: "HDAWG8"

* The `ref` property describes if the instrument uses Markers (`int`), Triggers (`ext`) or `none`.

    * `int` Enables sending Marker

    * `ext` Enables waiting for Marker

    * `none` Ignores waiting for Marker


* The `channelgrouping` property sets the HDAWG channel grouping value and impacts the amount
  of HDAWG channels per AWG must be used.

.. code-block:: python
    :linenos:
    :emphasize-lines: 5,17

    {
      "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
      "devices": [
        {
          "name": "hdawg0",
          "ref": "int",
          "channelgrouping": 0,
          "channel_0": {
            ...
          },
        },
      ]
    }

    ...

    instrument = zhinst.qcodes.HDAWG(name='hdawg0', serial='dev1234', ...)


.. autoclass:: quantify_scheduler.backends.types.zhinst.Device
    :members:
    :noindex:

* The `channel_{0..3}` properties of the hardware configuration are mapped to
  the :class:`~quantify_scheduler.backends.types.zhinst.Output` domain model. A single
  `channel` represents a complex output, consisting of two physical I/O channels on
  the Instrument.

* The `port` and `clock` properties map the Operations to physical and frequency space.

* The `mode` property specifies the channel mode: real or complex.

* The `modulation` property specifies if the uploaded waveforms are modulated.
  The backend supports:

    * Premodulation "premod"

    * No modulation "none"

* The `interm_freq` property specifies the inter-modulation frequency.

* The `line_trigger_delay` property specifies a delay which is added to the
  sequencer program in the form of a `wait(n)`. The delay is used manually adjust
  the sequencer start offset in time due to the cabling delays.

* The `markers` property specifies which markers to trigger on each sequencer iteration.
  The values are used as input for the `setTrigger` sequencer instruction.

* The `triggers` property specifies for a sequencer which digital trigger to wait for.
  The first value of the `triggers` array is used as input parameter for the
  `waitDigTrigger` sequencer instruction.

.. autoclass:: quantify_scheduler.backends.types.zhinst.Output
    :members:
    :noindex:


Tutorials
^^^^^^^^^

.. toctree::
    :maxdepth: 3

    T_verification_programs.py
