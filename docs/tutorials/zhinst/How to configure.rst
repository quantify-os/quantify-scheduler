.. _sec-zhinst-how-to-configure:

How to configure
================

The :mod:`~quantify_scheduler.backends.zhinst_backend` allows Zurich Instruments to be
configured individually or collectively by enabling master/slave configurations via 
Triggers and Markers.

Instruments can be configured by adding them to the :ref:`hardware configuration file<user-guide-example-zhinst-config>`.
The configuration file contains parameters about the Instruments and properties required 
to map :class:`~quantify_scheduler.types.Operation`\s, which act on qubits, onto physical
properties of the instrument.


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
  the name given to the QCodes Instrument during instantiation. 

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

    * Real-time modulation "modulate", UNTESTED!

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

.. code-block:: json
    :linenos:

    {
      "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
      "devices": [
        {
          "name": "hdawg0",
          "ref": "int",
          "channelgrouping": 0,
          "channel_0": {
            "port": "q0:mw",
            "clock": "q0.01",
            "mode": "complex",
            "modulation": "premod",
            "interm_freq": -50e6,
            "line_trigger_delay": 5e-9,
            "markers": [
              "AWG_MARKER1",
              "AWG_MARKER2"
            ]
          },
          "channel_1": {
            "port": "q1:mw",
            "clock": "q1.01",
            "mode": "complex",
            "modulation": "premod",
            "interm_freq": -50e6,
            "line_trigger_delay": 5e-9,
            "triggers": [
              1
            ]
          }
        },
      ]
    }

.. autoclass:: quantify_scheduler.backends.types.zhinst.Output
    :members:
    :noindex:
