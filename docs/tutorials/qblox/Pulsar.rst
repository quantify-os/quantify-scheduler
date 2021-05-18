.. _sec-qblox-pulsar:

Pulsar QCM/QRM
==============

Each device in the setup can be individually configured using the entry in the config. For instance:

.. jupyter-execute::
    :linenos:

    mapping_config = {
        "backend": "quantify.scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "type": "Pulsar_QCM",
            "mode": "complex",
            "ref": "int",
            "IP address": "192.168.0.2",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "lo0",
                "lo_freq": None,
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6
                }
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "lo_name": "lo1",
                "lo_freq": 7.2e9,
                "mixer_corrections": {
                    "amp_ratio": 0.9,
                    "phase_error": 7,
                    "offset_I": 0.001,
                    "offset_Q": -0.03
                },
                "seq1": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "interm_freq": None
                }
            }
        },
    }

Here we specify a setup containing only a `Pulsar QCM <https://www.qblox.com/pulsar>`_.

The first few entries in the dictionary contain settings and information for the entire device.
:code:`"type": "Pulsar_QCM"` specifies that this device is a `Pulsar QCM <https://www.qblox.com/pulsar>`_,
and :code:`"ref": "int"` sets the reference source to internal (as opposed to :code:`"ext"`). Under the entries
:code:`complex_output_0` (corresponding to O\ :sup:`1/2`) and :code:`complex_output_1` (corresponding to O\ :sup:`3/4`),
we set all the parameters that are configurable per output.

Output settings
^^^^^^^^^^^^^^^

Most notably under the :code:`complex_output_0`, we specify the sequencer settings.

.. code-block:: python
    :linenos:

    "seq0": {
        "port": "q0:mw",
        "clock": "q0.01",
        "interm_freq": 50e6
    }

Here we describe which port and clock the sequencer is associated with (see the :ref:`User guide <sec-user-guide>`
for more information on the role of ports and clocks within the Quantify-Scheduler). The other entry, :code:`interm_freq`,
specifies the intermediate frequency to use for I/Q modulation (in Hz).

I/Q modulation
^^^^^^^^^^^^^^

To perform upconversion using an I/Q mixer and an external local oscillator, simply specify a local oscillator in the config using the :code:`lo_name` entry.
:code:`complex_output_0` is connected to a local oscillator instrument named
:code:`lo0` and :code:`complex_output_1` to :code:`lo1`.
Since the Quantify-Scheduler aim is to only specify the final RF frequency when the signal arrives at the chip, rather than any parameters related to I/Q modulation, we specify this information here.

The backend assumes that upconversion happens according to the relation

.. math::

    f_{RF} = f_{IF} + f_{LO}

This means that in order to generate a certain :math:`f_{RF}`, we need to specify either an IF or an LO frequency. In the
dictionary, we therefore either set the :code:`lo_freq` or the :code:`interm_freq` and leave the other to be calculated by
the backend by specifying it as :code:`None`.

Note that the backend also supports correcting for mixer imperfections by digitally pre-distorting the waveforms. This is done
by setting the offsets on the I and Q output paths and correcting the amplitude and phase (specified in degrees) of the signals through
:func:`~quantify.scheduler.helpers.waveforms.apply_mixer_skewness_corrections`. The correction parameters need to be specified
in :code:`mixer_corrections` as done for :code:`complex_output_1` in the example config.

.. code-block:: python
    :linenos:

    "mixer_corrections": {
        "amp_ratio": 0.9,
        "phase_error": 7,
        "offset_I": 0.001,
        "offset_Q": -0.03
    }
