.. _sec-qblox-pulsar:

Pulsar QCM/QRM
==============

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # in the hidden cells we include some code that checks for correctness of the examples
    from tempfile import TemporaryDirectory

    from quantify_scheduler.operations import pulse_library
    from quantify_scheduler.compilation import determine_absolute_timing
    from quantify_scheduler.backends.qblox_backend import hardware_compile
    from quantify_scheduler import Schedule
    from quantify_scheduler.resources import ClockResource

    from quantify_core.data.handling import set_datadir

    temp_dir = TemporaryDirectory()
    set_datadir(temp_dir.name)

Each device in the setup can be individually configured using the entry in the config. For instance:

.. jupyter-execute::
    :hide-output:
    :linenos:

    mapping_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_name": "lo0",
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6
                }
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "lo_name": "lo1",
                "seq1": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "interm_freq": None
                }
            }
        },
        "lo0": {"instrument_type": "LocalOscillator", "lo_freq": None, "power": 20},
        "lo1": {"instrument_type": "LocalOscillator", "lo_freq": 7.2e9, "power": 20}
    }

.. jupyter-execute::
    :hide-code:
    :hide-output:

    test_sched = Schedule("test_sched")
    test_sched.add(
        pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
    )
    test_sched.add_resource(ClockResource(name="q0.01", freq=7e9))
    test_sched = determine_absolute_timing(test_sched)

    hardware_compile(test_sched, mapping_config)

Here we specify a setup containing only a `Pulsar QCM <https://www.qblox.com/pulsar>`_, with both outputs connected to a local oscillator sources.

The first few entries in the dictionary contain settings and information for the entire device.
:code:`"type": "Pulsar_QCM"` specifies that this device is a `Pulsar QCM <https://www.qblox.com/pulsar>`_,
and :code:`"ref": "internal"` sets the reference source to internal (as opposed to :code:`"external"`). Under the entries
:code:`complex_output_0` (corresponding to O\ :sup:`1/2`) and :code:`complex_output_1` (corresponding to O\ :sup:`3/4`),
we set all the parameters that are configurable per output.

The examples given below will be for a single Pulsar QCM, but the other devices can be configured similarly. In order to use a Pulsar QRM, QCM-RF or QRM-RF, change the :code:`"instrument_type"` entry to :code:`"Pulsar_QRM"`, :code:`"Pulsar_QCM_RF"` or :code:`"Pulsar_QRM_RF"`
respectively. Multiple devices can be added to the config, similar to how we added the local oscillators in the example given above.

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
the backend by specifying it as :code:`None`. Specifying both will raise an error if it violates :math:`f_{RF} = f_{IF} + f_{LO}`.

Mixer corrections
^^^^^^^^^^^^^^^^^

The backend also supports setting the parameters that are used by the hardware to correct for mixer imperfections in real-time.

We configure this by adding the lines

.. code-block:: python
    :linenos:

    "dc_mixer_offset_I": -0.054,
    "dc_mixer_offset_Q": -0.034,

to :code:`complex_output_0` (or :code:`complex_output_1`) in order to add a DC offset to the outputs to correct for feed-through of the local oscillator signal. And we add

.. code-block:: python
    :linenos:

    "mixer_amp_ratio": 0.9997,
    "mixer_phase_error_deg": -4.0,

To the sequencer configuration in order to correct to set the amplitude and phase correction to correct for imperfect rejection of the unwanted sideband.

Usage without an LO
^^^^^^^^^^^^^^^^^^^

In order to use the backend without an LO, we simply remove the :code:`"lo_name"` and all other related parameters. This includes the
mixer correction parameters as well as the frequencies.

.. jupyter-execute::
    :hide-output:
    :linenos:

    mapping_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "line_gain_db": 0,
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                }
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "seq1": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                }
            }
        },
    }

.. jupyter-execute::
    :hide-output:
    :hide-code:

    hardware_compile(test_sched, mapping_config)

Frequency multiplexing
^^^^^^^^^^^^^^^^^^^^^^

It is possible to do frequency multiplexing of the signals by adding multiple sequencers to the same output.

.. jupyter-execute::
    :hide-output:
    :linenos:

    mapping_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "line_gain_db": 0,
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                },
                "seq1": {
                    "port": "q0:mw",
                    "clock": "some_other_clock",
                }
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "seq2": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                }
            }
        },
    }

.. jupyter-execute::
    :hide-output:
    :hide-code:

    test_sched = Schedule("test_sched")
    test_sched.add(
        pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
    )
    test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))
    test_sched.add_resource(ClockResource(name="some_other_clock", freq=100e6))

    test_sched = determine_absolute_timing(test_sched)

    hardware_compile(test_sched, mapping_config)

In the given example, we added a second sequencer to output 0. Now any signal on port :code:`"q0:mw"` with clock :code:`"some_other_clock"` will be added digitally to the signal with the same port but clock :code:`"q0.01"`. The Qblox modules currently have six sequencers available, which sets the upper limit to our multiplexing capabilities.

.. note::

    We note that it is a requirement of the backend that each combination of a port and a clock is unique, i.e. it is possible to use the same port or clock multiple times in the hardware config but the combination of a port with a certain clock can only occur once.

Real mode
^^^^^^^^^

For the baseband modules, it is also possible to use the backend to generate signals for the outputs individually rather than using IQ pairs.

In order to do this, instead of :code:`"complex_output_X"`, we use :code:`"real_output_X"`. In case of a QCM, we have four of those outputs. The QRM has two available.

The resulting config looks like:

.. jupyter-execute::
    :hide-output:
    :linenos:

    mapping_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "real_output_0": {
                "line_gain_db": 0,
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                }
            },
            "real_output_1": {
                "line_gain_db": 0,
                "seq1": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                }
            },
            "real_output_2": {
                "line_gain_db": 0,
                "seq2": {
                    "port": "q2:mw",
                    "clock": "q2.01",
                }
            }
        },
    }

.. jupyter-execute::
    :hide-code:
    :hide-output:

    test_sched = Schedule("test_sched")
    test_sched.add(
        pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
    )
    test_sched.add(
        pulse_library.SquarePulse(amp=1, duration=1e-6, port="q1:mw", clock="q1.01")
    )
    test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))
    test_sched.add_resource(ClockResource(name="q1.01", freq=100e6))

    test_sched = determine_absolute_timing(test_sched)

    hardware_compile(test_sched, mapping_config)

When using real outputs, the backend automatically maps the signals to the correct output paths. We note that for real outputs, it is not allowed to use any pulses that have an imaginary component i.e. only real valued pulses are allowed. If you were to use a complex pulse, the backend will produce an error, e.g. square and ramp pulses are allowed but DRAG pulses not.

.. warning::

    When using real mode, we highly recommend using it in combination with the instrument coordinator as the outputs need to be configured correctly in order for this to function.

.. jupyter-execute::
    :hide-code:
    :hide-output:
    :raises: ValueError

    test_sched.add(
        pulse_library.DRAGPulse(
            G_amp=1, D_amp=1, duration=1e-6, port="q1:mw", clock="q1.01", phase=0
        )
    )

    test_sched = determine_absolute_timing(test_sched)

    hardware_compile(test_sched, mapping_config)


Experimental features
^^^^^^^^^^^^^^^^^^^^^

The Qblox backend contains some intelligence that allows it to generate certain specific waveforms from the pulse library using a more complicated series of sequencer instructions, which helps conserve waveform memory. Though in order to keep the backend fully transparent, all such advanced capabilities are disabled by default.

In order to enable the advanced capabilities we need to add line :code:`"instruction_generated_pulses_enabled": True` to the sequencer configuration.

.. jupyter-execute::
    :hide-output:
    :linenos:

    mapping_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "line_gain_db": 0,
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "instruction_generated_pulses_enabled": True
                }
            },
        },
    }

.. jupyter-execute::
    :hide-code:
    :hide-output:

    test_sched = Schedule("test_sched")
    test_sched.add(
        pulse_library.SquarePulse(amp=1, duration=1e-3, port="q0:mw", clock="q0.01")
    )

    test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))

    test_sched = determine_absolute_timing(test_sched)

    hardware_compile(test_sched, mapping_config)

Currently this has the following effects:

- Long square pulses get broken up into separate pulses with durations <= 1 us, which allows the modules to play square pulses longer than the waveform memory normally allows.
- Staircase pulses are generated using offset instructions instead of using waveform memory
