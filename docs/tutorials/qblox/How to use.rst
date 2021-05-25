.. _sec-qblox-how-to-configure:

Usage of the backend
====================

Configuring the backend is done by specifying a python dictionary (or loading it from a JSON file)
that describes your experimental setup. An example of such a config:

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

Here the entry :code:`"backend": "quantify.scheduler.backends.qblox_backend.hardware_compile"` specifies to the scheduler
that we are using the Qblox backend (specifically the :func:`~quantify.scheduler.backends.qblox_backend.hardware_compile` function).

Apart from the :code:`"backend"`, each entry in the dictionary corresponds to a device connected to the setup. In the example above, only a
:ref:`Pulsar QCM <sec-qblox-pulsar>` named :code:`"qcm0"` is specified.
