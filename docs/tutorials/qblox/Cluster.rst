.. _sec-qblox-cluster:

Cluster
=======

In the previous sections we explained how to configure the backend for use with the standalone `Pulsars <https://www.qblox.com/pulsar>`_, now we will explain how to adapt this config
to use one or multiple `Clusters <https://www.qblox.com/cluster>`_ instead.
Since the cluster modules behave similarly, we recommend first familiarizing yourself with the configuration for the :ref:`pulsars <_sec-qblox-pulsar>`.

We start by looking at an example config for a single cluster:

.. jupyter-execute::
    :hide-output:
    :linenos:

    mapping_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "cl_qcm0": {
                "complex_output_0": {
                    "line_gain_db": 0,
                    "lo_name": "lo0",
                    "seq0": {
                        "clock": "q4.01",
                        "interm_freq": 200000000.0,
                        "mixer_amp_ratio": 0.9999,
                        "mixer_phase_error_deg": -4.2,
                        "port": "q4:mw",
                    },
                },
                "instrument_type": "QCM",
            },
            "cl_qcm_rf0": {
                "complex_output_0": {
                    "line_gain_db": 0,
                    "seq0": {"clock": "q5.01", "interm_freq": 50000000.0, "port": "q5:mw"},
                },
                "instrument_type": "QCM_RF",
            },
            "instrument_type": "Cluster",
            "ref": "internal",
        },
    }
