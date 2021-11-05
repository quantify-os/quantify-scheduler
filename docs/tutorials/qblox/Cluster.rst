.. _sec-qblox-cluster:

Cluster
=======

.. jupyter-execute::
    :hide-code:

    # in the hidden cells we include some code that checks for correctness of the examples
    from tempfile import TemporaryDirectory

    from quantify_scheduler import Schedule
    from quantify_scheduler.operations import pulse_library
    from quantify_scheduler.compilation import determine_absolute_timing
    from quantify_scheduler.backends.qblox_backend import hardware_compile
    from quantify_scheduler.resources import ClockResource

    from quantify_core.data.handling import set_datadir

    temp_dir = TemporaryDirectory()
    set_datadir(temp_dir.name)

In the previous sections we explained how to configure the backend for use with the standalone `Pulsars <https://www.qblox.com/pulsar>`_, now we will explain how to adapt this config
to use one or multiple `Clusters <https://www.qblox.com/cluster>`_ instead.
Since the cluster modules behave similarly, we recommend first familiarizing yourself with the configuration for the :doc:`pulsars <Pulsar>`.

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
        "lo0": {"instrument_type": "LocalOscillator", "lo_freq": None, "power": 20},
    }

.. jupyter-execute::
    :hide-code:
    :hide-output:

    test_sched = Schedule("test_sched")
    test_sched.add(
        pulse_library.SquarePulse(amp=1, duration=1e-6, port="q4:mw", clock="q4.01")
    )
    test_sched.add(
        pulse_library.SquarePulse(amp=0.25, duration=1e-6, port="q5:mw", clock="q5.01")
    )
    test_sched.add_resource(ClockResource(name="q4.01", freq=7e9))
    test_sched.add_resource(ClockResource(name="q5.01", freq=8e9))
    test_sched = determine_absolute_timing(test_sched)

    hardware_compile(test_sched, mapping_config)

In the example, we notice that the cluster is specified using an instrument with :code:`"instrument_type": "Cluster"`. In the backend, the cluster instrument functions as a collection of
modules. The modules themselves can be configured identically to pulsars, except for the :code:`ref`, which has now become a cluster wide setting.

Valid values for :code:`"instrument_type"` for the modules are: :code:`QCM`, :code:`QRM`, :code:`QCM_RF` and :code:`QRM_RF`.
