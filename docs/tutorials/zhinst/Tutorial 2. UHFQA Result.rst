.. _sec-zhinst-2:

Tutorial 2. UHFQA Result
=========================

This tutorial describes how to use quantify-schedule to generate pulses and acquire result data,
also visualized in the Quantum Analyzer Result tab of LabOne, using the UHFQA's AWG.

For this tutorial lets use :func:`~quantify.scheduler.schedules.acquisition.ssb_integration_complex_schedule` to create a
pulse level Schedule. This utility function is used for debugging :class:`~quantify.scheduler.acquisition_library.SSBIntegrationComplex`
acquisition with pulses of a large fixed duration.

Requirements
^^^^^^^^^^^^

- Create a loopback by connecting the UHFQA output- to the input channel.

.. code-block:: python
    :linenos:

    from typing import Dict, Any, Callable
    import logging
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from zhinst.qcodes import UHFQA

    from quantify.scheduler.schedules.acquisition import ssb_integration_complex_schedule
    from quantify.scheduler.compilation import qcompile
    import quantify.scheduler.backends.zhinst_backend as zhinst_backend

    # Debug only
    # logging.getLogger().setLevel(logging.DEBUG)

.. code-block:: python
    :linenos:

    # Create a schedule using the `SSBIntegrationComplex` acquisition protocol
    # The `ssb_integration_complex_schedule` is a Schedule defined on pulse-level.
    # This schedule should only be used for testing UHFQA output.
    schedule = ssb_integration_complex_schedule(
        port="q0:res",
        clock="q0.ro",
        integration_time=1e-6,
        spec_pulse_amp=1,
        frequency=7.04e9
    )
    schedule.repetitions = 1

.. code-block:: python
    :linenos:
    :emphasize-lines: 24

    def load_example_json_scheme(filename: str) -> Dict[str, Any]:
        import quantify.scheduler.schemas.examples as es
        import os, inspect
        import json

        examples_path:str = inspect.getfile(es)
        config_file_path = os.path.abspath(os.path.join(examples_path, '..', filename))

        return json.loads(Path(config_file_path).read_text())

    # Load example configuration from quantify.scheduler.schemas.examples
    device_config_map = (load_example_json_scheme('transmon_test_config.json'))

    # Set the UHFQA `ref` trigger(Reference source trigger) option to "none"
    # The UHFQA is not controlling any marker or triggers
    zhinst_hardware_map: Dict[str, Any] = json.loads(
    """
    {
      "backend": "quantify.scheduler.backends.zhinst_backend.create_pulsar_backend",
      "devices": [
        {
          "name": "uhfqa0",
          "ref": "none",
          "channel_0": {
            "port": "q0:res",
            "clock": "q0.ro",
            "mode": "real",
            "lo_freq": 4.8e9,
            "interm_freq": -50e6
          }
        }
      ]
    }
    """
    )

.. code-block:: python
    :linenos:

    # Compile schedule with configurations
    schedule = qcompile(schedule, device_config_map, zhinst_hardware_map)

.. code-block:: python
    :linenos:

    # Instantiate ZI Instruments
    # Note that the device name in the hardware map must match the Instrument name.
    # for example: uhfqa0 or hdawg0
    uhfqa = UHFQA('uhfqa0', 'dev2299', host='localhost', interface='1GbE')

.. code-block:: python
    :linenos:

    # Run the backend setup
    acq_channel_resolvers_map = zhinst_backend.setup_zhinst_backend(schedule, zhinst_hardware_map)

.. code-block:: python
    :linenos:

    # Arm the UHFQA Quantum Analyzer Results unit
    uhfqa.arm(length=schedule.repetitions, averages=1)

    # Run the UHFQA AWG
    uhfqa.awg.run()
    uhfqa.awg.wait_done()

    # Resolve the results by querying the UHFQA monitor nodes
    acq_channel_results: Dict[int, Callable[..., Any]] = dict()
    for acq_channel, resolve in acq_channel_resolvers_map.items():
        acq_channel_results[acq_channel] = resolve()

.. code-block:: python
    :linenos:

    # Plot acquisition results
    labels = []
    for i, result in acq_channel_results.items():
        labels.append(f"acq_channel #{i} complex")
        real_vals = [val.real for val in result]
        imag_vals = [val.imag for val in result]

        print(result)

        plt.scatter(real_vals, imag_vals)

    plt.legend(labels)
