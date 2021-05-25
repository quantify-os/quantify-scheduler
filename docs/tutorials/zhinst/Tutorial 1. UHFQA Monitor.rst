.. _sec-zhinst-1:

Tutorial 1. UHFQA Monitor
=========================

This tutorial describes how to use :mod:`quantify.scheduler` to generate pulses and acquire monitor data,
also visualized in the Quantum Analyzer Input tab of LabOne, using the UHFQA's AWG.

For this tutorial lets use :func:`~quantify.scheduler.schedules.trace_schedules.trace_schedule` to create a
pulse level Schedule. This utility function is used for debugging :class:`~quantify.scheduler.acquisition_library.Trace` 
acquisition with pulses of a large fixed duration.

Requirements
^^^^^^^^^^^^

- Create a loopback by connecting the UHFQA output- to the input channel.

.. code-block:: python
    :linenos:

    from typing import Dict, Any
    import logging
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from qcodes.instrument.base import Instrument
    from zhinst.qcodes import UHFQA

    from quantify.scheduler.schedules.trace_schedules import trace_schedule
    from quantify.scheduler.compilation import qcompile

    # Debug only
    # logging.getLogger().setLevel(logging.DEBUG)

.. code-block:: python
    :linenos:

    # Create a schedule using the `Trace` acquisition protocol
    # The `trace_schedule` is a Schedule defined on pulse-level.
    # This schedule should only be used for testing UHFQA monitor output.
    schedule = trace_schedule(
        pulse_amp=1,
        pulse_duration=16e-9,
        pulse_delay=0,
        frequency=7.04e9,
        acquisition_delay=2e-9,
        integration_time=1e-6,
        port="q0:res", 
        clock="q0.ro", 
        init_duration=1e-5,
    )
    schedule.repetitions = 1

.. code-block:: python
    :linenos:
    :emphasize-lines: 24

    def load_example_json_scheme(filename: str) -> Dict[str, Any]:
        import quantify.scheduler.schemas.examples as examples
        path = Path(examples.__file__).parent.joinpath(filename)
        return json.loads(path.read_text())
    
    # Load example configuration from quantify.scheduler.schemas.examples
    device_config_map = (load_example_json_scheme('transmon_test_config.json'))

    # Set the UHFQA `ref` trigger(Reference source trigger) option to "none"
    # The UHFQA is not controlling any marker or triggers
    zhinst_hardware_map: Dict[str, Any] = json.loads(
    """
    {
      "backend": "quantify.scheduler.backends.zhinst_backend.compile_backend",
      "devices": [
        {
          "name": "uhfqa0",
          "type": "UHFQA", 
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

    # Compile schedule for the backend configuration
    zi_backend = qcompile(schedule, device_config_map, zhinst_hardware_map)

.. code-block:: python
    :linenos:

    # Instantiate ZI Instruments
    # Note that the device name in the hardware map must match the Instrument name.
    # for example: uhfqa0 or hdawg0
    uhfqa = UHFQA('uhfqa0', 'dev2299', host='localhost', interface='1GbE')

.. code-block:: python
    :linenos:
    
    # Configure the Instruments
    for instrument_name, settings_builder in zi_backend.settings.items():
        instrument = Instrument.find_instrument(instrument_name)
        zi_settings = settings_builder.build(instrument)

        # Apply settings to the Instrument
        zi_settings.apply()

        # Optionally serialize the settings to file storage
        root = Path('.')
        zi_settings.serialize(root)


.. code-block:: python
    :linenos:

    # Run the UHFQA AWG
    uhfqa.awg.run()
    uhfqa.awg.wait_done()

    # Resolve the results by querying the UHFQA monitor nodes
    acq_channel_results = dict()
    for acq_channel, resolve in acq_channel_resolvers_map.items():
        acq_channel_results[acq_channel] = resolve()

.. code-block:: python
    :linenos:

    # Plot acquisition results
    labels = []
    for i, result in acq_channel_results.items():
        labels.append(f"acq_channel #{i} real")
        plt.plot(result.real)

        labels.append(f"acq_channel #{i} imag")
        plt.plot(result.imag)

    plt.legend(labels)
