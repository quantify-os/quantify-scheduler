.. _sec-zhinst-4:

Tutorial 4. HDAWG and UHFQA
================================

This tutorial demonstrates how to setup a HDAWG in combination with an UHFQA.
In this setup the HDAWG will trigger the UHFQA using a marker.

Requirements
^^^^^^^^^^^^

- Connect HDAWG Marker 2 to UHFQA Trigger 2
- Connect HDAWG Output Channel 1 to UHFQA Input Channel 1
- Connect HDAWG Output Channel 2 to UHFQA Input Channel 2

.. code-block:: python
    :linenos:

    from typing import Dict, Any
    import logging
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from zhinst.qcodes import HDAWG, UHFQA

    from quantify.scheduler.schedules.timedomain_schedules import t1_sched
    from quantify.scheduler.compilation import qcompile

    # Debug only
    # logging.getLogger().setLevel(logging.DEBUG)

.. code-block:: python
    :linenos:

    # Create a T1 Schedule
    times = np.arange(0, 100e-6, 3e-6)
    schedule = t1_sched(times, "q0")
    schedule.repetitions = 1

.. code-block:: python
    :linenos:
    :emphasize-lines: 21,32-34,40,48-50

    def load_example_json_scheme(filename: str) -> Dict[str, Any]:
        import quantify.scheduler.schemas.examples as examples
        path = Path(examples.__file__).parent.joinpath(filename)
        return json.loads(path.read_text())
    
    # Load example configuration from quantify.scheduler.schemas.examples
    device_config_map = (load_example_json_scheme('transmon_test_config.json'))

    zhinst_hardware_map: Dict[str, Any] = json.loads(
    """
    {
      "backend": "quantify.scheduler.backends.zhinst_backend.compile_backend",
      "devices": [
        {
          "name": "hdawg0",
          "type": "HDAWG4",
          "ref": "int",
          "channelgrouping": 0,
          "channel_0": {
            "port": "q0:mw",
            "clock": "q0.01",
            "mode": "complex",
            "modulation": "none",
            "lo_freq": 4.8e9,
            "interm_freq": -50e6,
            "markers": [
              "AWG_MARKER2"
            ]
          }
        },
        {
          "name": "uhfqa0",
          "type": "UHFQA", 
          "ref": "ext",
          "channel_0": {
            "port": "q0:res",
            "clock": "q0.ro",
            "mode": "real",
            "modulation": "none",
            "lo_freq": 4.8e9,
            "interm_freq": -50e6,
            "triggers": [
              2
            ]
          }
        }
      ]
    }
    """
    )

.. code-block:: python
    :linenos:

    # Compile schedule with configurations
    zi_backend = qcompile(schedule, device_config_map, zhinst_hardware_map)

.. code-block:: python
    :linenos:

    # Instantiate ZI Instruments
    # Note that the device name in the hardware map must match the Instrument name.
    # for example: uhfqa0 or hdawg0
    hdawg = HDAWG('hdawg0', 'dev8161', host='localhost', interface='1GbE')
    uhfqa = UHFQA('uhfqa0', 'dev2299', host='localhost', interface='1GbE')

    # Enable UHFQA Trigger 2
    ZhinstHelpers.set(instrument=uhfqa, node="awgs/0/auxtriggers/1/channel", value=1)
    ZhinstHelpers.set(instrument=uhfqa, node="awgs/0/auxtriggers/1/slope", value=1)

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

    # arm the UHFQA Results
    n_acquisitions = len(times)
    uhfqa.arm(length=n_acquisitions, averages=1)

    # Start UHFQA AWG, waiting for trigger
    uhfqa.awg.run()

    # Start the HDAWG AWG(s)
    hdawg.awgs[0].run()

    # Await the experiment
    hdawg.awgs[0].wait_done()
    uhfqa.awg.wait_done()

    # qaresults_results = map(lambda c: c.result(), uhfqa.channels)
    # qamonitor_results = map(lambda index: ZhinstHelpers.get(uhfqa, f'qas/0/monitor/inputs/{index}/wave'), range(2))

    acq_channel_results = dict()
    for acq_channel, resolve in zi_backend.acquisition_resolvers.items():
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
