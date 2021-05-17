.. _sec-zhinst-3:

Tutorial 3. HDAWG
=================

This tutorial describes how to use :mod:`quantify.scheduler` to generate pulses using the HDAWG.

Requirements
^^^^^^^^^^^^

- HDAWG
- Scope

.. code-block:: python
    :linenos:

    from typing import Dict, Any
    import logging
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from zhinst.qcodes import HDAWG

    from quantify.scheduler.schedules.timedomain_schedules import t1_sched
    from quantify.scheduler.compilation import qcompile

    # Debug only
    # logging.getLogger().setLevel(logging.DEBUG)

.. code-block:: python
    :linenos:

    # Create a T1 Schedule
    schedule = t1_sched(np.arange(0, 100e-6, 3e-6), "q0")
    schedule.repetitions = 1

.. code-block:: python
    :linenos:

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
          "ref": "none",
          "channelgrouping": 0,
          "channel_0": {
            "port": "q0:mw",
            "clock": "q0.01",
            "mode": "complex",
            "modulation": "none",
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
    zi_backend = qcompile(schedule, device_config_map, zhinst_hardware_map)

.. code-block:: python
    :linenos:

    # Instantiate ZI Instruments
    # Note that the device name in the hardware map must match the Instrument name.
    # for example: uhfqa0 or hdawg0
    hdawg = HDAWG('hdawg0', 'dev8161', host='localhost', interface='1GbE')

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

    # Start the HDAWG AWG(s)
    hdawg.awgs[0].run()
    hdawg.awgs[0].wait_done()
