.. _sec-zhinst-3:

Tutorial 3. HDAWG
=================

This tutorial describes how to use quantify-schedule to generate pulses using the HDAWG.

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

    from quantify.scheduler.types import Schedule
    from quantify.scheduler.gate_library import Rxy, X, X90, Reset, Measure, CZ

    from quantify.scheduler.compilation import qcompile
    import quantify.scheduler.backends.zhinst_backend as zhinst_backend

    # Debug only
    # logging.getLogger().setLevel(logging.DEBUG)

.. code-block:: python
    :linenos:

    # Create a schedule
    schedule = Schedule("T1 Experiment", repetitions=1)
    times = np.arange(0, 100e-6, 3e-6)
    for tau in times:
        schedule.add(Reset("q0"))
        schedule.add(X("q0"), ref_pt="start")
        schedule.add(Measure("q0"), rel_time=tau)

.. code-block:: python
    :linenos:

    def load_example_json_scheme(filename: str) -> Dict[str, Any]:
        import quantify.scheduler.schemas.examples as es
        import os, inspect
        import json

        examples_path:str = inspect.getfile(es)
        config_file_path = os.path.abspath(os.path.join(examples_path, '..', filename))

        return json.loads(Path(config_file_path).read_text())

    # Load example configuration from quantify.scheduler.schemas.examples
    device_config_map = (load_example_json_scheme('transmon_test_config.json'))

    zhinst_hardware_map: Dict[str, Any] = json.loads(
    """
    {
      "backend": "quantify.scheduler.backends.zhinst_backend.create_pulsar_backend",
      "devices": [
        {
          "name": "hdawg0",
          "ref": "none",
          "channelgrouping": 0,
          "channel_0": {
            "port": "q0:mw",
            "clock": "q0.01",
            "mode": "complex",
            "modulation": "none",
            "line_gain_db": 0,
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
    hdawg = HDAWG('hdawg0', 'dev8161', host='localhost', interface='1GbE')

.. code-block:: python
    :linenos:

    # Run the backend setup
    zhinst_backend.setup_zhinst_backend(schedule, zhinst_hardware_map)

.. code-block:: python
    :linenos:

    # Start the HDAWG AWG(s)
    hdawg.awgs[0].run()
    hdawg.awgs[0].wait_done()
