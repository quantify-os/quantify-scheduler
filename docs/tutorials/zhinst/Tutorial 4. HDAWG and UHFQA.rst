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
    
    from typing import Dict, Any, Callable
    import logging
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from zhinst.qcodes import HDAWG, UHFQA

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
    :emphasize-lines: 22,33-35,41,49-51
    
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
          "ref": "int",
          "channelgrouping": 0,
          "channel_0": {
            "port": "q0:mw",
            "clock": "q0.01",
            "mode": "complex",
            "modulation": "none",
            "line_gain_db": 0,
            "lo_freq": 4.8e9,
            "interm_freq": -50e6,
            "markers": [
              "AWG_MARKER2"
            ]
          }
        },
        {
          "name": "uhfqa0",
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
    schedule = qcompile(schedule, device_config_map, zhinst_hardware_map)

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
    
    # Run the backend setup
    acq_channel_resolvers_map = zhinst_backend.setup_zhinst_backend(schedule, zhinst_hardware_map)

.. code-block:: python
    :linenos:

    # arm the UHFQA Results
    uhfqa.arm(length=schedule.repetitions, averages=1)

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
