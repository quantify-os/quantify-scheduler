.. _sec-qblox-how-to-configure:

Usage of the backend
====================

Configuring the backend is done by specifying a python dictionary (or loading it from a JSON file)
that describes your experimental setup. An example of such a config:

.. jupyter-execute::
    :hide-code:
    :linenos:

    import json
    import os, inspect
    from pathlib import Path
    from pprint import pprint
    import quantify_scheduler.schemas.examples as es

    esp = inspect.getfile(es)

    cfg_f = Path(esp).parent / 'qblox_test_mapping.json'

    with open(cfg_f, 'r') as f:
      qblox_test_mapping = json.load(f)

    pprint(qblox_test_mapping)

Here the entry :code:`"backend": "quantify_scheduler.backends.qblox_backend.hardware_compile"` specifies to the scheduler
that we are using the Qblox backend (specifically the :func:`~quantify_scheduler.backends.qblox_backend.hardware_compile` function).

Apart from the :code:`"backend"`, each entry in the dictionary corresponds to a device connected to the setup. In the other sections we will look at the specific instrument configurations in more detail.
