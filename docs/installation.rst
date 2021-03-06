.. highlight:: shell

============
Installation
============

Stable release
--------------

To install quantify-scheduler follow the :doc:`installation guide of quantify-core <quantify-core:installation>`.

Update to latest version
------------------------

To update to the latest version::

    $ pip install --upgrade quantify-scheduler

From sources
------------

The sources for quantify can be downloaded from the `GitLab repo <https://gitlab.com/quantify-os/quantify-scheduler>`_.

You can clone the public repository:

.. code-block:: console

    $ git clone git://gitlab.com/quantify-os/quantify-scheduler

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install --upgrade .


Setting up for local development
--------------------------------

Contributions are very welcome! To setup a an environment for local development see the instructions in the :doc:`installation guide of quantify-core <quantify-core:installation>`. You only need to replace :code:`quantify-core` with :code:`quantify-scheduler` in the provided commands.

If you need any help reach out to us by `creating a new issue <https://gitlab.com/quantify-os/quantify-scheduler/-/issues>`_.


Jupyter and plotly
-------------------

Quantify-scheduler uses the `ploty`_ graphing framework for some components, which can require some additional set-up
to run with a Jupyter environment - please see `this page for details.`_


.. _ploty: https://plotly.com/
.. _this page for details.: https://plotly.com/python/getting-started/#jupyter-notebook-support
