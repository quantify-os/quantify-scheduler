.. highlight:: shell

============
Installation
============

Stable release
--------------

To install quantify-scheduler follow the :doc:`installation guide of quantify-core <quantify-core:installation>`.

From sources
------------

The sources for quantify can be downloaded from the `GitLab repo <https://gitlab.com/quantify-os/quantify-scheduler>`_.

You can clone the public repository:

.. code-block:: console

    $ git clone git://gitlab.com/quantify-os/quantify-scheduler

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -e . --upgrade


Jupyter and plotly
-------------------

Quantify-scheduler uses the `ploty`_ graphing framework for some components, which can require some additional set-up
to run with a Jupyter environment - please see `this page for details.`_


.. _ploty: https://plotly.com/
.. _this page for details.: https://plotly.com/python/getting-started/#jupyter-notebook-support
