==================
quantify-scheduler
==================

.. image:: https://img.shields.io/badge/slack-chat-green.svg
    :target: https://join.slack.com/t/quantify-hq/shared_invite/zt-vao45946-f_NaRc4mvYQDQE_oYB8xSw
    :alt: Slack

.. image:: https://gitlab.com/quantify-os/quantify-scheduler/badges/develop/pipeline.svg
    :target: https://gitlab.com/quantify-os/quantify-scheduler/pipelines/
    :alt: Pipelines

.. image:: https://img.shields.io/pypi/v/quantify-scheduler.svg
    :target: https://pypi.org/pypi/quantify-scheduler
    :alt: PyPi

.. image:: https://app.codacy.com/project/badge/Grade/0c9cf5b6eb5f47ffbd2bb484d555c7e3
    :target: https://www.codacy.com/gl/quantify-os/quantify-scheduler/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-scheduler&amp;utm_campaign=Badge_Grade
    :alt: Code Quality

.. image:: https://app.codacy.com/project/badge/Coverage/0c9cf5b6eb5f47ffbd2bb484d555c7e3
    :target: https://www.codacy.com/gl/quantify-os/quantify-scheduler/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-scheduler&amp;utm_campaign=Badge_Coverage
    :alt: Coverage

.. image:: https://readthedocs.com/projects/quantify-quantify-scheduler/badge/?version=develop&token=ed6fdbf228e1369eacbeafdbad464f6de927e5dfb3a8e482ad0adcbea76fe74c
    :target: https://quantify-quantify-scheduler.readthedocs-hosted.com
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-BSD%204--Clause-blue.svg
    :target: https://gitlab.com/quantify-os/quantify-scheduler/-/blob/master/LICENSE
    :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style


.. figure:: https://orangeqs.com/logos/QUANTIFY_LANDSCAPE.svg
    :align: center
    :alt: Quantify logo

Quantify is a python based data acquisition platform focused on Quantum Computing and solid-state physics experiments.
It is build on top of `QCoDeS <https://qcodes.github.io/Qcodes/>`_ and is a spiritual successor of `PycQED <https://github.com/DiCarloLab-Delft/PycQED_py3>`_.
Quantify currently consists of `quantify-core <https://pypi.org/project/quantify-core/>`_ and `quantify-scheduler <https://pypi.org/project/quantify-scheduler/>`_.

Take a look at the documentation for quantify-scheduler: `last release <https://quantify-quantify-scheduler.readthedocs-hosted.com/>`_ (or `develop <https://quantify-quantify-scheduler.readthedocs-hosted.com/en/develop/>`_).

Quantify-scheduler is a python module for writing quantum programs featuring a hybrid gate-pulse control model with explicit timing control.
The control model allows quantum gate- and pulse-level descriptions to be combined in a clearly defined and hardware-agnostic way.
Quantify-scheduler is designed to allow experimentalists to easily define complex experiments, and produces synchronized pulse schedules to be distributed to control hardware.

.. caution::

    This is a pre-release **alpha version**, major changes are expected. Use for testing & development purposes only.

About
--------

Quantify-scheduler is maintained by The Quantify consortium consisting of Qblox and Orange Quantum Systems.

.. |_| unicode:: 0xA0
   :trim:


.. figure:: https://cdn.sanity.io/images/ostxzp7d/production/f9ab429fc72aea1b31c4b2c7fab5e378b67d75c3-132x31.svg
    :width: 200px
    :target: https://qblox.com
    :align: left

.. figure:: https://orangeqs.com/OQS_logo_with_text.svg
    :width: 200px
    :target: https://orangeqs.com
    :align: left

|_|


|_|

The software is free to use under the conditions specified in the license.


--------------------------

.. nothing-to-avoid-a-sphinx-warning:
