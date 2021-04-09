Zurich Instruments Backend
==========================

.. warning::
    The :mod:`quantify.scheduler.backends.zhinst_backend` is still under development. 
    Breaking changes at this stage are not excluded.

Please read this `Gitlab Issue <https://gitlab.com/quantify-os/quantify-scheduler/-/issues/88>`_ for open issues.

Introduction
^^^^^^^^^^^^

:mod:`quantify-scheduler` provides a stateless module: :mod:`~quantify.scheduler.backends.zhinst_backend`, 
that abstracts the complexity of setting up `Zurich Instruments <https://www.zhinst.com>`_ for 
experiments in a modern and automated fashion. :mod:`quantify-scheduler` combines Quantum Device- 
and Instrument properties with the :ref:`Schedule<sec-schedule>` during compilation to generate waveforms 
and sequencing instructions specifically for Zurich Instruments hardware. More information about 
`complilation` can be found in the :ref:`User Guide<sec-user-guide>`.

Using existing programming interfaces provided via :mod:`zhinst-qcodes` and :mod:`zhinst-toolkit`,
:mod:`quantify-scheduler` prepares the instruments that are present in the 
:ref:`hardware configuration file<Hardware configuration file>`. See more on how to configure the 
:mod:`~quantify.scheduler.backends.zhinst_backend` in the :ref:`How to configure<sec-zhinst-how-to-configure>` 
page.

Finnaly, after configuring and running :func:`~quantify.scheduler.backends.zhinst_backend.setup_zhinst_backend` 
successfully the instruments are prepared for execution. 

The Zurich Instruments backend provides:

- Automatic generation of Sequencer instructions.
- Waveform generation and modulation.
- Memory-efficient Sequencing with the CommandTable.
- Configuration for Triggers and Markers.
- Flexible configuration via JSON data structures.

Supported Instruments
^^^^^^^^^^^^^^^^^^^^^

- ✅ | HDAWG
- ✅ | UHFQA
- ⬜️ | MFLI
- ⬜️ | UHFLI


How to
^^^^^^

.. toctree::
    :maxdepth: 2

    How to configure
    Tutorial 1. UHFQA Monitor
    Tutorial 2. UHFQA Result
    Tutorial 3. HDAWG
    Tutorial 4. HDAWG and UHFQA