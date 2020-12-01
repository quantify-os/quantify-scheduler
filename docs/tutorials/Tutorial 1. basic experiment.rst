Tutorial 1. Basic experiments
================================

.. jupyter-kernel::
  :id: Tutorial 1. Basic experiment

.. tip::
    Following this Tutorial requires familiarity with the **core concepts** of Quantify-scheduler, we **highly recommended** to consult the (short) :ref:`User guide` before proceeding.


The benefit of allowing the user to mix the high-level gate description of a circuit with the lower-level pulse description can be understood through an example.
Below we first give an example of basic usage using `Bell violations`.
We next show the `Chevron` experiment in which the user is required to mix gate-type and pulse-type information when defining the :class:`~quantify.scheduler.Schedule`.

Basics: The Bell experiment
-----------------------------

As the first example, we want to perform the `Bell experiment <https://en.wikipedia.org/wiki/Bell%27s_theorem>`_ .
The goal of the Bell experiment is to create a Bell state :math:`|\Phi ^+\rangle=\frac{1}{2}(|00\rangle+|11\rangle)` followed by a measurement.
By rotating the measurement basis, or equavalently one of the qubits, it is possible to observe violations of the CSHS inequality.
If everything is done properly, one should observe the following oscillation:

.. jupyter-execute::
  :hide-code:

  import plotly.graph_objects as go
  import numpy as np

  x = np.linspace(0, 360, 361)
  y = np.cos(np.deg2rad(x-180))
  yc = np.minimum(x/90-1, -x/90+3)


  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x,y=y, name='Quantum'))
  fig.add_trace(go.Scatter(x=x,y=yc, name='Classical'))

  fig.update_layout(title='Bell experiment',
                     xaxis_title='Angle between detectors (deg)',
                     yaxis_title='Correlation')
  fig.show()


Bell circuit
~~~~~~~~~~~~~~~~
We create this experiment using :ref:`gates acting on qubits<Gate-level description>` .


We start by initializing an empty :class:`~quantify.scheduler.Schedule`

.. jupyter-execute::

  from quantify.scheduler import Schedule
  sched = Schedule('Bell experiment')
  sched

Under the hood, the :class:`~quantify.scheduler.Schedule` is based on a dictionary that can be serialized

.. jupyter-execute::

  sched.data

We also need to define the qubits.

.. jupyter-execute::

  q0, q1 = ('q0', 'q1') # we use strings because qubit resrouces have not been implemented yet.

Creating the circuit
^^^^^^^^^^^^^^^^^^^^^^

We will now add some operations to the schedule.
Because this experiment is most conveniently described on the gate level, we use operations defined in the :mod:`quantify.scheduler.gate_library` .

.. jupyter-execute::

    from quantify.scheduler.gate_library import Reset, Measure, CZ, Rxy, X90
    import numpy as np

    # we use a regular for loop as we have to unroll the changing theta variable here
    for theta in np.linspace(0, 360, 21):
        sched.add(Reset(q0, q1))
        sched.add(X90(q0))
        sched.add(X90(q1), ref_pt='start') # this ensures pulses are aligned
        sched.add(CZ(q0, q1))
        sched.add(Rxy(theta=theta, phi=0, qubit=q0))
        sched.add(Measure(q0, q1), label='M {:.2f} deg'.format(theta))


Visualizing the circuit
^^^^^^^^^^^^^^^^^^^^^^^^^

And we can use this to create a default visualizaton:

.. jupyter-execute::

  %matplotlib inline

  from quantify.scheduler.visualization.circuit_diagram import circuit_diagram_matplotlib
  f, ax = circuit_diagram_matplotlib(sched)
  # all gates are plotted, but it doesn't all fit in a matplotlib figure
  ax.set_xlim(-.5, 9.5)


Datastructure internals
^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the schedule, :class:`~quantify.scheduler.Operation` objects are also based on dicts.

.. jupyter-execute::

    rxy_theta = Rxy(theta=theta, phi=0, qubit=q0)
    rxy_theta.data



Let's take a look at the internals of the :class:`~quantify.scheduler.Schedule`.

.. jupyter-execute::

    sched

We can see that the number of unique operations is 24 corresponding to 4 operations that occur in every loop and 21 unique rotations for the different theta angles. (21+4 = 25 so we are missing something.

.. jupyter-execute::

    sched.data.keys()

The schedule consists of a hash table containing all the operations.
This allows efficient loading of pulses or gates to memory and also enables efficient adding of pulse type information as a compilation step.

.. jupyter-execute::

    from itertools import islice
    # showing the first 5 elements of the operation dict
    dict(islice(sched.data['operation_dict'].items(), 5))

The timing constraints are stored as a list of pulses.

.. jupyter-execute::

  sched.data['timing_constraints'][:6]

Compilation of a circuit diagram into pulses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Compilation of pulses onto physical hardware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Precise timing control: The Ramsey experiment
------------------------------------------------

.. todo::

  This tutorial should showcase in detail the timing options possible in the
  schedule.



A hybrid experiment: The Chevron
------------------------------------------------

.. todo::

  This tutorial should showcase hybridg schedules that mix pulse and gate level
  descriptions.

Of course different Qubits are driven with different techniques which must be defined. Here we have a pair of Transmon qubits,
which respond to microwave pulses:

.. jupyter-execute::

    #  q0 ro_pulse_modulation_freq should be 80e6, requires issue38 resolution
    device_test_cfg = {
          "qubits":
          {
              "q0": {"mw_amp180": 0.5, "mw_motzoi": -0.25, "mw_duration": 20e-9,
                     "mw_modulation_freq": 50e6, "mw_ef_amp180": 0.87, "mw_ch": "qcm0.s0",
                     "ro_ch": "qrm0.s0", "ro_pulse_amp": 0.5, "ro_pulse_modulation_freq": 80e6,
                     "ro_pulse_type": "square", "ro_pulse_duration": 150e-9,
                     "ro_acq_delay": 120e-9, "ro_acq_integration_time": 700e-9,
                     "ro_acq_weigth_type": "SSB",
                     "init_duration": 250e-6
                     },
              "q1": {"mw_amp180": 0.45, "mw_motzoi": -0.15, "mw_duration": 20e-9,
                     "mw_modulation_freq": 80e6, "mw_ef_amp180": 0.27, "mw_ch": "qcm1.s0",
                     "ro_ch": "qrm0.s1", "ro_pulse_amp": 0.5, "ro_pulse_modulation_freq": -23e6,
                     "ro_pulse_type": "square", "ro_pulse_duration": 100e-9,
                     "ro_acq_delay": 120e-9, "ro_acq_integration_time": 700e-9,
                     "ro_acq_weigth_type": "SSB",
                     "init_duration": 250e-6 }
          },
          "edges":
          {
              "q0-q1": {
                  "flux_duration": 40e-9,
                  "flux_ch_control": "qcm0.s1", "flux_ch_target": "qcm1.s1",
                  "flux_amp_control": 0.5,  "flux_amp_target": 0,
                  "phase_correction_control": 0,
                  "phase_correction_target": 0}
          }
      }


With this information, the compiler can now generate the waveforms required.

Resources
----------

Our gates and timings are now defined but we still need to describe how the various devices in our experiments are connected; Quantify uses the :class:`quantify.scheduler.types.Resource` to represent this.
FIXME: CompositeResource no longer exists, use port resource.
Of particular interest to us are the :class:`quantify.scheduler.resources.CompositeResource` and the :class:`quantify.scheduler.resources.Pulsar_QCM_sequencer`,
which represent a collection of Resources and a single Core on the Pulsar QCM:

.. jupyter-execute::

    # from quantify.scheduler.resources import CompositeResource, Pulsar_QCM_sequencer, Pulsar_QRM_sequencer
    # qcm0 = CompositeResource('qcm0', ['qcm0.s0', 'qcm0.s1'])
    # qcm0_s0 = Pulsar_QCM_sequencer('qcm0.s0', seq_idx=0)
    # qcm0_s1 = Pulsar_QCM_sequencer('qcm0.s1', seq_idx=1)

    # qcm1 = CompositeResource('qcm1', ['qcm1.s0', 'qcm1.s1'])
    # qcm1_s0 = Pulsar_QCM_sequencer('qcm1.s0', seq_idx=0)
    # qcm1_s1 = Pulsar_QCM_sequencer('qcm1.s1', seq_idx=1)

    # qrm0 = CompositeResource('qrm0', ['qrm0.s0', 'qrm0.s1'])
    # # Currently mocking a readout module using an acquisition module
    # qrm0_s0 = Pulsar_QRM_sequencer('qrm0.s0', seq_idx=0)
    # qrm0_s1 = Pulsar_QRM_sequencer('qrm0.s1', seq_idx=1)

    # sched.add_resources([qcm0, qcm0_s0, qcm0_s1, qcm1, qcm1_s0, qcm1_s1, qrm0, qrm0_s0, qrm0_s1])

With this information added, we can now compile the full program with an appropriate backend:

.. jupyter-execute::

  # from quantify.scheduler.compilation import qcompile
  # import quantify.scheduler.backends.pulsar_backend as pb
  # sched, config_dict = qcompile(sched, device_test_cfg, backend=pb.pulsar_assembler_backend)

Let's take a look at what our finished configuration looks like:

.. jupyter-execute::

    # config_dict

It contains a list of JSON files representing the configuration for each device. Now we are ready to deploy to hardware.

Visualization using a pulse diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The compiler also provides pulse schedule visualization, which can be useful for a quick verification that your schedule is as expected:

.. jupyter-execute::

  # from quantify.scheduler.visualization.pulse_scheme import pulse_diagram_plotly
  # fig = pulse_diagram_plotly(sched, ch_list=['qcm0.s0', 'qcm1.s0', 'qrm0.s0', 'qrm0.r0'])
  # fig.show()

By default :func:`quantify.scheduler.visualization.pulse_scheme.pulse_diagram_plotly` shows the first 8 channels encountered in in a schedule, but by specifying a list of channels, a more compact visualization can be created.

Connecting to Hardware
----------------------

The Pulsar QCM provides a QCodes based Python API. As well as interfacing with real hardware, it provides a mock driver we can use for testing and development, which we will
also use for demonstration purposes as part of this tutorial:

.. jupyter-execute::

    # # todo install from pypi when released
    # try:
    #     from pulsar_qcm.pulsar_qcm import pulsar_qcm_dummy
    #     from pulsar_qrm.pulsar_qrm import pulsar_qrm_dummy
    #     PULSAR_ASSEMBLER = True
    # except ImportError:
    #     PULSAR_ASSEMBLER = False

The Pulsar QCM backend provides a method for deploying our complete configuration to all our devices at once:

.. jupyter-execute::
    :raises:

    # if PULSAR_ASSEMBLER:
    #     _pulsars = []
    #     # first we need to create some Instruments representing the other devices in our configuration
    #     for qcm_name in ['qcm0', 'qcm1']:
    #         _pulsars.append(pulsar_qcm_dummy(qcm_name))
    #     for qrm_name in ['qrm0', 'qrm1']:
    #         _pulsars.append(pulsar_qrm_dummy(qrm_name))
    #     pb.configure_pulsar_sequencers(config_dict)

At this point, the assembler on the device will load the waveforms into memory and verify the program can be executed. We must next arm and then start the device:

.. jupyter-execute::
    :raises:

    # if PULSAR_ASSEMBLER:
    #     qcm0 = _pulsars[0]
    #     qrm0 = _pulsars[2]

    #     qcm0.arm_sequencer()
    #     qrm0.arm_sequencer()
    #     qcm0.start_sequencer()
    #     qrm0.start_sequencer()

Provided we have synchronized our Pulsars properly using the sync-line, our experiment will now run. Once it's complete,
it is necessary to stop the QRMs before we read any data they have acquired. We first instruct the QRM to move it's
acquisition to disk memory with a named identifier and number of samples. We then request the QRM to return these
acquisitions over the driver so we can do some processing in Python:

.. jupyter-execute::
    :raises:

    # if PULSAR_ASSEMBLER:
    #     seq_idx = 0
    #     qrm0.stop_sequencer()
    #     qrm0.store_acquisition(seq_idx, "meas_0", 4800)
    #     acq = qrm0.get_acquisitions(seq_idx)



.. seealso::

    The complete source code of this tutorial can be found in

    :jupyter-download:notebook:`Tutorial 1. Basic experiment`

    :jupyter-download:script:`Tutorial 1. Basic experiment`
