===========
Changelog
===========

Unreleased
----------

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* [Docs] pinned scanpydoc==0.6 due to conflicting docs dependencies (!84)
* Added utility for decomposing long square pulses (!83)

0.2.0 Hybrid pulse- gate-level control model (2021-01-14)
---------------------------------------------------------

* Major refactor of the scheduler resource code enabling hybrid pulse- gate-level control.
* Moved quantify.scheduler.types.Resource class to a separate quantify.scheduler.resources module.
* Adds a BasebandClockResource class within the newly created quantify.scheduler.resources module.
* Moved QRM and QCM related classes to the quantify.scheduler.backends.pulsar_backend module.
* In quantify.scheduler.compilation, rename of function '_determine_absolute_timing' to 'determine_absolute_timing'. Argument changed from clock_unit to time_unit.
* In quantify.scheduler.compilation, rename of function '_add_pulse_information_transmon' to 'add_pulse_information_transmon'.
* Added ramp waveform in quantify.scheduler.waveforms.
* Added schemas for operation and transmon_cfg.
* Added a basic hybrid visualisation for pulses using new addressing scheme.
* Operations check whether an operation is a valid gate or pulse.
* Refactor of visualization module. Moved quantify.scheduler.backends.visualization to quantify.scheduler.visualization module. Expect code breaking reorganization and changes to function names.
* Pulsar backend version now checks for QCM and QRM drivers version 0.1.2.

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* fix(pulse_scheme): Add tickformatstops for x-axis using SI-unit 'seconds'. Closes #39. (!39)
* Resolve "y-axis label is broken in plotly visualization after resources-refactor". Closes #45. (!38)
* Resources refactor (!28, !29, !30)
* Hybrid visualisation for pulses and circuit gate operations. Closes #22 and #6. (!27)
* Support Pulsar parameterisation from scheduler. Support feature for #29. (!2)
* Operation properties to check if an operation is a valid gate or pulse. Closes #28 (!25)
* Visualisation refactor. Closes #26. (!22)
* Windows job (!20)
* Changed Pulsar backend version check from 0.1.1 to 0.1.2. (!21)



0.1.0 (2020-10-21)
------------------
* Refactored scheduler functionality from quantify-core into quantify-scheduler
* Support for modifying Pulsar params via the sequencer #54 (!2)
* Simplification of compilation through `qcompile` (!1)
* Qubit resources can be parameters of gates #11 (!4)
* Circuit diagram visualization of operations without no pulse info raises exception #5 (!5)
* Pulsar backend verifies driver and firmware versions of hardware #14 (!6)
* Sequencer renamed to scheduler #15 (!7)
* Documentation update to reflect refactor #8 (!8)
* Refactor circuit diagram to be more usable !10 (relates to #6)
* Unify API docstrings to adhere to NumpyDocstring format !11
* Changes to addressing of where a pulse is played !9 (#10)
* Renamed doc -docs folder for consistency #18 (!12)
* Moved test folder outside of project #19 (!14)
* Add copyright notices and cleanup documenation #21 (!13)
* Add installation tip for plotly dependency in combination with jupyter #24 (!15)

.. note::

    * # denotes a closed issue.
    * ! denotes a merge request.
