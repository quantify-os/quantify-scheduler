===========
Changelog
===========

Unreleased
----------

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Feature: Added zhinst backend option to enable Calibration mode, Closes #103. (!123)
* Added acquisitions to circuit diagram (!93)
* Added Chirp and Staircase pulses; and efficient implementation in Qblox backend (!106)
* Added ControlStack (!70)
* Added Qblox ControlStack components (!112)
* Renamed input parameters of quantify_scheduler.schedules.* functions. (!136)


0.3.0 Multiple backends support (2021-05-20)
------------------------------------------------
* Added support for both Qblox and Zurich Instrument backends.
* Added convenience pylintrc configuration file.
* Added examples for timedomain and spectroscopy schedules.


Breaking changes
~~~~~~~~~~~~~~~~
* Major refactor of the Qblox backend. (For example, it's now `quantify_core.backends.qblox_backend` instead of the previous `quantify_core.backends.pulsar_backend`)
* Qblox backend requires strictly v0.3.2 of the qblox-instruments package.


Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Add mixer skewness corrections helper function. (!102)
* Added Qblox backend support. (!81)
* Compile backend with ZISettingsBuilder. (!87)
* Add vscode IDE config files. (!100)
* Add ZISettingsBuilder class. (!86)
* Added representation to gates in gate library and defined equality operation. (!101)
* Fix/operation duration. Fixes #107. (!89)
* Feat/long pulses fix validators name. (!90)
* Implemented long square pulses unrolling (for waveform-memory-limited devices). (!83)
* Changed Qblox-Instruments version to 0.3.2. (!88)
* Feature: Improve overall zhinst backend timing. (!77)
* Plotly cleanup. (!69)
* Pulsar backend version bump. (!82)
* Added zhinst backend support. (!49)
* Added example timedomain programs. (!71)
* Added example spectroscopy programs. (!64)
* Added pylintrc configuration file. (!55)
* Added repetitions property to Schedule. (!56)
* Added Acquisition Protocols. (!51)
* Hotfix for filename sanitization pulsar backend. (!61)
* Pulsar backend function sanitization. (!60)
* Potential fix time-out pulsar. (!58)
* Updated Pulsar backend version to v0.2.3.. (!57)
* Fixed datadir related bugs. (!54)
* Added Station implementation. (!52)
* Pulsar backend v0.2.2 check. (!48)
* Fix for issue with acq delay . (!45)
* Fix for issue #52. (!44)
* Add artificial detuning to Ramsey schedule (!120)



0.2.0 Hybrid pulse- gate-level control model (2021-01-14)
---------------------------------------------------------

* Major refactor of the scheduler resource code enabling hybrid pulse- gate-level control.
* Moved quantify_scheduler.types.Resource class to a separate quantify_scheduler.resources module.
* Adds a BasebandClockResource class within the newly created quantify_scheduler.resources module.
* Moved QRM and QCM related classes to the quantify_scheduler.backends.pulsar_backend module.
* In quantify_scheduler.compilation, rename of function '_determine_absolute_timing' to 'determine_absolute_timing'. Argument changed from clock_unit to time_unit.
* In quantify_scheduler.compilation, rename of function '_add_pulse_information_transmon' to 'add_pulse_information_transmon'.
* Added ramp waveform in quantify_scheduler.waveforms.
* Added schemas for operation and transmon_cfg.
* Added a basic hybrid visualisation for pulses using new addressing scheme.
* Operations check whether an operation is a valid gate or pulse.
* Refactor of visualization module. Moved quantify_scheduler.backends.visualization to quantify_scheduler.visualization module. Expect code breaking reorganization and changes to function names.
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
