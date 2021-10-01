=========
Changelog
=========

Unreleased
----------





Breaking changes
~~~~~~~~~~~~~~~~
* Mixer corrections in Qblox backend are broken because of the switch to real-time modulation. The support of mixer corrections in the Qblox firmware is under development.
* The `schedules.timedomain_schedules.allxy_sched` function no longer accepts the string "All" as an argument to the `element_select_idx` keyword.
* The `QuantumDevice.cfg_nr_averages` parameter was renamed to `QuantumDevice.cfg_sched_repetitions`
* The call signature of `gettables.ScheduleVectorAcqGettable` has been renamed to `gettables.ScheduleGettableSingleChannel`, and the call signature has been updated according to #36 to no longer accept several keyword arguments.

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Compilation - Added the ability to specify the BinMode at the quantum-circuit layer (#183, !180).
* Compilation - qcompile no longer modifies schedules (#102, !178).
* Control - Added a first version of the QuantumDevice object (#148, !180).
* Control - A single-qubit ScheduleGettable has been added (#36, !180).
* Docs - Added bibliography with sphinxcontrib-bibtex extension (!171).
* Docs - Fixed missing files in API reference (!176).
* InstrumentCoordinator - CompiledSchedule class added to specify interfaces of InstrumentCoordinator and compilation functions (#174, !177).
* InstrumentCoordinator - CompiledSchedule.last_schedule method added to provide access to last executed schedule (#167, !177).
* Qblox Backend - Added support for qblox_instruments version 0.4.0 (new acquisition path) (!143).
* Qblox Backend - Waveforms are now compared using the normalized data array rather than the parameterized description (!182).
* Qblox Backend - Support for append bin mode (#184, !180).
* Updated existing schedules to make use of the acquisition index (#180, !180).
* Added a function to extract acquisition metadata from a schedule (#179, !180).
* Qblox ICCs - Compensated integration time for Qblox QRM IC component (!199).
* Visualization - Allow user defined axis for plotting circuit diagram (!206)

0.4.0 InstrumentCoordinator and improvements to backends (2021-08-06)
---------------------------------------------------------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Change of namespace from quantify.scheduler.* to quantify_scheduler.*

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Changes the namespace from quantify.scheduler to quantify_scheduler (!124)
* InstrumentCoordinator - Add is_running property and wait_done method. Closes #133 (!140)
* InstrumentCoordinator - Add instrument coordinator reference parameter to transmon element (!152)
* InstrumentCoordinator - Prefix serialized settings for ZI ControlStack components. (!149)
* InstrumentCoordinator - Refactored ControlStack name to InstrumentCoordinator (!151)
* InstrumentCoordinator - Make use of InstrumentRefParameters (!144)
* InstrumentCoordinator - Add controlstack class (!70)
* InstrumentCoordinator - Add Zurich Instruments InstrumentCoordinatorComponent. (!99)
* InstrumentCoordinator - Add Qblox InstrumentCoordinatorComponent. (!112)
* InstrumentCoordinator - Avoid garbage collection for instrument coordinator components (!162)
* Qblox backend - Removed limit in Qblox backend that keeps the QCM sequencer count at 2 (!135)
* Qblox backend - Restructured compilation using external local oscillators. (!116)
* Qblox backend - Added Chirp and Staircase pulses; and efficient implementation for QD spin qubit experiments (!106)
* Qblox backend - Only run `start_sequencer` on pulsar instruments which have been armed (!156)
* Zhinst backend - Assert current with new sequence program to skip compilation (!131)
* Zhinst backend - Deserialize zhinst settings from JSON to ZISettingsBuilder (!130)
* Zhinst backend - Add waveform mixer skewness corrections (!103)
* Zhinst backend - Add backend option to enable Calibration mode (#103, !123)
* Zhinst backend - Replace weights string array with a numerical array in JSON format (!148)
* Zhinst backend - Add grouping of instrument settings (!133)
* Zhinst backend - Add qcompile tests for the zurich instruments backend (!118)
* Zhinst backend - Add repetitions parameter (!138)
* Zhinst backend - Fixes the bug where the seqc in the datadir is not copied to the webserver location. (!165)
* Fix for circuit diagram plotting failure after pulse scheduling (#157, !163)
* Fixed typo in the gate_info of the Y gate in the gate_library (!155)
* Add artificial detuning in Ramsey Schedule and bug fixes (!120)
* Use individual loggers per python file (!134)
* Recolour draw circuit diagram mpl (!96)
* Fix issues with timedomain schedules (!145)
* Renamed input parameters of quantify_scheduler.schedules.* functions. (!136)
* Added acquisitions to circuit diagram (!93)
* Add string representations to acquisition protocols of the acquisitions library (!114)
* Transmon element and config generation (!75)
* Rename operation_hash to operation_repr (!122)
* Add types.Schedule from_json conversion (!119)
* Add missing return types (!121)
* Add serialization to Operations (!110)



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
* Fix for issue with acq delay. (!45)
* Fix for issue #52. (!44)
* Add artificial detuning to Ramsey schedule (!120)
* Added support for the Qblox Pulsar QCM-RF/QRM-RF devices (!158)



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
