=========
Changelog
=========

0.5.2 Fixes to backends, and other incremental fixes  (2021-12-08)
------------------------------------------------------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Dependency on `jsonschema` has been replaced with `fastjsonschema`. (!284, !293)
* Zhinst hardware config json schema has changed. See the example schema. (!283)
* In `hardware_compile` function, the `hardware_map` is changed to `hardware_cfg` parameter. (!279)
* Remove enum tools dependency (!270)

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Compilation - The `determine_absolute_scheduling` function now sorts the list of labels in the timing constraints, and then a binary search (via `np.searchsorted`) is applied. (!272, !274)
* Compilation - Make `device_cfg` an optional argument of qcompile(!281)
* Compilation - renamed the hardware_mapping argument of qcompile into hardware_cfg (#165, !279)
* Compilation - Introduced the hardware_compile function to perform the hardware compilation returning a CompiledSchedule (#224, !279)
* Docs - Updating user guide to mention correctly the QuantumDevice and ScheduleGettable(s) available. (!209)
* Infrastructure - Adds rich package in the requirements since tutorials use it. (!276)
* Operations - The `locate` function now uses the `functools.lru_cache` to cache the result (only for python >= 3.8). For python 3.7, behaviour remains the same.  (!273, !275)
* Operations - Resolved a minor issue where identical Rxy rotations (for angles >360) would be treated as separate operations in a schedule (!263)
* Visualization - Adds a function `plot_acquisition_operations` which together with the new `AcquisitionOperation` class will help highlight acquisition pulses in the pulse diagrams. (!271, !277)
* Zhinst backend - Large parts of the Zhinst backend have been rewritten. This should resolve a range of issues. (!263)
    - Calculation of the timelines for different operations now makes using of a timing table, improving code readability and debugability.
    - Timing issues related to triggering should be resolved (#218)
    - The backend can now always use the same hardware configuration file (#214)
    - Acquisition is now done using the StartQA instruction (#213)
    - error handling in the Zhinst backend has been improved catching several exceptions at compile time of the schedule instead of manifesting in unexpected results during runtime.
    - Local oscillators through the ZI backend uses the GenericInstrumentCoordinatorComponent. Configures other parameters other than frequency. (!283, #204)
* Qblox backend - only check major and minor version when checking compatibility with the qblox_instruments package (!290)
    - Added support for the Qblox Downconverter (!297)
    - Added workaround for staircase_amplitude. (!292)
    - Fix looped acquisition integration time, fix acquire index offset by one (!291)
    - Qblox instruments version == 0.5.3 (!289)
    - Fix sequencer_sync_en not being reset in the qblox instrument coordinator component. (!285)
    - Fix rounding of time to samples in qblox backend (!282)
    - Fix pulse stitching at zero amplitude. (!280)
    - Allow instruction generated staircase with modulation (!278)
* Utilities - Improve JSON validation speed (!284)
* Utilities - Improve operation deserialization speed (!273)
* Bugfix - For calculating the pulse area, the mathematical area is used instead of area of sampled pulse. (!242, !286)
* Bugfix - Fix for plot window operations (!294)


0.5.1 Incremental fixes, refactoring, and addition of convenience methods and classes (2021-11-11)
--------------------------------------------------------------------------------------------------

Breaking changes
~~~~~~~~~~~~~~~~
* InstrumentCoordinator - `last_schedule` is now a property (!252).
* Structure - We have refactored the Operation and Schedule classes out of the types module and moved the different operation libraries (acquisition_library, gate_library, and pulse_library) (#217, !256).
    * `quantify_scheduler.types.Operation` -> `quantify_scheduler.operations.operation.Operation`, the import `quantify_scheduler.Operation` still works.
    * `quantify_scheduler.types.Schedule` -> `quantify_scheduler.schedules.schedule.Schedule`, the import `quantify_scheduler.Schedule` still works.
    * `quantify_scheduler.types.CompiledSchedule` -> `quantify_scheduler.schedules.schedule.CompiledSchedule`
    * `quantify_scheduler.types.ScheduleBase` -> `quantify_scheduler.schedules.schedule.ScheduleBase`
    * `quantify_scheduler.types.AcquisitionMetadata` -> `quantify_scheduler.schedules.schedule.AcquisitionMetadata`
    * `quantify_scheduler.acquisition_library` -> `quantify_scheduler.operations.acquisition_library`
    * `quantify_scheduler.gate_library` -> `quantify_scheduler.operations.gate_library`
    * `quantify_scheduler.pulse_library` -> `quantify_scheduler.operations.pulse_library`

Merged branches and closed issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Control - Add option to set output port in heterodyne_spec_sched (!262)
* Control - Expand SingleChannelScheduleGettable to support trace acquisitions (!248)
* Control - Update create_dc_compensation_pulse behaviour and docstring. (!244)
* Control - Refactor ScheduleGettableSingleChannel (!240, !249)
* Control - Reduce the default init_duration of spectroscopy schedules (!237)
* Generic ICC - Added a GenericInstrumentCoordinatorComponent. (!267)
* ICCs - InstrumentCoordinatorComponentBase now has a `force_set_parameter` as a ManualParameter to enable the user to switch the lazy_set behaviour when setting parameters of the instruments connected to the InstrumentCoordinatorComponent. (!267)
* Qblox ICCs - Adds a lazy_set behaviour by default when setting parameters with the same value to an instrument connected to the Qblox ICC. (!230)
* Visualization - made matplotlib schedule visualization methods accessible as methods `plot_circuit_diagram_mpl` and `plot_pulse_diagram_mpl` of the `Schedule` class (!253)
* Visualization - resolved a bug where a schedule was modified when drawing a circuit diagram (#197, !250)
* Visualization - Add support for window operation to transmon backend (!245)
* Infrastructure - Fix and enhance pre-commit + add to CI (!257, !265)
* Infrastructure - Added prospector config file for CI. (!261)
* Bugfix - Removed redundant `determine_absolute_timing` step in `qcompile`. (!259)
* Bugfix - Ramp pulse sampling utilizing `np.linspace` behaviour changed. (!258)
* Docs - Adds the new Quantify logo similar to quantify_core. (!266)
* Docs - Enhance documentation of public API for reimported modules [imports aliases] (!254)
* Docs - Fixes the funcparserlib error in rtd. (!251)
* Docs - Updated Qblox backend docs to include the new features. (!247)


0.5.0 Expanded feature sets hardware compilation backends (2021-10-25)
----------------------------------------------------------------------

Breaking changes
~~~~~~~~~~~~~~~~
* The `schedules.timedomain_schedules.allxy_sched` function no longer accepts the string "All" as an argument to the `element_select_idx` keyword.
* The `QuantumDevice.cfg_nr_averages` parameter was renamed to `QuantumDevice.cfg_sched_repetitions`
* The call signature of `gettables.ScheduleVectorAcqGettable` has been renamed to `gettables.ScheduleGettableSingleChannel`, and the call signature has been updated according to #36 to no longer accept several keyword arguments.
* Qblox Backend - The NCO phase is now reset at the start of a program (!213).
* Qblox Backend - Compilation now requires qblox_instruments version 0.5.0, 0.5.1 or 0.5.2 (!214, !221).

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
* Qblox Backend - Added support for real time mixer corrections rather than pre-distorting the uploaded waveforms (!192).
* Qblox Backend - Waveforms are now compared using the normalized data array rather than the parameterized description (!182).
* Qblox Backend - Support for append bin mode (#184, !180).
* Qblox Backend - Support for using real value pulses on arbitrary outputs added (!142).
* Qblox Backend - Compilation now supports 6 sequencers for both the QCM as well as the QRM (!142).
* Qblox Backend - Support for a cluster, along with its QCM, QRM, QCM-RF and QRM-RF modules (!164)
* Qblox Backend - Registers are now dynamically allocated during compilation (!195)
* Zhinst backend - No exception is raised when an LO that is in the config is not part of a schedule. (#203, !223)
* Zhinst backend - Instrument coordinator components for ZI will only be configured when the settings used to configure it have changed (#196, !227)
* Zhinst backend - Solved a bug that caused single-sideband demodulation to not be configured correctly when using the UHFQA (!227)
* Zhinst backend - Warnings raised during compilation of seqc programs will no longer raise an exception but will use logging.warning (!227)
* Zhinst backend - resolved a bug where the instrument coordinator cannot write waveforms to the UHFQA if it has never been used before (!227)
* Zhinst backend - resolved a bug where multiple identical measurements in a schedule would result in multiple integration weights being uploaded to the UFHQA (#207, !234)
* Zhinst backend - resolved a bug where the UHFQA would not be triggered properly when executing a schedule with multiple samples (batched mode) (#205, !234)
* Qblox ICCs - Compensated integration time for Qblox QRM IC component (!199).
* Qblox ICCs - Added error handling for error flags given by `get_sequencer_state` (!215)
* QuantumDevice - Added docstrings to the TransmonElement parameters (!216, !218)
* Qblox ICCs - QCoDeS parameters are now only set if they differ from the value in the cache (!230)
* Visualization - Allow user defined axis for plotting circuit diagram (!206)
* Visualization - Adds schedule plotting using matplotlib and a WindowOperation to help visualize pulse diagrams (!225, !232)
* Other - Added method `sample_schedule` to sample a `Schedule` (!212)
* Other - The `RampPulse` has an extra (optional) parameter `offset` (!211)
* Other - Updated existing schedules to make use of the acquisition index (#180, !180).
* Other - Added a function to extract acquisition metadata from a schedule (#179, !180).
* Other - The soft square waveform can now be evaluated with only one datapoint without raising an exception (!235)
* Other - Added a function that generates a square pulse that compensates DC components of a sequence of pulses (!173)

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
