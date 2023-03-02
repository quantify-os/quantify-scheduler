# Changelog

## 0.12.1

### Breaking changes

- Qblox backend - Strictly requires v0.9.x of the `qblox-instruments` package (!616)

## 0.12.0 (2023-02-28)

### Breaking changes

- Acquisition - `InstrumentCoordinator.retrieve_acquisition` returns an `xarray.Dataset` (!550, #362)
- Qblox backend - Set the `marker_ovr_en` QCoDeS parameter by default to `False` before schedule execution, so that the markers are always controlled using the `MarkerConfiguration` (!576)
- Qblox backend - Set `"downconverter_freq"` to `null` (json) or `None` to deactivate (`0` no longer deactivates it) (!574)
- Qblox backend - The NCO is now enabled when setting `sequencer.frequency` to `0` (`0` no longer disables it) (!574)
  - For baseband modules without external LO, NCO is still permanently disabled by setting `"interm_freq"` to `0` in the hardware config
- Visualization - Deprecate `visualization` module and its functions `circuit_diagram_matplotlib`, `pulse_diagram_matplotlib`, and `pulse_diagram_plotly` (!599, #397)
  - Create `_visualization` submodule within `schedules`
  - Make plots via `ScheduleBase` methods
  - Move visualization tests to `schedules` directory, and make tests for `ScheduleBase` plotting methods

### Merged branches and closed issues

- Compilation - Update `CompilationConfig` with high-level structure (!603, #332)
- Compilation - Add `determine_relative_latencies` that determines latencies for all port-clock combinations in the hardware config relative to the minimum latency (!566, #379)
- Documentation - Qblox backend reference guide overhaul and fix docs generation warnings (!587)
- JSON utilities - Introduce serialization of python objects to a dotted import string, next to the already existing deserialization (`import_python_object_from_string`) (!578, #393)
  - Serialization now happens automatically in `DataStructure`s, deserialization is implemented using Pydantic validators in the configuration models
- Operations - Introduce `SetClockFrequency` operation (!575, follow-up to !539 !543)
- Qblox backend - Introduce `qblox.helpers.determine_clock_lo_interm_freqs` and use in `QbloxBasebandModule.assign_frequencies` and `QbloxRFModule.assign_frequencies` (!574)
- Qblox backend - Compile `SetClockFrequency` operation to `set_freq` + `upd_param` of 8 ns (!575, follow-up to !539 !543)
- Qblox backend - Fix Qblox sync reset bug following !550 (!611)
- Schedules - Add fast NCO sweep schedules using `SetClockFrequency` (!542)
- ZI LabOne backend - Support distortion corrections (!600)

## 0.11.1 (2023-02-07)

### Breaking changes

- Qblox backend - `"input_att"` can be the property of `"complex_input_0"` and `"complex_output_0"`, but not both at the same time for QRM-RF (!596, !597)

## 0.11.0 (2023-02-03)

For help in migrating from deprecated methods, see [Quantify Deprecated Code Suggestions](examples/deprecated.md).

### Breaking changes

- Installation - Instead of `requirements.txt` and `requirements_dev.txt` `quantify-scheduler` uses optional requirements. Use `pip install "quantify-scheduler[dev]"` to install all of them. (!592)
- Compilation - Raise error upon clock being used in operation that was not added as resource to the schedule or device configuration (!538)
- Qblox ICCs - Replace `"acq_mapping"` by `"trace_acq_channel"` in the compiled schedule (!515)
- Qblox backend - Replace `"input_gain<n>"` by `"input_gain_<n>"` and `"input_att"` is the property of `"complex_input"` (!585)

### Merged branches and closed issues

- Acquisition - Data obtained with TriggerCount acquisition is formatted correctly (!530).
- Acquisition - Fix wrong assumption in input format (!564, follow-up for !530).
- Acquisition - Raise an exception if user tries to use same acquisition index and channel for different operations, and only extract data from used modules (!573)
- Compilation - Can optionally provide a `QuantumDevice` to `QuantifyCompiler`. This will be used as default `CompilationConfig` in `QuantifyCompiler.compile()` (!535)
- Compilation - Fix clock not being added to schedule from quantum device layer via new graph node `set_pulse_and_acquisition_clock` that verifies and sets clock frequency (!538, #371)
- Deprecation - Refactored tests to remove deprecated `qcompile`, refactored to `SerialCompiler` (!529, #368)
- Documentation - Sphinx build now compatible with qcodes==0.36.0 (!552)
- Documentation - Removed deprecated code from the Compiling to Hardware Tutorial (!582)
- NV centers - Avoid python warning when parsing docstring in `nv_element.py` (!562)
- NV centers - Dark ESR schedule combining all prior operations (!527)
- NV centers - `BasicElectronicNVElement` parameters accept physics-motivated values (!551)
- Qblox backend - Add preparation of acquisition settings and accompanying datastructures for NV centers (!567)
- Qblox backend - Add TriggerCount to `QRMAcquisitionManager` (!556)
- Qblox backend - Added method for gain configuration, overriding gain now raises ValueError (!533)
- Qblox backend - Provide sequencer setting to reset AWG offset and AWG gain to a known value (default 0 for offset and 1 for gain) before starting experiment (!544, #377)
- Qblox backend - Remove `mix_lo` from `SequencerSettings` (!896)
- Qblox backend - Typecast attenuations to `int`s before assigning them (!570)
- QuantumDevice - `BasicTransmonElement` can now be serialized to json string and deserialized via ``__getstate__/__init__`` (!510)
- Schedule Functions - Make experiment-related schedule functions available in `quantify_scheduler.schedules` (!572)
- Tests - Removed unused `lo0` and added `ttl_acq_threshold` in `qblox_test_mapping_nv_centers.json` so that `TestNVDarkESRSched` suite passes (!579, follow-up to !571)
- Visualization - Make box separation in circuit_diagram_matplotlib always equal to one (!589)
- Waveforms - Fix `sudden_net_zero` waveform generation function misunderstands `amp_B` (!549, #390)

## 0.10.1 (2022-12-20)

### Merged branches and closed issues

- Compilation - Reinstate `add_pulse_information_transmon` device compilation backend (removed in !526) (!557)
- Qblox backend - Drop no key `"distortion_corrections"` supplied log message level to debug (!560)

## 0.10.0 (2022-12-15)

### Breaking changes

- Deprecation - `add_pulse_information_transmon` is removed and `device_config` must now be of type `DeviceCompilationConfig` (!526)
- Qblox backend - Strictly requires v0.8.x of the `qblox-instruments` package (!512)
- Qblox backend, Operations - The phase of the measurement clock is set to zero at the start of each measurement by default (!434, #296)
- Qblox backend - `QRMAcquisitionManager` now truncates returned acquisitions to actual lengths (!478)
- Qblox backend - `mix_lo` flag now specifies if IQ mixing should be applied to LO (!482)
- Operations - ShiftClockPhase now uses `phase_shift` as keyword instead of `phase` (!434)

### Merged branches and closed issues

- Deprecation - Replaced `DeprecationWarning`s with `FutureWarning`s so they are shown to end-users by default (!536, counterpart to quantify-core!411)
- Deprecation - Remove code and test dependencies on deprecated `data` keyword in `Operations` (!545, #381)
- Documentation - Instrument naming requirements in qblox hardware config (!531)
- Documentation - Make class `__init__` docstring visible on Sphinx (!541, #314)
- Documentation - Improve parameter documentation for DeviceElements (!493)
- Documentation - Building sphinx documentation will now raise an error if one of the code cells fails to run (!514)
- Gate Library - Added Deprecation Warning for `acq_channel` Keyword in Measure (!491)
- Git - Changed git merge strategy to "union" for CHANGELOG.md and AUTHORS.md to reduce amount of merge conflicts (!495)
- Instrument Coordinator - Check if a parameter cache is valid before lazy setting (!505, #351)
- Instrument Coordinator - Changed argument of `GenericInstrumentCoordinatorComponent` from `name` to `instrument_reference`. (!497)
- NV centers - First operation `SpectroscopyOperation` with compilation for Qblox hardware (!471)
- NV centers - `Reset` operation with compilation for Qblox hardware (!485)
- NV centers - `Measure` operation using TriggerCount acquisition; only device compilation so far (!490)
- NV centers - `ChargeReset` operation only device compilation so far (!496)
- NV centers - `CRCount` operation using TriggerCount acquisition; only device compilation so far (!502)
- Qblox backend - Introduce `"sequence_to_file"` param in qblox hardware config to allow skipping writing sequence json files to disk (#108, !438)
- Qblox backend - Minor adjustments to `NcoPhaseShiftStrategy` to make compilation of `ShiftClockPhase` compatible with qblox-instruments==0.8.0 (!481)
- Qblox backend - `QbloxInstrumentCoordinatorComponentBase` accepts both `InstrumentModule` and `InstrumentChannel` as instrument reference to cluster module (!508)
- Qblox backend - Explicit error message when trying to do acquisitions on a QCM (!519)
- Qblox backend - Renamed `output_mode` to `io_mode` in `get_operation_strategy` (!497)
- Qblox backend - Added `TriggerCountAcquisitionStrategy` to acquisitions, generating the Q1ASM commands. (!540) 
- Tests - Refactored tests to remove duplicated `temp_dir` setup, and only use `tmp_test_data_dir` fixture (#370,  !525)
- Tests - Update tests to use `mock_setup_basic_transmon_with_standard_params` where needed (#369, !522)
- Tests - Tests refactoring, move to `mock_setup_basic_transmon_with_standard_params` and replace `qcompile` by `SerialCompiler` (!516)
- Validation - Replaced most of the asserts with raising proper exceptions so that they are raised in production environment too (#342, !499)
- Visualization - Updated `pulse_diagram_matplotlib` to be compatible with future quantify-core release (!517)
- Visualization - Show clock name in plotly pulse diagram (!547)

## 0.9.0 (2022-10-06)

### Breaking changes

- Deprecated methods removed:
  - `QuantumDevice`
    - `components` -> `elements`
    - `get_component` -> `get_element`
    - `add_component` -> `add_element`
    - `remove_component` -> `remove_element`
  - `ScheduleBase`
    - `plot_circuit_diagram_mpl` -> `plot_circuit_diagram`
    - `plot_pulse_diagram_mpl` -> `plot_pulse_diagram`
- Compilation - Compilation is now a graph. (#305, !407)
- Operations - Allow moving to `qcodes` >=0.34 (#300, !473)
    - Disallow `"_"` in `DeviceElement` names to comply with qcodes version 0.34
    - Enforces `"_"` as the separator between device elements in `Edge` names
    - Note: We are still on `qcodes` 0.33 due to pin in `quantify-core` package requirements
- Operations, Resources, and Schedulables - Deprecate the use of the `data` argument (!455)
- Qblox ICCs - Hotfix for storing scope acquisition (broken by !432) (!470)
- Qblox ICCs - Only activate markers and LOs of used outputs to prevent noise (!474)

### Merged branches and closed issues

- Docs - Support for `myst-nb` added (#301, !407)
- Docs - Sources are converted from restructured text format to MyST markdown. (!452)
- Docs - Add pin on `nbclient<0.6` for Read-the-Docs to build; Remove various old temp requirement pins (!477)
- Docs - Added documentation and unit tests for the Rxy, X, X90, Y and Y90 unitaries (#349)
- Gettables - Added a `ProfiledScheduleGettable` for profiling execution times of schedule steps. (!420, !469)
  - Please note: Setup in need of refactoring so interface is subject to change (see #320)
- Instrument Coordinator - Small fix for `search_settable_param` when scheduler is searching for qcodes parameters (!461)
- JSON utilities - Remove `repr` based serialization/deserialization methods (!445, #248)
- JSON utilities - Extend the capabilities of the ``__getstate__/__setstate__`` json serializer (!445, #248)
- Qblox ICCs - Added input/output gain/attenuation configurable hardware parameter (!458)
- Structure - Pydantic-based model is now used to validate latency corrections. (!467, #333)
- Zhinst backend - Raise a more understandable exception when compiling an acquisition with larger than allowed duration (!407).

## 0.8.0 Support for two qubit operations and basic CZ-gate implementation (2022-08-10)

### Breaking changes

- Operations - Pin `qcodes` package to \<0.34.0 due to breaking `Edge` naming (#300, !409)
- Qblox backend - Sequencers are now dynamically allocated. The hardware config file schema was changed. (!328)
    - For each instrument, the config now contains a `portclock_configs` entry, a list with a dictionary of settings per port-clock combination
    - See <https://quantify-quantify-scheduler.readthedocs-hosted.com/en/0.8.0/tutorials/qblox/recent.html>
- Qblox backend - Strictly requires v0.7.x of the `qblox-instruments` package (!449)
- Zhinst backend - Strictly requires v21.8.20515 of the `zhinst` package (!387)

### Merged branches and closed issues

- Compilation - Added `acq_protocol` optional parameter to the `Measure` gate. (!386)
- Compilation - Call `determine_absolute_timing` in `qcompile` when no `device_cfg` supplied. (!436)
- Compilation - Decrease test usage of deprecated transmon_test_config.json / add_pulse_information_transmon (!450)
- DRAG Pulse - Removed an extra G_amp factor from the Q component (derivative pulse). (#298, !406)
- Docs - Fix API reference pages on read-the-docs (#303, !413)
- Docs - Pin sphinx to 5.0.2 due to crash in napoleon (!437)
- Docs - Unpin sphinx >=5.1.1 (!445)
- Docs - Fix jsonschemas not rendered on read-the-docs (!448)
- Docs - Clarify port and clock concepts (!431)
- Docs - New scheduler tutorials: Schedules and Pulses; Compiling to Hardware; Operations and Qubits (!336, !439)
- Gettables - Added `generate_diagnostics_report` method to save the internal state of `ScheduleGettable` to a zip-file. (!408)
- Helpers - Moved `MockLocalOscillator` definition from tests to `helpers.mock_instruments.MockLocalOscillator` (!392, !336).
- JSON utilities - Add JSON serialization/deserialization methods based on `__getstate__`/`__setstate__` (!444)
- Operations - Added a `symmetric` key in the `gate_info` to flag symmetric operations. (!389)
- Operations - Introduce basic CZ-gate via `CompositeSquareEdge` (utilizing `quantify_scheduler.operations.pulse_factories.composite_square_pulse`) (!411)
    - Replaces the incomplete `SuddenNetZeroEdge` basic CZ-gate implementation
- Operations - Rxy theta rotations now fall into the domain of \[-180 to 180) degrees. (!433)
- QuantumDevice - Added implementation for `edges` in the quantum device config in order to support two qubit operations. (!389)
    - The `Edge` has been added as an abstract base class for edges to be added to a device.
- Qblox backend - Only add clocks to the schedule that are actually being used, avoids trying to assign frequencies for unused clocks (#278, !371)
- Qblox backend - Fix for supplying negative NCO phase (!393)
- Qblox backend - Fix compilation of ShiftClockPhase (!404, broken by merge of !328)
- Qblox backend - Fix for outputting signals on even output paths of qblox hardware in real_output_x mode (!397)
- Qblox backend - Make Qblox backend compatible with generic downconverter values in hardware_config (!418)
- Qblox backend - Fix for 90 degree phase shift on even output paths as a result of the !397 hotfix. (!412)
- Qblox backend - Fix cluster compatibility when converting old hwconfig to new specs (!419)
- Qblox backend - Latency corrections must now be defined in top layer of hw config (!400)
- Qblox backend - Fix combination of cluster and latency corrections when converting hw_configs to new specs  (!417)
- Qblox backend - Fix handling of composite pulses (#299, !411)
- Qblox backend - Implementation of distortion correction (#285, !388)
- Qblox backend - Fix incompatibility of distortion_correction parameters as numpy arrays (!426)
- Qblox backend - Remove all references to the inactive `line_gain_db` param (!435)
- Qblox ICCs - Fix for setting `scope_acq_sequencer_select` for QRM and QRM-RF (!432, !441)
- Qblox ICCs - Fix `ClusterComponent.prepare` mutating the schedule (!443)
- Schedules - Revert rename of `trace_schedule` done in !432 and rename new schedule using gates to `trace_schedule_circuit_layer` (!442)
- Schedules - Make `AcquisitionMetadata` a serializable class (!446)

## 0.7.0 Support for qblox-instruments v0.6.0, new BasicTransmonElement, change for triggers in Zhinst backend (2022-04-11)

### Breaking changes

- Qblox ICCs - Updated Qblox components for using the new unified-style qblox driver (see <https://gitlab.com/quantify-os/quantify-scheduler/-/wikis/Qblox-ICCs:-Interface-changes-in-using-qblox-instruments-v0.6.0>) (!377).
- Qblox backend - Strictly requires v0.6.0 of the qblox-instruments package (!377).
- Zhinst backend - Hardware config for the devices. Replaced keyword "triggers" to "trigger", and the value type from `List[int]` to `int`. E.g. old style, `"triggers": [2]`, new style, `"trigger": 2` (#264, !372).

### Merged branches and closed issues

- QuantumDevice - The `DeviceElement` has been added as an abstract base class for elements to be added to a device (#148, !374).
- QuantumDevice - The `BasicTransmonElement` has been added that generates a device config in a more structured manner (#246, !374).
- QuantumDevice - Fixed a bug in the `BasicTransmonElement` where operations had clock-frequencies (`float`) specified instead of clocks (`str`) (!379).
- QuantumDevice - The `TransmonElement` will be deprecated after version 0.8 (!374).

## 0.6.0 Full support for multiplexed readout, transmon element update, fixes to backends (2022-03-10)

### Breaking changes

- Compilation - Deprecated `add_pulse_information_transmon` in favor of `compilation.backends.circuit_to_device.compile_circuit_to_device` (#64, #67, !339).
- Compilation - attempting compilation with missing values in the `DeviceCompilationConfig` configuration will now raise validation errors. Be sure to set initial values when generating a config using the `QuantumDevice` object (!339)
- Compilation - Device compile making use of `.compile_circuit_to_device` no longer modifies the input schedule (#249, !339).
- Compilation - When specifying multiple timing constraints for a schedulable, the constraint specifying the latest time determines the absolute time of the shedulable (!309)
- Gettables - `ScheduleGettableSingleChannel` renamed to `ScheduleGettable` as it now supports multiple acquisition channels (!299).
- Hardware config - Removed the need for a `ic_` prefix from the hardware config (!312).
- Instrument Coordinator - IC now adds a `GenericInstrumentCoordinatorComponent` to itself on instantiation by default (!350)
- Instrument Coordinator - IC stop function has an `allow_failure` parameter which allows IC components attached to it to fail to stop with warning instead of raising errors. Allows for situations when some components cannot have a stop instruction sent before the prepare stage. (!359)
- Operations - The internal behavior of how acquisition channels and acquisition indices are configured in the `Measure` operation has changed slightly. See #262 for details. (!339).
- Operations - Added "operation_type" key to the schema. (!345)
- Structure - `Schedule.timing_constraints` has been renamed to `Schedule.schedulables`. It now points to a dictionary of schedulables rather than a list of dicts. (!309)
- Structure - Pydantic-based model is now used for the data structures. (!341)
- Visualization - Deprecated `plot_circuit_diagram_mpl` and `plot_pulse_diagram_mpl` in `ScheduleBase` in favour of `plot_circuit_diagram` and `plot_pulse_diagram` (!313)
- Qblox backend - Strictly requires v0.5.4 of the qblox-instruments package (!314)
- Zhinst backend - Due to !312, the csv files used to upload the waveforms to the UHFQA no longer use the `ic_` prefix in their filenames. (!334)
- Zhinst backend - Fixes bug when doing SSRO experiments. No more duplicated shots. Adds support for BinMode.APPEND during compilation. (#276, !358)
- Zhinst backend - Removed `latency` and `line_trigger_delay` keys in the channels of the devices for the Zhinst hardware config. (!363)
- Zhinst backend - Added `latency_corrections` main entry in the Zhinst hardware config for latency corrections on a port-clock combination basis. (!363)

### Merged branches and closed issues

- Compilation - Added a new compilation backend `compilation.backends.circuit_to_device.compile_circuit_to_device` for the quantum-circuit to quantum-device layer (#64, #67, !339).
- Compilation - Fixed `add_pulse_information_transmon` when using "Trace" acquisition mode (!300)
- Compilation - Fixed the deprecation warnings from pandas `DataFrame.append`. (!347)
- Docs - Pinning qcodes package to \<0.32.0 due to Read the Docs API reference failure (!361)
- Gettables - `ScheduleGettable` now first stops all instruments in IC during initialization (!324)
- Schedules - Adds a multiplexing verification schedule. (!329)
- Operations - Sudden Net Zero from Negirneac 2021 added to the `pulse_library` (!339)
- Operations - Docstrings for the X90, X, Y90, Y, and Rxy gate unitary have been aligned with literature. (#261, !305)
- Operations - Adds an optional "data" argument to staircase pulse. (!335)
- Pulse library - Added `ShiftClockPhase` operation that can be used to shift the phase of a clock during execution of a `Schedule` (!346)
- Pulse library - Added a numerically defined pulse. (!157)
- QuantumDevice - Unknown values are initialized as `float('nan')` (#274, !356)
- TransmonElement - Corrected the motzoi parameter range validator. (!351)
- Visualization - Adds visualisation of acquisitions to plotly pulse diagrams (!304)
- Visualization - Add `plot_pulse_diagram` and `plot_circuit_diagram` to schedule for easier method names, and enable plotly visualization directly from `ScheduleBase` (!313)
- Utilities - Migrates the utilities from quantify-core. (!357)
- Generic ICC - Adds support for nested parameters. (!330)
- Qblox ICCs - Stop now disables sync on all sequencers to prevent hanging during next run, where it gets re-enabled if needed (!324)
- Qblox ICCs - `_QRMAcquisitionManager._get_scope_data` now has correct return type (#232, !300)
- Qblox ICCs - Fixed bug where QRM scope mode sequencer does not get set correctly (!342)
- Qblox ICCs - Fixed reference source cluster issue when it is not being set correctly. (!323)
- Qblox backend - NCO phase now gets reset every averaging loop (!337)
- Qblox backend - Enables RF output switch at the start of a program. (!344)
- Qblox backend - Added logic for changing the NCO phase during execution of a `Schedule` (!346)
- Qblox backend - Added ability to correct for latency by delaying program execution on a per sequencer basis (!325)
- Qblox backend - Compilation with local oscillators changed to work with generic instrument coordinator components (!306)
- Qblox backend - Refactored operation handling and greatly increased test coverage (!301).
- Qblox backend - Made max duration of wait instructions (!319).
- Qblox backend - Fixed an issue with the downconverter frequency correction. (!318)
- Qblox backend - Temporary fix for a floating point rounding error when calculating the length of pulses. (#284, !365)
- Zhinst backend - Fixed the ZI resolver return typehint. (!307)
- Zhinst backend - Fixed an issue when compiling seqc programs for multiple sequencers end up overwriting the first sequencer. (!340, #260)

## 0.5.2 Fixes to backends, and other incremental fixes  (2021-12-08)

### Breaking changes

- Dependency on `jsonschema` has been replaced with `fastjsonschema`. (!284, !293)
- Zhinst hardware config json schema has changed. See the example schema. (!283)
- In `hardware_compile` function, the `hardware_map` is changed to `hardware_cfg` parameter. (!279)
- Remove enum tools dependency (!270)

### Merged branches and closed issues

- Compilation - The `determine_absolute_scheduling` function now sorts the list of labels in the timing constraints, and then a binary search (via `np.searchsorted`) is applied. (!272, !274)
- Compilation - Make `device_cfg` an optional argument of qcompile(!281)
- Compilation - renamed the hardware_mapping argument of qcompile into hardware_cfg (#165, !279)
- Compilation - Introduced the hardware_compile function to perform the hardware compilation returning a CompiledSchedule (#224, !279)
- Docs - Updating user guide to mention correctly the QuantumDevice and ScheduleGettable(s) available. (!209)
- Infrastructure - Adds rich package in the requirements since tutorials use it. (!276)
- Operations - The `locate` function now uses the `functools.lru_cache` to cache the result (only for python >= 3.8). For python 3.7, behaviour remains the same.  (!273, !275)
- Operations - Resolved a minor issue where identical Rxy rotations (for angles >360) would be treated as separate operations in a schedule (!263)
- Visualization - Adds a function `plot_acquisition_operations` which together with the new `AcquisitionOperation` class will help highlight acquisition pulses in the pulse diagrams. (!271, !277)
- Zhinst backend - Large parts of the Zhinst backend have been rewritten. This should resolve a range of issues. (!263)
    - Calculation of the timelines for different operations now makes using of a timing table, improving code readability and debugability.
    - Timing issues related to triggering should be resolved (#218)
    - The backend can now always use the same hardware configuration file (#214)
    - Acquisition is now done using the StartQA instruction (#213)
    - error handling in the Zhinst backend has been improved catching several exceptions at compile time of the schedule instead of manifesting in unexpected results during runtime.
    - Local oscillators through the ZI backend uses the GenericInstrumentCoordinatorComponent. Configures other parameters other than frequency. (!283, #204)
- Qblox backend - only check major and minor version when checking compatibility with the qblox_instruments package (!290)
    - Added support for the Qblox Downconverter (!297)
    - Added workaround for staircase_amplitude. (!292)
    - Fix looped acquisition integration time, fix acquire index offset by one (!291)
    - Qblox instruments version == 0.5.3 (!289)
    - Fix sequencer_sync_en not being reset in the qblox instrument coordinator component. (!285)
    - Fix rounding of time to samples in qblox backend (!282)
    - Fix pulse stitching at zero amplitude. (!280)
    - Allow instruction generated staircase with modulation (!278)
- Utilities - Improve JSON validation speed (!284)
- Utilities - Improve operation deserialization speed (!273)
- Bugfix - For calculating the pulse area, the mathematical area is used instead of area of sampled pulse. (!242, !286)
- Bugfix - Fix for plot window operations (!294)

## 0.5.1 Incremental fixes, refactoring, and addition of convenience methods and classes (2021-11-11)

### Breaking changes

- InstrumentCoordinator - `last_schedule` is now a property (!252).
- Structure - We have refactored the Operation and Schedule classes out of the types module and moved the different operation libraries (acquisition_library, gate_library, and pulse_library) (#217, !256).
    - `quantify_scheduler.types.Operation` -> `quantify_scheduler.operations.operation.Operation`, the import `quantify_scheduler.Operation` still works.
    - `quantify_scheduler.types.Schedule` -> `quantify_scheduler.schedules.schedule.Schedule`, the import `quantify_scheduler.Schedule` still works.
    - `quantify_scheduler.types.CompiledSchedule` -> `quantify_scheduler.schedules.schedule.CompiledSchedule`
    - `quantify_scheduler.types.ScheduleBase` -> `quantify_scheduler.schedules.schedule.ScheduleBase`
    - `quantify_scheduler.types.AcquisitionMetadata` -> `quantify_scheduler.schedules.schedule.AcquisitionMetadata`
    - `quantify_scheduler.acquisition_library` -> `quantify_scheduler.operations.acquisition_library`
    - `quantify_scheduler.gate_library` -> `quantify_scheduler.operations.gate_library`
    - `quantify_scheduler.pulse_library` -> `quantify_scheduler.operations.pulse_library`

### Merged branches and closed issues

- Control - Add option to set output port in heterodyne_spec_sched (!262)
- Control - Expand SingleChannelScheduleGettable to support trace acquisitions (!248)
- Control - Update create_dc_compensation_pulse behaviour and docstring. (!244)
- Control - Refactor ScheduleGettableSingleChannel (!240, !249)
- Control - Reduce the default init_duration of spectroscopy schedules (!237)
- Generic ICC - Added a GenericInstrumentCoordinatorComponent. (!267)
- ICCs - InstrumentCoordinatorComponentBase now has a `force_set_parameter` as a ManualParameter to enable the user to switch the lazy_set behaviour when setting parameters of the instruments connected to the InstrumentCoordinatorComponent. (!267)
- Qblox ICCs - Adds a lazy_set behaviour by default when setting parameters with the same value to an instrument connected to the Qblox ICC. (!230)
- Visualization - made matplotlib schedule visualization methods accessible as methods `plot_circuit_diagram_mpl` and `plot_pulse_diagram_mpl` of the `Schedule` class (!253)
- Visualization - resolved a bug where a schedule was modified when drawing a circuit diagram (#197, !250)
- Visualization - Add support for window operation to transmon backend (!245)
- Infrastructure - Fix and enhance pre-commit + add to CI (!257, !265)
- Infrastructure - Added prospector config file for CI. (!261)
- Bugfix - Removed redundant `determine_absolute_timing` step in `qcompile`. (!259)
- Bugfix - Ramp pulse sampling utilizing `np.linspace` behaviour changed. (!258)
- Docs - Adds the new Quantify logo similar to quantify_core. (!266)
- Docs - Enhance documentation of public API for reimported modules \[imports aliases\] (!254)
- Docs - Fixes the funcparserlib error in rtd. (!251)
- Docs - Updated Qblox backend docs to include the new features. (!247)

## 0.5.0 Expanded feature sets hardware compilation backends (2021-10-25)

### Breaking changes

- The `schedules.timedomain_schedules.allxy_sched` function no longer accepts the string "All" as an argument to the `element_select_idx` keyword.
- The `QuantumDevice.cfg_nr_averages` parameter was renamed to `QuantumDevice.cfg_sched_repetitions`
- The call signature of `gettables.ScheduleVectorAcqGettable` has been renamed to `gettables.ScheduleGettableSingleChannel`, and the call signature has been updated according to #36 to no longer accept several keyword arguments.
- Qblox Backend - The NCO phase is now reset at the start of a program (!213).
- Qblox Backend - Compilation now requires qblox_instruments version 0.5.0, 0.5.1 or 0.5.2 (!214, !221).

### Merged branches and closed issues

- Compilation - Added the ability to specify the BinMode at the quantum-circuit layer (#183, !180).
- Compilation - qcompile no longer modifies schedules (#102, !178).
- Control - Added a first version of the QuantumDevice object (#148, !180).
- Control - A single-qubit ScheduleGettable has been added (#36, !180).
- Docs - Added bibliography with sphinxcontrib-bibtex extension (!171).
- Docs - Fixed missing files in API reference (!176).
- InstrumentCoordinator - CompiledSchedule class added to specify interfaces of InstrumentCoordinator and compilation functions (#174, !177).
- InstrumentCoordinator - CompiledSchedule.last_schedule method added to provide access to last executed schedule (#167, !177).
- Qblox Backend - Added support for qblox_instruments version 0.4.0 (new acquisition path) (!143).
- Qblox Backend - Added support for real time mixer corrections rather than pre-distorting the uploaded waveforms (!192).
- Qblox Backend - Waveforms are now compared using the normalized data array rather than the parameterized description (!182).
- Qblox Backend - Support for append bin mode (#184, !180).
- Qblox Backend - Support for using real value pulses on arbitrary outputs added (!142).
- Qblox Backend - Compilation now supports 6 sequencers for both the QCM as well as the QRM (!142).
- Qblox Backend - Support for a cluster, along with its QCM, QRM, QCM-RF and QRM-RF modules (!164)
- Qblox Backend - Registers are now dynamically allocated during compilation (!195)
- Zhinst backend - No exception is raised when an LO that is in the config is not part of a schedule. (#203, !223)
- Zhinst backend - Instrument coordinator components for ZI will only be configured when the settings used to configure it have changed (#196, !227)
- Zhinst backend - Solved a bug that caused single-sideband demodulation to not be configured correctly when using the UHFQA (!227)
- Zhinst backend - Warnings raised during compilation of seqc programs will no longer raise an exception but will use logging.warning (!227)
- Zhinst backend - resolved a bug where the instrument coordinator cannot write waveforms to the UHFQA if it has never been used before (!227)
- Zhinst backend - resolved a bug where multiple identical measurements in a schedule would result in multiple integration weights being uploaded to the UFHQA (#207, !234)
- Zhinst backend - resolved a bug where the UHFQA would not be triggered properly when executing a schedule with multiple samples (batched mode) (#205, !234)
- Qblox ICCs - Compensated integration time for Qblox QRM IC component (!199).
- Qblox ICCs - Added error handling for error flags given by `get_sequencer_state` (!215)
- QuantumDevice - Added docstrings to the TransmonElement parameters (!216, !218)
- Qblox ICCs - QCoDeS parameters are now only set if they differ from the value in the cache (!230)
- Visualization - Allow user defined axis for plotting circuit diagram (!206)
- Visualization - Adds schedule plotting using matplotlib and a WindowOperation to help visualize pulse diagrams (!225, !232)
- Other - Added method `sample_schedule` to sample a `Schedule` (!212)
- Other - The `RampPulse` has an extra (optional) parameter `offset` (!211)
- Other - Updated existing schedules to make use of the acquisition index (#180, !180).
- Other - Added a function to extract acquisition metadata from a schedule (#179, !180).
- Other - The soft square waveform can now be evaluated with only one datapoint without raising an exception (!235)
- Other - Added a function that generates a square pulse that compensates DC components of a sequence of pulses (!173)

## 0.4.0 InstrumentCoordinator and improvements to backends (2021-08-06)

### Breaking changes

- Change of namespace from quantify.scheduler.\* to quantify_scheduler.\*

### Merged branches and closed issues

- Changes the namespace from quantify.scheduler to quantify_scheduler (!124)
- InstrumentCoordinator - Add is_running property and wait_done method. Closes #133 (!140)
- InstrumentCoordinator - Add instrument coordinator reference parameter to transmon element (!152)
- InstrumentCoordinator - Prefix serialized settings for ZI ControlStack components. (!149)
- InstrumentCoordinator - Refactored ControlStack name to InstrumentCoordinator (!151)
- InstrumentCoordinator - Make use of InstrumentRefParameters (!144)
- InstrumentCoordinator - Add controlstack class (!70)
- InstrumentCoordinator - Add Zurich Instruments InstrumentCoordinatorComponent. (!99)
- InstrumentCoordinator - Add Qblox InstrumentCoordinatorComponent. (!112)
- InstrumentCoordinator - Avoid garbage collection for instrument coordinator components (!162)
- Qblox backend - Removed limit in Qblox backend that keeps the QCM sequencer count at 2 (!135)
- Qblox backend - Restructured compilation using external local oscillators. (!116)
- Qblox backend - Added Chirp and Staircase pulses; and efficient implementation for QD spin qubit experiments (!106)
- Qblox backend - Only run `start_sequencer` on pulsar instruments which have been armed (!156)
- Zhinst backend - Assert current with new sequence program to skip compilation (!131)
- Zhinst backend - Deserialize zhinst settings from JSON to ZISettingsBuilder (!130)
- Zhinst backend - Add waveform mixer skewness corrections (!103)
- Zhinst backend - Add backend option to enable Calibration mode (#103, !123)
- Zhinst backend - Replace weights string array with a numerical array in JSON format (!148)
- Zhinst backend - Add grouping of instrument settings (!133)
- Zhinst backend - Add qcompile tests for the zurich instruments backend (!118)
- Zhinst backend - Add repetitions parameter (!138)
- Zhinst backend - Fixes the bug where the seqc in the datadir is not copied to the webserver location. (!165)
- Fix for circuit diagram plotting failure after pulse scheduling (#157, !163)
- Fixed typo in the gate_info of the Y gate in the gate_library (!155)
- Add artificial detuning in Ramsey Schedule and bug fixes (!120)
- Use individual loggers per python file (!134)
- Recolour draw circuit diagram mpl (!96)
- Fix issues with timedomain schedules (!145)
- Renamed input parameters of quantify_scheduler.schedules.\* functions. (!136)
- Added acquisitions to circuit diagram (!93)
- Add string representations to acquisition protocols of the acquisitions library (!114)
- Transmon element and config generation (!75)
- Rename operation_hash to operation_repr (!122)
- Add types.Schedule from_json conversion (!119)
- Add missing return types (!121)
- Add serialization to Operations (!110)

## 0.3.0 Multiple backends support (2021-05-20)

- Added support for both Qblox and Zurich Instrument backends.
- Added convenience pylintrc configuration file.
- Added examples for timedomain and spectroscopy schedules.

### Breaking changes

- Major refactor of the Qblox backend. (For example, it's now `quantify_core.backends.qblox_backend` instead of the previous `quantify_core.backends.pulsar_backend`)
- Qblox backend requires strictly v0.3.2 of the qblox-instruments package.

### Merged branches and closed issues

- Add mixer skewness corrections helper function. (!102)
- Added Qblox backend support. (!81)
- Compile backend with ZISettingsBuilder. (!87)
- Add vscode IDE config files. (!100)
- Add ZISettingsBuilder class. (!86)
- Added representation to gates in gate library and defined equality operation. (!101)
- Fix/operation duration. Fixes #107. (!89)
- Feat/long pulses fix validators name. (!90)
- Implemented long square pulses unrolling (for waveform-memory-limited devices). (!83)
- Changed Qblox-Instruments version to 0.3.2. (!88)
- Feature: Improve overall zhinst backend timing. (!77)
- Plotly cleanup. (!69)
- Pulsar backend version bump. (!82)
- Added zhinst backend support. (!49)
- Added example timedomain programs. (!71)
- Added example spectroscopy programs. (!64)
- Added pylintrc configuration file. (!55)
- Added repetitions property to Schedule. (!56)
- Added Acquisition Protocols. (!51)
- Hotfix for filename sanitization pulsar backend. (!61)
- Pulsar backend function sanitization. (!60)
- Potential fix time-out pulsar. (!58)
- Updated Pulsar backend version to v0.2.3.. (!57)
- Fixed datadir related bugs. (!54)
- Added Station implementation. (!52)
- Pulsar backend v0.2.2 check. (!48)
- Fix for issue with acq delay. (!45)
- Fix for issue #52. (!44)
- Add artificial detuning to Ramsey schedule (!120)
- Added support for the Qblox Pulsar QCM-RF/QRM-RF devices (!158)

## 0.2.0 Hybrid pulse- gate-level control model (2021-01-14)

- Major refactor of the scheduler resource code enabling hybrid pulse- gate-level control.
- Moved quantify_scheduler.types.Resource class to a separate quantify_scheduler.resources module.
- Adds a BasebandClockResource class within the newly created quantify_scheduler.resources module.
- Moved QRM and QCM related classes to the quantify_scheduler.backends.pulsar_backend module.
- In quantify_scheduler.compilation, rename of function '\_determine_absolute_timing' to 'determine_absolute_timing'. Argument changed from clock_unit to time_unit.
- In quantify_scheduler.compilation, rename of function '\_add_pulse_information_transmon' to 'add_pulse_information_transmon'.
- Added ramp waveform in quantify_scheduler.waveforms.
- Added schemas for operation and transmon_cfg.
- Added a basic hybrid visualisation for pulses using new addressing scheme.
- Operations check whether an operation is a valid gate or pulse.
- Refactor of visualization module. Moved quantify_scheduler.backends.visualization to quantify_scheduler.visualization module. Expect code breaking reorganization and changes to function names.
- Pulsar backend version now checks for QCM and QRM drivers version 0.1.2.

### Merged branches and closed issues

- fix(pulse_scheme): Add tickformatstops for x-axis using SI-unit 'seconds'. Closes #39. (!39)
- Resolve "y-axis label is broken in plotly visualization after resources-refactor". Closes #45. (!38)
- Resources refactor (!28, !29, !30)
- Hybrid visualisation for pulses and circuit gate operations. Closes #22 and #6. (!27)
- Support Pulsar parameterisation from scheduler. Support feature for #29. (!2)
- Operation properties to check if an operation is a valid gate or pulse. Closes #28 (!25)
- Visualisation refactor. Closes #26. (!22)
- Windows job (!20)
- Changed Pulsar backend version check from 0.1.1 to 0.1.2. (!21)

## 0.1.0 (2020-10-21)

- Refactored scheduler functionality from quantify-core into quantify-scheduler
- Support for modifying Pulsar params via the sequencer #54 (!2)
- Simplification of compilation through `qcompile` (!1)
- Qubit resources can be parameters of gates #11 (!4)
- Circuit diagram visualization of operations without no pulse info raises exception #5 (!5)
- Pulsar backend verifies driver and firmware versions of hardware #14 (!6)
- Sequencer renamed to scheduler #15 (!7)
- Documentation update to reflect refactor #8 (!8)
- Refactor circuit diagram to be more usable !10 (relates to #6)
- Unify API docstrings to adhere to NumpyDocstring format !11
- Changes to addressing of where a pulse is played !9 (#10)
- Renamed doc -docs folder for consistency #18 (!12)
- Moved test folder outside of project #19 (!14)
- Add copyright notices and cleanup documenation #21 (!13)
- Add installation tip for plotly dependency in combination with jupyter #24 (!15)

```{note}
- \# denotes a closed issue.
- ! denotes a merge request.
```
