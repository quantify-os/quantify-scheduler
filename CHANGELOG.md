# Changelog

## 0.18.0 (2023-12-22)

### Release highlights

**General updates**
- Add device elements and edges to quantum devices without keeping an explicit reference to them, e.g. do `quantum_device.add_element(BasicTransmonElement("q0"))`.
- `DeviceCompilationConfig` updated: `backend` key was removed.

**Qblox backend improvements**
- New features
  - Square pulses now support complex value pulses, via complex valued amplitude.
- Fixes
  - Marker pulse functionality corrected on RF modules (baseband not affected).
  - `ScheduleGettable` option `always_initialize=False` fixed, speeding up repeated execution by skipping compilation and initializing instruments.

### Breaking changes

- Compilation 
  - Changes to the `CompilationConfig` generation in the `QuantumDevice` to support parsing custom (backend-specific) `HardwareCompilationConfig` datastructures. (!840)
  - The `backend` field in the `HardwareCompilationConfig` was replaced by the `config_type` field, which contains a (string) reference to the backend-specific `HardwareCompilationConfig` datastructure.
  - The `backend` field was removed from the `DeviceCompilationConfig`.
  - The `compilation_passes` field was moved from the `SerialCompilationConfig` into the `DeviceCompilationConfig` and `HardwareCompilationConfig` datastructures.
    - Move the default device `compilation_passes` from the `QuantumDevice.generate_device_config()` to the `DeviceCompilationConfig` datastructure to ensure backwards compatibility (!884).
  - Migration:
    - `DeviceCompilationConfig`: If you are loading a stored `DeviceCompilationConfig` (instead of relying on the device config generation of the `QuantumDevice`), remove the `"backend"` key.
    - `HardwareCompilationConfig`: If you are already using the new-style `HardwareCompilationConfig`, change the `"backend"` key to `"config_type"`.
      - For Qblox: `"config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig"`,
      - For Zurich Instruments: `"config_type": "quantify_scheduler.backends.zhinst_backend.ZIHardwareCompilationConfig"`.

- Operations 
  - Modify the parameters of the `VoltageOffset` operation (!863):
    - Deprecate the `duration` parameter. The duration of a `VoltageOffset` is always 0. Using the `duration` parameter results in a `FutureWarning`.
    - Make the `port` parameter non-optional, and the `clock` parameter by default `BasebandClockResource.IDENTITY`.
      - Note: This changes the order of the arguments, please check `VoltageOffset` for the new signature. 
  - `VoltageOffset` and `StitchedPulse` code moved to Qblox backend (!863):
     - `VoltageOffset` from `quantify_scheduler.operations.pulse_library` to `quantify_scheduler.backends.qblox.operations.pulse_library`.
     - `quantify_scheduler.operations.stitched_pulse` to `quantify_scheduler.backends.qblox.operations.stitched_pulse`.
     - `staircase_pulse`, `long_square_pulse` and `long_ramp_pulse` from `quantify_scheduler.operations.pulse_factories` to `quantify_scheduler.backends.qblox.operations.pulse_factories`.

- Pulses 
  - The phase argument for `SquarePulse` has been removed. (!867)

- Qblox backend 
  - Fix missing signal on O2 and O4 outputs of baseband modules in real mode (reverting !803). (!891)
  - Fix to allow running `ScheduleGettable` with option `always_initialize=False`. (!868)
    - Arming the sequencers is now done via `InstrumentCoordinator.start()` instead of `InstrumentCoordinator.prepare()`.

- Schedulables 
  - Rename `Schedulable["operation_repr"]` to `Schedulable["operation_id"]` (!775, #438)

### Merged branches and closed issues

- Compilation
  - Implement `Connectivity` datastructure for specifying connections between ports on the quantum device and on the control hardware in the `HardwareCompilationConfig`. (!734)
  - Allow additional third-party instruments with custom compilation nodes in hardware backends. (!837)
  - Allow specifying one-to-many, many-to-one, and many-to-many connections in the `Connectivity`. (!841)
  - Improve errors and warnings when compiling subschedules and/or loops. (!847)
  - Add helper functions and validators to convert old-style hardware config dicts to new-style `HardwareCompilationConfig` datastructures. (!843)
  - Allow `MarkerPulse`s to be appended to other operations. (!867)

- Documentation 
  - Move all `__init__` docstrings to class description and minor docstring changes. (!785)
  - Add a warning banner to documentation when on an old or on a development version of quantify. (!864)
  - Improve formatting by replacing single backticks with double backticks where needed. (!866)

- Infrastructure
  - Add ability to run profiling via the CI pipeline and manually in a notebook. (!854)
  - Add new test notebook for performance tests. (!862)

- Operations 
  - Make `staircase_pulse`, `long_square_pulse` and `long_ramp_pulse` compatible with use in control flow on Qblox hardware. These now end on a pulse with 0 voltage offset, to remove 4ns timing mismatch when they are used in control flow. (!857)
  - Add `acq_channel` argument to `Measure` operation and make `acq_channel` device element accept hashable types. (!869)

- Qblox backend
  - Refactors (!759, !870)
    - Move to/rename helpers in `StaticHardwareProperties`:
      - `get_io_info` => `_get_io_mode`,
      - `output_name_to_output_indices` + `input_name_to_input_indices` => `io_name_to_connected_io_indices`,
      - `output_map` => `io_name_to_digital_marker`.
    - Rename properties in `Sequencer`:
      - `connected_outputs` => `connected_output_indices`,
      - `connected_inputs` => `connected_input_indices`.
    - Substitute `io_mode` string literals by `ChannelMode` enums.
    - Remove `"imag"` sequencer mode from Qblox backend, rename `io_name` to `channel_name` and `path0`/`path1` to `path_I`/`path_Q`. (!870)
  - Bugfixes
    - Fix `MarkerPulse` playback on QRM-RF and QCM-RF. (!828)
      - Marker bit index values for addressing outputs need to be swapped on QCM-RF, not QRM-RF (done via `MarkerPulseStrategy._fix_marker_bit_output_addressing_qcm_rf`).
    - Fix for waveform gain/offset instructions and optimization with waveform uploading. (!860)
    - Fix (temporary) for reshaping of acquisition data of looped measurements in `BinMode.APPEND`. (!850)

- QuantumDevice
  - Store element and edge instrument references in `QuantumDevice`. (!855, #442)

- Schedules 
  - Prevent `FutureWarning` when creating `Schedule.timing_table` and sort by `abs_time`. (!852) 
  - Fix missing resources in nested schedule. (!877)

### Compatibility info

**Qblox**

| quantify-scheduler |                      qblox-instruments                       |                               Cluster firmware                                |
|--------------------|:------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
| v0.18.0            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |

**Zurich Instruments**
- `zhinst==21.8.20515`, `zhinst-qcodes==0.1.4`, `zhinst-toolkit==0.1.5`

## 0.17.1 (2023-11-02)

### Release highlights

- Hotfix
  - Reverted change in QuantumDevice that broke serialization/deserialization using a `quantify_core.data.handling.snapshot`. 
- New features
  - QuantumDevice: User-friendly QuantumDevice json serialization/deserialization methods.
    - `QuantumDevice.to_json()`/`QuantumDevice.to_json_file()` and `QuantumDevice.from_json()`/`QuantumDevice.from_json_file()`. 
  - New schedule: `timedomain_schedules.cpmg_sched`, CPMG schedule-generating function for dynamical decoupling experiments.

### Breaking changes

- QuantumDevice - Revert !813: `ManualParameter` `elements` and `edges` have been changed back from `dict` to `list`. (!846)
  - The change broke serialization/deserialization of QuantumDevice using a `quantify_core.data.handling.snapshot`, see #452, via `quantify_core.utilities.experiment_helpers.load_settings_onto_instrument`.

### Merged branches and closed issues

- Gettable 
  - Change reference timezone included in the diagnostics report from "Europe/Amsterdam" to UTC. (!849)
  - Include versions of the installed dependency packages of quantify-scheduler in the diagnostics report (`ScheduleGettable.initialize_and_get_with_report`). (!832)
  - Replace invalid utf-8 characters with "?" in Qblox hardware logs included in the diagnostics report. (!853)

- Qblox backend 
  - Absolute amplitude tolerance of waveform index suppression set to `2/constants.IMMEDIATE_SZ_GAIN` which prevents uploading of all gain-zero waveforms. (!842)

- QuantumDevice 
  - Serialize by adding `__getstate__` and `__setstate__` methods to the class, includes `DeviceElement`, `Edge` and `cfg_sched_repetitions`. (!802)

- Schedules 
  - Add CPMG schedule function `timedomain_schedules.cpmg_sched` for dynamical decoupling experiments. (!805)

- Utilities 
  - Add profiling notebooks to Developer guide. (!845)

### Compatibility info

**Qblox**

| quantify-scheduler |                      qblox-instruments                       |                               Cluster firmware                                |
|--------------------|:------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
| v0.17.1            | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |

**Zurich Instruments**
- `zhinst==21.8.20515`, `zhinst-qcodes==0.1.4`, `zhinst-toolkit==0.1.5`

## 0.17.0 (2023-10-20)

### Release highlights

- New features
  - Subschedules & repetition loops
    - A schedule can now be added to a schedule just like an operation.
    - Looping of an operation via new `control_flow` argument in `Schedule.add`.
    - Various limitations apply, please consult the documentation: [Reference guide: Control flow](https://quantify-os.org/docs/quantify-scheduler/reference/control_flow.html).
    - Also, currently plotting and timing table is not working:
      - Subschedules: circuit diagram does not work,
      - Repetition loops: not shown in circuit diagram, timing table does not work.
  - Diagnostics report: `ScheduleGettable.initialize_and_get_with_report` saves information from an experiment in a report zipfile.
    - For Qblox instruments, this includes hardware logs.
  - New operation: `GaussPulse`, an operation equivalent to the `DRAGPulse` but with derived amplitude zero.
  - New schedule: `two_qubit_transmon_schedules.chevron_cz_sched`, two-qubit schedule-generating function for CZ tuneup.
- Qblox backend improvements
  - Schedule compilation time decreased by 30-75%!
- Documentation
  -  We have moved to: https://quantify-os.org/docs/quantify-scheduler.
  - https://quantify-quantify-scheduler.readthedocs-hosted.com will be permanently redirected to the new location.

### Breaking changes

- Deprecation - The deprecated `instruction_generated_pulses_enabled` Qblox hardware configuration parameter, and the two classes related to it (`StitchedSquarePulseStrategy` and `StaircasePulseStrategy`) have been removed. The deprecated code suggestions have been updated. (!811)
- QuantumDevice - `ManualParameter` `elements` and `edges` have been changed from `list` to `dict`. (!813)
  - Before, these were lists with instrument names, so one could do `element_name = quantum_device.elements()[0]` and `edge_name = quantum_device.edges()[0]`.
  - Now, these are dicts with instrument names as keys and the `DeviceElement` and `Edge` instances as values, so one would need to change to `element_name = list(quantum_device.elements())[0]` and `edge_name = list(quantum_device.edges())[0]`.
  - Also see [Tutorial: Operations and Qubits - Device configuration](https://quantify-os.org/docs/quantify-scheduler/tutorials/Operations%20and%20Qubits.html#device-configuration).

### Merged branches and closed issues

- Compilation:
  - Add `keep_original_schedule` compilation option into the `CompilationConfig`, controlling whether a copy of the schedule is made at the start of compilation. (!816)
    - As an intermediate step, `keep_original_schedule` was added as an argument to several compilation functions (!771), but this was later moved to the `CompilationConfig`.
    - Also, directly calling `determine_absolute_timing` or `compile_circuit_to_device` (instead of from the compiler) is now deprecated.
  - Add a `scheduling_strategy` parameter to `QuantumDevice` and `DeviceCompilationConfig` classes to enable new strategies for calculating absolute timing in `compilation.determine_absolute_timing`.  (!736)
  - Added an optional `reference_magnitude` parameter to `VoltageOffset` operations. (!797)
  - Enforce always adding (possibly empty) `HardwareOptions` to the `HardwareCompilationConfig`. (!812)
  - Hotfix for !812: Fix backwards compatibility old-style hardware config for custom backend. (!818)
  
- Documentation:
  - Add a short explanation and example of the `NumericalWeightedIntegrationComplex` protocol to the Acquisitions tutorial. (!791)
  - Add a short explanation and examples of the `StitchedPulse` and `StitchedPulseBuilder` to the Schedules and Pulses tutorial. (!766)
  - Color change for code block emphasized lines. (!741)
  - Deploy documentation to quantify-os.org. (!796)
  - Explain in Qblox Cluster docs the possibility of using `"{complex,real}_output_<n>"` hardware config keys for both playback and acquisitions. (!763)
  - Fix missing images in Jupyter cell outputs in documentation deployed using Gitlab Pages. (!772, #404, counterpart of quantify-core!480)
  - Split source and build folders to simplify using [`sphinx-autobuild`](https://github.com/executablebooks/sphinx-autobuild) for its editing. (!774)
  - Update the deprecated code suggestions table. (!815)
  
- Gettable:
  - Add `ScheduleGettable.initialize_and_get_with_report` that saves information from an experiment in a report zipfile for diagnostics. (!672)
    - For Qblox instruments, this includes hardware logs, retrieved via `InstrumentCoordinator.retrieve_hardware_logs` from `qblox-instruments.ConfigurationManager`.
    - For Qblox instruments, add serial numbers and version info (via `get_idn` and `_get_mods_info`) to the report zipfile for diagnostics. (!787)
  
- Infrastructure:
  - Add `jinja2` as a dependency to quantify-scheduler (needed for `pandas.DataFrame`). (!777)
  - Bump `quantify_core` version dependency to 0.7.1 to include the fix to `without()` in quantify-core!438. (!795)
  - Improve the speed of `make_hash` (affecting `Schedule.add`) and some compilation steps. (!770)
  - Upgrade to `pydantic` V2 functionality (instead of importing from the legacy V1 module). (!714) 
  - Use new qcodes syntax for adding parameters. (!758)
  
- Operations:
  - Adjust pulse info and acquisition info definitions to take the class name. (!809)
  - Introduce the `GaussPulse`, an operation equivalent to the `DRAGPulse` but with zero derived amplitude, as well as its factory function `rxy_gauss_pulse`. (!793)
  
- Plotting:
  - Fix error while plotting numerical pulses with non-zero `rel_time`. (!783)
  
- Qblox backend:
  - Remove unnecessary deep copies from the schedule for a 30-75% performance improvement. (!771)
  - Improve compilation time for updating ports and clocks. (!830)
  - Make QASM fields aligning optional, disable by default. (!741)
  - Prevent repeated port-clock combinations. (!799)
  - Remove code referencing RF pulsars (these devices do not exist). (!748)
  - Add `debug_mode` compilation config parameter to align the q1asm program (replaces the `align_qasm_fields` setting); set to `True` for diagnostic report. (!822)
  
- Schedule:
  - A schedule can now be added to another schedule. It will be treated as one big operation. (!709)
  - Added looping: An inner schedule can be repeated inside of the schedule. This feature has limitations, please refer to the [control flow documentation](https://quantify-os.org/docs/quantify-scheduler/reference/control_flow.html). (!709, !819)
  
- Schedules:
  - Added two-qubit schedule-generating function `two_qubit_transmon_schedules.chevron_cz_sched` for CZ tuneup. (!700).
  
- Security:
  - Add `check=True` flag to all subprocess calls (see also Ruff rule PLW1510). (!767)

### Compatibility info

**Qblox**

| quantify-scheduler |                      qblox-instruments                       |                               Cluster firmware                                |
|--------------------|:------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
| v0.17.0            | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |

**Zurich Instruments**
- `zhinst==21.8.20515`, `zhinst-qcodes==0.1.4`, `zhinst-toolkit==0.1.5`

## 0.16.1 (2023-09-29)

### Release highlights

- Urgent Qblox hotfix
  - Fixes a bug on Qblox baseband modules where no output is measured on O2 and O4 outputs when the io-mode in the hardware configuration is set to `real_output_{1,3}`.

### Merged branches and closed issues

- Qblox backend - Fix missing signal on O2 and O4 outputs of baseband modules. (!803)
- Documentation - Switch to `pydata-sphinx-theme`. (!778)

### Compatibility info

- Qblox: `qblox-instruments==0.11.x` ([Cluster firmware v0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0)) and ([Cluster firmware v0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1))
- ZI:    `zhinst==21.8.20515` `zhinst-qcodes==0.1.4` `zhinst-toolkit==0.1.5`

## 0.16.0 (2023-08-17)

### Release highlights

- New features
  - New acquisition protocol: [ThresholdedAcquisition](https://quantify-os.org/docs/quantify-scheduler/tutorials/Acquisitions.html#thresholded-acquisition)
    - Currently only supported by the Qblox backend

- Qblox backend improvements
  - All-zero waveforms are no longer stored in the waveform dictionary and no longer uploaded to the hardware
  - Fixed accidentally changing multiple instances of same type of operation in a schedule
  - Fixed append mode for weighted acquisitions doing average instead of append
  - Fixed simultaneous playback on both outputs of a QCM-RF
  - Compatibility with v0.11 of the `qblox-instruments` package

### Breaking changes

- Operations - Prevent collisions by changing logic for checking Operation uniqueness to use `Operation.hash` instead of `str(Operation)`. (!738, #209)
  - `Operation.hash` now returns `str` rather than `int`.
  - `str(Operation)` is no longer required to be unique (except with the ZI backend).
  - `schedule.operations.keys()` can no longer be used to get a `repr` of the `Operation`s.
- Qblox backend - Made Q1ASM generation functions `acquire_append` and `acquire_average` private. (!739)
- Qblox backend - `to_grid_time` helper function would raise `ValueError` if time supplied is not a multiple of grid time, now additionally checking that time is within the tolerance of 1 picosecond of grid time instead of silently rounding to nanoseconds (!751)
- Qblox backend - Rename `is_within_grid_time ` helper function to `is_within_half_grid_time` !(753)
- Qblox backend - Strictly requires v0.11.x of the `qblox-instruments` package (!723)

### Merged branches and closed issues

- Acquisition - New acquisition protocol for thresholded acquisitions: `ThresholdedAcquisition` (!617)
- Compilation - Made the `HardwareCompilationConfig` datastructure backend-specific (!708)
- Compilation - Change default `ref_pt` and `ref_pt_new` to `None` in `Schedule` and `Schedulable`; `compilation.determine_absolute_timing` will then assume `end` and `start` respectively, preserving previous behavior (!733)
- Documentation - Fix broken list bullets on Read-the-Docs by enforcing `sphinx-rtd-theme>=1.2.2` (!743)
- Documentation - Fixes to acquisitions tutorial (!732)
- Documentation - Restore Qblox documentation on using the old-style hardware config (!761)
- Documentation - Fix broken xarray display in docs (propagate from quantify-core!470) (!762)
- Infrastructure - Pin version of `dataclasses-json` due to breaking pipelines (!727) and unpin again (!735)
- Instrument Coordinator - Improve error message for missing IC components (!718)
- Qblox backend - Fix weighted acquisition in append mode. (!725)
- Qblox backend - Prevent uploading "null" (i.e. all-zero) waveforms (!711)
- Qblox backend - Fix playback on both outputs of a QCM-RF (!742)
- Qblox backend - Added warning if waveform playback or acquisition is interrupted by another waveform or acquisition, respectively (!744, #436)
- Qblox backend - Update setting of sequencer qcodes parameters responsible for connection with physical input and output parameters, due to new channel map setup in `qblox-instruments==0.11`. (!723)
- Qblox backend - Hotfix for !723: turn on channel map parameters associated with inputs for the cases of output io names (e.g. `complex_output_0`) defined in readout modules. (!760)
- Schedulables - Store references to `Schedulables` in timing constraints as `string` rather than `Schedulable`. (!717)
- Schedule, Schedulable - Refactor logic for checking schedulable uniqueness within the schedule and reference schedulable existence within schedule out of `Schedulable` to `Schedule`. (!724)
- Waveforms - Fix such that `interpolated_complex_waveform` does not extrapolate except for rounding errors. (!710)
- Zhinst backend - Decapitalize `"dc_mixer_offset_I"` and `"dc_mixer_offset_Q"` in `backends.types.zhinst.Output` validator to fix compatibility with old-style hardware config dicts. (!740)

### Compatibility info

- Qblox: `qblox-instruments==0.11.x` ([Cluster firmware v0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0))
- ZI:    `zhinst==21.8.20515` `zhinst-qcodes==0.1.4` `zhinst-toolkit==0.1.5`

## 0.15.0 (2023-07-13)

### Release highlights

- Added New tutorials on performing acquisitions:
  - Tutorial: Acquisitions - how to add acquisitions to schedules, and how to retrieve acquisition results using the Instrument Coordinator
  - Tutorial: ScheduleGettable - how to perform acquisitions with ScheduleGettable
- Fixed `sudden_net_zero` waveform generation
- Added `Rz`, `Z` and `Z90` gate to gate library and `BasicTransmonElement`
- Improved pulse diagram, mostly in the `matplotlib` backend
- Added new-style validated hardware config: Restructured `CompilationConfig` by adding `HardwareCompilationConfig` datastructure that contains `HardwareDescription`, `Connectivity`, and `HardwareOptions`
  - The old-style unvalidated hardware config is still fully supported
  - Currently, the new-style hardware config is being converted to the old-style hardware config before processing by the hardware backends

### Breaking changes

- Acquisition - Trigger count protocol changes: `ScheduleGettable` and `InstrumentCoordinator` return data reconciliation; using `counts` instead of `acq_index` as a dimension when `BinMode.AVERAGE` is used (!703)
- Instrument Coordinator - Dimension names of the datasets returned by the instrument coordinator have changed. If you used them explicitly in processing, you can fix that by extracting the names dynamically from the dataset. The exact names of the dimensions are not guaranteed to be stable in the future. (!608)
- Qblox backend - Remove overwriting of IF frequency to `None` when `mix_lo=False` (!699)
- Qblox backend - Compile `ShiftClockPhase` operation to `set_ph_delta` + `upd_param`, extending duration from 0 to 8 ns (!704, #432)
- Visualization - The keyword argument `plot_kwargs` in `Schedule.plot_pulse_diagram()` has been replaced by `**backend_kwargs`, making it possible to directly specify keyword arguments (!664).

### Merged branches and closed issues

- Compilation - Minor refactor of `circuit_to_device` to be compatible with `numpy>=1.25` (!706)
- Compilation - Amended `ReferenceMagnitude` set method to ensure that all unit parameters are not overwritten when one of the parameters is set to `nan` (!695, #429).
- Compilation - Changed units of amplitude parameters in device elements to dimensionless, for consistency with the new `ReferenceMagnitude` interface (!691).
- Compilation - Restructured `CompilationConfig` by adding the `HardwareCompilationConfig` datastructure that contains `HardwareDescription`, `Connectivity`, and `HardwareOptions` (!680)
- Documentation - A new `ScheduleGettable` tutorial has been added (!686).
- Documentation - New acquisitions tutorial (!694)
- Documentation - Update broken `qblox-instruments` documentation URLs (!696)
- Documentation - Utilize `hvplot` and `bokeh` for part of data visualization in documentation to overcome issues with `matplotlib` (!712).
- Gate Library - Added `Rz`, `Z` and `Z90` gate to gate library, `BasicTransmonElement` and tested the new gates in `test_gate_library.py` (!697, #290)
- Gettables - The shape of the data returned by the instrument coordinator components for different acquisition protocols is semi-formalized and validated in the code of `ScheduleGettable.retrieve_acquisition()`. Data returned by Qblox and ZI LabOne backends is adjusted accordingly. (!608)
- JSON utilities - `DataStructure` can serialize Numpy arrays using the new `quantify_scheduler.structure.NDArray` field type. (!701)
- Schedulables - Raises a more readable error when the reference point is a `Schedulable` that is not in the `Schedule` (!707)
- Schedules - Remove one of the `CRCount` operations in `nv_dark_esr_sched_nco` from the NCO frequency loop to avoid redundancy (!643)
- Visualization - Large refactor of the pulse diagram, mostly in the matplotlib backend (!664).
  - The `matplotlib` backend no longer plots 0 V points in between pulses, leading to significant performance improvements in some cases.
  - In both the `matplotlib` and the plotly backend, lines are now shaded from the line to the x-axis.
  - For the `matplotlib` backend, an extra keyword argument `multiple_subplots` (bool) is added. If True, each port used in the schedule gets its own subplot, similar to how it's done in the `plotly` backend.
  - In the `plotly` backend, the time slider at the bottom of the figure has been removed. This was necessary to allow the y-axis of the bottom-most plot to be interactively re-scalable as well.
- Waveforms - Fix `sudden_net_zero` waveform generation. Rounding of pulse times will now no longer lead to an incorrect SNZ pulse. I.e., the last sample of the first pulse and the first sample of the second pulse will remain correctly scaled, and the integral correction will have an amplitude such that the integral of the pulse is zero. (!581, #310)

### Compatibility info

- Qblox: `qblox-instruments==0.9.0` ([Cluster firmware v0.4.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.4.0)) and `qblox-instruments==0.10.x` ([Cluster firmware v0.5.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.5.0))
- ZI:    `zhinst==21.8.20515` `zhinst-qcodes==0.1.4` `zhinst-toolkit==0.1.5`

## 0.14.0 (2023-06-02)

### Release highlights

- New features
  - **Introducing** `ReferenceMagnitude`. A new parameter called has been introduced for pulses, enabling the flexible specification of amplitudes and powers across various orders of magnitude. This parameter allows users to utilize variable attenuations, among other techniques, to precisely control the amplitudes and powers of the pulses. **Important note** this parameter is not yet implemented for any of the available backends. Future updates are expected to include support for ReferenceMagnitude in the compilation backends.

- Qblox backend improvements
  - **Introducing the** `MarkerPulse`. This feature simplifies the specification of digital pulses with precise timing, facilitating the control of third-party electronics. For more information, see [Digital I/O](https://quantify-os.org/docs/quantify-scheduler/reference/qblox/Cluster.html#digital-i-o).
  - **Improved Compilation Time**. The compilation time has been significantly improved, achieving approximately 10 times faster execution. Notably, a 32 times speedup has been observed when running 2-qubit Chevron schedules.
  - **Reduced Acquisition Time**. The minimum time between acquisitions has been lowered from 1000 ns to 300 ns, enhancing the efficiency of data acquisition.

### Breaking changes

- Compilation - Introduced new `ReferenceMagnitude` parameter for pulses to allow specification of amplitudes and powers over different orders of magnitude (using variable attenuations, for example) (!652)
  - Currently, only the user interface is implemented - changes to the backends will be made later (#413)
  - The code is backwards compatible, i.e., all schedules remain valid, but pulse amplitudes are not backwards compatible and will need adjustment / re-calibrating
- Qblox backend - The compiled offset value in `AwgOffsetStrategy` is adjusted to match the changes to pulse amplitudes in !652. A pulse with a given amplitude `A` and a `VoltageOffset` with offset `A` will now produce the same voltage at the hardware level (!683).
- Qblox backend - Lowering the minimum time between acquisitions to 300 ns (!676, #369)
- Deprecation - Deprecated code that had been scheduled to be removed after version 0.13 has been removed. The deprecated code suggestions have been updated (!679).
- Visualization - The function `quantify_scheduler.waveforms.modulate_wave` has been removed as it duplicated `quantify_scheduler.helpers.waveforms.modulate_waveform` in an incorrect manner (!679).

### Merged branches and closed issues

- Compilation - Allow to subclass `CompiledSchedule` in graph compilation (!663).
- Compilation - Support batched frequencies in schedule resources (!670)
- Compilation - Move `MixerCorrections` to `CompilationConfig.hardware_options` (!669)
- Compilation - Move `PowerScaling` (gain/attenuation) to `CompilationConfig.hardware_options` (!673)
- Documentation - Add `UML_class_diagrams` notebook for visualizing class hierarchies (!653)
- Gettables - When `ScheduleGettable.get()` is called and the associated `InstrumentCoordinator` returns nothing (for example, if the hardware configuration was never set), this will no longer raise a KeyError from the xarray module. Instead, a more helpful error is raised before the data processing (!671).
- Operations - The function `convert_to_numerical_pulse` has been added, which can be used to convert `StitchedPulse` to `NumericalPulse` (!665).
  - In addition, this function is called in the pulse diagram plotting functions and the ZI LabOne backend to allow resp. plotting and compiling of `StitchedPulse`.
- Qblox backend - Raise `RuntimeWarning` instead of `NotImplementedError` upon using `reference_magnitude` parameter (introduced in !652) (!684)
- Qblox backend - Compilation uses `math.isclose` instead of `numpy.isclose` in certain cases to improve compile time (!682)
- Qblox backend - Add `MarkerPulse` to pulse library, and implement Qblox backend compilation for this pulse (!628)
- Qblox backend - Fix bug where LO/IF frequency `nan` is treated as overconstrained mixer (!690, #423)

### Compatibility info

- Qblox: `qblox-instruments==0.9.0` ([Cluster firmware v0.4.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.4.0)) and `qblox-instruments==0.10.x` ([Cluster firmware v0.5.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.5.0))
- ZI:    `zhinst==21.8.20515` `zhinst-qcodes==0.1.4` `zhinst-toolkit==0.1.5`

## 0.13.0 (2023-05-05)

### Release highlights

- This release introduces a new acquisition protocol: `NumericalWeightedIntegrationComplex`, that allows users perform a weighted integration acquisition.
- The schedule library now has a new schedule that performs an NCO frequency sweep in a dark ESR experiment on an NV-center.
- A lot of code that was marked as deprecated has been removed. Please checkout our [deprecated code suggestions](https://quantify-os.org/docs/quantify-scheduler/v0.13.0/examples/deprecated.html) on how to modify you code to ammend any problems caused by this removal.
- Many improvements and small bug fixes.

### Breaking changes

- Deprecation - Deprecated code that had been scheduled to be removed after version 0.12 has been removed. The deprecated code suggestions have been updated (!667).
- Qblox backend - Markers will not be pulled high at the start of sequences anymore; removed `MarkerConfiguration` from the Qblox backend; moved `output_map` to `StaticHardwareProperties`, the RF output switches are controlled by the `output_map` (!662)
- Qblox backend - Deprecate the `instruction_generated_pulses_enabled` hardware configuration setting, as well as the `StitchedSquarePulseStrategy` and `StaircasePulseStrategy`. The newly introduced `StitchedPulse`, as well as the helper functions in `quantify_scheduler.pulse_factories` can be used instead. `SquarePulse`s with a duration >1 microsecond (a constant in the Qblox backend) are compiled to AWG offset instructions (!637).

### Merged branches and closed issues

- Compilation - Move `ModulationFrequencies` to `CompilationConfig.hardware_options` (!660)
- Compilation - Update structure of `HardwareOptions` datastructure with fields `latency_corrections: Dict[str, LatencyCorrection]` and `distortion_corrections: Dict[str, DistortionCorrection]` (!650).
- Compilation - Move `DistortionCorrections` to `CompilationConfig.hardware_options` (!648)
- Compilation - Move `LatencyCorrections` to `CompilationConfig.hardware_options` and use field name `corrections` instead of `latencies` (!633).
- Compilation - `CompilationNode`s take the full `CompilationConfig` as input (!615, #405)
- Deprecation - Replace `device_compile` and `hardware_compile` by `SerialCompiler`  in NV center tests (!651)
- Documentation - Fix documentation generation warnings and errors (!658)
- Documentation - Acquisition data format in user guide (!646)
- Gettables - Clean up code syntax (!638)
- Git - Change back to default merge strategy for CHANGELOG.md (!659).
- Operations - Introduce the `StitchedPulse`, an Operation that can be composed of AWG offset instructions and waveforms, and the `StitchedPulseBuilder` which can be used to create `StitchedPulse`s (!588, !666).
  - Additionally, helper functions `long_square_pulse`, `long_ramp_pulse` and `staircase_pulse` are introduced in `quantify_scheduler.pulse_factories`, to more easily generate the operations for these common use-cases.
- Operations, Qblox backend - Introduce the `VoltageOffset` operation, for use in the `StitchedPulse`, and modify the compilation steps to compile this operation (!588).
- Qblox backend - Add the `marker_debug_mode_enable` parameter to the hardware configuration, which toggles markers at the start of operations on inputs and outputs where it is set to True (!606).
- Qblox backend - Renamed `hw_mapping` input parameter to `instrument_cfg` in `InstrumentCompiler` and up (!644).
- Qblox backend - Let the compiler raise an error when the waveforms specified in the schedule are too large to be uploaded to a sequencer (!625).
- Qblox backend - Forbid repeated acquisition index in schedule (!655, partially revert !542)
- Qblox backend - Rename the `MAX_SAMPLE_SIZE_ACQUISITIONS` constant to `MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS`, and modify the docstring to clarify that this constant only refers to scope trace acquisitions (!649).
- Qblox backend - Forbid repeated acquisition index in schedule (!655, !657, partially revert !542)
- Qblox backend - `Measure` can now use the `NumericalWeightedIntegrationComplex` protocol. The `DispersiveMeasurement` has been expanded with optional weight parameters for use in this protocol (!612).
- Schedulables - Make name uniqueness check more efficient and readable. (!631)
- Schedules - Add `nv_dark_esr_sched_nco` spectroscopy schedule using SetClockFrequency Operation to sweep the NCO frequency (!639)
- Tests - Move Qblox Pulsar entries from `qblox_test_mapping.json` to pytest fixtures (!632)
- Tests - Qblox acquisition testcases with dummy data (!654)
- Typing - More lenient typehints (!640)
- Visualization - Introduce the `x_range` keyword for the matplotlib backend in `Schedule.plot_pulse_diagram`. This will cut off any points outside the given range when creating the plot. This can be used to reduce memory usage when plotting a small section of a long pulse sequence (!629).
- ZI LabOne backend - Return datasets from UHFQA instrument coordinator component (which fixes the broken backend) (#410, !623).

### Compatibility info

- Qblox: `qblox-instruments==0.9.0` ([Cluster firmware v0.4.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.4.0)) and `qblox-instruments==0.10.x` ([Cluster firmware v0.5.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.5.0))
- ZI:    `zhinst==21.8.20515` `zhinst-qcodes==0.1.4` `zhinst-toolkit==0.1.5`

## 0.12.3 (2023-03-09)

### Merged branches and closed issues

- Documentation - Replace deprecated code in the Operations and Qubits tutorial (!602)
- Workflow - `ruff` and `pyright` are enabled for linting the code (in `pre-commit` and in CI pipeline). All existing code is added to exclusion list because it does not comply with them yet. (!614)

## 0.12.2 (2023-03-05)

### Breaking changes

- Qblox backend - Fix imaginary and real part of the acquisition by swapping them, and fix trigger count formatting (!619)

### Merged branches and closed issues

- Qblox backend - raise DriverVersionError when importing qblox module with incompatible qblox-instruments version (!620)

## 0.12.1 (2023-03-02)

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

For help in migrating from deprecated methods, see [Quantify Deprecated Code Suggestions](../examples/deprecated.md).

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
