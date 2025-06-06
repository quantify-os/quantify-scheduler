# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler classes for Qblox backend."""

from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np

from quantify_scheduler.backends.qblox import compiler_abc
from quantify_scheduler.backends.qblox.analog import (
    AnalogModuleCompiler,
    BasebandModuleCompiler,
    RFModuleCompiler,
)
from quantify_scheduler.backends.qblox.constants import (
    MAX_NUMBER_OF_INSTRUCTIONS_QCM,
    MAX_NUMBER_OF_INSTRUCTIONS_QRC,
    MAX_NUMBER_OF_INSTRUCTIONS_QRM,
    MAX_NUMBER_OF_INSTRUCTIONS_QTM,
)
from quantify_scheduler.backends.qblox.enums import (
    QbloxFilterConfig,
    QbloxFilterMarkerDelay,
)
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler
from quantify_scheduler.backends.types.common import (
    HardwareDistortionCorrection,
)
from quantify_scheduler.backends.types.qblox import (
    AnalogModuleSettings,
    BoundedParameter,
    ClusterSettings,
    OpInfo,
    QbloxRealTimeFilter,
    RFModuleSettings,
    StaticAnalogModuleProperties,
    StaticTimetagModuleProperties,
    TimetagModuleSettings,
)
from quantify_scheduler.enums import TimeRef

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox_backend import (
        ChannelPath,
        _ClusterCompilationConfig,
        _ClusterModuleCompilationConfig,
        _LocalOscillatorCompilationConfig,
        _SequencerCompilationConfig,
    )
    from quantify_scheduler.resources import Resource


class LocalOscillatorCompiler(compiler_abc.InstrumentCompiler):
    """
    Implementation of an
    :class:`~quantify_scheduler.backends.qblox.compiler_abc.InstrumentCompiler`
    that compiles for a generic LO. The main
    difference between this class and the other compiler classes is that it doesn't take
    pulses and acquisitions.


    Parameters
    ----------
    name
        QCoDeS name of the device it compiles for.
    total_play_time
        Total time execution of the schedule should go on for. This parameter is
        used to ensure that the different devices, potentially with different clock
        rates, can work in a synchronized way when performing multiple executions of
        the schedule.
    instrument_cfg
        The compiler config referring to this instrument.

    """

    def __init__(
        self,
        name: str,
        total_play_time: float,
        instrument_cfg: _LocalOscillatorCompilationConfig,
    ) -> None:
        super().__init__(
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self.freq_param_name = "frequency"
        self._frequency = instrument_cfg.frequency
        self.power_param_name = "power"
        self._power = instrument_cfg.hardware_description.power

    @property
    def frequency(self) -> float | None:
        """
        Getter for the frequency.

        Returns
        -------
        :
            The current frequency.

        """
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        """
        Sets the lo frequency for this device if no frequency is specified, but raises
        an exception otherwise.

        Parameters
        ----------
        value
            The frequency to set it to.

        Raises
        ------
        ValueError
            Occurs when a frequency has been previously set and attempting to set the
            frequency to a different value than what it is currently set to. This would
            indicate an invalid configuration in the hardware mapping.

        """
        if self._frequency is not None and value != self._frequency:
            raise ValueError(
                f"Attempting to set LO {self.name} to frequency {value}, "
                f"while it has previously already been set to "
                f"{self._frequency}!"
            )
        self._frequency = value

    def compile(
        self,
        debug_mode: bool,  # noqa: ARG002 Debug_mode not used for this class
        repetitions: int = 1,  # noqa: ARG002 Repetitions not used for this class
    ) -> dict[str, Any] | None:
        """
        Compiles the program for the LO InstrumentCoordinator component.

        Parameters
        ----------
        debug_mode
            Debug mode can modify the compilation process,
            so that debugging of the compilation process is easier.
        repetitions
            Number of times execution the schedule is repeated.

        Returns
        -------
        :
            Dictionary containing all the information the InstrumentCoordinator
            component needs to set the parameters appropriately.

        """
        if self._frequency is None:
            return None
        return {
            f"{self.name}.{self.freq_param_name}": self._frequency,
            f"{self.name}.{self.power_param_name}": self._power,
        }


class QCMCompiler(BasebandModuleCompiler):
    """QCM specific implementation of the qblox compiler."""

    _settings_type = AnalogModuleSettings
    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = False  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QCM  # type: ignore
    static_hw_properties: StaticAnalogModuleProperties = StaticAnalogModuleProperties(
        instrument_type="QCM",
        max_awg_output_voltage=2.5,
        mixer_dc_offset_range=BoundedParameter(min_val=-2.5, max_val=2.5, units="V"),
        channel_name_to_digital_marker={
            "digital_output_0": 0b0001,
            "digital_output_1": 0b0010,
            "digital_output_2": 0b0100,
            "digital_output_3": 0b1000,
        },
    )

    def _configure_hardware_distortion_corrections(self) -> None:
        """Assign distortion corrections to settings of instrument compiler."""
        distortion_configs = self._get_distortion_configs_per_output()
        self._configure_distortion_correction_latency_compensations(distortion_configs)

        for output in distortion_configs:
            output_settings = self._settings.distortion_corrections[output]
            marker_debug_mode_enable = distortion_configs[output]["marker_debug_mode_enable"]
            if not isinstance(distortion_configs[output]["distortion_corrections"], list):
                dc_list = [distortion_configs[output]["distortion_corrections"]]
            else:
                dc_list = distortion_configs[output]["distortion_corrections"]
            for dc in dc_list:
                for key in dc.model_fields_set:
                    value = getattr(dc, key)
                    for i in range(4):
                        if key == f"exp{i}_coeffs" and value is not None:
                            if len(value) != 2 or value[0] < 6 or value[1] < -1 or value[1] >= 1:
                                raise ValueError(
                                    "The exponential overshoot correction has two "
                                    "coefficients with ranges of [6,inf) and [-1,1)."
                                )
                            self._configure_filter(
                                getattr(output_settings, f"exp{i}"),
                                value,
                                marker_debug_mode_enable,
                            )
                    if key == "fir_coeffs" and value is not None:
                        if len(value) != 32 or np.any(value) < -2 or np.any(value) >= 2:
                            raise ValueError(
                                "The FIR filter has 32 coefficients with a range of [-2,2) each."
                            )
                        self._configure_filter(output_settings.fir, value, marker_debug_mode_enable)

    def _get_distortion_configs_per_output(self) -> dict[int, dict]:
        module_distortion_configs = {}
        corrections = self.instrument_cfg.hardware_options.distortion_corrections
        if corrections is not None:
            for portclock in corrections:
                if (path := self.instrument_cfg.portclock_to_path.get(portclock, None)) is not None:
                    correction_cfg = corrections[portclock]
                    # `correction_cfg` can also be a `SoftwareDistortionCorrection`
                    if isinstance(correction_cfg, (HardwareDistortionCorrection, list)):
                        output_name = path.channel_name
                        output_number = int(output_name.split("_")[-1])
                        channel_description = getattr(
                            self.instrument_cfg.hardware_description, output_name
                        )
                        marker_debug_mode_enable = (
                            channel_description.marker_debug_mode_enable
                            if channel_description is not None
                            else False
                        )
                        if module_distortion_configs.get(output_number) is None:
                            if isinstance(correction_cfg, HardwareDistortionCorrection):
                                output_number = int(output_name.split("_")[-1])
                                module_distortion_configs[output_number] = {
                                    "distortion_corrections": correction_cfg,
                                    "marker_debug_mode_enable": marker_debug_mode_enable,
                                }
                            elif isinstance(correction_cfg, list):
                                output_number = 2 * int(output_name.split("_")[-1])
                                module_distortion_configs[output_number] = {
                                    "distortion_corrections": correction_cfg[0],
                                    "marker_debug_mode_enable": marker_debug_mode_enable,
                                }
                                output_number += 1
                                module_distortion_configs[output_number] = {
                                    "distortion_corrections": correction_cfg[1],
                                    "marker_debug_mode_enable": marker_debug_mode_enable,
                                }
                        else:
                            raise ValueError(
                                f"Attempting to set distortion corrections to"
                                f"{output_name} using portclock {portclock}, while it"
                                f"has previously already been set on this output."
                            )
        return module_distortion_configs

    def _configure_filter(
        self,
        filt: QbloxRealTimeFilter,
        coefficient: float,
        marker_debug_mode_enable: bool,
    ) -> None:
        filt.coeffs = coefficient
        filt.config = QbloxFilterConfig.ENABLED
        if marker_debug_mode_enable:
            filt.marker_delay = QbloxFilterMarkerDelay.DELAY_COMP


class QRMCompiler(BasebandModuleCompiler):
    """QRM specific implementation of the qblox compiler."""

    _settings_type = RFModuleSettings
    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = True  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QRM  # type: ignore
    static_hw_properties: StaticAnalogModuleProperties = StaticAnalogModuleProperties(
        instrument_type="QRM",
        max_awg_output_voltage=0.5,
        mixer_dc_offset_range=BoundedParameter(min_val=-0.5, max_val=0.5, units="V"),
        channel_name_to_digital_marker={
            "digital_output_0": 0b0001,
            "digital_output_1": 0b0010,
            "digital_output_2": 0b0100,
            "digital_output_3": 0b1000,
        },
    )


class QCMRFCompiler(RFModuleCompiler):
    """QCM-RF specific implementation of the qblox compiler."""

    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = False  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QCM  # type: ignore
    static_hw_properties: StaticAnalogModuleProperties = StaticAnalogModuleProperties(
        instrument_type="QCM_RF",
        max_awg_output_voltage=None,
        mixer_dc_offset_range=BoundedParameter(min_val=-50, max_val=50, units="mV"),
        channel_name_to_digital_marker={
            "complex_output_0": 0b0001,
            "complex_output_1": 0b0010,
            # Note: indices 2 and 3 are for outputs 1 and 0.
            "digital_output_1": 0b0100,
            "digital_output_0": 0b1000,
        },
        default_markers={
            "complex_output_0": 0b0001,
            "complex_output_1": 0b0010,
            # Note: indices 2 and 3 are for outputs 1 and 0.
            "digital_output_1": 0b0011,
            "digital_output_0": 0b0011,
        },
    )


class QRMRFCompiler(RFModuleCompiler):
    """QRM-RF specific implementation of the qblox compiler."""

    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = True  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QRM  # type: ignore
    static_hw_properties: StaticAnalogModuleProperties = StaticAnalogModuleProperties(
        instrument_type="QRM_RF",
        max_awg_output_voltage=None,
        mixer_dc_offset_range=BoundedParameter(min_val=-50, max_val=50, units="mV"),
        channel_name_to_digital_marker={
            # bit index 0 is inactive
            "complex_output_0": 0b0010,
            "digital_output_0": 0b0100,
            "digital_output_1": 0b1000,
        },
        default_markers={
            "complex_input_0": 0b0010,
            "complex_output_0": 0b0010,
            "digital_output_0": 0b0010,
            "digital_output_1": 0b0010,
        },
    )


class QRCCompiler(RFModuleCompiler):
    """QRC specific implementation of the qblox compiler."""

    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = True  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QRC  # type: ignore
    static_hw_properties = StaticAnalogModuleProperties(  # type: ignore
        instrument_type="QRC",
        max_awg_output_voltage=None,
        mixer_dc_offset_range=BoundedParameter(min_val=-50, max_val=50, units="mV"),
        channel_name_to_digital_marker={
            "digital_output_0": 0b1,
        },
        default_nco_en=True,
    )


class QTMCompiler(compiler_abc.ClusterModuleCompiler):
    """
    QTM specific implementation of the qblox compiler.

    Parameters
    ----------
    name
        Name of the `QCoDeS` instrument this compiler object corresponds to.
    total_play_time
        Total time execution of the schedule should go on for. This parameter is
        used to ensure that the different devices, potentially with different clock
        rates, can work in a synchronized way when performing multiple executions of
        the schedule.
    instrument_cfg
        The instrument compilation config referring to this device.

    """

    static_hw_properties: StaticTimetagModuleProperties = (  # type: ignore
        StaticTimetagModuleProperties(
            instrument_type="QTM",
        )
    )

    def __init__(
        self,
        name: str,
        total_play_time: float,
        instrument_cfg: _ClusterModuleCompilationConfig,
    ) -> None:
        super().__init__(
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self.sequencers: dict[str, TimetagSequencerCompiler] = {}

        self._settings: TimetagModuleSettings = (  # type: ignore
            TimetagModuleSettings.extract_settings_from_mapping(instrument_cfg)
        )

    @property
    def max_number_of_instructions(self) -> int:
        """The maximum number of Q1ASM instructions supported by this module type."""
        return MAX_NUMBER_OF_INSTRUCTIONS_QTM

    @property
    def supports_acquisition(self) -> bool:
        """Specifies whether the device can perform acquisitions."""
        return True

    def _construct_sequencer_compiler(
        self,
        index: int,  # noqa: ARG002 ignore unused argument
        sequencer_cfg: _SequencerCompilationConfig,
    ) -> TimetagSequencerCompiler:
        def get_index_from_channel_name() -> int:
            """
            Get the sequencer index.

            The QTM has no channel map yet, so the sequencer index = the channel index,
            and there is always only one channel index.
            """
            input_idx = self.static_hw_properties._get_connected_input_indices(
                channel_name,
                channel_name_measure,
            )
            if len(input_idx) > 0:
                return input_idx[0]
            else:
                # If it's not an input channel, it must be an output channel.
                output_idx = self.static_hw_properties._get_connected_output_indices(channel_name)
                return output_idx[0]

        channel_name = sequencer_cfg.channel_name
        channel_name_measure = sequencer_cfg.channel_name_measure
        return TimetagSequencerCompiler(
            parent=self,
            index=get_index_from_channel_name(),
            static_hw_properties=self.static_hw_properties,
            sequencer_cfg=sequencer_cfg,
        )

    def prepare(self, **kwargs) -> None:  # noqa: ARG002 other kwargs are ignored
        """
        Performs the logic needed before being able to start the compilation. In effect,
        this means assigning the pulses and acquisitions to the sequencers and
        calculating the relevant frequencies in case an external local oscillator is
        used.
        """
        self._set_time_ref_channel(self._op_infos, self.instrument_cfg.portclock_to_path)
        self._construct_all_sequencer_compilers()
        self.distribute_data()
        for seq in self.sequencers.values():
            seq.prepare()

    @staticmethod
    def _set_time_ref_channel(
        op_infos: dict[tuple[str, str], list[OpInfo]], portclock_to_path: dict[str, ChannelPath]
    ) -> None:
        """
        Set the time_ref_channel for all Timetag operations using TimeRef.PORT.

        Needs to be called before `SequencerCompiler._prepare_acq_settings()`.

        It is not validated that there is indeed a timetag acquisition on the port that was
        referenced, as this is not necessary for the schedule to run without errors.
        """
        all_op_infos = chain.from_iterable(op_infos.values())
        all_acqs_with_time_ref_port = [
            op
            for op in all_op_infos
            if op.is_acquisition and op.data.get("time_ref") == TimeRef.PORT
        ]
        for acq in all_acqs_with_time_ref_port:
            # For each acquisition, "time_ref_port" must be connected to exactly one channel path.
            # This means that, for all port-clocks that contains the same port as "time_ref_port",
            # we check if the channel path is the same.
            matching_port_clocks = [
                f"{port}-{clock}" for port, clock in op_infos if port == acq.data["time_ref_port"]
            ]
            if len(matching_port_clocks) == 0:
                raise ValueError(
                    f"Found no channels connected to time_ref_port={acq.data['time_ref_port']} on "
                    f"the same module as the acquisition port={acq.data['port']}"
                )
            if (
                len(matching_port_clocks) > 1
                and len(set(portclock_to_path[pc] for pc in matching_port_clocks)) > 1
            ):
                raise ValueError(
                    "Found multiple channels connected to time_ref_port="
                    f"{acq.data['time_ref_port']}"
                )

            # All port-clocks in matching_port_clocks map to the same channel path in
            # portclock_to_path.
            ref_channel = portclock_to_path[matching_port_clocks[0]]
            acq.data["time_ref_channel"] = ref_channel.channel_idx


class ClusterCompiler(compiler_abc.InstrumentCompiler):
    """
    Compiler class for a Qblox cluster.

    Parameters
    ----------
    name
        Name of the `QCoDeS` instrument this compiler object corresponds to.
    total_play_time
        Total time execution of the schedule should go on for.
    instrument_cfg
        The instrument compiler config referring to this device.

    """

    compiler_classes: dict[str, type] = {
        "QCM": QCMCompiler,
        "QRM": QRMCompiler,
        "QCM_RF": QCMRFCompiler,
        "QRM_RF": QRMRFCompiler,
        "QRC": QRCCompiler,
        "QTM": QTMCompiler,
    }
    """References to the individual module compiler classes that can be used by the
    cluster."""

    def __init__(
        self,
        name: str,
        total_play_time: float,
        instrument_cfg: _ClusterCompilationConfig,
    ) -> None:
        super().__init__(
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self.instrument_cfg: _ClusterCompilationConfig  # Help typechecker
        self._settings: ClusterSettings = ClusterSettings.extract_settings_from_mapping(
            instrument_cfg
        )
        self._op_infos: dict[tuple[str, str], list[OpInfo]] = defaultdict(list)
        self.instrument_compilers = self._construct_module_compilers()
        self.portclock_to_path = instrument_cfg.portclock_to_path

    def add_op_info(self, port: str, clock: str, op_info: OpInfo) -> None:
        """
        Assigns a certain pulse or acquisition to this device.

        Parameters
        ----------
        port
            The port this waveform is sent to (or acquired from).
        clock
            The clock for modulation of the pulse or acquisition. Can be a BasebandClock.
        op_info
            Data structure containing all the information regarding this specific
            pulse or acquisition operation.

        """
        self._op_infos[(port, clock)].append(op_info)

    def _construct_module_compilers(self) -> dict[str, AnalogModuleCompiler]:
        """
        Constructs the compilers for the modules inside the cluster.

        Returns
        -------
        :
            A dictionary with the name of the module as key and the value its
            compiler.

        """
        module_compilers = {}
        module_configs = self.instrument_cfg._extract_module_compilation_configs()

        for module_idx, cfg in module_configs.items():
            module_name = f"{self.name}_module{module_idx}"
            compiler_type: type = self.compiler_classes[cfg.hardware_description.instrument_type]
            module_compilers[module_name] = compiler_type(
                name=module_name,
                total_play_time=self.total_play_time,
                instrument_cfg=cfg,
            )
        return module_compilers

    def prepare(
        self,
        external_los: dict[str, LocalOscillatorCompiler] | None = None,
        schedule_resources: dict[str, Resource] | None = None,
        **kwargs,  # noqa: ARG002 other kwargs are ignored
    ) -> None:
        """
        Prepares the instrument compiler for compilation by assigning the data.

        Parameters
        ----------
        external_los
            Optional LO compiler objects representing external LOs, whose LO frequency
            will be determined and set.
        schedule_resources
            Mapping from clock name to clock resource, which contains the clock frequency.
        kwargs:
            Potential keyword arguments for other compiler classes.

        """
        self._validate_external_trigger_sync()
        self.distribute_data()
        for compiler in self.instrument_compilers.values():
            compiler.prepare(external_los=external_los, schedule_resources=schedule_resources)

    def _validate_external_trigger_sync(self) -> None:
        """
        Validate _ClusterCompilationConfig.sync_on_external_trigger.

        If the slot and channel used for external trigger sync are also in portclock_to_path,
        validate that the settings do not conflict.
        """
        if self._settings.sync_on_external_trigger is None:
            return

        ext_trig_sync = self._settings.sync_on_external_trigger

        # First check modules defined in the hardware config. Users do not specify the CMM in the
        # hardware config, so if the slot is present in the hardware config, we only need to check
        # whether it is a QTM.
        if ext_trig_sync.slot in self.instrument_cfg.hardware_description.modules:
            if (
                self.instrument_cfg.hardware_description.modules[ext_trig_sync.slot].instrument_type
                != "QTM"
            ):
                raise ValueError(
                    f"Slot {ext_trig_sync.slot} specified in the `sync_on_external_trigger` "
                    "settings contains a module that is not a QTM or CMM. External trigger "
                    "synchronization only works with these two module types."
                )
        elif ext_trig_sync.channel != 1:
            # The user may be keeping a module outside of the hardware config intentionally, so
            # allow but warn just in case.
            warnings.warn(
                f"Slot {ext_trig_sync.slot} specified in the sync_on_external_trigger settings is "
                "not present in the hardware description. If this is a CMM module, only channel 1 "
                f"can be used, but channel {ext_trig_sync.channel} was specified."
            )

        # Gather all port-clock pairs with ports connected to the same channel as specified in
        # external trigger sync...
        port_clocks_to_check: dict[str, ChannelPath] = {}
        for port_clock, path in self.portclock_to_path.items():
            if (
                path.module_idx == ext_trig_sync.slot
                and path.channel_idx == ext_trig_sync.channel - 1
            ):
                port_clocks_to_check[port_clock] = path

        # ... and check whether they have any conflicting settings.
        for port_clock, path in port_clocks_to_check.items():
            if "digital_input" not in path.channel_name:
                raise ValueError(
                    f"Slot {path.module_idx} channel {path.channel_idx} is present in the "
                    f"connectivity as {path}, which is not a 'digital_input'."
                )

            dig_thresholds = self.instrument_cfg.hardware_options.digitization_thresholds
            if dig_thresholds is None or port_clock not in dig_thresholds:
                if ext_trig_sync.input_threshold is None:
                    raise ValueError(
                        f"No input threshold was set for {path}. Please specify an input "
                        "threshold, either via the 'sync_on_external_trigger' settings or the "
                        "hardware options."
                    )
            elif dig_thresholds[port_clock].analog_threshold != ext_trig_sync.input_threshold:
                raise ValueError(
                    f"Channel {path} has an associated 'analog_threshold="
                    f"{dig_thresholds[port_clock].analog_threshold}' "
                    "which is different from 'sync_on_external_trigger.input_threshold="
                    f"{ext_trig_sync.input_threshold}'"
                )

    def distribute_data(self) -> None:
        """
        Distributes the pulses and acquisitions assigned to the cluster over the
        individual module compilers.
        """
        for compiler in self.instrument_compilers.values():
            for portclock in compiler.portclocks:
                port, clock = portclock.split("-")
                portclock_tuple = (port, clock)
                if portclock_tuple in self._op_infos:
                    for pulse in self._op_infos[portclock_tuple]:
                        compiler.add_op_info(port, clock, pulse)

    def compile(self, debug_mode: bool, repetitions: int = 1) -> dict[str, Any]:
        """
        Performs the compilation.

        Parameters
        ----------
        debug_mode
            Debug mode can modify the compilation process,
            so that debugging of the compilation process is easier.
        repetitions
            Amount of times to repeat execution of the schedule.

        Returns
        -------
        :
            The part of the compiled instructions relevant for this instrument.

        """
        program: dict[str, Any] = {"settings": self._settings}

        sequence_to_file = self.instrument_cfg.hardware_description.sequence_to_file

        for compiler in self.instrument_compilers.values():
            instrument_program = compiler.compile(
                repetitions=repetitions,
                sequence_to_file=sequence_to_file,
                debug_mode=debug_mode,
            )
            if instrument_program is not None and len(instrument_program) > 0:
                program[compiler.name] = instrument_program

        return program
