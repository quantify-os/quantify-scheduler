# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utilty classes for Qblox analog modules."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Iterator

from quantify_scheduler.backends.qblox import (
    constants,
    helpers,
    instrument_compilers,
    q1asm_instructions,
)
from quantify_scheduler.backends.qblox.compiler_abc import (
    ClusterModuleCompiler,
    SequencerCompiler,
)
from quantify_scheduler.backends.qblox.enums import (
    ChannelMode,
    DistortionCorrectionLatencyEnum,
    QbloxFilterConfig,
    QbloxFilterMarkerDelay,
)
from quantify_scheduler.backends.qblox.operation_handling.acquisitions import (
    SquareAcquisitionStrategy,
)
from quantify_scheduler.backends.qblox.operation_handling.pulses import (
    MarkerPulseStrategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    NcoPhaseShiftStrategy,
    NcoResetClockPhaseStrategy,
    NcoSetClockFrequencyStrategy,
)
from quantify_scheduler.backends.types.qblox import (
    AnalogModuleSettings,
    AnalogSequencerSettings,
    BasebandModuleSettings,
    OpInfo,
    RFModuleSettings,
    StaticAnalogModuleProperties,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.pulse_library import SetClockFrequency

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.operation_handling.base import (
        IOperationStrategy,
    )
    from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
    from quantify_scheduler.schedules.schedule import AcquisitionMetadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class NcoOperationTimingError(ValueError):
    """Exception thrown if there are timing errors for NCO operations."""


class AnalogSequencerCompiler(SequencerCompiler):
    """
    Class that performs the compilation steps on the sequencer level, for QCM and
    QRM-type modules.

    Parameters
    ----------
    parent
        A reference to the module compiler this sequencer belongs to.
    index
        Index of the sequencer.
    portclock
        Tuple that specifies the unique port and clock combination for this
        sequencer. The first value is the port, second is the clock.
    static_hw_properties
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
    settings
        The settings set to this sequencer.
    latency_corrections
        Dict containing the delays for each port-clock combination.
    qasm_hook_func
        Allows the user to inject custom Q1ASM code into the compilation, just prior to
        returning the final string.
    lo_name
        The name of the local oscillator instrument connected to the same output via
        an IQ mixer. This is used for frequency calculations.
    downconverter_freq
        .. warning::
            Using ``downconverter_freq`` requires custom Qblox hardware, do not use otherwise.

        Frequency of the external downconverter if one is being used.
        Defaults to ``None``, in which case the downconverter is inactive.
    mix_lo
        Boolean flag for IQ mixing with LO.
        Defaults to ``True`` meaning IQ mixing is applied.
    marker_debug_mode_enable
        Boolean flag to indicate if markers should be pulled high at the start of operations.
        Defaults to False, which means the markers will not be used during the sequence.
    """

    def __init__(
        self,
        parent: AnalogModuleCompiler,
        index: int,
        portclock: tuple[str, str],
        static_hw_properties: StaticAnalogModuleProperties,
        settings: AnalogSequencerSettings,
        latency_corrections: dict[str, float],
        distortion_corrections: dict[int, Any] | None = None,
        qasm_hook_func: Callable | None = None,
        lo_name: str | None = None,
        downconverter_freq: float | None = None,
        mix_lo: bool = True,
        marker_debug_mode_enable: bool = False,
    ) -> None:
        self.static_hw_properties: StaticAnalogModuleProperties  # help type checker
        self._settings: AnalogSequencerSettings  # help type checker
        super().__init__(
            parent=parent,
            index=index,
            portclock=portclock,
            static_hw_properties=static_hw_properties,
            settings=settings,
            latency_corrections=latency_corrections,
            distortion_corrections=distortion_corrections,
            qasm_hook_func=qasm_hook_func,
        )
        self.associated_ext_lo = lo_name
        self.downconverter_freq = downconverter_freq
        self.mix_lo = mix_lo
        self._marker_debug_mode_enable = marker_debug_mode_enable

        self._default_marker = (
            self.static_hw_properties.channel_name_to_digital_marker.get(
                self._settings.channel_name, self.static_hw_properties.default_marker
            )
        )

    @property
    def frequency(self) -> float | None:
        """
        The frequency used for modulation of the pulses.

        Returns
        -------
        :
            The frequency.
        """
        return self._settings.modulation_freq

    @frequency.setter
    def frequency(self, freq: float) -> None:
        """
        Assigns a modulation frequency to the sequencer.

        Parameters
        ----------
        freq
            The frequency to be used for modulation.

        Raises
        ------
        ValueError
            Attempting to set the modulation frequency to a new value even though a
            different value has been previously assigned.
        """
        if (
            self._settings.modulation_freq is not None
            and not math.isnan(self._settings.modulation_freq)
            and not math.isclose(self._settings.modulation_freq, freq)
        ):
            raise ValueError(
                f"Attempting to set the modulation frequency of '{self.name}' of "
                f"'{self.parent.name}' to {freq:e}, while it has previously been set "
                f"to {self._settings.modulation_freq:e}."
            )

        self._settings.modulation_freq = freq
        self._settings.nco_en = freq is not None

    def _prepare_acq_settings(
        self,
        acquisitions: list[IOperationStrategy],
        acq_metadata: AcquisitionMetadata,
    ) -> None:
        """
        Sets sequencer settings that are specific to certain acquisitions.
        For example for a TTL acquisition strategy.

        Parameters
        ----------
        acquisitions
            List of the acquisitions assigned to this sequencer.
        acq_metadata
            Acquisition metadata.
        """
        acquisition_infos: list[OpInfo] = list(
            map(lambda acq: acq.operation_info, acquisitions)
        )
        if acq_metadata.acq_protocol == "TriggerCount":
            self._settings.ttl_acq_auto_bin_incr_en = (
                acq_metadata.bin_mode == BinMode.AVERAGE
            )
            if len(self.connected_input_indices) == 1:
                self._settings.ttl_acq_input_select = self.connected_input_indices[0]
            elif len(self.connected_input_indices) > 1:
                raise ValueError(
                    f"Please make sure you use a single real input for this "
                    f"portclock combination. "
                    f"Found: {len(self.connected_input_indices)} connected. "
                    f"TTL acquisition does not support multiple inputs."
                    f"Problem occurred for port {self.port} with"
                    f"clock {self.clock}, which corresponds to {self.name} of "
                    f"{self.parent.name}."
                )

        elif acq_metadata.acq_protocol == "ThresholdedAcquisition":
            self._settings.thresholded_acq_rotation = acquisition_infos[0].data.get(
                "acq_rotation"
            )

            integration_length = acquisition_infos[0].data.get("duration", 0.0) * 1e9
            self._settings.thresholded_acq_threshold = (
                acquisition_infos[0].data.get("acq_threshold", 0.0) * integration_length
            )
            for info in acquisition_infos:
                if (address := info.data.get("feedback_trigger_address")) is not None:
                    self._settings.thresholded_acq_trigger_en = True
                    self._settings.thresholded_acq_trigger_address = address

        self._settings.integration_length_acq = (
            self._get_integration_length_from_acquisitions()
        )

    def _get_integration_length_from_acquisitions(self) -> int | None:
        """
        Get the (validated) integration_length sequencer setting.

        Get the duration of all SSB integration acquisitions assigned to this sequencer
        and validate that they are all the same.
        """
        integration_length = None
        for op_strat in self.op_strategies:
            if not isinstance(op_strat, SquareAcquisitionStrategy):
                continue

            acq_duration_ns = round(op_strat.operation_info.duration * 1e9)
            if acq_duration_ns % constants.GRID_TIME != 0:
                raise ValueError(
                    "Attempting to perform square acquisition with a duration of "
                    f"{acq_duration_ns} ns. Please ensure the duration is a multiple "
                    f"of {constants.GRID_TIME} ns.\n\nException caused by "
                    f"{repr(op_strat)}."
                )
            if integration_length is None:
                integration_length = acq_duration_ns
            elif integration_length != acq_duration_ns:
                raise ValueError(
                    f"Attempting to set an integration_length of {acq_duration_ns} "
                    f"ns, while this was previously determined to be "
                    f"{integration_length}. Please check whether all square "
                    "acquisitions in the schedule have the same duration."
                )

        return integration_length

    def prepare(self) -> None:
        """
        Perform necessary operations on this sequencer's data before
        :meth:`~quantify_scheduler.backends.qblox.compiler_abc.SequencerCompiler.compile`
        is called.
        """
        self._update_set_clock_frequency_operations()
        self.op_strategies = self._replace_marker_pulses(self.op_strategies)
        self._check_nco_operation_timing(self._get_ordered_operations())
        super().prepare()

    def _update_set_clock_frequency_operations(self) -> None:
        for op_strat in self.op_strategies:
            if op_strat.operation_info.name == SetClockFrequency.__name__:
                op_strat.operation_info.data.update(
                    {
                        "interm_freq_old": self.frequency,
                    }
                )

    @staticmethod
    def _replace_marker_pulses(
        op_strategies: list[IOperationStrategy],
    ) -> list[IOperationStrategy]:
        """Replaces MarkerPulse operations by explicit high and low operations."""
        new_op_strategies: list[IOperationStrategy] = []
        for op_strategy in op_strategies:
            if isinstance(op_strategy, MarkerPulseStrategy):
                high_op_info = OpInfo(
                    name=op_strategy.operation_info.name,
                    data=op_strategy.operation_info.data.copy(),
                    timing=op_strategy.operation_info.timing,
                )
                duration = op_strategy.operation_info.data["duration"]
                high_op_info.data["enable"] = True
                high_op_info.data["duration"] = 0
                new_op_strategies.append(
                    MarkerPulseStrategy(
                        operation_info=high_op_info,
                        channel_name=op_strategy.channel_name,
                    )
                )

                low_op_info = OpInfo(
                    name=op_strategy.operation_info.name,
                    data=op_strategy.operation_info.data.copy(),
                    timing=op_strategy.operation_info.timing + duration,
                )
                low_op_info.data["enable"] = False
                low_op_info.data["duration"] = 0
                new_op_strategies.append(
                    MarkerPulseStrategy(
                        operation_info=low_op_info,
                        channel_name=op_strategy.channel_name,
                    )
                )
            else:
                new_op_strategies.append(op_strategy)

        return new_op_strategies

    @staticmethod
    def _check_nco_operation_timing(
        ordered_op_strategies: list[IOperationStrategy],
    ) -> None:
        """Check whether this sequencer's operation adhere to NCO timing restrictions."""
        last_freq_upd_time = -constants.NCO_SET_FREQ_WAIT
        last_phase_upd_time = -constants.NCO_SET_PH_DELTA_WAIT
        for op in ordered_op_strategies:
            timing = round(op.operation_info.timing * 1e9)
            if isinstance(op, NcoSetClockFrequencyStrategy):
                if (diff := timing - last_freq_upd_time) < constants.NCO_SET_FREQ_WAIT:
                    raise NcoOperationTimingError(
                        f"Operation {op.operation_info} occurred {diff} ns after the "
                        "previous frequency update. The minimum time between frequency "
                        f"updates must be {constants.NCO_SET_FREQ_WAIT} ns."
                    )
                else:
                    last_freq_upd_time = timing

            if isinstance(op, (NcoPhaseShiftStrategy, NcoResetClockPhaseStrategy)):
                timing = round(op.operation_info.timing * 1e9)
                if (
                    diff := timing - last_phase_upd_time
                ) < constants.NCO_SET_PH_DELTA_WAIT:
                    raise NcoOperationTimingError(
                        f"Operation {op.operation_info} occurred {diff} ns after the "
                        "previous phase update. The minimum time between phase "
                        f"updates must be {constants.NCO_SET_PH_DELTA_WAIT} ns."
                    )
                else:
                    last_phase_upd_time = timing

            if isinstance(
                op,
                (
                    NcoSetClockFrequencyStrategy,
                    NcoPhaseShiftStrategy,
                    NcoResetClockPhaseStrategy,
                ),
            ):
                try:
                    helpers.to_grid_time(timing * 1e-9, constants.NCO_TIME_GRID)
                except ValueError as e:
                    raise NcoOperationTimingError(
                        f"NCO related operation {op.operation_info} must be on "
                        f"{constants.NCO_TIME_GRID} ns time grid"
                    ) from e

    def _write_pre_wait_sync_instructions(self, qasm: QASMProgram) -> None:
        """
        Write instructions to the QASM program that must come before the first wait_sync.

        The duration must be equal for all module types.
        """
        qasm.set_marker(self._default_marker)

    def _write_repetition_loop_header(self, qasm: QASMProgram) -> None:
        """
        Write the Q1ASM that should appear at the start of the repetition loop.

        The duration must be equal for all module types.
        """
        qasm.emit(q1asm_instructions.RESET_PHASE)
        qasm.emit(
            q1asm_instructions.UPDATE_PARAMETERS,
            constants.MIN_TIME_BETWEEN_OPERATIONS,
        )

    def _insert_qasm(
        self, op_strategy: IOperationStrategy, qasm_program: QASMProgram
    ) -> None:
        """
        Get Q1ASM instruction(s) from ``op_strategy`` and insert them into ``qasm_program``.

        Optionally wrap pulses and acquisitions in marker pulses depending on the
        ``marker_debug_mode_enable`` setting.
        """
        if self._marker_debug_mode_enable:
            valid_operation = (
                op_strategy.operation_info.is_acquisition
                or op_strategy.operation_info.data.get("wf_func") is not None
            )
            if valid_operation:
                qasm_program.set_marker(self._decide_markers(op_strategy))
                op_strategy.insert_qasm(qasm_program)
                qasm_program.set_marker(self._default_marker)
                qasm_program.emit(
                    q1asm_instructions.UPDATE_PARAMETERS,
                    constants.MIN_TIME_BETWEEN_OPERATIONS,
                )
                qasm_program.elapsed_time += constants.MIN_TIME_BETWEEN_OPERATIONS

        else:
            op_strategy.insert_qasm(qasm_program)

    def _decide_markers(self, operation: IOperationStrategy) -> int:
        """
        Helper method to decide what markers should be pulled high when enable_marker is
        set to True.  Checks what module and operation are being processed, then builds
        a bit string accordingly.

        Note that with the current quantify structure a sequencer cannot have connected
        inputs and outputs simultaneously.  Therefore, the QRM baseband module pulls
        both input or output markers high when doing an operation, as it is impossible
        during compile time to find out what physical port is being used.

        Parameters
        ----------
        operation
            The operation currently being processed by the sequence.

        Returns
        -------
            A bit string passed on to the set_mrk function of the Q1ASM object.
        """
        marker_bit_string = 0
        instrument_type = self.static_hw_properties.instrument_type
        if instrument_type == "QCM":
            for output in self.connected_output_indices:
                marker_bit_string |= 1 << output
        elif instrument_type == "QRM":
            marker_bit_string = (
                0b1100 if operation.operation_info.is_acquisition else 0b0011
            )

        # For RF modules, the first two indices correspond to path enable/disable.
        # Therefore, the index of the output is shifted by 2.
        elif instrument_type == "QCM_RF":
            # connected outputs are either 0,1 or 2,3 corresponding to
            # marker bitstrings 0b0100 and 0b1000 respectively
            output = self.connected_output_indices[0] // 2
            marker_bit_string |= 1 << (output + 2)
            marker_bit_string |= self._default_marker
        elif instrument_type == "QRM_RF":
            marker_bit_string = (
                0b1011 if operation.operation_info.is_acquisition else 0b0111
            )
        return marker_bit_string


class AnalogModuleCompiler(ClusterModuleCompiler, ABC):
    """
    Base class for QCM and QRM-type modules.

    This class is defined as an abstract base class since the distinctions between the
    different devices are defined in subclasses.
    Effectively, this base class contains the functionality shared by all Qblox
    devices and serves to avoid repeated code between them.

    Parameters
    ----------
    parent: :class:`quantify_scheduler.backends.qblox.instrument_compilers.ClusterCompiler`
        Reference to the parent object.
    name
        Name of the `QCoDeS` instrument this compiler object corresponds to.
    total_play_time
        Total time execution of the schedule should go on for. This parameter is
        used to ensure that the different devices, potentially with different clock
        rates, can work in a synchronized way when performing multiple executions of
        the schedule.
    instrument_cfg
        The part of the hardware configuration dictionary referring to this device. This is one
        of the inner dictionaries of the overall hardware config.
    latency_corrections
        Dict containing the delays for each port-clock combination. This is specified in
        the top layer of hardware config.
    """

    _settings: AnalogModuleSettings  # type: ignore

    def __init__(
        self,
        parent: instrument_compilers.ClusterCompiler,
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
        distortion_corrections: dict[int, Any] | None = None,
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
            latency_corrections=latency_corrections,
            distortion_corrections=distortion_corrections,
        )
        self.sequencers: dict[str, AnalogSequencerCompiler] = {}

    @property
    @abstractmethod
    def static_hw_properties(self) -> StaticAnalogModuleProperties:
        """
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
        """

    def _construct_sequencer_compiler(
        self,
        index: int,
        portclock: tuple[str, str],
        channel_name: str,
        sequencer_cfg: dict[str, Any],
        channel_cfg: dict[str, Any],
    ) -> AnalogSequencerCompiler:
        """Create an instance of :class:`AnalogSequencerCompiler`."""
        settings = AnalogSequencerSettings.initialize_from_config_dict(
            sequencer_cfg=sequencer_cfg,
            channel_name=channel_name,
            connected_output_indices=self.static_hw_properties._get_connected_output_indices(
                channel_name
            ),
            connected_input_indices=self.static_hw_properties._get_connected_input_indices(
                channel_name
            ),
        )
        return AnalogSequencerCompiler(
            parent=self,
            index=index,
            portclock=portclock,
            static_hw_properties=self.static_hw_properties,
            settings=settings,
            latency_corrections=self.latency_corrections,
            qasm_hook_func=sequencer_cfg.get("qasm_hook_func"),
            lo_name=channel_cfg.get("lo_name"),
            mix_lo=channel_cfg.get("mix_lo", True),
            marker_debug_mode_enable=channel_cfg.get("marker_debug_mode_enable", False),
            downconverter_freq=channel_cfg.get("downconverter_freq"),
        )

    @abstractmethod
    def assign_frequencies(self, sequencer: AnalogSequencerCompiler) -> None:
        """
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, and an LO frequency to each output (if applicable).
        """

    def _set_lo_interm_freqs(
        self,
        freqs: helpers.Frequencies,
        sequencer: AnalogSequencerCompiler,
        compiler_lo_baseband: (
            instrument_compilers.LocalOscillatorCompiler | None
        ) = None,
        lo_freq_setting_rf: str | None = None,
    ) -> None:
        """
        Sets the LO/IF frequencies, for baseband and RF modules.

        Parameters
        ----------
        freqs
            LO, IF, and clock frequencies, supplied via an :class:`.helpers.Frequencies`
            object.
        sequencer
            The sequencer for which frequences are to be set.
        compiler_lo_baseband
            For baseband modules, supply the
            :class:`.instrument_compilers.LocalOscillatorCompiler` instrument compiler
            of which the frequency is to be set.
        lo_freq_setting_rf
            For RF modules, supply the name of the LO frequency param from the
            :class:`.RFModuleSettings` that is to be set.

        Raises
        ------
        ValueError
            In case neither LO frequency nor IF has been supplied.
        ValueError
            In case both LO frequency and IF have been supplied and do not adhere to
            :math:`f_{RF} = f_{LO} + f_{IF}`.
        ValueError
            In case of RF, when the LO frequency was already set to a different value.
        """
        if freqs.LO is not None:
            if compiler_lo_baseband is not None:
                compiler_lo_baseband.frequency = freqs.LO

            elif lo_freq_setting_rf is not None:
                previous_lo_freq = getattr(self._settings, lo_freq_setting_rf)

                if (
                    previous_lo_freq is not None
                    and not math.isnan(previous_lo_freq)
                    and not math.isclose(freqs.LO, previous_lo_freq)
                ):
                    raise ValueError(
                        f"Attempting to set '{lo_freq_setting_rf}' to frequency "
                        f"'{freqs.LO:e}', while it has previously already been set to "
                        f"'{previous_lo_freq:e}'!"
                    )

                setattr(self._settings, lo_freq_setting_rf, freqs.LO)

        if freqs.IF is not None:
            sequencer.frequency = freqs.IF

    @abstractmethod
    def assign_attenuation(self) -> None:
        """
        An abstract method that should be overridden. Meant to assign
        attenuation settings from the hardware configuration if there is any.
        """

    def prepare(self) -> None:
        """
        Performs the logic needed before being able to start the compilation. In effect,
        this means assigning the pulses and acquisitions to the sequencers and
        calculating the relevant frequencies in case an external local oscillator is
        used.
        """
        self._configure_input_gains()
        self._configure_mixer_offsets()
        self._configure_hardware_distortion_corrections()
        self._construct_all_sequencer_compilers()
        for seq in self.sequencers.values():
            self.assign_frequencies(seq)
        self.distribute_data()
        for seq in self.sequencers.values():
            seq.prepare()
        self._ensure_single_scope_mode_acquisition_sequencer()
        self.assign_attenuation()

    def _configure_input_gains(self) -> None:
        """
        Configures input gain of module settings.
        Loops through all valid channel names and checks for gain values in hw config.
        Throws a ValueError if a gain value gets modified.
        """
        in0_gain, in1_gain = None, None

        for channel_name in helpers.find_channel_names(self.instrument_cfg):
            channel_mapping = self.instrument_cfg.get(channel_name, None)

            if channel_name.startswith(ChannelMode.COMPLEX):
                in0_gain = channel_mapping.get("input_gain_I", None)
                in1_gain = channel_mapping.get("input_gain_Q", None)

            elif channel_name.startswith(ChannelMode.REAL):
                # The next code block is for backwards compatibility.
                in_gain = channel_mapping.get("input_gain", None)
                if in_gain is None:
                    in0_gain = channel_mapping.get("input_gain_0", None)
                    in1_gain = channel_mapping.get("input_gain_1", None)
                else:
                    in0_gain = in_gain
                    in1_gain = in_gain

            if in0_gain is not None:
                if (
                    self._settings.in0_gain is None
                    or in0_gain == self._settings.in0_gain
                ):
                    self._settings.in0_gain = in0_gain
                else:
                    raise ValueError(
                        f"Overwriting gain of {channel_name} of module {self.name} "
                        f"to in0_gain: {in0_gain}.\nIt was previously set to "
                        f"in0_gain: {self._settings.in0_gain}."
                    )

            if in1_gain is not None:
                if (
                    self._settings.in1_gain is None
                    or in1_gain == self._settings.in1_gain
                ):
                    self._settings.in1_gain = in1_gain
                else:
                    raise ValueError(
                        f"Overwriting gain of {channel_name} of module {self.name} "
                        f"to in1_gain: {in1_gain}.\nIt was previously set to "
                        f"in1_gain: {self._settings.in1_gain}."
                    )

    def _configure_mixer_offsets(self) -> None:
        """
        Configures offset of input, uses calc_from_units_volt found in helper file.
        Raises an exception if a value outside the accepted voltage range is given.
        """
        supported_channels = ("complex_output_0", "complex_output_1")
        for output_idx, channel_name in enumerate(supported_channels):
            if channel_name not in self.instrument_cfg:
                continue

            output_cfg = self.instrument_cfg[channel_name]
            voltage_range = self.static_hw_properties.mixer_dc_offset_range
            if output_idx == 0:
                self._settings.offset_ch0_path_I = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_I", output_cfg
                )
                self._settings.offset_ch0_path_Q = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_Q", output_cfg
                )
            else:
                self._settings.offset_ch1_path_I = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_I", output_cfg
                )
                self._settings.offset_ch1_path_Q = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_Q", output_cfg
                )

    def _configure_distortion_correction_latency_compensations(self) -> None:
        channel_names = helpers.find_channel_names(self.instrument_cfg)
        for channel_name in channel_names:
            io_mapping = self.instrument_cfg.get(channel_name, None)
            if io_mapping is None:
                continue

            dc_comp = io_mapping.pop(
                "distortion_correction_latency_compensation",
                DistortionCorrectionLatencyEnum.NO_DELAY_COMP,
            )
            marker_debug_mode_enable = io_mapping.get("marker_debug_mode_enable", False)
            output_indices = self.static_hw_properties._get_connected_output_indices(
                channel_name
            )
            if output_indices is None:
                output_indices = ()
            for output in output_indices:
                if ChannelMode.DIGITAL in channel_name:
                    self._configure_dc_latency_comp_for_marker(output, dc_comp)
                else:
                    if marker_debug_mode_enable:
                        if f"digital_output_{output}" in channel_names:
                            raise ValueError(
                                f"digital_output_{output} cannot be used along with "
                                "marker_debug_mode_enable on the same digital output."
                            )
                        else:
                            if output in self.distortion_corrections:
                                self.distortion_corrections[output][
                                    "marker_debug_mode_enable"
                                ] = True
                            marker_output = output
                            if (
                                channel_name
                                in self.static_hw_properties.channel_name_to_digital_marker
                            ):
                                marker_output = (
                                    self.static_hw_properties.channel_name_to_digital_marker[
                                        channel_name
                                    ]
                                    + 1
                                ) % 2
                            self._configure_dc_latency_comp_for_marker(
                                marker_output, dc_comp
                            )
                    self._configure_dc_latency_comp_for_output(output, dc_comp)

    def _configure_dc_latency_comp_for_output(self, output: int, dc_comp: int) -> None:
        self._settings.distortion_corrections[output].bt.config = (
            QbloxFilterConfig.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.BT
            else QbloxFilterConfig.BYPASSED
        )
        for i, mask in enumerate(
            [
                DistortionCorrectionLatencyEnum.EXP0,
                DistortionCorrectionLatencyEnum.EXP1,
                DistortionCorrectionLatencyEnum.EXP2,
                DistortionCorrectionLatencyEnum.EXP3,
            ]
        ):
            getattr(self._settings.distortion_corrections[output], f"exp{i}").config = (
                QbloxFilterConfig.DELAY_COMP
                if dc_comp & mask
                else QbloxFilterConfig.BYPASSED
            )
        self._settings.distortion_corrections[output].fir.config = (
            QbloxFilterConfig.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.FIR
            else QbloxFilterConfig.BYPASSED
        )

    def _configure_dc_latency_comp_for_marker(self, output: int, dc_comp: int) -> None:
        self._settings.distortion_corrections[output].bt.marker_delay = (
            QbloxFilterMarkerDelay.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.BT
            else QbloxFilterMarkerDelay.BYPASSED
        )
        for i, mask in enumerate(
            [
                DistortionCorrectionLatencyEnum.EXP0,
                DistortionCorrectionLatencyEnum.EXP1,
                DistortionCorrectionLatencyEnum.EXP2,
                DistortionCorrectionLatencyEnum.EXP3,
            ]
        ):
            getattr(
                self._settings.distortion_corrections[output], f"exp{i}"
            ).marker_delay = (
                QbloxFilterMarkerDelay.DELAY_COMP
                if dc_comp & mask
                else QbloxFilterMarkerDelay.BYPASSED
            )
        self._settings.distortion_corrections[output].fir.marker_delay = (
            QbloxFilterMarkerDelay.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.FIR
            else QbloxFilterMarkerDelay.BYPASSED
        )

    def _configure_hardware_distortion_corrections(self) -> None:
        """Assign distortion corrections to settings of instrument compiler."""
        self._configure_distortion_correction_latency_compensations()

    def _ensure_single_scope_mode_acquisition_sequencer(self) -> None:
        """
        Raises an error if multiple sequencers use scope mode acquisition,
        because that's not supported by the hardware.
        Also, see
        :func:`~quantify_scheduler.instrument_coordinator.components.qblox._QRMComponent._determine_scope_mode_acquisition_sequencer_and_qblox_acq_index`
        which also ensures the program that gets uploaded to the hardware satisfies this
        requirement.

        Raises
        ------
        ValueError
            Multiple sequencers have to perform trace acquisition. This is not
            supported by the hardware.
        """  # noqa: E501 line too long

        def is_scope_acquisition(acquisition: OpInfo) -> bool:
            return acquisition.data["protocol"] == "Trace"

        scope_acq_seq = None
        for seq in self.sequencers.values():
            op_infos = [
                op.operation_info
                for op in seq.op_strategies
                if op.operation_info.is_acquisition
            ]

            has_scope = any(map(is_scope_acquisition, op_infos))
            if has_scope:
                if scope_acq_seq is not None:
                    helpers.single_scope_mode_acquisition_raise(
                        sequencer_0=scope_acq_seq,
                        sequencer_1=seq.index,
                        module_name=self.name,
                    )
                scope_acq_seq = seq.index


class BasebandModuleCompiler(AnalogModuleCompiler):
    """
    Abstract class with all the shared functionality between the QRM and QCM baseband
    modules.
    """

    def __init__(
        self,
        parent: instrument_compilers.ClusterCompiler,
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
        distortion_corrections: dict[int, Any] | None = None,
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
            latency_corrections=latency_corrections,
            distortion_corrections=distortion_corrections,
        )
        self._settings: BasebandModuleSettings = (  # type: ignore
            BasebandModuleSettings.extract_settings_from_mapping(instrument_cfg)
        )

    def assign_frequencies(self, sequencer: AnalogSequencerCompiler) -> None:
        """
        Determines LO/IF frequencies and assigns them, for baseband modules.

        In case of **no** external local oscillator, the NCO is given the same
        frequency as the clock -- unless NCO was permanently disabled via
        `"interm_freq": 0` in the hardware config.

        In case of **an** external local oscillator and `sequencer.mix_lo` is
        ``False``, the LO is given the same frequency as the clock
        (via :func:`.helpers.determine_clock_lo_interm_freqs`).
        """
        compiler_container = self.parent.parent
        if (
            sequencer.clock not in compiler_container.resources
            or ChannelMode.DIGITAL in sequencer.settings.channel_name
        ):
            return

        clock_freq = compiler_container.resources[sequencer.clock]["freq"]
        if sequencer.associated_ext_lo is None:
            # Set NCO frequency to the clock frequency, unless NCO was permanently
            # disabled via `"interm_freq": 0` in the hardware config
            if sequencer.frequency != 0:
                sequencer.frequency = clock_freq
        else:
            # In using external local oscillator, determine clock and LO/IF freqs,
            # and then set LO/IF freqs, and enable NCO (via setter)
            if (
                compiler_lo := compiler_container.instrument_compilers.get(
                    sequencer.associated_ext_lo
                )
            ) is None:
                raise RuntimeError(
                    f"External local oscillator '{sequencer.associated_ext_lo}' set to "
                    f"be used by '{sequencer.name}' of '{self.name}' not found! Make "
                    f"sure it is present in the hardware configuration."
                )
            try:
                freqs = helpers.determine_clock_lo_interm_freqs(
                    freqs=helpers.Frequencies(
                        clock=clock_freq,
                        LO=compiler_lo.frequency,
                        IF=sequencer.frequency,
                    ),
                    downconverter_freq=sequencer.downconverter_freq,
                    mix_lo=sequencer.mix_lo,
                )
            except Exception as error:  # Adding sequencer info to exception message
                raise error.__class__(
                    f"{error} (for '{sequencer.name}' of '{self.name}' "
                    f"with port '{sequencer.port}' and clock '{sequencer.clock}')"
                )
            self._set_lo_interm_freqs(
                freqs=freqs, sequencer=sequencer, compiler_lo_baseband=compiler_lo
            )

    def assign_attenuation(self) -> None:
        """
        Meant to assign attenuation settings from the hardware configuration, if there
        is any. For baseband modules there is no attenuation parameters currently.
        """

    def _configure_dc_latency_comp_for_marker(self, output: int, dc_comp: int) -> None:
        self._settings.distortion_corrections[output].bt.marker_delay = (
            QbloxFilterMarkerDelay.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.BT
            else QbloxFilterMarkerDelay.BYPASSED
        )
        for i, mask in enumerate(
            [
                DistortionCorrectionLatencyEnum.EXP0,
                DistortionCorrectionLatencyEnum.EXP1,
                DistortionCorrectionLatencyEnum.EXP2,
                DistortionCorrectionLatencyEnum.EXP3,
            ]
        ):
            getattr(
                self._settings.distortion_corrections[output], f"exp{i}"
            ).marker_delay = (
                QbloxFilterMarkerDelay.DELAY_COMP
                if dc_comp & mask
                else QbloxFilterMarkerDelay.BYPASSED
            )
        self._settings.distortion_corrections[output].fir.marker_delay = (
            QbloxFilterMarkerDelay.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.FIR
            else QbloxFilterMarkerDelay.BYPASSED
        )


class RFModuleCompiler(AnalogModuleCompiler):
    """
    Abstract class with all the shared functionality between the QRM-RF and QCM-RF
    modules.
    """

    def __init__(
        self,
        parent: instrument_compilers.ClusterCompiler,
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
        distortion_corrections: dict[int, Any] | None = None,
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
            latency_corrections=latency_corrections,
            distortion_corrections=distortion_corrections,
        )
        self._settings: RFModuleSettings = (  # type: ignore
            RFModuleSettings.extract_settings_from_mapping(instrument_cfg)
        )

    def assign_frequencies(self, sequencer: AnalogSequencerCompiler) -> None:
        """Determines LO/IF frequencies and assigns them for RF modules."""
        compiler_container = self.parent.parent
        if (
            not sequencer.connected_output_indices
            or sequencer.clock not in compiler_container.resources
            or ChannelMode.DIGITAL in sequencer.settings.channel_name
        ):
            return

        for lo_idx in RFModuleCompiler._get_connected_lo_indices(sequencer):
            lo_freq_setting_name = f"lo{lo_idx}_freq"
            try:
                freqs = helpers.determine_clock_lo_interm_freqs(
                    freqs=helpers.Frequencies(
                        clock=compiler_container.resources[sequencer.clock]["freq"],
                        LO=getattr(self._settings, lo_freq_setting_name),
                        IF=sequencer.frequency,
                    ),
                    downconverter_freq=sequencer.downconverter_freq,
                    mix_lo=True,
                )
            except Exception as error:  # Adding sequencer info to exception message
                raise error.__class__(
                    f"{error} (for '{sequencer.name}' of '{self.name}' "
                    f"with port '{sequencer.port}' and clock '{sequencer.clock}')"
                )
            self._set_lo_interm_freqs(
                freqs=freqs,
                sequencer=sequencer,
                lo_freq_setting_rf=lo_freq_setting_name,
            )

    @staticmethod
    def _get_connected_lo_indices(sequencer: AnalogSequencerCompiler) -> Iterator[int]:
        """
        Identify the LO the sequencer is outputting.
        Use the sequencer output to module output correspondence, and then
        use the fact that LOX is connected to module output X.
        """
        for sequencer_output_index in sequencer.connected_output_indices:
            if sequencer_output_index % 2 != 0:
                # We will only use real output 0 and 2, as they are part of the same
                # complex outputs as real output 1 and 3
                continue

            module_output_index = 0 if sequencer_output_index == 0 else 1
            yield module_output_index

    def assign_attenuation(self) -> None:
        """
        Assigns attenuation settings from the hardware configuration.

        Floats that are a multiple of 1 are converted to ints.
        This is needed because the :func:`quantify_core.measurement.control.grid_setpoints`
        converts setpoints to floats when using an attenuation as settable.
        """

        def _convert_to_int(value: float, label: str) -> int | None:
            if value is not None:
                if not math.isclose(value % 1, 0):
                    raise ValueError(
                        f'Trying to set "{label}" to non-integer value {value}'
                    )
                return int(value)

        complex_input_0 = self.instrument_cfg.get("complex_input_0", {})
        complex_output_0 = self.instrument_cfg.get("complex_output_0", {})

        input_att = complex_input_0.get("input_att", None)
        if (input_att_output := complex_output_0.get("input_att", None)) is not None:
            if input_att is not None:
                raise ValueError(
                    f"'input_att' is defined for both 'complex_input_0' and "
                    f"'complex_output_0' on module '{self.name}', which is prohibited. "
                    f"Make sure you define it at a single place."
                )
            input_att = input_att_output
        self._settings.in0_att = _convert_to_int(input_att, label="in0_att")

        self._settings.out0_att = _convert_to_int(
            complex_output_0.get("output_att", None),
            label="out0_att",
        )
        complex_output_1 = self.instrument_cfg.get("complex_output_1", {})
        self._settings.out1_att = _convert_to_int(
            complex_output_1.get("output_att", None),
            label="out1_att",
        )

    def _configure_dc_latency_comp_for_marker(self, output: int, dc_comp: int) -> None:
        output += 2  # In RF modules, the two marker indices are 2 & 3.
        self._settings.distortion_corrections[output].bt.marker_delay = (
            QbloxFilterMarkerDelay.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.BT
            else QbloxFilterMarkerDelay.BYPASSED
        )
        for i, mask in enumerate(
            [
                DistortionCorrectionLatencyEnum.EXP0,
                DistortionCorrectionLatencyEnum.EXP1,
                DistortionCorrectionLatencyEnum.EXP2,
                DistortionCorrectionLatencyEnum.EXP3,
            ]
        ):
            getattr(
                self._settings.distortion_corrections[output], f"exp{i}"
            ).marker_delay = (
                QbloxFilterMarkerDelay.DELAY_COMP
                if dc_comp & mask
                else QbloxFilterMarkerDelay.BYPASSED
            )
        self._settings.distortion_corrections[output].fir.marker_delay = (
            QbloxFilterMarkerDelay.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.FIR
            else QbloxFilterMarkerDelay.BYPASSED
        )
