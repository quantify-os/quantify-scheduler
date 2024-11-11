# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utilty classes for Qblox analog modules."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator

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
from quantify_scheduler.backends.qblox.exceptions import NcoOperationTimingError
from quantify_scheduler.backends.qblox.operation_handling.acquisitions import (
    SquareAcquisitionStrategy,
)
from quantify_scheduler.backends.qblox.operation_handling.factory_analog import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    NcoPhaseShiftStrategy,
    NcoResetClockPhaseStrategy,
    NcoSetClockFrequencyStrategy,
)
from quantify_scheduler.backends.qblox.qasm_program import get_marker_binary
from quantify_scheduler.backends.types.qblox import (
    AnalogModuleSettings,
    AnalogSequencerSettings,
    BasebandModuleSettings,
    ComplexChannelDescription,
    ComplexInputGain,
    DigitalChannelDescription,
    InputAttenuation,
    OpInfo,
    OutputAttenuation,
    RealInputGain,
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
    from quantify_scheduler.backends.qblox_backend import (
        _ClusterModuleCompilationConfig,
        _SequencerCompilationConfig,
    )
    from quantify_scheduler.resources import Resource
    from quantify_scheduler.schedules.schedule import AcquisitionMetadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
    static_hw_properties
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
    sequencer_cfg
        The instrument compiler config associated to this device.

    """

    def __init__(
        self,
        parent: AnalogModuleCompiler,
        index: int,
        static_hw_properties: StaticAnalogModuleProperties,
        sequencer_cfg: _SequencerCompilationConfig,
    ) -> None:
        self.static_hw_properties: StaticAnalogModuleProperties  # help type checker
        super().__init__(
            parent=parent,
            index=index,
            static_hw_properties=static_hw_properties,
            sequencer_cfg=sequencer_cfg,
        )

        self._settings: AnalogSequencerSettings = (  # type: ignore  (override type)
            AnalogSequencerSettings.initialize_from_compilation_config(
                sequencer_cfg=sequencer_cfg,
                connected_output_indices=static_hw_properties._get_connected_output_indices(
                    sequencer_cfg.channel_name,
                ),
                connected_input_indices=static_hw_properties._get_connected_input_indices(
                    sequencer_cfg.channel_name,
                    sequencer_cfg.channel_name_measure,
                ),
            )
        )

        self.associated_ext_lo = sequencer_cfg.lo_name
        self.downconverter_freq = (
            sequencer_cfg.hardware_description.downconverter_freq
            if isinstance(sequencer_cfg.hardware_description, ComplexChannelDescription)
            else None
        )
        self.mix_lo = (
            sequencer_cfg.hardware_description.mix_lo
            if not isinstance(sequencer_cfg.hardware_description, DigitalChannelDescription)
            else None
        )
        self._marker_debug_mode_enable = (
            (sequencer_cfg.hardware_description.marker_debug_mode_enable)
            if not isinstance(sequencer_cfg.hardware_description, DigitalChannelDescription)
            else None
        )

        self._default_marker = self.static_hw_properties.channel_name_to_digital_marker.get(
            self._settings.channel_name,
            self.static_hw_properties.default_marker,
        )

    @property
    def settings(self) -> AnalogSequencerSettings:
        """
        Gives the current settings. Overridden from the parent class for type hinting.

        Returns
        -------
        :
            The settings set to this sequencer.

        """
        return self._settings

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

    def get_operation_strategy(
        self,
        operation_info: OpInfo,
    ) -> IOperationStrategy:
        """
        Determines and instantiates the correct strategy object.

        Parameters
        ----------
        operation_info
            The operation we are building the strategy for.
        channel_name
            Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).

        Returns
        -------
        :
            The instantiated strategy object.

        """
        return get_operation_strategy(operation_info, self.settings.channel_name)

    def add_operation_strategy(self, op_strategy: IOperationStrategy) -> None:
        """
        Adds the operation strategy to the sequencer compiler.

        Parameters
        ----------
        op_strategy
            The operation strategy.

        """
        if op_strategy.operation_info.data.get("marker_pulse", False):
            # A digital pulse always uses one output.
            op_strategy.operation_info.data["output"] = self.connected_output_indices[0]
            op_strategy.operation_info.data["default_marker"] = (
                self.static_hw_properties.default_marker
            )

        super().add_operation_strategy(op_strategy)

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

        def _verify_param_range(
            param_name: str,
            val: float | None,
            min_value: float,
            max_value: float,
            portclock: tuple[str, str],
        ) -> None:
            if val is None:
                return

            if val < min_value or val > max_value:
                raise ValueError(
                    f"Attempting to configure {param_name} to {val} for the sequencer "
                    f"specified with portclock '{portclock}' and while the "
                    f"hardware requires it to be between {min_value} and {max_value}."
                )

        acquisition_infos: list[OpInfo] = list(map(lambda acq: acq.operation_info, acquisitions))
        if acq_metadata.acq_protocol == "TriggerCount":
            self._settings.ttl_acq_auto_bin_incr_en = acq_metadata.bin_mode == BinMode.DISTRIBUTION
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
            acq_rotation = acquisition_infos[0].data.get("acq_rotation")
            _verify_param_range(
                param_name="acq_rotation",
                val=acq_rotation,
                min_value=constants.MIN_PHASE_ROTATION_ACQ,
                max_value=constants.MAX_PHASE_ROTATION_ACQ,
                portclock=self.portclock,
            )
            self._settings.thresholded_acq_rotation = acq_rotation

            acq_threshold = acquisition_infos[0].data.get("acq_threshold", 0.0)
            _verify_param_range(
                param_name="acq_threshold",
                val=acq_threshold,
                min_value=constants.MIN_DISCRETIZATION_THRESHOLD_ACQ,
                max_value=constants.MAX_DISCRETIZATION_THRESHOLD_ACQ,
                portclock=self.portclock,
            )
            integration_length = acquisition_infos[0].data.get("duration", 0.0) * 1e9
            self._settings.thresholded_acq_threshold = acq_threshold * integration_length

            for info in acquisition_infos:
                if (address := info.data.get("feedback_trigger_address")) is not None:
                    self._settings.thresholded_acq_trigger_en = True
                    self._settings.thresholded_acq_trigger_address = address

        self._settings.integration_length_acq = self._get_integration_length_from_acquisitions()

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
        if not self._settings.allow_off_grid_nco_ops:
            self._assert_nco_operation_timing_on_grid(self._get_ordered_operations())
        super().prepare()

    def _update_set_clock_frequency_operations(self) -> None:
        for op_strat in self.op_strategies:
            if op_strat.operation_info.name == SetClockFrequency.__name__:
                op_strat.operation_info.data.update(
                    {
                        "interm_freq_old": self.frequency,
                    }
                )

    def _get_latency_correction_ns(self, latency_correction: float) -> int:
        # NCO grid alignment for NCO operations, _without_ latency corrections, is
        # already checked in `_assert_nco_operation_timing_on_grid`. Therefore here we
        # only check the latency corrections. Method overridden from superclass because
        # only QRM/QCM modules have NCO operations.
        latency_correction = super()._get_latency_correction_ns(latency_correction)
        if not self._settings.allow_off_grid_nco_ops:
            try:
                helpers.to_grid_time(latency_correction * 1e-9, constants.NCO_TIME_GRID)
            except ValueError as e:
                raise NcoOperationTimingError(
                    f"The latency correction value of {latency_correction} ns for "
                    f"{self.port}-{self.clock} does not align with the grid time of "
                    f"{constants.NCO_TIME_GRID} ns for NCO operations. The latency "
                    "corrections must adhere to this grid time to ensure proper "
                    "alignment of all later operations in the schedule."
                ) from e

        return latency_correction

    @staticmethod
    def _assert_nco_operation_timing_on_grid(
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
                if (diff := timing - last_phase_upd_time) < constants.NCO_SET_PH_DELTA_WAIT:
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

    def _assert_total_play_time_on_nco_grid(self) -> None:
        if self._settings.allow_off_grid_nco_ops:
            return

        try:
            helpers.to_grid_time(self.parent.total_play_time, constants.NCO_TIME_GRID)
        except ValueError as e:
            raise NcoOperationTimingError(
                "The schedule is repeated with a duration of "
                f"{round(self.parent.total_play_time * 1e9)} ns per iteration, "
                "which does not align with the grid time of "
                f"{constants.NCO_TIME_GRID} ns for NCO operations. The duration "
                "must adhere to this grid time to ensure proper alignment of NCO "
                "operations for each iteration."
            ) from e

    def _write_pre_wait_sync_instructions(self, qasm: QASMProgram) -> None:
        """
        Write instructions to the QASM program that must come before the first wait_sync.

        The duration must be equal for all module types.
        """
        qasm.emit(
            q1asm_instructions.SET_MARKER,
            get_marker_binary(self._default_marker),
            comment=f"set markers to {self._default_marker}",
        )

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

    def _insert_qasm(self, op_strategy: IOperationStrategy, qasm_program: QASMProgram) -> None:
        """
        Get Q1ASM instruction(s) from ``op_strategy`` and insert them into ``qasm_program``.

        Optionally wrap pulses and acquisitions in marker pulses depending on the
        ``marker_debug_mode_enable`` setting.
        """
        if self._marker_debug_mode_enable and (
            op_strategy.operation_info.is_acquisition
            or op_strategy.operation_info.data.get("wf_func") is not None
        ):
            marker = self._decide_markers(op_strategy)
            qasm_program.emit(
                q1asm_instructions.SET_MARKER,
                get_marker_binary(marker),
                comment=f"set markers to {marker}",
            )
            op_strategy.insert_qasm(qasm_program)
            qasm_program.emit(
                q1asm_instructions.SET_MARKER,
                get_marker_binary(self._default_marker),
                comment=f"set markers to {self._default_marker}",
            )
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
            marker_bit_string = 0b1100 if operation.operation_info.is_acquisition else 0b0011

        # For RF modules, the first two indices correspond to path enable/disable.
        # Therefore, the index of the output is shifted by 2.
        elif instrument_type == "QCM_RF":
            # connected outputs are either 0,1 or 2,3 corresponding to
            # marker bitstrings 0b0100 and 0b1000 respectively
            output = self.connected_output_indices[0] // 2
            marker_bit_string |= 1 << (output + 2)
            marker_bit_string |= self._default_marker
        elif instrument_type == "QRM_RF":
            marker_bit_string = 0b1011 if operation.operation_info.is_acquisition else 0b0111
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
    name
        Name of the `QCoDeS` instrument this compiler object corresponds to.
    total_play_time
        Total time execution of the schedule should go on for. This parameter is
        used to ensure that the different devices, potentially with different clock
        rates, can work in a synchronized way when performing multiple executions of
        the schedule.
    instrument_cfg
        The instrument compiler config referring to this device.

    """

    _settings: AnalogModuleSettings  # type: ignore

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
        self.sequencers: dict[str, AnalogSequencerCompiler] = {}

    @property
    @abstractmethod
    def static_hw_properties(self) -> StaticAnalogModuleProperties:
        """
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
        """

    def _construct_sequencer_compiler(
        self, index: int, sequencer_cfg: _SequencerCompilationConfig
    ) -> AnalogSequencerCompiler:
        """Create an instance of :class:`AnalogSequencerCompiler`."""
        return AnalogSequencerCompiler(
            parent=self,
            index=index,
            static_hw_properties=self.static_hw_properties,
            sequencer_cfg=sequencer_cfg,
        )

    @abstractmethod
    def assign_frequencies(
        self,
        sequencer: AnalogSequencerCompiler,
        external_lo: instrument_compilers.LocalOscillatorCompiler | None,
        clock_frequency: float,
    ) -> None:
        """
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, and an LO frequency to each output (if applicable).

        Parameters
        ----------
        sequencer
            Sequencer compiler object whose NCO frequency will be determined and set.
        external_lo
            Optional LO compiler object representing an external LO, whose LO frequency
            will be determined and set.
        clock_frequency
            Clock frequency of the clock assigned to the sequencer compiler.

        """

    @abstractmethod
    def assign_attenuation(self) -> None:
        """
        An abstract method that should be overridden. Meant to assign
        attenuation settings from the hardware configuration if there is any.
        """

    def prepare(
        self,
        external_los: dict[str, instrument_compilers.LocalOscillatorCompiler] | None = None,
        schedule_resources: dict[str, Resource] | None = None,
        **kwargs,  # noqa: ARG002  (unused arg necessary to fit signature)
    ) -> None:
        """
        Performs the logic needed before being able to start the compilation. In effect,
        this means assigning the pulses and acquisitions to the sequencers and
        calculating the relevant frequencies in case an external local oscillator is
        used.

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
        self._configure_input_gains()
        self._configure_mixer_offsets()
        self._configure_hardware_distortion_corrections()
        self._construct_all_sequencer_compilers()
        for seq in self.sequencers.values():
            if (
                schedule_resources is not None
                and seq.clock in schedule_resources
                and ChannelMode.DIGITAL not in seq.settings.channel_name
            ):
                if seq.associated_ext_lo is None or external_los is None:
                    external_lo = None
                else:
                    external_lo = external_los[seq.associated_ext_lo]
                self.assign_frequencies(
                    seq,
                    external_lo,
                    schedule_resources[seq.clock]["freq"],
                )
        self.distribute_data()
        for seq in self.sequencers.values():
            seq.prepare()
        self._ensure_single_scope_mode_acquisition_sequencer()
        self.assign_attenuation()

    def _configure_input_gains(self) -> None:
        """
        Configures input gain of module settings.
        Loops through all valid channel names and checks for gain values in hw config.
        """
        in0_gain, in1_gain = None, None
        # These variables are for checking if gain is being overwritten
        init_in0_gain, init_in1_gain = None, None
        init_portclock = None

        for portclock, path in self.portclock_to_path.items():
            if self.instrument_cfg.hardware_options is None:
                continue

            input_gains = self.instrument_cfg.hardware_options.input_gain
            if input_gains is None:
                continue

            if portclock in input_gains:
                input_gain = input_gains[portclock]
                if isinstance(input_gain, ComplexInputGain):
                    in0_gain = input_gain.gain_I
                    in1_gain = input_gain.gain_Q
                elif isinstance(input_gain, RealInputGain):
                    if int(path.channel_name[-1]) == 0:
                        in0_gain = input_gain
                    elif int(path.channel_name[-1]) == 1:
                        in1_gain = input_gain

                if init_portclock and (
                    (
                        in0_gain is not None
                        and init_in0_gain is not None
                        and in0_gain != init_in0_gain
                    )
                    or (
                        in1_gain is not None
                        and init_in1_gain is not None
                        and in1_gain != init_in1_gain
                    )
                ):
                    raise ValueError(
                        f"Found non-unique input gains on module {self.name}. "
                        f"Please ensure that all `input_gain` values on the input `I` channel are "
                        f"the same, and all `input_gain` values on the input `Q` channel are the "
                        f"same in the hardware options for all port clock combinations with this "
                        f"module. \n\n"
                        f"For more information, please visit "
                        f"https://quantify-os.org/docs/quantify-scheduler/latest/reference/qblox/Cluster.html#gain-configuration"
                    )

                init_in0_gain = in0_gain
                init_in1_gain = in1_gain
                init_portclock = deepcopy(portclock)

            self._settings.in0_gain = in0_gain
            self._settings.in1_gain = in1_gain

    def _configure_mixer_offsets(self) -> None:
        """
        Configures offset of input, uses calc_from_units_volt found in helper file.
        Raises an exception if a value outside the accepted voltage range is given.
        """
        supported_outputs = ("complex_output_0", "complex_output_1")
        for output_idx, output_label in enumerate(supported_outputs):
            for portclock, path in self.portclock_to_path.items():
                if output_label == path.channel_name:
                    if self.instrument_cfg.hardware_options.mixer_corrections is not None:
                        mixer_corrections = (
                            self.instrument_cfg.hardware_options.mixer_corrections.get(
                                portclock, None
                            )
                            if output_label == path.channel_name
                            else None
                        )
                    else:
                        mixer_corrections = None

                    if mixer_corrections:
                        offset_i = mixer_corrections.dc_offset_i
                        offset_q = mixer_corrections.dc_offset_q
                        voltage_range = self.static_hw_properties.mixer_dc_offset_range
                        if output_idx == 0:
                            self._settings.offset_ch0_path_I = helpers.calc_from_units_volt(
                                voltage_range,
                                self.name,
                                "dc_mixer_offset_I",
                                offset_i,
                            )
                            self._settings.offset_ch0_path_Q = helpers.calc_from_units_volt(
                                voltage_range,
                                self.name,
                                "dc_mixer_offset_Q",
                                offset_q,
                            )
                            self._settings.out0_lo_freq_cal_type_default = (
                                mixer_corrections.auto_lo_cal
                            )
                        else:
                            self._settings.offset_ch1_path_I = helpers.calc_from_units_volt(
                                voltage_range,
                                self.name,
                                "dc_mixer_offset_I",
                                offset_i,
                            )
                            self._settings.offset_ch1_path_Q = helpers.calc_from_units_volt(
                                voltage_range,
                                self.name,
                                "dc_mixer_offset_Q",
                                offset_q,
                            )
                            self._settings.out1_lo_freq_cal_type_default = (
                                mixer_corrections.auto_lo_cal
                            )

    def _configure_distortion_correction_latency_compensations(
        self, distortion_configs: dict[int, Any] | None = None
    ) -> None:
        channel_names = [path.channel_name for path in self.portclock_to_path.values()]
        hardware_description = self.instrument_cfg.hardware_description

        for description in hardware_description.model_fields_set:
            channel_name = ""
            channel_description = None
            if description in channel_names:
                channel_name = description
                channel_description = getattr(hardware_description, description)

            if channel_description is None:
                continue

            dc_comp = channel_description.distortion_correction_latency_compensation
            output_indices = self.static_hw_properties._get_connected_output_indices(channel_name)
            if output_indices is None:
                output_indices = ()
            for output in output_indices:
                if ChannelMode.DIGITAL in channel_name:
                    self._configure_dc_latency_comp_for_marker(output, dc_comp)
                else:
                    if channel_description.marker_debug_mode_enable:
                        if f"digital_output_{output}" in channel_names:
                            raise ValueError(
                                f"digital_output_{output} cannot be used along with "
                                "marker_debug_mode_enable on the same digital output."
                            )

                        distortion_configs = distortion_configs or {}
                        if output in distortion_configs:
                            distortion_configs[output]["marker_debug_mode_enable"] = True
                        marker_output = output
                        if channel_name in self.static_hw_properties.channel_name_to_digital_marker:
                            marker_output = (
                                self.static_hw_properties.channel_name_to_digital_marker[
                                    channel_name
                                ]
                                + 1
                            ) % 2
                        self._configure_dc_latency_comp_for_marker(marker_output, dc_comp)
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
                QbloxFilterConfig.DELAY_COMP if dc_comp & mask else QbloxFilterConfig.BYPASSED
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
            getattr(self._settings.distortion_corrections[output], f"exp{i}").marker_delay = (
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
                op.operation_info for op in seq.op_strategies if op.operation_info.is_acquisition
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
        name: str,
        total_play_time: float,
        instrument_cfg: _ClusterModuleCompilationConfig,
    ) -> None:
        super().__init__(
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self._settings: BasebandModuleSettings = (  # type: ignore
            BasebandModuleSettings.extract_settings_from_mapping(instrument_cfg)
        )

    def assign_frequencies(
        self,
        sequencer: AnalogSequencerCompiler,
        external_lo: instrument_compilers.LocalOscillatorCompiler | None,
        clock_frequency: float,
    ) -> None:
        """
        Determines LO/IF frequencies and assigns them, for baseband modules.

        In case of **no** external local oscillator, the NCO is given the same
        frequency as the clock -- unless NCO was permanently disabled via
        `"interm_freq": 0` in the hardware config.

        In case of **an** external local oscillator and `sequencer.mix_lo` is
        ``False``, the LO is given the same frequency as the clock
        (via :func:`.helpers.determine_clock_lo_interm_freqs`).

        Parameters
        ----------
        sequencer
            Sequencer compiler object whose NCO frequency will be determined and set.
        external_lo
            Optional LO compiler object representing an external LO, whose LO frequency
            will be determined and set.
        clock_frequency
            Clock frequency of the clock assigned to the sequencer compiler.

        Raises
        ------
        ValueError
            If the NCO and/or LO frequencies cannot be determined, are invalid, or are
            inconsistent with the clock frequency.

        """
        if external_lo is None:
            if sequencer.settings.nco_en:
                sequencer.frequency = clock_frequency
            # Early return: we do not validate further since there is no way we could
            # retrieve an NCO frequency if it was not already set, and if it was not set
            # then nco_en was set to False.
            return

        # In using external local oscillator, determine clock and LO/IF freqs,
        # and then set LO/IF freqs, and enable NCO (via setter)
        try:
            freqs = helpers.determine_clock_lo_interm_freqs(
                freqs=helpers.Frequencies(
                    clock=clock_frequency,
                    LO=external_lo.frequency,
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

        external_lo.frequency = freqs.LO
        sequencer.frequency = freqs.IF

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
            getattr(self._settings.distortion_corrections[output], f"exp{i}").marker_delay = (
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
        name: str,
        total_play_time: float,
        instrument_cfg: _ClusterModuleCompilationConfig,
    ) -> None:
        super().__init__(
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self._settings: RFModuleSettings = (  # type: ignore
            RFModuleSettings.extract_settings_from_mapping(instrument_cfg)
        )

    def assign_frequencies(
        self,
        sequencer: AnalogSequencerCompiler,
        external_lo: instrument_compilers.LocalOscillatorCompiler | None,  # noqa: ARG002
        clock_frequency: float,
    ) -> None:
        """
        Determines LO/IF frequencies and assigns them for RF modules.

        Parameters
        ----------
        sequencer
            Sequencer compiler object whose NCO frequency will be determined and set.
        external_lo
            Optional LO compiler object representing an external LO. Not used for RF
            modules, since they use the LO frequency in the module settings.
        clock_frequency
            Clock frequency of the clock assigned to the sequencer compiler.

        Raises
        ------
        ValueError
            If the NCO and/or LO frequencies cannot be determined, are invalid, or are
            inconsistent with the clock frequency.

        """
        for lo_idx in RFModuleCompiler._get_connected_lo_indices(sequencer):
            try:
                freqs = helpers.determine_clock_lo_interm_freqs(
                    freqs=helpers.Frequencies(
                        clock=clock_frequency,
                        LO=self._get_lo_frequency(lo_idx),
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

            self._set_lo_frequency(lo_idx, freqs.LO)

            # Calling the frequency setter inside the for-loop helps catch bugs where
            # two different frequencies could accidentally be set on the same sequencer.
            sequencer.frequency = freqs.IF

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

    def _get_lo_frequency(self, lo_idx: int) -> float | None:
        """
        Get the LO frequency from the settings.

        Parameters
        ----------
        lo_idx : int
            The LO index.

        Returns
        -------
        float | None
            The frequency, or None if it has not been determined yet.

        Raises
        ------
        IndexError
            If the derived class instance does not contain an LO with that index.

        """
        if lo_idx == 0:
            return self._settings.lo0_freq
        if lo_idx == 1 and self.static_hw_properties.instrument_type == "QCM_RF":
            return self._settings.lo1_freq
        raise IndexError(
            f"Module {self.name} of type "
            f"{self.static_hw_properties.instrument_type} does not have an LO "
            f"index {lo_idx}."
        )

    def _set_lo_frequency(self, lo_idx: int, frequency: float) -> None:
        """
        Set the LO frequency from the settings.

        Parameters
        ----------
        lo_idx : int
            The LO index.

        frequency : float
            The frequency.

        Raises
        ------
        IndexError
            If the derived class instance does not contain an LO with that index.

        """
        previous_lo_freq = self._get_lo_frequency(lo_idx)

        if (
            previous_lo_freq is not None
            and not math.isnan(previous_lo_freq)
            and not math.isclose(frequency, previous_lo_freq)
        ):
            raise ValueError(
                f"Attempting to set 'lo{lo_idx}' to frequency "
                f"'{frequency:e}', while it has previously already been set to "
                f"'{previous_lo_freq:e}'!"
            )

        if lo_idx == 0:
            self._settings.lo0_freq = frequency
        elif lo_idx == 1 and self.static_hw_properties.instrument_type == "QCM_RF":
            self._settings.lo1_freq = frequency
        else:
            raise IndexError(
                f"Module {self.name} of type "
                f"{self.static_hw_properties.instrument_type} does not have an LO "
                f"index {lo_idx}."
            )

    def assign_attenuation(self) -> None:
        """
        Assigns attenuation settings from the hardware configuration.

        Floats that are a multiple of 1 are converted to ints.
        This is needed because the :func:`quantify_core.measurement.control.grid_setpoints`
        converts setpoints to floats when using an attenuation as settable.
        """

        def _convert_to_int(
            value: InputAttenuation | OutputAttenuation | None, label: str
        ) -> int | None:
            if value is None:
                return None
            if not math.isclose(value % 1, 0):
                raise ValueError(f'Trying to set "{label}" to non-integer value {value}')
            return int(value)

        in0_att = out0_att = out1_att = None

        input_att_cfg = self.instrument_cfg.hardware_options.input_att

        output_att_cfg = self.instrument_cfg.hardware_options.output_att

        for (
            portclock,
            path,
        ) in self.instrument_cfg.portclock_to_path.items():
            if path.channel_name == "complex_input_0" and input_att_cfg is not None:
                in0_att = input_att_cfg.get(portclock)

        for (
            portclock,
            path,
        ) in self.instrument_cfg.portclock_to_path.items():
            if path.channel_name == "complex_output_0":
                if input_att_cfg is not None:
                    in0_att_from_output = input_att_cfg.get(portclock)
                    if in0_att_from_output is not None:
                        if in0_att is not None:
                            raise ValueError(
                                f"'input_att' is defined for both 'complex_input_0' and "
                                f"'complex_output_0' on module '{self.name}', which is prohibited. "
                                f"Make sure you define it at a single place."
                            )
                        in0_att = in0_att_from_output

                if output_att_cfg is not None:
                    out0_att = output_att_cfg.get(portclock)

            if path.channel_name == "complex_output_1" and output_att_cfg is not None:
                out1_att = output_att_cfg.get(portclock)

        self._settings.in0_att = _convert_to_int(in0_att, label="in0_att")
        self._settings.out0_att = _convert_to_int(out0_att, label="out0_att")
        self._settings.out1_att = _convert_to_int(out1_att, label="out1_att")

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
            getattr(self._settings.distortion_corrections[output], f"exp{i}").marker_delay = (
                QbloxFilterMarkerDelay.DELAY_COMP
                if dc_comp & mask
                else QbloxFilterMarkerDelay.BYPASSED
            )
        self._settings.distortion_corrections[output].fir.marker_delay = (
            QbloxFilterMarkerDelay.DELAY_COMP
            if dc_comp & DistortionCorrectionLatencyEnum.FIR
            else QbloxFilterMarkerDelay.BYPASSED
        )
