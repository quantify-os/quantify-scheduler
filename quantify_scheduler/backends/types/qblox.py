# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Python dataclasses for compilation to Qblox hardware."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    get_args,
)

from dataclasses_json import DataClassJsonMixin
from pydantic import Field, field_validator
from pydantic.functional_validators import model_validator
from typing_extensions import Annotated

from quantify_scheduler.backends.qblox import q1asm_instructions
from quantify_scheduler.backends.qblox.constants import (
    DEFAULT_MIXER_AMP_RATIO,
    DEFAULT_MIXER_PHASE_ERROR_DEG,
    MAX_MIXER_AMP_RATIO,
    MAX_MIXER_PHASE_ERROR_DEG,
    MIN_MIXER_AMP_RATIO,
    MIN_MIXER_PHASE_ERROR_DEG,
)
from quantify_scheduler.backends.qblox.enums import (
    ChannelMode,
    DistortionCorrectionLatencyEnum,
    LoCalEnum,
    QbloxFilterConfig,
    QbloxFilterMarkerDelay,
    SidebandCalEnum,
    TimetagTraceType,
)
from quantify_scheduler.backends.types.common import (
    HardwareDescription,
    HardwareDistortionCorrection,
    HardwareOptions,
    IQMixerDescription,
    LocalOscillatorDescription,
    MixerCorrections,
    OpticalModulatorDescription,
    SoftwareDistortionCorrection,
)
from quantify_scheduler.enums import (
    TimeRef,  # noqa: TCH001 pydantic needs them
    TimeSource,  # noqa: TCH001 pydantic needs them
)
from quantify_scheduler.structure.model import DataStructure

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox_backend import (
        _ClusterModuleCompilationConfig,
        _SequencerCompilationConfig,
    )


class ValidationWarning(UserWarning):
    """Warning type for dubious arguments passed to pydantic models."""


@dataclass(frozen=True)
class BoundedParameter:
    """Specifies a certain parameter with a fixed max and min in a certain unit."""

    min_val: float
    """Min value allowed."""
    max_val: float
    """Max value allowed."""
    units: str
    """Units in which the parameter is specified."""


@dataclass(frozen=True)
class StaticHardwareProperties:
    """Specifies the fixed hardware properties needed in the backend."""

    instrument_type: str
    """The type of instrument."""
    max_sequencers: int
    """The amount of sequencers available."""

    def _get_connected_io_indices(self, mode: str, channel_idx: str) -> tuple[int, ...]:
        """Return the connected input/output indices associated to this channel name."""
        idx = int(channel_idx)
        return (2 * idx, 2 * idx + 1) if mode == ChannelMode.COMPLEX else (idx,)

    def _get_connected_output_indices(self, channel_name: str) -> tuple[int, ...]:
        """Return the connected output indices associated to this channel name."""
        mode, io, idx = channel_name.split("_")
        return self._get_connected_io_indices(mode, idx) if "output" in io else ()

    def _get_connected_input_indices(
        self, channel_name: str, channel_name_measure: Union[list[str], None]
    ) -> tuple[int, ...]:
        """Return the connected input indices associated to this channel name."""
        mode, io, idx = channel_name.split("_")
        if "input" in io:
            if channel_name_measure is None:
                return self._get_connected_io_indices(mode, idx)
        elif channel_name_measure is not None:
            if len(channel_name_measure) == 1:
                mode_measure, _, idx_measure = channel_name_measure[0].split("_")
                return self._get_connected_io_indices(mode_measure, idx_measure)
            else:
                # Edge case for compatibility with hardware config version 0.1 (SE-427)
                return (0, 1)

        return ()


@dataclass(frozen=True)
class StaticAnalogModuleProperties(StaticHardwareProperties):
    """Specifies the fixed hardware properties needed in the backend for QRM/QCM modules."""

    max_awg_output_voltage: Optional[float]
    """Maximum output voltage of the awg."""
    mixer_dc_offset_range: BoundedParameter
    """Specifies the range over which the dc offsets can be set that are used for mixer
    calibration."""
    default_marker: int = 0
    """The default marker value to set at the beginning of programs.
    Important for RF instruments that use the set_mrk command to enable/disable the RF output."""
    channel_name_to_digital_marker: Dict[str, int] = dataclasses_field(default_factory=dict)
    """A mapping from channel_name to digital marker setting.
    Specifies which marker bit needs to be set at start if the
    output (as a string ex. `complex_output_0`) contains a pulse."""


@dataclass(frozen=True)
class StaticTimetagModuleProperties(StaticHardwareProperties):
    """Specifies the fixed hardware properties needed in the backend for QTM modules."""


@dataclass(frozen=True)
class OpInfo(DataClassJsonMixin):
    """
    Data structure describing a pulse or acquisition and containing all the information
    required to play it.
    """

    name: str
    """Name of the operation that this pulse/acquisition is part of."""
    data: dict
    """The pulse/acquisition info taken from the ``data`` property of the
    pulse/acquisition in the schedule."""
    timing: float
    """The start time of this pulse/acquisition.
    Note that this is a combination of the start time "t_abs" of the schedule
    operation, and the t0 of the pulse/acquisition which specifies a time relative
    to "t_abs"."""

    @property
    def duration(self) -> float:
        """The duration of the pulse/acquisition."""
        return self.data["duration"]

    @property
    def is_acquisition(self) -> bool:
        """Returns ``True`` if this is an acquisition, ``False`` otherwise."""
        return "acq_channel" in self.data or "bin_mode" in self.data

    @property
    def is_real_time_io_operation(self) -> bool:
        """
        Returns ``True`` if the operation is a non-idle pulse (i.e., it has a
        waveform), ``False`` otherwise.
        """
        return (
            self.is_acquisition or self.is_parameter_update or self.data.get("wf_func") is not None
        )

    @property
    def is_offset_instruction(self) -> bool:
        """
        Returns ``True`` if the operation describes a DC offset operation,
        corresponding to the Q1ASM instruction ``set_awg_offset``.
        """
        return "offset_path_I" in self.data or "offset_path_Q" in self.data

    @property
    def is_parameter_instruction(self) -> bool:
        """
        Return ``True`` if the instruction is a parameter, like a voltage offset.

        From the Qblox documentation: "parameter operation instructions" are latched and
        only updated when the upd_param, play, acquire, acquire_weighed or acquire_ttl
        instructions are executed.

        Please refer to
        https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#q1-instructions
        for the full list of these instructions.
        """
        return (
            self.is_offset_instruction
            or "phase_shift" in self.data
            or "reset_clock_phase" in self.data
            or "clock_freq_new" in self.data
            or "marker_pulse" in self.data
            or "timestamp" in self.data
        )

    @property
    def is_parameter_update(self) -> bool:
        """
        Return ``True`` if the operation is a parameter update, corresponding to the
        Q1ASM instruction ``upd_param``.
        """
        return self.data.get("instruction", "") == q1asm_instructions.UPDATE_PARAMETERS

    @property
    def is_loop(self) -> bool:
        """
        Return ``True`` if the operation is a loop, corresponding to the Q1ASM
        instruction ``loop``.
        """
        return self.data.get("repetitions", None) is not None

    @property
    def is_control_flow_end(self) -> bool:
        """Return ``True`` if the operation is a control flow end."""
        return self.data.get("control_flow_end", None) is True

    def __str__(self) -> str:
        type_label: str = "Acquisition" if self.is_acquisition else "Pulse"
        return f'{type_label} "{self.name}" (t0={self.timing}, duration={self.duration})'

    def __repr__(self) -> str:
        repr_string = (
            f"{'Acquisition' if self.is_acquisition else 'Pulse'} "
            f"{str(self.name)} (t={self.timing} to "
            f"{self.timing + self.duration})\ndata={self.data}"
        )
        return repr_string


@dataclass(frozen=True)
class LOSettings(DataClassJsonMixin):
    """Dataclass containing all the settings for a generic LO instrument."""

    power: Dict[str, float]
    """Power of the LO source."""
    frequency: Dict[str, Optional[float]]
    """The frequency to set the LO to."""


_ModuleSettingsT = TypeVar("_ModuleSettingsT", bound="BaseModuleSettings")
"""
Custom type to allow correct type inference from ``extract_settings_from_mapping`` for
child classes.
"""


@dataclass
class QbloxRealTimeFilter(DataClassJsonMixin):
    """An individual real time filter on Qblox hardware."""

    coeffs: Optional[Union[float, List[float]]] = None
    """Coefficient(s) of the filter.
       Can be None if there is no filter
       or if it is inactive."""
    config: QbloxFilterConfig = QbloxFilterConfig.BYPASSED
    """Configuration of the filter.
       One of 'BYPASSED', 'ENABLED',
       or 'DELAY_COMP'."""
    marker_delay: QbloxFilterMarkerDelay = QbloxFilterMarkerDelay.BYPASSED
    """State of the marker delay.
       One of 'BYPASSED' or 'ENABLED'."""


@dataclass
class DistortionSettings(DataClassJsonMixin):
    """Distortion correction settings for all Qblox modules."""

    bt: QbloxRealTimeFilter = dataclasses_field(default_factory=QbloxRealTimeFilter)
    """The bias tee correction filter."""
    exp0: QbloxRealTimeFilter = dataclasses_field(default_factory=QbloxRealTimeFilter)
    """The exponential overshoot correction 1 filter."""
    exp1: QbloxRealTimeFilter = dataclasses_field(default_factory=QbloxRealTimeFilter)
    """The exponential overshoot correction 2 filter."""
    exp2: QbloxRealTimeFilter = dataclasses_field(default_factory=QbloxRealTimeFilter)
    """The exponential overshoot correction 3 filter."""
    exp3: QbloxRealTimeFilter = dataclasses_field(default_factory=QbloxRealTimeFilter)
    """The exponential overshoot correction 4 filter."""
    fir: QbloxRealTimeFilter = dataclasses_field(default_factory=QbloxRealTimeFilter)
    """The FIR filter."""


@dataclass
class BaseModuleSettings(DataClassJsonMixin):
    """Shared settings between all the Qblox modules."""

    offset_ch0_path_I: Optional[float] = None
    """The DC offset on the path_I of channel 0."""
    offset_ch0_path_Q: Optional[float] = None
    """The DC offset on the path_Q of channel 0."""
    offset_ch1_path_I: Optional[float] = None
    """The DC offset on path_I of channel 1."""
    offset_ch1_path_Q: Optional[float] = None
    """The DC offset on path_Q of channel 1."""
    in0_gain: Optional[int] = None
    """The gain of input 0."""
    in1_gain: Optional[int] = None
    """The gain of input 1."""
    distortion_corrections: List[DistortionSettings] = dataclasses_field(
        default_factory=lambda: [DistortionSettings() for _ in range(4)]
    )
    """distortion correction settings"""

    @classmethod
    def extract_settings_from_mapping(
        cls: type[_ModuleSettingsT],
        mapping: _ClusterModuleCompilationConfig,  # noqa: ARG003 not used
        **kwargs,
    ) -> _ModuleSettingsT:
        """
        Factory method that takes all the settings defined in the mapping and generates
        an instance of this class.

        Parameters
        ----------
        mapping
            The mapping dict to extract the settings from
        **kwargs
            Additional keyword arguments passed to the constructor. Can be used to
            override parts of the mapping dict.

        """
        return cls(**kwargs)


@dataclass
class AnalogModuleSettings(BaseModuleSettings):
    """Shared settings between all QCM/QRM modules."""

    offset_ch0_path_I: Optional[float] = None
    """The DC offset on the path_I of channel 0."""
    offset_ch0_path_Q: Optional[float] = None
    """The DC offset on the path_Q of channel 0."""
    offset_ch1_path_I: Optional[float] = None
    """The DC offset on path_I of channel 1."""
    offset_ch1_path_Q: Optional[float] = None
    """The DC offset on path_Q of channel 1."""
    out0_lo_freq_cal_type_default: LoCalEnum = LoCalEnum.OFF
    """
    Setting that controls whether the mixer of channel 0 is calibrated upon changing the
    LO and/or intermodulation frequency.
    """
    out1_lo_freq_cal_type_default: LoCalEnum = LoCalEnum.OFF
    """
    Setting that controls whether the mixer of channel 1 is calibrated upon changing the
    LO and/or intermodulation frequency.
    """
    in0_gain: Optional[int] = None
    """The gain of input 0."""
    in1_gain: Optional[int] = None
    """The gain of input 1."""


@dataclass
class BasebandModuleSettings(AnalogModuleSettings):
    """
    Settings for a baseband module.

    Class exists to ensure that the cluster baseband modules don't need special
    treatment in the rest of the code.
    """


@dataclass
class RFModuleSettings(AnalogModuleSettings):
    """
    Global settings for the module to be set in the InstrumentCoordinator component.
    This is kept separate from the settings that can be set on a per sequencer basis,
    which are specified in :class:`~.AnalogSequencerSettings`.
    """

    lo0_freq: Optional[float] = None
    """The frequency of Output 0 (O0) LO. If left `None`, the parameter will not be set.
    """
    lo1_freq: Optional[float] = None
    """The frequency of Output 1 (O1) LO. If left `None`, the parameter will not be set.
    """
    out0_att: Optional[int] = None
    """The attenuation of Output 0."""
    out1_att: Optional[int] = None
    """The attenuation of Output 1."""
    in0_att: Optional[int] = None
    """The attenuation of Input 0."""

    @classmethod
    def extract_settings_from_mapping(
        cls, mapping: _ClusterModuleCompilationConfig, **kwargs: Optional[dict]
    ) -> RFModuleSettings:
        """
        Factory method that takes all the settings defined in the mapping and generates
        an :class:`~.RFModuleSettings` object from it.

        Parameters
        ----------
        mapping
            The compiler config to extract the settings from
        **kwargs
            Additional keyword arguments passed to the constructor. Can be used to
            override parts of the mapping dict.

        """
        rf_settings = {}

        for portclock, path in mapping.portclock_to_path.items():
            modulation_frequencies = mapping.hardware_options.modulation_frequencies

            if modulation_frequencies is not None:
                pc_freqs = modulation_frequencies.get(portclock)
                lo_freq = pc_freqs.lo_freq if pc_freqs is not None else None
                if path.channel_name == "complex_output_0":
                    rf_settings["lo0_freq"] = lo_freq
                elif path.channel_name == "complex_output_1":
                    rf_settings["lo1_freq"] = lo_freq

        combined_settings = {**rf_settings, **kwargs}
        return cls(**combined_settings)


@dataclass
class TimetagModuleSettings(BaseModuleSettings):
    """
    Global settings for the module to be set in the InstrumentCoordinator component.
    This is kept separate from the settings that can be set on a per sequencer basis,
    which are specified in :class:`~.TimetagSequencerSettings`.
    """


@dataclass
class SequencerSettings(DataClassJsonMixin):
    """
    Sequencer level settings.

    In the Qblox driver these settings are typically recognized by parameter names of
    the form ``"{module}.sequencer{index}.{setting}"`` (for allowed values see
    `Cluster QCoDeS parameters
    <https://docs.qblox.com/en/main/api_reference/sequencer.html#cluster-qcodes-parameters>`__).
    These settings are set once and will remain unchanged after, meaning that these
    correspond to the "slow" QCoDeS parameters and not settings that are changed
    dynamically by the sequencer.

    These settings are mostly defined in the hardware configuration under each
    port-clock key combination or in some cases through the device configuration
    (e.g. parameters related to thresholded acquisition).
    """

    sync_en: bool
    """Enables party-line synchronization."""
    channel_name: str
    """Specifies the channel identifier of the hardware config (e.g. `complex_output_0`)."""
    channel_name_measure: Union[list[str], None]
    """Extra channel name necessary to define a `Measure` operation."""
    connected_output_indices: Tuple[int, ...]
    """Specifies the indices of the outputs this sequencer produces waveforms for."""
    connected_input_indices: Tuple[int, ...]
    """Specifies the indices of the inputs this sequencer collects data for."""
    sequence: Optional[Dict[str, Any]] = None
    """JSON compatible dictionary holding the waveforms and program for the
    sequencer."""
    seq_fn: Optional[str] = None
    """Filename of JSON file containing a dump of the waveforms and program."""
    thresholded_acq_trigger_address: Optional[int] = None
    """Sets the feedback trigger address to be used by conditional playback."""
    thresholded_acq_trigger_en: Optional[bool] = None
    """Enables the sequencer to record acquisitions."""
    thresholded_acq_trigger_invert: bool = False
    """
    If you want to set a trigger when the acquisition result is 1,
    the parameter must be set to false and vice versa.
    """

    @classmethod
    def initialize_from_compilation_config(
        cls,
        sequencer_cfg: _SequencerCompilationConfig,
        connected_output_indices: tuple[int, ...],
        connected_input_indices: tuple[int, ...],
    ) -> SequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer compilation config.

        Parameters
        ----------
        sequencer_cfg
            The sequencer compilation_config.
        connected_output_indices
            Specifies the indices of the outputs this sequencer produces waveforms for.
        connected_input_indices
            Specifies the indices of the inputs this sequencer collects data for.

        Returns
        -------
        : SequencerSettings
            A SequencerSettings instance with initial values.

        """
        return cls(
            sync_en=True,
            channel_name=sequencer_cfg.channel_name,
            channel_name_measure=sequencer_cfg.channel_name_measure,
            connected_output_indices=connected_output_indices,
            connected_input_indices=connected_input_indices,
        )


@dataclass
class AnalogSequencerSettings(SequencerSettings):
    """
    Sequencer level settings.

    In the Qblox driver these settings are typically recognized by parameter names of
    the form ``"{module}.sequencer{index}.{setting}"`` (for allowed values see
    `Cluster QCoDeS parameters
    <https://docs.qblox.com/en/master/api_reference/sequencer.html#cluster-qcodes-parameters>`__).
    These settings are set once and will remain unchanged after, meaning that these
    correspond to the "slow" QCoDeS parameters and not settings that are changed
    dynamically by the sequencer.

    These settings are mostly defined in the hardware configuration under each
    port-clock key combination or in some cases through the device configuration
    (e.g. parameters related to thresholded acquisition).
    """

    nco_en: bool = False
    """Specifies whether the NCO will be used or not."""
    init_offset_awg_path_I: float = 0.0
    """Specifies what value the sequencer offset for AWG path_I will be reset to
    before the start of the experiment."""
    init_offset_awg_path_Q: float = 0.0
    """Specifies what value the sequencer offset for AWG path_Q will be reset to
    before the start of the experiment."""
    init_gain_awg_path_I: float = 1.0
    """Specifies what value the sequencer gain for AWG path_I will be reset to
    before the start of the experiment."""
    init_gain_awg_path_Q: float = 1.0
    """Specifies what value the sequencer gain for AWG path_Q will be reset to
    before the start of the experiment."""
    modulation_freq: Optional[float] = None
    """Specifies the frequency of the modulation."""
    mixer_corr_phase_offset_degree: Optional[float] = None
    """The phase shift to apply between the I and Q channels, to correct for quadrature
    errors."""
    mixer_corr_gain_ratio: Optional[float] = None
    """The gain ratio to apply in order to correct for imbalances between the I and Q
    paths of the mixer."""
    auto_sideband_cal: SidebandCalEnum = SidebandCalEnum.OFF
    """
    Setting that controls whether the mixer is calibrated upon changing the
    intermodulation frequency.
    """
    integration_length_acq: Optional[int] = None
    """Integration length for acquisitions. Must be a multiple of 4 ns."""
    thresholded_acq_threshold: Optional[float] = None
    """The sequencer discretization threshold for discretizing the phase rotation result."""
    thresholded_acq_rotation: Optional[float] = None
    """The sequencer integration result phase rotation in degrees."""
    ttl_acq_input_select: Optional[int] = None
    """Selects the input used to compare against
    the threshold value in the TTL trigger acquisition path."""
    ttl_acq_threshold: Optional[float] = None
    """
    For QRM modules only, sets the threshold value with which to compare the input ADC
    values of the selected input path.
    """
    ttl_acq_auto_bin_incr_en: Optional[bool] = None
    """Selects if the bin index is automatically incremented when acquiring multiple triggers."""
    allow_off_grid_nco_ops: Optional[bool] = None
    """
    Flag to allow NCO operations to play at times that are not aligned with the NCO
    grid.
    """

    @classmethod
    def initialize_from_compilation_config(
        cls,
        sequencer_cfg: _SequencerCompilationConfig,
        connected_output_indices: tuple[int, ...],
        connected_input_indices: tuple[int, ...],
    ) -> AnalogSequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer compilation config.

        Parameters
        ----------
        sequencer_cfg
            The sequencer compilation_config.
        connected_output_indices
            Specifies the indices of the outputs this sequencer produces waveforms for.
        connected_input_indices
            Specifies the indices of the inputs this sequencer collects data for.

        Returns
        -------
        : AnalogSequencerSettings
            A AnalogSequencerSettings instance with initial values.

        """
        modulation_freq = (
            sequencer_cfg.modulation_frequencies.interm_freq
            if sequencer_cfg.modulation_frequencies is not None
            else None
        )
        # Allow NCO to be permanently disabled via `"interm_freq": 0` in the hardware config
        nco_en: bool = not (
            modulation_freq == 0
            or isinstance(sequencer_cfg.hardware_description, DigitalChannelDescription)
            or len(connected_output_indices) == 0
        )

        # TODO: there must be a way to make this nicer
        init_offset_awg_path_I = (
            sequencer_cfg.sequencer_options.init_offset_awg_path_I
            if sequencer_cfg.sequencer_options is not None
            else 0.0
        )
        init_offset_awg_path_Q = (
            sequencer_cfg.sequencer_options.init_offset_awg_path_Q
            if sequencer_cfg.sequencer_options is not None
            else 0.0
        )
        init_gain_awg_path_I = (
            sequencer_cfg.sequencer_options.init_gain_awg_path_I
            if sequencer_cfg.sequencer_options is not None
            else 1.0
        )
        init_gain_awg_path_Q = (
            sequencer_cfg.sequencer_options.init_gain_awg_path_Q
            if sequencer_cfg.sequencer_options is not None
            else 1.0
        )
        mixer_phase_error = (
            sequencer_cfg.mixer_corrections.phase_error
            if sequencer_cfg.mixer_corrections is not None
            else None
        )
        mixer_amp_ratio = (
            sequencer_cfg.mixer_corrections.amp_ratio
            if sequencer_cfg.mixer_corrections is not None
            else None
        )
        auto_sideband_cal = (
            sequencer_cfg.mixer_corrections.auto_sideband_cal
            if sequencer_cfg.mixer_corrections is not None
            else SidebandCalEnum.OFF
        )
        ttl_acq_threshold = (
            sequencer_cfg.sequencer_options.ttl_acq_threshold
            if sequencer_cfg.sequencer_options is not None
            else None
        )

        allow_off_grid_nco_ops = sequencer_cfg.allow_off_grid_nco_ops

        return cls(
            nco_en=nco_en,
            sync_en=True,
            channel_name=sequencer_cfg.channel_name,
            channel_name_measure=sequencer_cfg.channel_name_measure,
            connected_output_indices=connected_output_indices,
            connected_input_indices=connected_input_indices,
            init_offset_awg_path_I=init_offset_awg_path_I,
            init_offset_awg_path_Q=init_offset_awg_path_Q,
            init_gain_awg_path_I=init_gain_awg_path_I,
            init_gain_awg_path_Q=init_gain_awg_path_Q,
            modulation_freq=modulation_freq,
            mixer_corr_phase_offset_degree=mixer_phase_error,
            mixer_corr_gain_ratio=mixer_amp_ratio,
            ttl_acq_threshold=ttl_acq_threshold,
            auto_sideband_cal=auto_sideband_cal,
            allow_off_grid_nco_ops=allow_off_grid_nco_ops,
        )


@dataclass
class TimetagSequencerSettings(SequencerSettings):
    """
    Sequencer level settings.

    In the Qblox driver these settings are typically recognized by parameter names of
    the form ``"{module}.sequencer{index}.{setting}"`` (for allowed values see
    `Cluster QCoDeS parameters
    <https://docs.qblox.com/en/master/api_reference/sequencer.html#cluster-qcodes-parameters>`__).
    These settings are set once and will remain unchanged after, meaning that these
    correspond to the "slow" QCoDeS parameters and not settings that are changed
    dynamically by the sequencer.

    These settings are mostly defined in the hardware configuration under each
    port-clock key combination or in some cases through the device configuration
    (e.g. parameters related to thresholded acquisition).
    """

    digitization_thresholds: Optional[DigitizationThresholds] = None
    """The settings that determine when an analog voltage is counted as a pulse."""
    time_source: Optional[TimeSource] = None
    """Selects the timetag data source for timetag acquisitions."""
    time_ref: Optional[TimeRef] = None
    """Selects the time reference that the timetag is recorded in relation to."""
    scope_trace_type: Optional[TimetagTraceType] = None
    """Set to True if the program on this sequencer contains a scope/trace acquisition."""
    trace_acq_duration: Optional[int] = None
    """Duration of the trace acquisition (if any) done with this sequencer."""

    def __post_init__(self) -> None:
        self._validate_io_indices_no_channel_map()

    def _validate_io_indices_no_channel_map(self) -> None:
        """
        There is no channel map in the QTM yet, so there can be only one connected
        index: either input or output.
        """
        if len(self.connected_input_indices) > 1 or len(self.connected_output_indices) > 1:
            raise ValueError(
                "Too many connected inputs or outputs for a QTM sequencer. "
                f"{self.connected_input_indices=}, {self.connected_output_indices=}."
            )

        if len(self.connected_output_indices) == 1 and len(self.connected_input_indices) == 1:
            raise ValueError(
                "A QTM sequencer cannot be connected to both an output and an input " "port."
            )

    @classmethod
    def initialize_from_compilation_config(
        cls,
        sequencer_cfg: _SequencerCompilationConfig,
        connected_output_indices: tuple[int, ...],
        connected_input_indices: tuple[int, ...],
    ) -> TimetagSequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer compilation config.

        Parameters
        ----------
        sequencer_cfg
            The sequencer compilation config.
        connected_output_indices
            Specifies the indices of the outputs this sequencer produces waveforms for.
        connected_input_indices
            Specifies the indices of the inputs this sequencer collects data for.

        Returns
        -------
        : SequencerSettings
            A SequencerSettings instance with initial values.

        """
        return cls(
            sync_en=True,
            channel_name=sequencer_cfg.channel_name,
            channel_name_measure=sequencer_cfg.channel_name_measure,
            connected_output_indices=connected_output_indices,
            connected_input_indices=connected_input_indices,
            digitization_thresholds=sequencer_cfg.digitization_thresholds,
        )


class QbloxBaseDescription(HardwareDescription):
    """Base class for a Qblox hardware description."""

    ref: Union[Literal["internal"], Literal["external"]]
    """The reference source for the instrument."""
    sequence_to_file: bool = False
    """Write sequencer programs to files for (all modules in this) instrument."""


class ComplexChannelDescription(DataStructure):
    """
    Information needed to specify an complex input/output in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    marker_debug_mode_enable: bool = False
    """
    Setting to send 4 ns trigger pulse on the marker
    located next to the I/O port along with each operation.
    The marker will be pulled high at the same time as the module starts playing or acquiring.
    """
    mix_lo: bool = True
    """Whether IQ mixing with a local oscillator is enabled for this channel.
    Effectively always ``True`` for RF modules."""
    downconverter_freq: Optional[float] = None
    """
    Downconverter frequency that should be taken into account w
    hen determining the modulation frequencies for this channel.
    Only relevant for users with custom Qblox downconverter hardware.
    """
    distortion_correction_latency_compensation: int = DistortionCorrectionLatencyEnum.NO_DELAY_COMP
    """
    Delay compensation setting that either
    delays the signal by the amount chosen by the settings or not.
    """


class RealChannelDescription(DataStructure):
    """
    Information needed to specify a real input/output in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    marker_debug_mode_enable: bool = False
    """
    Setting to send 4 ns trigger pulse on the marker located
    next to the I/O port along with each operation.
    The marker will be pulled high at the same time as the module starts playing or acquiring.
    """
    mix_lo: bool = True
    """Whether IQ mixing with a local oscillator is enabled for this channel.
    Effectively always ``True`` for RF modules."""
    distortion_correction_latency_compensation: int = DistortionCorrectionLatencyEnum.NO_DELAY_COMP
    """
    Delay compensation setting that either
    delays the signal by the amount chosen by the settings or not.
    """


class DigitalChannelDescription(DataStructure):
    """
    Information needed to specify a digital (marker) output
    (for :class:`~.quantify_scheduler.operations.pulse_library.MarkerPulse`) in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    distortion_correction_latency_compensation: int = DistortionCorrectionLatencyEnum.NO_DELAY_COMP
    """
    Delay compensation setting that either
    delays the signal by the amount chosen by the settings or not.
    """


class DescriptionAnnotationsGettersMixin:
    """Provide the functionality of retrieving valid channel names by inheriting this class."""

    @classmethod
    def get_valid_channels(cls) -> List[str]:
        """Return all the valid channel names for this hardware description."""
        channel_description_types = [
            ComplexChannelDescription.__name__,
            RealChannelDescription.__name__,
            DigitalChannelDescription.__name__,
        ]

        channel_names = []
        for description_name, description_type in cls.__annotations__.items():
            for channel_description_type in channel_description_types:
                if channel_description_type in description_type:
                    channel_names.append(description_name)
                    break

        return channel_names

    @classmethod
    def get_instrument_type(cls) -> str:
        """Return the instrument type indicated in this hardware description."""
        return get_args(cls.model_fields["instrument_type"].annotation)[0]  # type: ignore

    @classmethod
    def validate_channel_names(cls, channel_names: Iterable[str]) -> None:
        """Validate channel names specified in the Connectivity."""
        valid_names = cls.get_valid_channels()
        for name in channel_names:
            if name not in valid_names:
                raise ValueError(
                    "Invalid channel name specified for module of type "
                    f"{cls.get_instrument_type()}: {name}"
                )


class QRMDescription(DataStructure, DescriptionAnnotationsGettersMixin):
    """
    Information needed to specify a QRM in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    instrument_type: Literal["QRM"]
    """The instrument type of this module."""
    sequence_to_file: bool = False
    """Write sequencer programs to files, for this module."""
    complex_output_0: Optional[ComplexChannelDescription] = None
    """Description of the complex output channel on this QRM, corresponding to ports O1 and O2."""
    complex_input_0: Optional[ComplexChannelDescription] = None
    """Description of the complex input channel on this QRM, corresponding to ports I1 and I2."""
    real_output_0: Optional[RealChannelDescription] = None
    """Description of the real output channel on this QRM, corresponding to port O1."""
    real_output_1: Optional[RealChannelDescription] = None
    """Description of the real output channel on this QRM, corresponding to port O2."""
    real_input_0: Optional[RealChannelDescription] = None
    """Description of the real input channel on this QRM, corresponding to port I1."""
    real_input_1: Optional[RealChannelDescription] = None
    """Description of the real output channel on this QRM, corresponding to port I2."""
    digital_output_0: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M1."""
    digital_output_1: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M2."""
    digital_output_2: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M3."""
    digital_output_3: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M4."""


class QCMDescription(DataStructure, DescriptionAnnotationsGettersMixin):
    """
    Information needed to specify a QCM in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    instrument_type: Literal["QCM"]
    """The instrument type of this module."""
    sequence_to_file: bool = False
    """Write sequencer programs to files, for this module."""
    complex_output_0: Optional[ComplexChannelDescription] = None
    """Description of the complex output channel on this QRM, corresponding to ports O1 and O2."""
    complex_output_1: Optional[ComplexChannelDescription] = None
    """Description of the complex output channel on this QRM, corresponding to ports O3 and O4."""
    real_output_0: Optional[RealChannelDescription] = None
    """Description of the real output channel on this QRM, corresponding to port O1."""
    real_output_1: Optional[RealChannelDescription] = None
    """Description of the real output channel on this QRM, corresponding to port O2."""
    real_output_2: Optional[RealChannelDescription] = None
    """Description of the real output channel on this QRM, corresponding to port O3."""
    real_output_3: Optional[RealChannelDescription] = None
    """Description of the real output channel on this QRM, corresponding to port O4."""
    digital_output_0: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M1."""
    digital_output_1: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M2."""
    digital_output_2: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M3."""
    digital_output_3: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M4."""


class QRMRFDescription(DataStructure, DescriptionAnnotationsGettersMixin):
    """
    Information needed to specify a QRM-RF in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    instrument_type: Literal["QRM_RF"]
    """The instrument type of this module."""
    sequence_to_file: bool = False
    """Write sequencer programs to files, for this module."""
    complex_output_0: Optional[ComplexChannelDescription] = None
    """Description of the complex output channel on this QRM, corresponding to port O1."""
    complex_input_0: Optional[ComplexChannelDescription] = None
    """Description of the complex input channel on this QRM, corresponding to port I1."""
    digital_output_0: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M1."""
    digital_output_1: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M2."""


class QCMRFDescription(DataStructure, DescriptionAnnotationsGettersMixin):
    """
    Information needed to specify a QCM-RF in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    instrument_type: Literal["QCM_RF"]
    """The instrument type of this module."""
    sequence_to_file: bool = False
    """Write sequencer programs to files, for this module."""
    complex_output_0: Optional[ComplexChannelDescription] = None
    """Description of the complex output channel on this QRM, corresponding to port O1."""
    complex_output_1: Optional[ComplexChannelDescription] = None
    """Description of the complex output channel on this QRM, corresponding to port O2."""
    digital_output_0: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M1."""
    digital_output_1: Optional[DigitalChannelDescription] = None
    """Description of the digital (marker) output channel on this QRM, corresponding to port M2."""


class QTMDescription(DataStructure, DescriptionAnnotationsGettersMixin):
    """
    Information needed to specify a QTM in the
    :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`.
    """

    instrument_type: Literal["QTM"]
    """The instrument type of this module."""
    sequence_to_file: bool = False
    """Write sequencer programs to files, for this module."""
    digital_input_0: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 1, specified as input."""
    digital_input_1: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 2, specified as input."""
    digital_input_2: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 3, specified as input."""
    digital_input_3: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 4, specified as input."""
    digital_input_4: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 5, specified as input."""
    digital_input_5: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 6, specified as input."""
    digital_input_6: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 7, specified as input."""
    digital_input_7: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 8, specified as input."""
    digital_output_0: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 1, specified as output."""
    digital_output_1: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 2, specified as output."""
    digital_output_2: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 3, specified as output."""
    digital_output_3: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 4, specified as output."""
    digital_output_4: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 5, specified as output."""
    digital_output_5: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 6, specified as output."""
    digital_output_6: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 7, specified as output."""
    digital_output_7: Optional[DigitalChannelDescription] = None
    """Description of the digital channel corresponding to port 8, specified as output."""

    @classmethod
    def validate_channel_names(cls, channel_names: Iterable[str]) -> None:
        """Validate channel names specified in the Connectivity."""
        super().validate_channel_names(channel_names)

        used_inputs = set(int(n.lstrip("digital_input_")) for n in channel_names if "input" in n)
        used_outputs = set(int(n.lstrip("digital_output_")) for n in channel_names if "output" in n)

        if overlap := used_inputs & used_outputs:
            raise ValueError(
                "The configuration for the QTM module contains channel names with port "
                "numbers that are assigned as both input and output. This is not "
                "allowed. Conflicting channel names:\n"
                + "\n".join(f"digital_input_{n}\ndigital_output_{n}" for n in overlap)
            )


ClusterModuleDescription = Annotated[
    Union[
        QRMDescription,
        QCMDescription,
        QRMRFDescription,
        QCMRFDescription,
        QTMDescription,
    ],
    Field(discriminator="instrument_type"),
]
"""
Specifies a Cluster module and its instrument-specific settings.

The supported instrument types are:
:class:`~.QRMDescription`,
:class:`~.QCMDescription`,
:class:`~.QRMRFDescription`,
:class:`~.QCMRFDescription`,
:class:`~.QTMDescription`,
"""


class ClusterDescription(QbloxBaseDescription):
    """Information needed to specify a Cluster in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["Cluster"]  # type: ignore  # (valid override)
    """The instrument type, used to select this datastructure
    when parsing a :class:`~.CompilationConfig`."""
    modules: Dict[int, ClusterModuleDescription] = {}
    """Description of the modules of this Cluster, using slot index as key."""
    ip: Optional[str] = None
    """Unique identifier (typically the ip address) used to connect to the cluster"""


QbloxHardwareDescription = Annotated[
    Union[
        ClusterDescription,
        LocalOscillatorDescription,
        IQMixerDescription,
        OpticalModulatorDescription,
    ],
    Field(discriminator="instrument_type"),
]
"""
Specifies a piece of Qblox hardware and its instrument-specific settings.
"""


RealInputGain = int
"""
Input gain settings for a real input connected to a port-clock combination.

This gain value will be set on the QRM input ports
that are connected to this port-clock combination.

.. admonition:: Example
    :class: dropdown

    .. code-block:: python

        hardware_compilation_config.hardware_options.input_gain = {
            "q0:res-q0.ro": RealInputGain(2),
        }
"""


class ComplexInputGain(DataStructure):
    """
    Input gain settings for a complex input connected to a port-clock combination.

    This gain value will be set on the QRM input ports
    that are connected to this port-clock combination.

    .. admonition:: Example
        :class: dropdown

        .. code-block:: python

            hardware_compilation_config.hardware_options.input_gain = {
                "q0:res-q0.ro": ComplexInputGain(
                    gain_I=2,
                    gain_Q=3
                ),
            }
    """

    gain_I: int  # noqa: N815, capital I allowed here
    """Gain setting on the input receiving the I-component data for this port-clock combination."""
    gain_Q: int  # noqa: N815, capital Q allowed here
    """Gain setting on the input receiving the Q-component data for this port-clock combination."""


OutputAttenuation = int
"""
Output attenuation setting for a port-clock combination.

This attenuation value will be set on each control-hardware output
port that is connected to this port-clock combination.

.. admonition:: Example
    :class: dropdown

    .. code-block:: python

        hardware_compilation_config.hardware_options.output_att = {
            "q0:res-q0.ro": OutputAttenuation(10),
        }
"""


InputAttenuation = int
"""
Input attenuation setting for a port-clock combination.

This attenuation value will be set on each control-hardware output
port that is connected to this port-clock combination.

.. admonition:: Example
    :class: dropdown

    .. code-block:: python

        hardware_compilation_config.hardware_options.input_att = {
            "q0:res-q0.ro": InputAttenuation(10),
        }
"""


class QbloxMixerCorrections(MixerCorrections):
    """
    Mixer correction settings with defaults set to None, and extra mixer correction
    settings for _automated_ mixer correction.

    These settings will be set on each control-hardware output
    port that is connected to this port-clock combination.

    .. admonition:: Example
        :class: dropdown

        .. code-block:: python

            hardware_compilation_config.hardware_options.mixer_corrections = {
                "q0:res-q0.ro": {
                    auto_lo_cal="on_lo_interm_freq_change",
                    auto_sideband_cal="on_interm_freq_change"
                },
            }
    """

    dc_offset_i: Optional[float] = None  # type: ignore  # (optional due to AMC)
    """The DC offset on the I channel used for this port-clock combination."""
    dc_offset_q: Optional[float] = None  # type: ignore  # (optional due to AMC)
    """The DC offset on the Q channel used for this port-clock combination."""
    amp_ratio: float = Field(
        default=DEFAULT_MIXER_AMP_RATIO, ge=MIN_MIXER_AMP_RATIO, le=MAX_MIXER_AMP_RATIO
    )
    """The mixer gain ratio used for this port-clock combination."""
    phase_error: float = Field(
        default=DEFAULT_MIXER_PHASE_ERROR_DEG,
        ge=MIN_MIXER_PHASE_ERROR_DEG,
        le=MAX_MIXER_PHASE_ERROR_DEG,
    )
    """The mixer phase error used for this port-clock combination."""
    auto_lo_cal: LoCalEnum = LoCalEnum.OFF
    """
    Setting that controls whether the mixer is calibrated upon changing the LO and/or
    intermodulation frequency.
    """

    auto_sideband_cal: SidebandCalEnum = SidebandCalEnum.OFF
    """
    Setting that controls whether the mixer is calibrated upon changing the
    intermodulation frequency.
    """

    @model_validator(mode="before")
    @classmethod
    def warn_if_mixed_auto_and_manual_calibration(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Warn if there is mixed usage of automatic mixer calibration (the auto_*
        settings) and manual mixer correction settings.
        """
        # This is a "before" mode pydantic validator because we use
        # validate_assignment=True, which means an "after" mode validator would fall
        # into an infinite recursion loop.
        if data.get("auto_lo_cal", LoCalEnum.OFF) != LoCalEnum.OFF and not (
            data.get("dc_offset_i") is None and data.get("dc_offset_q") is None
        ):
            warnings.warn(
                f"Setting `auto_lo_cal={data['auto_lo_cal']}` will overwrite settings "
                f"`dc_offset_i={data.get('dc_offset_i')}` and "
                f"`dc_offset_q={data.get('dc_offset_q')}`. To suppress this warning, do not "
                "set either `dc_offset_i` or `dc_offset_q` for this port-clock.",
                ValidationWarning,
            )
            data["dc_offset_i"] = None
            data["dc_offset_q"] = None

        if data.get("auto_sideband_cal", SidebandCalEnum.OFF) != SidebandCalEnum.OFF and not (
            data.get("amp_ratio") is None and data.get("phase_error") is None
        ):
            warnings.warn(
                f"Setting `auto_sideband_cal={data['auto_sideband_cal']}` will "
                f"overwrite settings `amp_ratio={data.get('amp_ratio')}` and "
                f"`phase_error={data.get('phase_error')}`. To suppress this warning, do not "
                "set either `amp_ratio` or `phase_error` for this port-clock.",
                ValidationWarning,
            )
            data["amp_ratio"] = DEFAULT_MIXER_AMP_RATIO
            data["phase_error"] = DEFAULT_MIXER_PHASE_ERROR_DEG

        return data


class SequencerOptions(DataStructure):
    """
    Configuration options for a sequencer.

    For allowed values, also see `Cluster QCoDeS parameters
    <https://docs.qblox.com/en/main/api_reference/sequencer.html#cluster-qcodes-parameters>`__.

    .. admonition:: Example
        :class: dropdown

        .. code-block:: python

            hardware_compilation_config.hardware_options.sequencer_options = {
                "q0:res-q0.ro": {
                    "init_offset_awg_path_I": 0.1,
                    "init_offset_awg_path_Q": -0.1,
                    "init_gain_awg_path_I": 0.9,
                    "init_gain_awg_path_Q": 1.0,
                    "ttl_acq_threshold": 0.5
                    "qasm_hook_func": foo
                }
            }
    """

    init_offset_awg_path_I: float = Field(default=0.0, ge=-1.0, le=1.0)
    """Specifies what value the sequencer offset for AWG path_I will be reset to
    before the start of the experiment."""
    init_offset_awg_path_Q: float = Field(default=0.0, ge=-1.0, le=1.0)
    """Specifies what value the sequencer offset for AWG path_Q will be reset to
    before the start of the experiment."""
    init_gain_awg_path_I: float = Field(default=1.0, ge=-1.0, le=1.0)
    """Specifies what value the sequencer gain for AWG path_I will be reset to
    before the start of the experiment."""
    init_gain_awg_path_Q: float = Field(default=1.0, ge=-1.0, le=1.0)
    """Specifies what value the sequencer gain for AWG path_Q will be reset to
    before the start of the experiment."""
    ttl_acq_threshold: Optional[float] = None
    """
    For QRM modules only, the threshold value with which to compare the input ADC values
    of the selected input path.
    """
    qasm_hook_func: Optional[Callable] = None
    """
    Function to inject custom qasm instructions after the compiler inserts the
    footer and the stop instruction in the generated qasm program.
    """

    @field_validator(
        "init_offset_awg_path_I",
        "init_offset_awg_path_Q",
        "init_gain_awg_path_I",
        "init_gain_awg_path_Q",
    )
    def _init_setting_limits(cls, init_setting):  # noqa: N805
        # if connectivity contains a hardware config with latency corrections
        if init_setting < -1.0 or init_setting > 1.0:
            raise ValueError(
                f"Trying to set init gain/awg setting to {init_setting} "
                f"in the SequencerOptions. Must be between -1.0 and 1.0."
            )
        return init_setting


class QbloxHardwareDistortionCorrection(HardwareDistortionCorrection):
    """A hardware distortion correction specific to the Qblox backend."""

    bt_coeffs: Optional[List[float]] = None
    """Coefficient of the bias tee correction."""
    exp0_coeffs: Optional[List[float]] = None
    """Coefficients of the exponential overshoot/undershoot correction 1."""
    exp1_coeffs: Optional[List[float]] = None
    """Coefficients of the exponential overshoot/undershoot correction 2."""
    exp2_coeffs: Optional[List[float]] = None
    """Coefficients of the exponential overshoot/undershoot correction 3."""
    exp3_coeffs: Optional[List[float]] = None
    """Coefficients of the exponential overshoot/undershoot correction 4."""
    fir_coeffs: Optional[List[float]] = None
    """Coefficients for the FIR filter."""


class DigitizationThresholds(DataStructure):
    """The settings that determine when an analog voltage is counted as a pulse."""

    in_threshold_primary: Optional[float] = None
    """
    For QTM modules only, this is the voltage threshold above which an input signal is
    registered as high.
    """


class QbloxHardwareOptions(HardwareOptions):
    """
    Datastructure containing the hardware options for each port-clock combination.

    .. admonition:: Example
        :class: dropdown

        Here, the HardwareOptions datastructure is created by parsing a
        dictionary containing the relevant information.

        .. jupyter-execute::

            import pprint
            from quantify_scheduler.schemas.examples.utils import (
                load_json_example_scheme
            )

        .. jupyter-execute::

            from quantify_scheduler.backends.types.qblox import (
                QbloxHardwareOptions
            )
            qblox_hw_options_dict = load_json_example_scheme(
                "qblox_hardware_config_transmon.json")["hardware_options"]
            pprint.pprint(qblox_hw_options_dict)

        The dictionary can be parsed using the :code:`model_validate` method.

        .. jupyter-execute::

            qblox_hw_options = QbloxHardwareOptions.model_validate(qblox_hw_options_dict)
            qblox_hw_options
    """

    input_gain: Optional[Dict[str, Union[RealInputGain, ComplexInputGain]]] = None
    """
    Dictionary containing the input gain settings (values) that should be applied
    to the inputs that are connected to a certain port-clock combination (keys).
    """
    output_att: Optional[Dict[str, OutputAttenuation]] = None
    """
    Dictionary containing the attenuation settings (values) that should be applied
    to the outputs that are connected to a certain port-clock combination (keys).
    """
    input_att: Optional[Dict[str, InputAttenuation]] = None
    """
    Dictionary containing the attenuation settings (values) that should be applied
    to the inputs that are connected to a certain port-clock combination (keys).
    """
    mixer_corrections: Optional[Dict[str, QbloxMixerCorrections]] = None  # type: ignore
    """
    Dictionary containing the qblox-specific mixer corrections (values) that should be
    used for signals on a certain port-clock combination (keys).
    """
    sequencer_options: Optional[Dict[str, SequencerOptions]] = None
    """
    Dictionary containing the options (values) that should be set
    on the sequencer that is used for a certain port-clock combination (keys).
    """
    distortion_corrections: Optional[  # type: ignore
        Dict[
            str,
            Union[
                SoftwareDistortionCorrection,
                QbloxHardwareDistortionCorrection,
                List[QbloxHardwareDistortionCorrection],
            ],
        ]
    ] = None
    digitization_thresholds: Optional[Dict[str, DigitizationThresholds]] = None
    """
    Dictionary containing the digitization threshold settings for QTM modules. These are
    the settings that determine the voltage thresholds above which input signals are
    registered as high.
    """
