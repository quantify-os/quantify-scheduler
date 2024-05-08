# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Python dataclasses for compilation to Qblox hardware."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from typing import (
    Any,
    Callable,
    Dict,
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
from typing_extensions import Annotated

from quantify_scheduler.backends.qblox import constants, q1asm_instructions
from quantify_scheduler.backends.qblox.enums import (
    DistortionCorrectionLatencyEnum,
    QbloxFilterConfig,
    QbloxFilterMarkerDelay,
)
from quantify_scheduler.backends.types.common import (
    Connectivity,
    HardwareDescription,
    HardwareDistortionCorrection,
    HardwareOptions,
    IQMixerDescription,
    LocalOscillatorDescription,
    SoftwareDistortionCorrection,
)
from quantify_scheduler.structure.model import DataStructure


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
    channel_name_to_connected_io_indices: Dict[str, tuple[int, ...]]
    """Specifies the connected io indices per channel_name identifier."""

    def _get_connected_output_indices(self, channel_name) -> tuple[int, ...]:
        """
        Return the connected output indices associated with the output name
        specified in the hardware config.
        """
        return (
            self.channel_name_to_connected_io_indices[channel_name]
            if "output" in channel_name
            else ()
        )

    def _get_connected_input_indices(self, channel_name) -> tuple[int, ...]:
        """
        Return the connected input indices associated with the input name
        specified in the hardware config.
        """
        return (
            self.channel_name_to_connected_io_indices[channel_name]
            if "input" in channel_name
            else ()
        )


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
    channel_name_to_digital_marker: Dict[str, int] = dataclasses_field(
        default_factory=dict
    )
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
            self.is_acquisition
            or self.is_parameter_update
            or self.data.get("wf_func") is not None
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
        https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/cluster/q1_sequence_processor.html#q1-instructions
        for the full list of these instructions.
        """
        return (
            self.is_offset_instruction
            or "phase_shift" in self.data
            or "reset_clock_phase" in self.data
            or "clock_freq_new" in self.data
            or "marker_pulse" in self.data
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
    def is_return_stack(self) -> bool:
        """
        Return ``True`` if the operation is a return stack, corresponding to the Q1ASM
        instruction ``return_stack``.
        """
        return self.data.get("return_stack", None) is not None

    def __str__(self):
        type_label: str = "Acquisition" if self.is_acquisition else "Pulse"
        return (
            f'{type_label} "{self.name}" (t0={self.timing}, duration={self.duration})'
        )

    def __repr__(self):
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

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> LOSettings:
        """
        Factory method for the LOSettings from a mapping dict. The required format is
        {"frequency": {parameter_name: value}, "power": {parameter_name: value}}. For
        convenience {"frequency": value, "power": value} is also allowed.

        Parameters
        ----------
        mapping
            The part of the mapping dict relevant for this instrument.

        Returns
        -------
        :
            Instantiated LOSettings from the mapping dict.
        """
        if "power" not in mapping:
            raise KeyError(
                "Attempting to compile settings for a local oscillator but 'power' is "
                "missing from settings. 'power' is required as an entry for Local "
                "Oscillators."
            )
        if "generic_icc_name" in mapping:
            generic_icc_name = mapping["generic_icc_name"]
            if generic_icc_name != constants.GENERIC_IC_COMPONENT_NAME:
                raise NotImplementedError(
                    f"Specified name '{generic_icc_name}' as a generic instrument "
                    f"coordinator component, but the Qblox backend currently only "
                    f"supports using the default name "
                    f"'{constants.GENERIC_IC_COMPONENT_NAME}'"
                )

        power_entry: Union[float, Dict[str, float]] = mapping["power"]
        if not isinstance(power_entry, dict):  # floats allowed for convenience
            power_entry = {"power": power_entry}
        freq_entry: float | None | Dict[str, float | None] = mapping.get("frequency")
        if not isinstance(freq_entry, dict):
            freq_entry = {"frequency": freq_entry}

        return cls(power=power_entry, frequency=freq_entry)


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
        cls: type[_ModuleSettingsT], mapping: Dict[str, Any], **kwargs: Optional[dict]
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
        del mapping  # not used
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
        cls, mapping: Dict[str, Any], **kwargs: Optional[dict]
    ) -> RFModuleSettings:
        """
        Factory method that takes all the settings defined in the mapping and generates
        an :class:`~.RFModuleSettings` object from it.

        Parameters
        ----------
        mapping
            The mapping dict to extract the settings from
        **kwargs
            Additional keyword arguments passed to the constructor. Can be used to
            override parts of the mapping dict.
        """
        rf_settings = {}

        complex_output_0 = mapping.get("complex_output_0")
        complex_output_1 = mapping.get("complex_output_1")
        if complex_output_0:
            rf_settings["lo0_freq"] = complex_output_0.get("lo_freq")
        if complex_output_1:
            rf_settings["lo1_freq"] = complex_output_1.get("lo_freq")

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
    <https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/sequencer.html#cluster-qcodes-parameters>`__).
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
    If you want to set a trigger when the acquisition result is 1, the parameter must be set to false 
    and vice versa.
    """

    @classmethod
    def initialize_from_config_dict(
        cls,
        sequencer_cfg: Dict[str, Any],  # noqa: ARG003 ignore unused argument
        channel_name: str,
        connected_output_indices: tuple[int, ...],
        connected_input_indices: tuple[int, ...],
    ) -> SequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer configuration dictionary.

        Parameters
        ----------
        sequencer_cfg : dict
            The sequencer configuration dict.
        channel_name
            Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).
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
            channel_name=channel_name,
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
    <https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/sequencer.html#cluster-qcodes-parameters>`__).
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
    mixer_corr_phase_offset_degree: float = 0.0
    """The phase shift to apply between the I and Q channels, to correct for quadrature
    errors."""
    mixer_corr_gain_ratio: float = 1.0
    """The gain ratio to apply in order to correct for imbalances between the I and Q
    paths of the mixer."""
    integration_length_acq: Optional[int] = None
    """Integration length for acquisitions. Must be a multiple of 4 ns."""
    thresholded_acq_threshold: Optional[float] = None
    """The sequencer discretization threshold for discretizing the phase rotation result."""
    thresholded_acq_rotation: Optional[float] = None
    """The sequencer integration result phase rotation in degrees."""
    ttl_acq_input_select: Optional[int] = None
    """Selects the input used to compare against the threshold value in the TTL trigger acquisition path."""
    ttl_acq_threshold: Optional[float] = None
    """Sets the threshold value with which to compare the input ADC values of the selected input path."""
    ttl_acq_auto_bin_incr_en: Optional[bool] = None
    """Selects if the bin index is automatically incremented when acquiring multiple triggers."""

    @classmethod
    def initialize_from_config_dict(
        cls,
        sequencer_cfg: Dict[str, Any],
        channel_name: str,
        connected_output_indices: tuple[int, ...],
        connected_input_indices: tuple[int, ...],
    ) -> AnalogSequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer configuration dictionary.

        Parameters
        ----------
        sequencer_cfg : dict
            The sequencer configuration dict.
        channel_name
            Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).
        connected_output_indices
            Specifies the indices of the outputs this sequencer produces waveforms for.
        connected_input_indices
            Specifies the indices of the inputs this sequencer collects data for.

        Returns
        -------
        : AnalogSequencerSettings
            A AnalogSequencerSettings instance with initial values.
        """
        T = TypeVar("T", int, float)

        def extract_and_verify_range(
            param_name: str,
            settings: Dict[str, Any],
            default_value: T | None,
            min_value: T,
            max_value: T,
        ) -> T:
            val = settings.get(param_name, default_value)
            if val is None:
                return val
            elif val < min_value or val > max_value:
                raise ValueError(
                    f"Attempting to configure {param_name} to {val} for the sequencer "
                    f"specified with port {settings.get('port', '[port invalid!]')} and"
                    f" clock {settings.get('clock', '[clock invalid!]')}, while the "
                    f"hardware requires it to be between {min_value} and {max_value}."
                )
            return val

        modulation_freq: Optional[float] = sequencer_cfg.get("interm_freq")
        nco_en: bool = (
            modulation_freq is not None and modulation_freq != 0
        )  # Allow NCO to be permanently disabled via `"interm_freq": 0` in the hardware config

        init_offset_awg_path_I = extract_and_verify_range(
            param_name="init_offset_awg_path_I",
            settings=sequencer_cfg,
            default_value=cls.init_offset_awg_path_I,
            min_value=-1.0,
            max_value=1.0,
        )

        init_offset_awg_path_Q = extract_and_verify_range(
            param_name="init_offset_awg_path_Q",
            settings=sequencer_cfg,
            default_value=cls.init_offset_awg_path_Q,
            min_value=-1.0,
            max_value=1.0,
        )

        init_gain_awg_path_I = extract_and_verify_range(
            param_name="init_gain_awg_path_I",
            settings=sequencer_cfg,
            default_value=cls.init_gain_awg_path_I,
            min_value=-1.0,
            max_value=1.0,
        )

        init_gain_awg_path_Q = extract_and_verify_range(
            param_name="init_gain_awg_path_Q",
            settings=sequencer_cfg,
            default_value=cls.init_gain_awg_path_Q,
            min_value=-1.0,
            max_value=1.0,
        )

        mixer_phase_error = extract_and_verify_range(
            param_name="mixer_phase_error_deg",
            settings=sequencer_cfg,
            default_value=0.0,
            min_value=constants.MIN_MIXER_PHASE_ERROR_DEG,
            max_value=constants.MAX_MIXER_PHASE_ERROR_DEG,
        )

        mixer_amp_ratio = extract_and_verify_range(
            param_name="mixer_amp_ratio",
            settings=sequencer_cfg,
            default_value=1.0,
            min_value=constants.MIN_MIXER_AMP_RATIO,
            max_value=constants.MAX_MIXER_AMP_RATIO,
        )

        thresholded_acq_threshold = extract_and_verify_range(
            param_name="thresholded_acq_threshold",
            settings=sequencer_cfg,
            default_value=cls.thresholded_acq_threshold,
            min_value=constants.MIN_DISCRETIZATION_THRESHOLD_ACQ,
            max_value=constants.MAX_DISCRETIZATION_THRESHOLD_ACQ,
        )

        thresholded_acq_rotation = extract_and_verify_range(
            param_name="thresholded_acq_rotation",
            settings=sequencer_cfg,
            default_value=cls.thresholded_acq_rotation,
            min_value=constants.MIN_PHASE_ROTATION_ACQ,
            max_value=constants.MAX_PHASE_ROTATION_ACQ,
        )

        ttl_acq_threshold = sequencer_cfg.get("ttl_acq_threshold")

        return cls(
            nco_en=nco_en,
            sync_en=True,
            channel_name=channel_name,
            connected_output_indices=connected_output_indices,
            connected_input_indices=connected_input_indices,
            init_offset_awg_path_I=init_offset_awg_path_I,
            init_offset_awg_path_Q=init_offset_awg_path_Q,
            init_gain_awg_path_I=init_gain_awg_path_I,
            init_gain_awg_path_Q=init_gain_awg_path_Q,
            modulation_freq=modulation_freq,
            mixer_corr_phase_offset_degree=mixer_phase_error,
            mixer_corr_gain_ratio=mixer_amp_ratio,
            thresholded_acq_rotation=thresholded_acq_rotation,
            thresholded_acq_threshold=thresholded_acq_threshold,
            ttl_acq_threshold=ttl_acq_threshold,
        )


@dataclass
class TimetagSequencerSettings(SequencerSettings):
    """
    Sequencer level settings.

    In the Qblox driver these settings are typically recognized by parameter names of
    the form ``"{module}.sequencer{index}.{setting}"`` (for allowed values see
    `Cluster QCoDeS parameters
    <https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/sequencer.html#cluster-qcodes-parameters>`__).
    These settings are set once and will remain unchanged after, meaning that these
    correspond to the "slow" QCoDeS parameters and not settings that are changed
    dynamically by the sequencer.

    These settings are mostly defined in the hardware configuration under each
    port-clock key combination or in some cases through the device configuration
    (e.g. parameters related to thresholded acquisition).
    """

    @classmethod
    def initialize_from_config_dict(
        cls,
        sequencer_cfg: Dict[str, Any],  # noqa: ARG003 ignore unused argument
        channel_name: str,
        connected_output_indices: tuple[int, ...],
        connected_input_indices: tuple[int, ...],
    ) -> TimetagSequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer configuration dictionary.

        Parameters
        ----------
        sequencer_cfg : dict
            The sequencer configuration dict.
        channel_name
            Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).
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
            channel_name=channel_name,
            connected_output_indices=connected_output_indices,
            connected_input_indices=connected_input_indices,
        )


class QbloxBaseDescription(HardwareDescription):
    """Base class for a Qblox hardware description."""

    ref: Union[Literal["internal"], Literal["external"]]
    """The reference source for the instrument."""
    sequence_to_file: bool = False
    """Write sequencer programs to files for (all modules in this) instrument."""


class ComplexChannelDescription(DataStructure):
    """Information needed to specify an complex input/output in the :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`."""

    marker_debug_mode_enable: bool = False
    """
    Setting to send 4 ns trigger pulse on the marker located next to the I/O port along with each operation.
    The marker will be pulled high at the same time as the module starts playing or acquiring.
    """
    mix_lo: bool = True
    """Whether IQ mixing with a local oscillator is enabled for this channel. Effectively always ``True`` for RF modules."""
    downconverter_freq: Optional[float] = None
    """
    Downconverter frequency that should be taken into account when determining the modulation frequencies for this channel.
    Only relevant for users with custom Qblox downconverter hardware.
    """
    distortion_correction_latency_compensation: int = (
        DistortionCorrectionLatencyEnum.NO_DELAY_COMP
    )
    """
    Delay compensation setting that either delays the signal by the amount chosen by the settings or not.
    """


class RealChannelDescription(DataStructure):
    """Information needed to specify a real input/output in the :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`."""

    marker_debug_mode_enable: bool = False
    """
    Setting to send 4 ns trigger pulse on the marker located next to the I/O port along with each operation.
    The marker will be pulled high at the same time as the module starts playing or acquiring.
    """
    distortion_correction_latency_compensation: int = (
        DistortionCorrectionLatencyEnum.NO_DELAY_COMP
    )
    """
    Delay compensation setting that either delays the signal by the amount chosen by the settings or not.
    """


class DigitalChannelDescription(DataStructure):
    """Information needed to specify a digital (marker) output (for :class:`~.quantify_scheduler.operations.pulse_library.MarkerPulse`) in the :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`."""

    distortion_correction_latency_compensation: int = (
        DistortionCorrectionLatencyEnum.NO_DELAY_COMP
    )
    """
    Delay compensation setting that either delays the signal by the amount chosen by the settings or not.
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
        return get_args(cls.model_fields["instrument_type"].annotation)[0]


class QRMDescription(DataStructure, DescriptionAnnotationsGettersMixin):
    """Information needed to specify a QRM in the :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`."""

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
    """Information needed to specify a QCM in the :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`."""

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
    """Information needed to specify a QRM-RF in the :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`."""

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
    """Information needed to specify a QCM-RF in the :class:`~.quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`."""

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


ClusterModuleDescription = Annotated[
    Union[QRMDescription, QCMDescription, QRMRFDescription, QCMRFDescription],
    Field(discriminator="instrument_type"),
]
"""
Specifies a Cluster module and its instrument-specific settings.

The supported instrument types are:
:class:`~.QRMDescription`,
:class:`~.QCMDescription`,
:class:`~.QRMRFDescription`,
:class:`~.QCMRFDescription`,
"""


class ClusterDescription(QbloxBaseDescription):
    """Information needed to specify a Cluster in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["Cluster"]
    """The instrument type, used to select this datastructure when parsing a :class:`~.CompilationConfig`."""
    modules: Dict[int, ClusterModuleDescription] = {}
    """Description of the modules of this Cluster, using slot index as key."""


QbloxHardwareDescription = Annotated[
    Union[
        ClusterDescription,
        LocalOscillatorDescription,
        IQMixerDescription,
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
    Input gain settings for a real input connected to a port-clock combination.

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

    gain_I: int
    """Gain setting on the input receiving the I-component data for this port-clock combination."""
    gain_Q: int
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


class SequencerOptions(DataStructure):
    """
    Configuration options for a sequencer.

    For allowed values, also see `Cluster QCoDeS parameters
    <https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/sequencer.html#cluster-qcodes-parameters>`__.

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
    ttl_acq_threshold: Optional[float] = None
    """Threshold value with which to compare the input ADC values of the selected input path."""
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
                "qblox_hardware_compilation_config.json")["hardware_options"]
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
    sequencer_options: Optional[Dict[str, SequencerOptions]] = None
    """
    Dictionary containing the options (values) that should be set
    on the sequencer that is used for a certain port-clock combination (keys).
    """
    distortion_corrections: Optional[
        Dict[
            str,
            Union[
                SoftwareDistortionCorrection,
                QbloxHardwareDistortionCorrection,
                List[QbloxHardwareDistortionCorrection],
            ],
        ]
    ] = None


QbloxHardwareOptions.model_rebuild()


class _LocalOscillatorCompilerConfig(DataStructure):
    """Configuration values for a :class:`quantify_scheduler.backends.qblox.instrument_compilers.LocalOscillatorCompiler`."""

    instrument_type: Literal["LocalOscillator"]
    """The type of the instrument described by this config."""
    hardware_description: LocalOscillatorDescription
    """Description of the physical setup of this local oscillator."""
    frequency: Union[float, None] = None
    """The frequency of this local oscillator."""


class _ClusterCompilerConfig(DataStructure):
    """Configuration values for a :class:`~.ClusterCompiler`."""

    instrument_type: Literal["Cluster"]
    """The type of the instrument described by this config."""
    ref: Union[Literal["internal"], Literal["external"]]
    """The reference source for the cluster."""
    sequence_to_file: bool = False
    """Write sequencer programs to files for (all modules in this) cluster."""
    modules: Dict[int, _ClusterModuleCompilerConfig] = {}
    """Compiler configs of the modules of this cluster, using slot index as key."""
    portclock_to_path: Dict[str, str] = {}
    """Mapping between portclocks and their associated channel name paths (e.g. cluster0.module1.complex_output_0)."""


class _ClusterModuleCompilerConfig(DataStructure):
    """Configuration values for a :class:`~.ClusterModuleCompiler`."""

    instrument_type: Union[
        Literal["QCM"], Literal["QRM"], Literal["QCM_RF"], Literal["QRM_RF"]
    ]
    """The type of the instrument described by this config."""
    hardware_description: ClusterModuleDescription
    """Description of the physical setup of this module."""
    hardware_options: QbloxHardwareOptions
    """Options that are used in compiling the instructions for the hardware."""
    connectivity: Connectivity
    """Datastructure representing how ports on the quantum device are connected to ports on the control hardware."""
    portclock_to_path: Dict[str, str] = {}
    """Mapping between portclocks and their associated channel name paths (e.g. cluster0.module1.complex_output_0)."""
    channel_to_lo: Dict[str, str] = {}
    """Mapping between channel names and the name of the local oscillator they are connected to."""
