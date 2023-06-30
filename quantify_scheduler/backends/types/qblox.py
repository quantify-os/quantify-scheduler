# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Python dataclasses for compilation to Qblox hardware."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from typing import Any, Dict, Literal, Optional, Tuple, TypeVar, Union, List

from dataclasses_json import DataClassJsonMixin
from pydantic import Field
from typing_extensions import Annotated

from quantify_scheduler.backends.qblox import constants, q1asm_instructions
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
    """
    Specifies the fixed hardware properties needed in the backend.
    """

    instrument_type: str
    """The type of instrument."""
    max_sequencers: int
    """The amount of sequencers available."""
    max_awg_output_voltage: Optional[float]
    """Maximum output voltage of the awg."""
    mixer_dc_offset_range: BoundedParameter
    """Specifies the range over which the dc offsets can be set that are used for mixer
    calibration."""
    valid_ios: List[str]
    """Specifies the complex/real output identifiers supported by this device."""
    default_marker: int = 0
    """The default marker value to set at the beginning of programs.
    Important for RF instruments that use the marker to enable the RF output."""
    output_map: Dict[str, int] = dataclasses_field(default_factory=dict)
    """A mapping from output name to marker setting.
    Specifies which marker bit needs to be set at start if the
    output (as a string ex. `complex_output_0`) contains a pulse."""


@dataclass(frozen=True)
class OpInfo(DataClassJsonMixin):
    """
    Data structure describing a pulse or acquisition and containing all the information
    required to play it.
    """

    name: str
    """Name of the operation that this pulse/acquisition is part of."""
    data: dict
    """The pulse/acquisition info taken from the `data` property of the
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
        return "acq_index" in self.data

    @property
    def is_real_time_io_operation(self) -> bool:
        """Returns ``True`` if the operation is a non-idle pulse (i.e., it has a
        waveform), ``False`` otherwise.
        """
        return (
            self.is_acquisition
            or self.is_parameter_update
            or self.data.get("wf_func") is not None
        )

    @property
    def is_offset_instruction(self) -> bool:
        """Returns ``True`` if the operation describes a DC offset operation,
        corresponding to the Q1ASM instruction ``set_awg_offset``.
        """
        return "offset_path_0" in self.data or "offset_path_1" in self.data

    @property
    def is_parameter_update(self) -> bool:
        """Return ``True`` if the operation is a parameter update, corresponding to the
        Q1ASM instruction ``upd_param``.
        """
        return self.data.get("instruction", "") == q1asm_instructions.UPDATE_PARAMETERS

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
    """
    Dataclass containing all the settings for a generic LO instrument.
    """

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
        freq_entry: Union[float, Dict[str, Optional[float]]] = mapping["frequency"]
        if not isinstance(freq_entry, dict):
            freq_entry = {"frequency": freq_entry}

        return cls(power=power_entry, frequency=freq_entry)


@dataclass
class BaseModuleSettings(DataClassJsonMixin):
    """Shared settings between all the Qblox modules."""

    offset_ch0_path0: Union[float, None] = None
    """The DC offset on the path 0 of channel 0."""
    offset_ch0_path1: Union[float, None] = None
    """The DC offset on the path 1 of channel 0."""
    offset_ch1_path0: Union[float, None] = None
    """The DC offset on path 0 of channel 1."""
    offset_ch1_path1: Union[float, None] = None
    """The DC offset on path 1 of channel 1."""
    in0_gain: Union[int, None] = None
    """The gain of input 0."""
    in1_gain: Union[int, None] = None
    """The gain of input 1."""


@dataclass
class BasebandModuleSettings(BaseModuleSettings):
    """
    Settings for a baseband module.

    Class exists to ensure that the cluster baseband modules don't need special
    treatment in the rest of the code.
    """

    @classmethod
    def extract_settings_from_mapping(
        cls, mapping: Dict[str, Any], **kwargs: Optional[dict]
    ) -> BasebandModuleSettings:
        """
        Factory method that takes all the settings defined in the mapping and generates
        a :class:`~.BasebandModuleSettings` object from it.

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
class PulsarSettings(BaseModuleSettings):
    """
    Global settings for the Pulsar to be set in the InstrumentCoordinator component.
    This is kept separate from the settings that can be set on a per sequencer basis,
    which are specified in :class:`~.SequencerSettings`.
    """

    ref: str = "internal"
    """The reference source. Should either be ``"internal"`` or ``"external"``, will
    raise an exception in the instrument coordinator component otherwise."""

    @classmethod
    def extract_settings_from_mapping(
        cls, mapping: Dict[str, Any], **kwargs: Optional[dict]
    ) -> PulsarSettings:
        """
        Factory method that takes all the settings defined in the mapping and generates
        a :class:`~.PulsarSettings` object from it.

        Parameters
        ----------
        mapping
            The mapping dict to extract the settings from
        **kwargs
            Additional keyword arguments passed to the constructor. Can be used to
            override parts of the mapping dict.
        """
        ref: str = mapping["ref"]
        if ref != "internal" and ref != "external":
            raise ValueError(
                f"Attempting to configure ref to {ref}. "
                f"The only allowed values are 'internal' and 'external'."
            )
        return cls(ref=ref, **kwargs)


@dataclass
class RFModuleSettings(BaseModuleSettings):
    """
    Global settings for the module to be set in the InstrumentCoordinator component.
    This is kept separate from the settings that can be set on a per sequencer basis,
    which are specified in :class:`~.SequencerSettings`.
    """

    lo0_freq: Union[float, None] = None
    """The frequency of Output 0 (O0) LO. If left `None`, the parameter will not be set.
    """
    lo1_freq: Union[float, None] = None
    """The frequency of Output 1 (O1) LO. If left `None`, the parameter will not be set.
    """
    out0_att: Union[int, None] = None
    """The attenuation of Output 0."""
    out1_att: Union[int, None] = None
    """The attenuation of Output 1."""
    in0_att: Union[int, None] = None
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
class PulsarRFSettings(RFModuleSettings, PulsarSettings):
    """
    Settings specific for a Pulsar RF. Effectively, combines the Pulsar specific
    settings with the RF specific settings.
    """

    @classmethod
    def extract_settings_from_mapping(
        cls, mapping: Dict[str, Any], **kwargs: Optional[dict]
    ) -> PulsarRFSettings:
        """
        Factory method that takes all the settings defined in the mapping and generates
        a :class:`~.PulsarRFSettings` object from it.

        Parameters
        ----------
        mapping
            The mapping dict to extract the settings from
        **kwargs
            Additional keyword arguments passed to the constructor. Can be used to
            override parts of the mapping dict.
        """
        rf_settings = RFModuleSettings.extract_settings_from_mapping(mapping)
        pulsar_settings = PulsarSettings.extract_settings_from_mapping(mapping)
        combined_settings = {
            **rf_settings.to_dict(),
            **pulsar_settings.to_dict(),
            **kwargs,
        }
        return cls(**combined_settings)


@dataclass
class SequencerSettings(DataClassJsonMixin):
    # pylint: disable=too-many-instance-attributes
    """
    Sequencer level settings.

    In the drivers these settings are typically recognized by parameter names of the
    form ``"sequencer_{index}_{setting}"``. These settings are set
    once at the start and will remain unchanged after. Meaning that these correspond to
    the "slow" QCoDeS parameters and not settings that are changed dynamically by the
    sequencer.
    """

    nco_en: bool
    """Specifies whether the NCO will be used or not."""
    sync_en: bool
    """Enables party-line synchronization."""
    connected_outputs: Optional[Union[Tuple[int], Tuple[int, int]]]
    """Specifies which physical outputs this sequencer produces waveform data for."""
    connected_inputs: Optional[Union[Tuple[int], Tuple[int, int]]]
    """Specifies which physical inputs this sequencer collects data for."""
    init_offset_awg_path_0: float = 0.0
    """Specifies what value the sequencer offset for AWG path 0 will be reset to
    before the start of the experiment."""
    init_offset_awg_path_1: float = 0.0
    """Specifies what value the sequencer offset for AWG path 1 will be reset to
    before the start of the experiment."""
    init_gain_awg_path_0: float = 1.0
    """Specifies what value the sequencer gain for AWG path 0 will be reset to
    before the start of the experiment."""
    init_gain_awg_path_1: float = 1.0
    """Specifies what value the sequencer gain for AWG path 0 will be reset to
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
    sequence: Optional[Dict[str, Any]] = None
    """JSON compatible dictionary holding the waveforms and program for the
    sequencer."""
    seq_fn: Optional[str] = None
    """Filename of JSON file containing a dump of the waveforms and program."""
    ttl_acq_input_select: Optional[int] = None
    """Selects the input used to compare against the threshold value in the TTL trigger acquisition path."""
    ttl_acq_threshold: Optional[float] = None
    """"Sets the threshold value with which to compare the input ADC values of the selected input path."""
    ttl_acq_auto_bin_incr_en: Optional[bool] = None
    """Selects if the bin index is automatically incremented when acquiring multiple triggers."""

    @classmethod
    def initialize_from_config_dict(
        cls,
        seq_settings: Dict[str, Any],
        connected_outputs: Optional[Union[Tuple[int], Tuple[int, int]]],
        connected_inputs: Optional[Union[Tuple[int], Tuple[int, int]]],
    ) -> SequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer configuration dictionary.

        Parameters
        ----------
        seq_settings
            The sequencer configuration dict.
        connected_outputs
            The outputs connected to the sequencer.
        connected_inputs
            The inputs connected to the sequencer.

        Returns
        -------
        :
            The class with initial values.
        """

        T = TypeVar("T", int, float)

        def extract_and_verify_range(
            param_name: str,
            settings: Dict[str, Any],
            default_value: T,
            min_value: T,
            max_value: T,
        ) -> T:
            val = settings.get(param_name, default_value)
            if val < min_value or val > max_value:
                raise ValueError(
                    f"Attempting to configure {param_name} to {val} for the sequencer "
                    f"specified with port {settings.get('port', '[port invalid!]')} and"
                    f" clock {settings.get('clock', '[clock invalid!]')}, while the "
                    f"hardware requires it to be between {min_value} and {max_value}."
                )
            return val

        modulation_freq: Optional[float] = seq_settings.get("interm_freq", None)
        nco_en: bool = (
            modulation_freq is not None and modulation_freq != 0
        )  # Allow NCO to be permanently disabled via `"interm_freq": 0` in the hardware config

        mixer_amp_ratio = extract_and_verify_range(
            param_name="mixer_amp_ratio",
            settings=seq_settings,
            default_value=1.0,
            min_value=constants.MIN_MIXER_AMP_RATIO,
            max_value=constants.MAX_MIXER_AMP_RATIO,
        )
        mixer_phase_error = extract_and_verify_range(
            param_name="mixer_phase_error_deg",
            settings=seq_settings,
            default_value=0.0,
            min_value=constants.MIN_MIXER_PHASE_ERROR_DEG,
            max_value=constants.MAX_MIXER_PHASE_ERROR_DEG,
        )
        ttl_acq_threshold = seq_settings.get("ttl_acq_threshold", None)

        init_offset_awg_path_0 = extract_and_verify_range(
            param_name="init_offset_awg_path_0",
            settings=seq_settings,
            default_value=cls.init_offset_awg_path_0,
            min_value=-1.0,
            max_value=1.0,
        )
        init_offset_awg_path_1 = extract_and_verify_range(
            param_name="init_offset_awg_path_1",
            settings=seq_settings,
            default_value=cls.init_offset_awg_path_1,
            min_value=-1.0,
            max_value=1.0,
        )
        init_gain_awg_path_0 = extract_and_verify_range(
            param_name="init_gain_awg_path_0",
            settings=seq_settings,
            default_value=cls.init_gain_awg_path_0,
            min_value=-1.0,
            max_value=1.0,
        )
        init_gain_awg_path_1 = extract_and_verify_range(
            param_name="init_gain_awg_path_1",
            settings=seq_settings,
            default_value=cls.init_gain_awg_path_1,
            min_value=-1.0,
            max_value=1.0,
        )

        settings = cls(
            nco_en=nco_en,
            sync_en=True,
            connected_outputs=connected_outputs,
            connected_inputs=connected_inputs,
            init_offset_awg_path_0=init_offset_awg_path_0,
            init_offset_awg_path_1=init_offset_awg_path_1,
            init_gain_awg_path_0=init_gain_awg_path_0,
            init_gain_awg_path_1=init_gain_awg_path_1,
            modulation_freq=modulation_freq,
            mixer_corr_gain_ratio=mixer_amp_ratio,
            mixer_corr_phase_offset_degree=mixer_phase_error,
            ttl_acq_threshold=ttl_acq_threshold,
        )
        return settings


class QbloxBaseDescription(DataStructure):
    """Base class for a Qblox hardware description."""

    hardware_type: Literal["Qblox"]
    """The hardware type, used to select this datastructure when parsing a :class:`~.CompilationConfig`."""
    ref: Union[Literal["internal"], Literal["external"]]
    """The reference source for the instrument."""
    sequence_to_file: bool = True
    """Write sequencer programs to files for (all modules in this) instrument."""


class ClusterModuleDescription(DataStructure):
    """Information needed to specify a Cluster module in the :class:`~.CompilationConfig`."""

    module_type: Union[
        Literal["QCM"], Literal["QRM"], Literal["QCM_RF"], Literal["QRM_RF"]
    ]
    """The module (instrument) type."""
    sequence_to_file: bool = True
    """Write sequencer programs to files, for this module."""


class ClusterDescription(QbloxBaseDescription):
    """Information needed to specify a Cluster in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["Cluster"]
    """The instrument type, used to select this datastructure when parsing a :class:`~.CompilationConfig`."""
    modules: Dict[int, ClusterModuleDescription]
    """Description of the modules of this Cluster, using slot index as key."""


class PulsarQCMDescription(QbloxBaseDescription):
    """Information needed to specify a Pulsar QCM in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["Pulsar_QCM"]
    """The instrument type, used to select this datastructure when parsing a :class:`~.CompilationConfig`."""


class PulsarQRMDescription(QbloxBaseDescription):
    """Information needed to specify a Pulsar QRM in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["Pulsar_QRM"]
    """The instrument type, used to select this datastructure when parsing a :class:`~.CompilationConfig`."""


QbloxHardwareDescription = Annotated[
    Union[ClusterDescription, PulsarQCMDescription, PulsarQRMDescription],
    Field(discriminator="instrument_type"),
]
"""
Specifies a piece of Qblox hardware and its instrument-specific settings.

Currently, the supported instrument types are: 
:class:`~.ClusterDescription`,
:class:`~.PulsarQCMDescription`,
:class:`~.PulsarQRMDescription`
"""
