# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Python dataclasses for compilation to Qblox hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from dataclasses_json import DataClassJsonMixin

from quantify_scheduler.backends.qblox import constants


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
class MarkerConfiguration:
    """Specifies the marker configuration set during the execution of the sequencer
    program."""

    start: int
    """The setting set in the header at the start of the program. Should set all the
    marker outputs high and turn on the output switches."""
    end: int
    """Setting set in the footer at the end of the program."""


@dataclass(frozen=True)
class StaticHardwareProperties:
    """
    Specifies the fixed hardware properties needed in the backend.
    """

    max_sequencers: int
    """The amount of sequencers available."""
    max_awg_output_voltage: float
    """Maximum output voltage of the awg."""
    marker_configuration: MarkerConfiguration
    """The marker configuration to use."""
    mixer_dc_offset_range: BoundedParameter
    """Specifies the range over which the dc offsets can be set that are used for mixer
    calibration."""


@dataclass
class QASMRuntimeSettings:
    """
    Settings that can be changed dynamically by the sequencer during execution of the
    schedule. This is in contrast to the relatively static :class:`~.SequencerSettings`.
    """

    awg_gain_0: float
    """Gain set to the AWG output path 0. Value should be in the range -1.0 < param <
    1.0. Else an exception will be raised during compilation."""
    awg_gain_1: float
    """Gain set to the AWG output path 1. Value should be in the range -1.0 < param <
    1.0. Else an exception will be raised during compilation."""
    awg_offset_0: float = 0.0
    """Offset applied to the AWG output path 0. Value should be in the range -1.0 <
    param < 1.0. Else an exception will be raised during compilation."""
    awg_offset_1: float = 0.0
    """Offset applied to the AWG output path 1. Value should be in the range -1.0 <
    param < 1.0. Else an exception will be raised during compilation."""


@dataclass
class OpInfo(DataClassJsonMixin):
    """
    Data structure containing all the information describing a pulse or acquisition
    needed to play it.
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
    uuid: Optional[str] = None
    """A unique identifier for this pulse/acquisition."""
    pulse_settings: Optional[QASMRuntimeSettings] = None
    """Settings that are to be set by the sequencer before playing this
    pulse/acquisition. This is used for parameterized behavior e.g. setting a gain
    parameter to change the pulse amplitude, instead of changing the waveform. This
    allows to reuse the same waveform multiple times despite a difference in
    amplitude."""
    bin_idx_register: Optional[str] = None
    """The register used to keep track of the bin index, only not None for append mode
    acquisitions."""

    @property
    def duration(self) -> float:
        """
        The duration of the pulse/acquisition.

        Returns
        -------
        :
            The duration of the pulse/acquisition.
        """
        return self.data["duration"]

    @property
    def is_acquisition(self):
        """
        Returns true if this is an acquisition, false if it's a pulse.

        Returns
        -------
        :
            Is this an acquisition?
        """
        return "acq_index" in self.data

    def __repr__(self):
        repr_string = 'Acquisition "' if self.is_acquisition else 'Pulse "'
        repr_string += f"{str(self.name)} - {str(self.uuid)}"
        repr_string += f'" (t={self.timing} to {self.timing+self.duration})'
        repr_string += f" data={self.data}"
        return repr_string


@dataclass
class LOSettings(DataClassJsonMixin):
    """
    Dataclass containing all the settings for a generic LO instrument.
    """

    power: float
    """Power of the LO source."""
    lo_freq: Optional[float]
    """The frequency to set the LO to."""

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> LOSettings:
        """
        Factory method for the LOSettings from a mapping dict.

        Parameters
        ----------
        mapping
            The part of the mapping dict relevant for this instrument.

        Returns
        -------
        :
            Instantiated LOSettings from the mapping dict.
        """
        return cls(power=mapping["power"], lo_freq=mapping["lo_freq"])


@dataclass
class BaseModuleSettings(DataClassJsonMixin):
    """Shared settings between all the Qblox modules."""

    scope_mode_sequencer: Optional[str] = None
    """The name of the sequencer that triggers scope mode Acquisitions. Only a single
    sequencer can perform trace acquisition. This setting gets set as a QCoDeS parameter
    on the driver as well as used for internal checks. Having multiple sequencers
    perform trace acquisition will result in an exception being raised."""
    offset_ch0_path0: Union[float, None] = None
    """The DC offset on the path 0 of channel 0."""
    offset_ch0_path1: Union[float, None] = None
    """The DC offset on the path 1 of channel 0."""
    offset_ch1_path0: Union[float, None] = None
    """The DC offset on path 0 of channel 1."""
    offset_ch1_path1: Union[float, None] = None
    """The DC offset on path 1 of channel 1."""


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
        assert ref in ("internal", "external")
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
    connected_outputs: Union[Tuple[int], Tuple[int, int]]
    """Specifies which physical outputs this sequencer produces waveform data for."""
    modulation_freq: float = None
    """Specifies the frequency of the modulation."""
    mixer_corr_phase_offset_degree: float = 0.0
    """The phase shift to apply between the I and Q channels, to correct for quadrature
    errors."""
    mixer_corr_gain_ratio: float = 1.0
    """The gain ratio to apply in order to correct for imbalances between the I and Q
    paths of the mixer."""
    integration_length_acq: Optional[int] = None
    """Integration length for acquisitions. Must be a multiple of 4 ns."""

    @classmethod
    def initialize_from_config_dict(
        cls,
        seq_settings: Dict[str, Any],
        connected_outputs: Union[Tuple[int], Tuple[int, int]],
    ) -> SequencerSettings:
        """
        Instantiates an instance of this class, with initial parameters determined from
        the sequencer configuration dictionary.

        Parameters
        ----------
        seq_settings
            The sequencer configuration dict.

        Returns
        -------
        :
            The class with initial values.
        """

        def extract_and_verify_range(
            param_name: str,
            settings: Dict[str, Any],
            default_value: float,
            min_value: float,
            max_value: float,
        ) -> float:
            val: float = settings.get(param_name, default_value)
            if val < min_value or val > max_value:
                raise ValueError(
                    f"Attempting to configure {param_name} to {val} for the sequencer "
                    f"specified with port {settings.get('port', '[port invalid!]')} and"
                    f" clock {settings.get('clock', '[clock invalid!]')}, while the "
                    f"hardware requires it to be between {min_value} and {max_value}."
                )
            return val

        modulation_freq: Union[float, None] = seq_settings.get("interm_freq", None)
        nco_en: bool = modulation_freq != 0 and modulation_freq is not None

        mixer_amp_ratio = extract_and_verify_range(
            "mixer_amp_ratio",
            seq_settings,
            1.0,
            constants.MIN_MIXER_AMP_RATIO,
            constants.MAX_MIXER_AMP_RATIO,
        )
        mixer_phase_error = extract_and_verify_range(
            "mixer_phase_error_deg",
            seq_settings,
            0.0,
            constants.MIN_MIXER_PHASE_ERROR_DEG,
            constants.MAX_MIXER_PHASE_ERROR_DEG,
        )

        settings = cls(
            nco_en=nco_en,
            sync_en=True,
            connected_outputs=connected_outputs,
            modulation_freq=modulation_freq,
            mixer_corr_gain_ratio=mixer_amp_ratio,
            mixer_corr_phase_offset_degree=mixer_phase_error,
        )
        return settings
