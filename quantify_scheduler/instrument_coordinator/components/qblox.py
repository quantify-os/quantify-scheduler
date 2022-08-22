# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing Qblox InstrumentCoordinator Components."""
from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
from qblox_instruments import (
    Cluster,
    SequencerState,
    SequencerStatus,
    SequencerStatusFlags,
)
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.types.qblox import (
    BaseModuleSettings,
    RFModuleSettings,
    SequencerSettings,
)
from quantify_scheduler.instrument_coordinator.components import base
from quantify_scheduler.instrument_coordinator.utility import lazy_set
from quantify_scheduler.schedules.schedule import AcquisitionMetadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass(frozen=True)
class _SequencerStateInfo:
    message: str
    """The text to pass as the logging message."""
    logging_level: int
    """The logging level to use."""

    @staticmethod
    def get_logging_level(flag: SequencerStatusFlags) -> int:
        """Define the logging level per SequencerStatusFlags flag."""
        if (
            flag is SequencerStatusFlags.ACQ_SCOPE_DONE_PATH_0
            or flag is SequencerStatusFlags.ACQ_SCOPE_DONE_PATH_0
            or flag is SequencerStatusFlags.ACQ_BINNING_DONE
        ):
            return logging.DEBUG

        if (
            flag is SequencerStatusFlags.DISARMED
            or flag is SequencerStatusFlags.FORCED_STOP
            or flag is SequencerStatusFlags.ACQ_SCOPE_OVERWRITTEN_PATH_0
            or flag is SequencerStatusFlags.ACQ_SCOPE_OVERWRITTEN_PATH_1
        ):
            return logging.INFO

        if (
            flag is SequencerStatusFlags.ACQ_SCOPE_OUT_OF_RANGE_PATH_0
            or flag is SequencerStatusFlags.ACQ_SCOPE_OUT_OF_RANGE_PATH_1
            or flag is SequencerStatusFlags.ACQ_BINNING_OUT_OF_RANGE
        ):
            return logging.WARNING

        if (
            flag is SequencerStatusFlags.SEQUENCE_PROCESSOR_Q1_ILLEGAL_INSTRUCTION
            or flag
            is SequencerStatusFlags.SEQUENCE_PROCESSOR_RT_EXEC_ILLEGAL_INSTRUCTION
            or flag is SequencerStatusFlags.SEQUENCE_PROCESSOR_RT_EXEC_COMMAND_UNDERFLOW
            or flag is SequencerStatusFlags.AWG_WAVE_PLAYBACK_INDEX_INVALID_PATH_0
            or flag is SequencerStatusFlags.AWG_WAVE_PLAYBACK_INDEX_INVALID_PATH_1
            or flag is SequencerStatusFlags.ACQ_WEIGHT_PLAYBACK_INDEX_INVALID_PATH_0
            or flag is SequencerStatusFlags.ACQ_WEIGHT_PLAYBACK_INDEX_INVALID_PATH_1
            or flag is SequencerStatusFlags.ACQ_BINNING_FIFO_ERROR
            or flag is SequencerStatusFlags.ACQ_BINNING_COMM_ERROR
            or flag is SequencerStatusFlags.ACQ_INDEX_INVALID
            or flag is SequencerStatusFlags.ACQ_BIN_INDEX_INVALID
            or flag is SequencerStatusFlags.CLOCK_INSTABILITY
            or flag is SequencerStatusFlags.OUTPUT_OVERFLOW
        ):
            return logging.ERROR

        return logging.DEBUG


_SEQUENCER_STATE_FLAG_INFO: Dict[SequencerStatusFlags, _SequencerStateInfo] = {
    flag: _SequencerStateInfo(
        message=flag.value, logging_level=_SequencerStateInfo.get_logging_level(flag)
    )
    for flag in SequencerStatusFlags
}
"""Used to link all flags returned by the hardware to logging message and
logging level."""


@dataclass(frozen=True)
class _StaticHardwareProperties:
    """Dataclass that holds all the static differences between the different Qblox
    devices that are relevant for configuring them correctly."""

    settings_type: Type[BaseModuleSettings]
    """The settings dataclass to use that the hardware needs to configure to."""
    has_internal_lo: bool
    """Specifies if an internal lo source is available."""
    number_of_sequencers: int
    """The number of sequencers the hardware has available."""
    number_of_output_paths: int
    """The number of output paths that can be used."""


_QCM_BASEBAND_PROPERTIES = _StaticHardwareProperties(
    settings_type=BaseModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QCM,
    number_of_output_paths=4,
)
_QRM_BASEBAND_PROPERTIES = _StaticHardwareProperties(
    settings_type=BaseModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QRM,
    number_of_output_paths=2,
)
_QCM_RF_PROPERTIES = _StaticHardwareProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QCM,
    number_of_output_paths=4,
)
_QRM_RF_PROPERTIES = _StaticHardwareProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QRM,
    number_of_output_paths=2,
)


class QbloxInstrumentCoordinatorComponentBase(base.InstrumentCoordinatorComponentBase):
    """Qblox InstrumentCoordinator component base class."""

    def __init__(
        self, instrument: Union[Instrument, InstrumentChannel], **kwargs
    ) -> None:
        """
        Create a new instance of QbloxInstrumentCoordinatorComponentBase base class.
        """
        super().__init__(instrument, **kwargs)

        self.instrument_channel = (
            instrument if isinstance(instrument, InstrumentChannel) else None
        )

        if instrument.is_rf_type is not self._hardware_properties.has_internal_lo:
            raise RuntimeError(
                "QbloxInstrumentCoordinatorComponentBase not compatible with the "
                "provided instrument. Please confirm whether your device "
                "is an RF module or a baseband module (having or not having an "
                "internal LO)."
            )

        self._seq_name_to_idx_map = {
            f"seq{idx}": idx
            for idx in range(self._hardware_properties.number_of_sequencers)
        }

    @property
    def instrument(self) -> Union[Instrument, InstrumentChannel]:
        """
        For Cluster modules we return a reference to its `InstrumentChannel` in the
        Cluster instrument; for Pulsar modules we return the `instrument` reference
        """
        if self.instrument_channel is not None:
            return self.instrument_channel

        return super().instrument

    def _set_parameter(
        self,
        instrument: Union[Instrument, InstrumentChannel],
        parameter_name: str,
        val: Any,
    ) -> None:
        """
        Sets the parameter directly or using the lazy set, depending on the value of
        `force_set_parameters`.

        Parameters
        ----------
        instrument
            The instrument or instrument channel that holds the parameter to set,
            e.g. `self.instrument` or `self.instrument[f"sequencer{idx}"]`.
        parameter_name
            The name of the parameter to set.
        val
            The new value of the parameter.
        """
        if self.force_set_parameters():
            instrument.set(parameter_name, val)
        else:
            lazy_set(instrument, parameter_name, val)

    @property
    def is_running(self) -> bool:
        """
        Finds if any of the sequencers is currently running.

        Returns
        -------
        :
            True if any of the sequencers reports the `SequencerStatus.RUNNING` status.
        """
        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            seq_state = self.instrument.get_sequencer_state(seq_idx)
            if seq_state.status is SequencerStatus.RUNNING:
                return True
        return False

    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Blocks the instrument until all the sequencers are done running.

        Parameters
        ----------
        timeout_sec
            The timeout in seconds. N.B. the instrument takes the timeout in minutes
            (int), therefore it is rounded down to whole minutes with a minimum of 1.
        """
        timeout_min = timeout_sec // 60
        if timeout_min == 0:
            timeout_min = 1
        for idx in range(self._hardware_properties.number_of_sequencers):
            state: SequencerState = self.instrument.get_sequencer_state(
                sequencer=idx, timeout=timeout_min
            )
            if state.flags:
                for flag in state.flags:
                    if flag not in _SEQUENCER_STATE_FLAG_INFO:
                        logger.error(
                            f"[{self.name}|seq{idx}] Encountered flag {flag} in "
                            f"returned value by `get_sequencer_state` which is not "
                            f"defined in {self.__module__}. Please refer to the Qblox "
                            f"instruments documentation for more info."
                        )
                    else:
                        flag_info = _SEQUENCER_STATE_FLAG_INFO[flag]
                        msg = f"[{self.name}|seq{idx}] {flag} - {flag_info.message}"
                        logger.log(level=flag_info.logging_level, msg=msg)

    def start(self) -> None:
        """
        Starts execution of the schedule.
        """
        for idx in range(self._hardware_properties.number_of_sequencers):
            state = self.instrument.get_sequencer_state(idx)
            if state.status is SequencerStatus.ARMED:
                self.instrument.start_sequencer(idx)

    def stop(self) -> None:
        """
        Stops all execution.
        """
        for idx in range(self._hardware_properties.number_of_sequencers):
            # disable sync to prevent hanging on next run if instrument is not used.
            self._set_parameter(self.instrument[f"sequencer{idx}"], "sync_en", False)
        self.instrument.stop_sequencer()

    @abstractmethod
    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """

    def _configure_sequencer_settings(
        self, seq_idx: int, settings: SequencerSettings
    ) -> None:
        """
        Configures all sequencer-specific settings.

        Parameters
        ----------
        seq_idx
            Index of the sequencer to configure.
        settings
            The settings to configure it to.
        """
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"], "sync_en", settings.sync_en
        )

        nco_en: bool = settings.nco_en
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"], "mod_en_awg", nco_en
        )
        if nco_en:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "nco_freq",
                settings.modulation_freq,
            )
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "mixer_corr_phase_offset_degree",
            settings.mixer_corr_phase_offset_degree,
        )
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "mixer_corr_gain_ratio",
            settings.mixer_corr_gain_ratio,
        )

        for output_idx in range(self._hardware_properties.number_of_output_paths):
            connected: bool = output_idx in settings.connected_outputs
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                self._get_channel_map_parameter_name(output_index=output_idx),
                connected,
            )

    @staticmethod
    def _get_channel_map_parameter_name(output_index: int) -> str:
        path_idx = output_index % 2  # even or odd output
        return f"channel_map_path{path_idx}_out{output_index}_en"

    def _arm_all_sequencers_in_program(self, program: Dict[str, Any]):
        """Arms all the sequencers that are part of the program."""
        for seq_name in program:
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
                self.instrument.arm_sequencer(sequencer=seq_idx)

    @property
    @abstractmethod
    def _hardware_properties(self) -> _StaticHardwareProperties:
        """
        Holds all the differences between the different modules.

        Returns
        -------
        :
            A dataclass with all the hardware properties for this specific module.
        """


class QCMComponent(QbloxInstrumentCoordinatorComponentBase):
    """
    QCM specific InstrumentCoordinator component.
    """

    _hardware_properties = _QCM_BASEBAND_PROPERTIES

    def __init__(self, instrument: Instrument, **kwargs) -> None:
        """Create a new instance of QCMComponent."""
        if not instrument.is_qcm_type:
            raise TypeError(
                f"Trying to create QCMComponent from non-QCM instrument "
                f'of type "{type(instrument)}".'
            )
        super().__init__(instrument, **kwargs)

    def retrieve_acquisition(self) -> None:
        """
        Retrieves the previous acquisition.

        Returns
        -------
        :
            QCM returns None since the QCM has no acquisition.
        """
        return None

    def prepare(self, options: Dict[str, dict]) -> None:
        """
        Uploads the waveforms and programs to the sequencers and
        configures all the settings required. Keep in mind that values set directly
        through the driver may be overridden (e.g. the offsets will be set according to
        the specified mixer calibration parameters).

        Parameters
        ----------
        options
            Program to upload to the sequencers. The key is a sequencer, e.g.,
            :code:`"seq0"`, or :code:`"settings"`,
            the value is the global settings dict or a sequencer-specific configuration.
        """
        program = copy.deepcopy(options)

        if "settings" in program:
            settings_entry = program.pop("settings")
            module_settings = self._hardware_properties.settings_type.from_dict(
                settings_entry
            )
            self._configure_global_settings(module_settings)
        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"], "sync_en", False
            )

        for seq_name, seq_cfg in program.items():
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            if "settings" in seq_cfg:
                seq_settings = SequencerSettings.from_dict(seq_cfg["settings"])
                self._configure_sequencer_settings(
                    seq_idx=seq_idx, settings=seq_settings
                )

            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "sequence",
                seq_cfg["seq_fn"],
            )

        self._arm_all_sequencers_in_program(program)

    def _configure_global_settings(self, settings: BaseModuleSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        # configure mixer correction offsets
        if settings.offset_ch0_path0 is not None:
            self._set_parameter(
                self.instrument, "out0_offset", settings.offset_ch0_path0
            )
        if settings.offset_ch0_path1 is not None:
            self._set_parameter(
                self.instrument, "out1_offset", settings.offset_ch0_path1
            )
        if settings.offset_ch1_path0 is not None:
            self._set_parameter(
                self.instrument, "out2_offset", settings.offset_ch1_path0
            )
        if settings.offset_ch1_path1 is not None:
            self._set_parameter(
                self.instrument, "out3_offset", settings.offset_ch1_path1
            )


class QRMComponent(QbloxInstrumentCoordinatorComponentBase):
    """
    QRM specific InstrumentCoordinator component.
    """

    _hardware_properties = _QRM_BASEBAND_PROPERTIES

    def __init__(self, instrument: Instrument, **kwargs) -> None:
        """Create a new instance of QRMComponent."""
        if not instrument.is_qrm_type:
            raise TypeError(
                f"Trying to create QRMComponent from non-QRM instrument "
                f'of type "{type(instrument)}".'
            )
        super().__init__(instrument, **kwargs)

        self._acquisition_manager: Optional[_QRMAcquisitionManager] = None
        """Holds all the acquisition related logic."""

    def retrieve_acquisition(self) -> Optional[Dict[AcquisitionIndexing, Any]]:
        """
        Retrieves the latest acquisition results.

        Returns
        -------
        :
            The acquired data.
        """
        if self._acquisition_manager is None:  # No acquisition has been prepared.
            return None
        return self._acquisition_manager.retrieve_acquisition()

    def prepare(self, options: Dict[str, dict]) -> None:
        """
        Uploads the waveforms and programs to the sequencers and
        configures all the settings required. Keep in mind that values set directly
        through the driver may be overridden (e.g. the offsets will be set according to
        the specified mixer calibration parameters).

        Parameters
        ----------
        options
            Program to upload to the sequencers. The key is a sequencer, e.g.,
            :code:`"seq0"`, or :code:`"settings"`,
            the value is the global settings dict or a sequencer-specific configuration.
        """
        program = copy.deepcopy(options)
        if "acq_metadata" in program:
            acq_metadata = program.pop("acq_metadata")
        if "acq_mapping" in program:  # Resets everything to do with acquisition.

            acq_mapping = program.pop("acq_mapping")
            self._acquisition_manager = _QRMAcquisitionManager(
                self,
                number_of_sequencers=self._hardware_properties.number_of_sequencers,
                acquisition_mapping=acq_mapping,
                acquisition_metadata=acq_metadata,
            )
        else:
            self._acquisition_manager = None

        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"], "sync_en", False
            )

        if "settings" in program:
            settings_entry = program.pop("settings")
            module_settings = self._hardware_properties.settings_type.from_dict(
                settings_entry
            )
            if self._acquisition_manager is not None:
                self._acquisition_manager.scope_mode_sequencer = (
                    module_settings.scope_mode_sequencer
                )
            self._configure_global_settings(module_settings)

        for path in [
            0,
            1,
        ]:
            self._set_parameter(
                self.instrument, f"scope_acq_trigger_mode_path{path}", "sequencer"
            )
            self._set_parameter(
                self.instrument, f"scope_acq_avg_mode_en_path{path}", True
            )

        for seq_name, seq_cfg in program.items():
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )
            if "settings" in seq_cfg:
                seq_settings = SequencerSettings.from_dict(seq_cfg["settings"])
                self._configure_sequencer_settings(
                    seq_idx=seq_idx, settings=seq_settings
                )

            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "sequence",
                seq_cfg["seq_fn"],
            )

        self._arm_all_sequencers_in_program(program)

    def _configure_global_settings(self, settings: BaseModuleSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        if settings.scope_mode_sequencer is not None:
            self._set_parameter(
                self.instrument,
                "scope_acq_sequencer_select",
                settings.scope_mode_sequencer,
            )

        # configure mixer correction offsets
        if settings.offset_ch0_path0 is not None:
            self._set_parameter(
                self.instrument, "out0_offset", settings.offset_ch0_path0
            )
        if settings.offset_ch0_path1 is not None:
            self._set_parameter(
                self.instrument, "out1_offset", settings.offset_ch0_path1
            )
        # configure gain
        if settings.in0_gain is not None:
            self._set_parameter(self.instrument, "in0_gain", settings.in0_gain)
        if settings.in1_gain is not None:
            self._set_parameter(self.instrument, "in1_gain", settings.in1_gain)

    def _configure_sequencer_settings(
        self, seq_idx: int, settings: SequencerSettings
    ) -> None:
        super()._configure_sequencer_settings(seq_idx, settings)
        if settings.integration_length_acq is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "integration_length_acq",
                settings.integration_length_acq,
            )
            self._acquisition_manager.integration_length_acq = (
                settings.integration_length_acq
            )
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"], "demod_en_acq", settings.nco_en
        )


class QCMRFComponent(QCMComponent):
    """
    QCM-RF specific InstrumentCoordinator component.
    """

    _hardware_properties = _QCM_RF_PROPERTIES

    def _configure_global_settings(self, settings: RFModuleSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        if settings.lo0_freq is not None:
            self._set_parameter(self.instrument, "out0_lo_freq", settings.lo0_freq)
        if settings.lo1_freq is not None:
            self._set_parameter(self.instrument, "out1_lo_freq", settings.lo1_freq)

        # configure mixer correction offsets
        if settings.offset_ch0_path0 is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path0", settings.offset_ch0_path0
            )
        if settings.offset_ch0_path1 is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path1", settings.offset_ch0_path1
            )
        if settings.offset_ch1_path0 is not None:
            self._set_parameter(
                self.instrument, "out1_offset_path0", settings.offset_ch1_path0
            )
        if settings.offset_ch1_path1 is not None:
            self._set_parameter(
                self.instrument, "out1_offset_path1", settings.offset_ch1_path1
            )
        # configure attenuation
        if settings.out0_att is not None:
            self._set_parameter(self.instrument, "out0_att", settings.out0_att)
        if settings.out1_att is not None:
            self._set_parameter(self.instrument, "out1_att", settings.out1_att)


class QRMRFComponent(QRMComponent):
    """
    QRM-RF specific InstrumentCoordinator component.
    """

    _hardware_properties = _QRM_RF_PROPERTIES

    def _configure_global_settings(self, settings: RFModuleSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        if settings.scope_mode_sequencer is not None:
            self._set_parameter(
                self.instrument,
                "scope_acq_sequencer_select",
                settings.scope_mode_sequencer,
            )

        if settings.lo0_freq is not None:
            self._set_parameter(self.instrument, "out0_in0_lo_freq", settings.lo0_freq)

        # configure mixer correction offsets
        if settings.offset_ch0_path0 is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path0", settings.offset_ch0_path0
            )
        if settings.offset_ch0_path1 is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path1", settings.offset_ch0_path1
            )
        # configure gain and attenuation
        if settings.out0_att is not None:
            self._set_parameter(self.instrument, "out0_att", settings.out0_att)
        if settings.in0_att is not None:
            self._set_parameter(self.instrument, "in0_att", settings.in0_att)


class PulsarQCMComponent(QCMComponent):
    """A component for a baseband Pulsar QCM."""

    def prepare(self, options: Dict[str, dict]) -> None:
        super().prepare(options)
        reference_source: str = options["settings"]["ref"]
        self._set_parameter(self.instrument, "reference_source", reference_source)


class PulsarQRMComponent(QRMComponent):
    """A component for a baseband Pulsar QRM."""

    def prepare(self, options: Dict[str, dict]) -> None:
        super().prepare(options)
        reference_source: str = options["settings"]["ref"]
        self._set_parameter(self.instrument, "reference_source", reference_source)


AcquisitionIndexing = namedtuple("AcquisitionIndexing", "acq_channel acq_index")
"""
Named tuple to clarify how the indexing of acquisitions works inside the
`_QRMAcquisitionManager`.
"""


class _QRMAcquisitionManager:
    """
    Utility class that handles the acquisitions performed with the QRM.

    An instance of this class is meant to exist only for a single prepare-start-
    retrieve_acquisition cycle to prevent stateful behavior.
    """

    def __init__(
        self,
        parent: QRMComponent,
        number_of_sequencers: int,
        acquisition_mapping: Dict[Tuple[int, int], Tuple[str, str]],
        acquisition_metadata: AcquisitionMetadata,
    ):
        """
        Constructor for `_QRMAcquisitionManager`.

        Parameters
        ----------
        parent
            Reference to the parent QRM IC component.
        number_of_sequencers
            The number of sequencers capable of acquisitions.
        acquisition_mapping
            The acquisition mapping extracted from the schedule, this mapping links the
            `acq_channel` and `acq_index` to the sequencer name and acquisition
            protocol. The key is a tuple (`acq_channel`, `acq_index`), the values
            (seq_name, protocol).
        acquisition_metadata
            Provides a summary of the used channels bins and acquisition protocols.
        """
        self.parent: QRMComponent = parent
        self.number_of_sequencers: int = number_of_sequencers
        self.acquisition_mapping: Dict[
            Tuple[int, int], Tuple[str, str]
        ] = acquisition_mapping
        self.acquisition_metadata: AcquisitionMetadata = acquisition_metadata

        self.scope_mode_sequencer: Optional[str] = None
        self.integration_length_acq: Optional[int] = None
        self.seq_name_to_idx_map = {
            f"seq{idx}": idx for idx in range(number_of_sequencers)
        }

    @property
    def instrument(self):
        """Returns the QRM driver from the parent IC component."""
        return self.parent.instrument

    def retrieve_acquisition(self) -> Dict[AcquisitionIndexing, Any]:
        """
        Retrieves all the acquisition data in the correct format.

        Returns
        -------
        :
            The acquisitions with the protocols specified in the `acq_mapping` as a
            `dict` with the `(acq_channel, acq_index)` as keys.
        """

        protocol_to_function_mapping = {
            "weighted_integrated_complex": self._get_integration_data,
            "ssb_integration_complex": self._get_integration_amplitude_data,
            "trace": self._get_scope_data,
            # NB thresholded protocol is still missing since there is nothing in
            # the acquisition library for it yet.
        }
        self._store_scope_acquisition()

        formatted_acquisitions: Dict[AcquisitionIndexing, Any] = {}

        for seq_idx in range(self.number_of_sequencers):
            if f"seq{seq_idx}" not in self.acquisition_metadata:
                continue
            acq_metadata = self.acquisition_metadata[f"seq{seq_idx}"]
            acquisition_function: Callable = protocol_to_function_mapping[
                acq_metadata.acq_protocol
            ]

            # retrieve the raw data from the qrm sequencer module
            acquisitions = self.instrument.get_acquisitions(seq_idx)
            for acq_channel, acq_indices in acq_metadata.acq_indices.items():
                # the acquisition_function retrieves the right part of the acquisitions
                # data structure returned by the qrm
                i_vals, q_vals = acquisition_function(
                    acquisitions=acquisitions, acq_channel=acq_channel
                )

                # the Qblox compilation backend verifies that the
                # acquisition indices start at 0 and increment in steps of 1.
                # this enables us to simply stride over the bin_idx as if they
                # correspond to acq_indices.
                for acq_idx in acq_indices:
                    acq_stride = len(acq_indices)
                    # N.B. the stride idx ensures that in append mode all data
                    # corresponding to the same acq_index appears in the
                    # same acq_ch, acq_idx part of the returned formatted acquisitions.
                    formatted_acquisitions[
                        AcquisitionIndexing(acq_channel=acq_channel, acq_index=acq_idx)
                    ] = (
                        i_vals[acq_idx::acq_stride],
                        q_vals[acq_idx::acq_stride],
                    )

        return formatted_acquisitions

    def _store_scope_acquisition(self):
        sequencer_index = self.scope_mode_sequencer

        if sequencer_index is None:
            return

        if sequencer_index > self.number_of_sequencers:
            raise ValueError(
                f"Attempting to retrieve scope mode data from sequencer "
                f"{sequencer_index}. A QRM has only "
                f"{self.number_of_sequencers} sequencers."
            )
        scope_ch_and_idx = self._get_scope_channel_and_index()
        if scope_ch_and_idx is not None:
            acq_channel, _ = scope_ch_and_idx
            acq_name = self._channel_index_to_channel_name(acq_channel)
            self.instrument.store_scope_acquisition(sequencer_index, acq_name)

    def _get_protocol(self, acq_channel, acq_index) -> str:
        """
        Returns the acquisition protocol corresponding to acq_channel with
        acq_index.
        """
        return self.acquisition_mapping[(acq_channel, acq_index)][1]

    def _get_sequencer_index(self, acq_channel, acq_index) -> int:
        """
        Returns the seq idx corresponding to acq_channel with
        acq_index.
        """
        seq_name = self.acquisition_mapping[(acq_channel, acq_index)][0]
        return self.seq_name_to_idx_map[seq_name]

    def _get_scope_channel_and_index(self) -> Optional[AcquisitionIndexing]:
        """
        Returns the first `(acq_channel, acq_index)` pair that uses `"trace"`
        acquisition. Returns `None` if none of them do.
        """
        ch_and_idx: Optional[AcquisitionIndexing] = None
        for key, value in self.acquisition_mapping.items():
            if value[1] == "trace":
                if ch_and_idx is not None:
                    # Pylint seems to not care we explicitly check for None
                    # pylint: disable=unpacking-non-sequence
                    acq_channel, acq_index = ch_and_idx
                    raise RuntimeError(
                        f"A scope mode acquisition is defined for both acq_channel "
                        f"{acq_channel} with acq_index {acq_index} as well as "
                        f"acq_channel {key[0]} with acq_index {key[1]}. Only a single "
                        f"trace acquisition is allowed per QRM."
                    )
                ch_and_idx = AcquisitionIndexing(acq_channel=key[0], acq_index=key[1])
        return ch_and_idx

    def _get_scope_data(
        self, acquisitions: dict, acq_channel: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the scope mode acquisition associated with an `acq_channel`.

        Parameters
        ----------
        acq_channel
            The acq_channel to get the scope mode acquisition for.

        Returns
        -------
        scope_data_i
            The scope mode data for `path0`.
        scope_data_q
            The scope mode data for `path1`.
        """
        acq_name = self._channel_index_to_channel_name(acq_channel)
        scope_data = acquisitions[acq_name]["acquisition"]["scope"]
        for path_label in ("path0", "path1"):
            if scope_data[path_label]["out-of-range"]:
                logger.warning(
                    f"The scope mode data of {path_label} of {self.parent.name} with "
                    f"acq_channel={acq_channel}  was out-of-range."
                )
        # NB hardware already divides by avg_count for scope mode
        scope_data_i = np.array(scope_data["path0"]["data"])
        scope_data_q = np.array(scope_data["path1"]["data"])
        return scope_data_i, scope_data_q

    def _get_integration_data(
        self, acquisitions: dict, acq_channel: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the integrated acquisition data associated with an `acq_channel`.

        Parameters
        ----------
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The `acq_channel` from which to get the data.

        Returns
        -------
        i_data
            The integrated data for path0.
        q_data
            The integrated data for path1.
        """

        bin_data = self._get_bin_data(acquisitions, acq_channel)

        i_data, q_data = (
            np.array(bin_data["integration"]["path0"]),
            np.array(bin_data["integration"]["path1"]),
        )

        return i_data, q_data

    def _get_integration_amplitude_data(
        self, acquisitions: dict, acq_channel: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the integration data but normalized to the integration time (number of
        samples summed). The return value is thus the amplitude of the demodulated
        signal directly and has volt units (i.e. same units as a single sample of the
        integrated signal).

        Parameters
        ----------
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The `acq_channel` from which to get the data.

        Returns
        -------
        data_i
            Array containing I-quadrature data.
        data_q
            Array containing Q-quadrature data.
        """
        if self.integration_length_acq is None:
            raise RuntimeError(
                "Retrieving data failed. Expected the integration length to be defined,"
                " but it is `None`."
            )
        compensated_data_i, compensated_data_q = self._get_integration_data(
            acquisitions=acquisitions, acq_channel=acq_channel
        )
        compensated_data_i, compensated_data_q = (
            compensated_data_i / self.integration_length_acq,
            compensated_data_q / self.integration_length_acq,
        )
        return compensated_data_i, compensated_data_q

    def _get_threshold_data(
        self, acquisitions: dict, acq_channel: int = 0, acq_index: int = 0
    ) -> float:
        """
        Retrieves the thresholded acquisition data associated with `acq_channel` and
        `acq_index`.

        Parameters
        ----------
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acq_channel to get the thresholded acquisition data for.
        acq_index
            The acq_index to get the thresholded acquisition data for.

        Returns
        -------
        :
            The value of the thresholded acquisition for `acq_channel` and `acq_index`.
            Should always be 0.0 <= val <= 1.0.
        """
        bin_data = self._get_bin_data(acquisitions, acq_channel)
        data = bin_data["threshold"]

        if acq_index > len(data):
            raise ValueError(
                f"Attempting to access acq_index {acq_index} on "
                f"{self.parent.name} but only {len(data)} values found "
                f"in acquisition data."
            )
        return data[acq_index]

    @staticmethod
    def _channel_index_to_channel_name(acq_channel: int) -> str:
        """Returns the name of the acquisition from the acq_channel."""
        return str(acq_channel)

    @classmethod
    def _get_bin_data(cls, acquisitions: dict, acq_channel: int = 0) -> dict:
        """Returns the bin entry of the acquisition data dict."""
        acq_name = cls._channel_index_to_channel_name(acq_channel)
        channel_data = acquisitions[acq_name]
        if channel_data["index"] != acq_channel:
            raise RuntimeError(
                f"Name does not correspond to a valid acquisition for name {acq_name}, "
                f'which has index {channel_data["index"]}.'
            )
        return channel_data["acquisition"]["bins"]


ClusterModule = Union[QCMComponent, QRMComponent, QCMRFComponent, QRMRFComponent]
"""Type that combines all the possible modules for a cluster."""


class ClusterComponent(base.InstrumentCoordinatorComponentBase):
    """
    Class that represents an instrument coordinator component for a Qblox cluster.
    """

    def __init__(self, instrument: Cluster, **kwargs) -> None:
        """
        Create a new instance of the ClusterComponent. Automatically adds installed
        modules using name `"<cluster_name>_module<slot>"`.

        Parameters
        ----------
        instrument
            Reference to the cluster driver object.
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init__(instrument, **kwargs)
        self._cluster_modules: Dict[str, ClusterModule] = {}

        for instrument_channel in instrument.modules:
            try:
                icc_class: type = {
                    (True, False): QCMComponent,
                    (True, True): QCMRFComponent,
                    (False, False): QRMComponent,
                    (False, True): QRMRFComponent,
                }[(instrument_channel.is_qcm_type, instrument_channel.is_rf_type)]
            except KeyError:
                continue

            self._cluster_modules[instrument_channel.name] = icc_class(
                instrument_channel
            )

    @property
    def is_running(self) -> bool:
        """Returns true if any of the modules are currently running."""
        return any(comp.is_running for comp in self._cluster_modules.values())

    def start(self) -> None:
        """Starts all the modules in the cluster."""
        for comp in self._cluster_modules.values():
            comp.start()

    def stop(self) -> None:
        """Stops all the modules in the cluster."""
        for comp in self._cluster_modules.values():
            comp.stop()

    def _configure_cmm_settings(self, settings: Dict[str, Any]):
        """
        Sets all the settings of the CMM (Cluster Management Module) that have been
        provided by the backend.

        Parameters
        ----------
        settings
            A dictionary containing all the settings to set.
        """
        if "reference_source" in settings:
            if self.force_set_parameters():
                self.instrument.set("reference_source", settings["reference_source"])
            else:
                lazy_set(
                    self.instrument, "reference_source", settings["reference_source"]
                )

    def prepare(self, options: Dict[str, dict]) -> None:
        """
        Prepares the cluster component for execution of a schedule.

        Parameters
        ----------
        options
            The compiled instructions to configure the cluster to.
        """
        for name, comp_options in options.items():
            if name == "settings":
                self._configure_cmm_settings(settings=comp_options)
            elif name in self._cluster_modules:
                self._cluster_modules[name].prepare(comp_options)
            else:
                raise KeyError(
                    f"Attempting to prepare module {name} of cluster {self.name}, while"
                    f" module has not been added to the cluster component."
                )

    def retrieve_acquisition(self) -> Optional[Dict[Tuple[int, int], Any]]:
        """
        Retrieves all the data from the instruments.

        Returns
        -------
        :
            The acquired data or ``None`` if no acquisitions have been performed.
        """
        acquisitions: Dict[Tuple[int, int], Any] = {}
        for comp in self._cluster_modules.values():
            comp_acq = comp.retrieve_acquisition()
            if comp_acq is not None:
                acquisitions.update(comp_acq)

        return acquisitions if len(acquisitions) > 0 else None

    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Blocks until all the components are done executing their programs.

        Parameters
        ----------
        timeout_sec
            The time in seconds until the instrument is considered to have timed out.
        """
        for comp in self._cluster_modules.values():
            comp.wait_done(timeout_sec=timeout_sec)
