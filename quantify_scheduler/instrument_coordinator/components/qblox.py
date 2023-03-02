# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing Qblox InstrumentCoordinator Components."""
from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import numpy as np
from qblox_instruments import (
    Cluster,
    SequencerState,
    SequencerStatus,
    SequencerStatusFlags,
)
from qcodes.instrument import Instrument, InstrumentModule
from xarray import DataArray, Dataset

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.helpers import (
    single_scope_mode_acquisition_raise,
)
from quantify_scheduler.backends.types.qblox import (
    BaseModuleSettings,
    RFModuleSettings,
    SequencerSettings,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.instrument_coordinator.components import base
from quantify_scheduler.instrument_coordinator.utility import (
    check_already_existing_acquisition,
    lazy_set,
)
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
            or flag is SequencerStatusFlags.TRIGGER_NETWORK_CONFLICT
            or flag is SequencerStatusFlags.TRIGGER_NETWORK_MISSED_INTERNAL_TRIGGER
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
    number_of_input_paths: int
    """"The number of input paths that can be used."""


_QCM_BASEBAND_PROPERTIES = _StaticHardwareProperties(
    settings_type=BaseModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QCM,
    number_of_output_paths=4,
    number_of_input_paths=0,
)
_QRM_BASEBAND_PROPERTIES = _StaticHardwareProperties(
    settings_type=BaseModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QRM,
    number_of_output_paths=2,
    number_of_input_paths=2,
)
_QCM_RF_PROPERTIES = _StaticHardwareProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QCM,
    number_of_output_paths=4,
    number_of_input_paths=0,
)
_QRM_RF_PROPERTIES = _StaticHardwareProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QRM,
    number_of_output_paths=2,
    number_of_input_paths=2,
)


class QbloxInstrumentCoordinatorComponentBase(base.InstrumentCoordinatorComponentBase):
    """Qblox InstrumentCoordinator component base class."""

    def __init__(
        self, instrument: Union[Instrument, InstrumentModule], **kwargs
    ) -> None:
        """
        Create a new instance of QbloxInstrumentCoordinatorComponentBase base class.
        """
        super().__init__(instrument, **kwargs)

        self._instrument_module = (
            instrument if isinstance(instrument, InstrumentModule) else None
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
    def instrument(self) -> Union[Instrument, InstrumentModule]:
        """
        If the instrument behind this instance of
        `QbloxInstrumentCoordinatorComponentBase` is an `InstrumentModule` (e.g. the
        module within the `qblox_instrument.Cluster`), it is returned. Otherwise, the
        reference to the `instrument` is returned (e.g. for a stand-alone
        `qblox_instruments.Pulsar`).
        """
        if self._instrument_module is not None:
            return self._instrument_module

        return super().instrument

    def _set_parameter(
        self,
        instrument: Union[Instrument, InstrumentModule],
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

        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"], "mod_en_awg", settings.nco_en
        )
        if settings.nco_en:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "nco_freq",
                settings.modulation_freq,
            )

        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "offset_awg_path0",
            settings.init_offset_awg_path_0,
        )
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "offset_awg_path1",
            settings.init_offset_awg_path_1,
        )

        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "gain_awg_path0",
            settings.init_gain_awg_path_0,
        )
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "gain_awg_path1",
            settings.init_gain_awg_path_1,
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

        if settings.connected_outputs is not None:
            for output_idx in range(self._hardware_properties.number_of_output_paths):
                connected: bool = output_idx in settings.connected_outputs
                self._set_parameter(
                    self.instrument[f"sequencer{seq_idx}"],
                    self._get_channel_map_parameter_name(output_index=output_idx),
                    connected,
                )

        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"], "sequence", settings.sequence
        )

    @staticmethod
    def _get_channel_map_parameter_name(output_index: int) -> str:
        path_idx = output_index % 2  # even or odd output
        return f"channel_map_path{path_idx}_out{output_index}_en"

    def _arm_all_sequencers_in_program(self, program: Dict[str, Any]):
        """Arms all the sequencers that are part of the program."""
        for seq_name in program["sequencers"]:
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

    def prepare(self, program: Dict[str, dict]) -> None:
        """
        Uploads the waveforms and programs to the sequencers and
        configures all the settings required. Keep in mind that values set directly
        through the driver may be overridden (e.g. the offsets will be set according to
        the specified mixer calibration parameters).

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        """

        if (settings_entry := program.get("settings")) is not None:
            module_settings = self._hardware_properties.settings_type.from_dict(
                settings_entry
            )
            self._configure_global_settings(module_settings)

        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"], "sync_en", False
            )

        for seq_name, seq_cfg in program["sequencers"].items():
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            self._configure_sequencer_settings(
                seq_idx=seq_idx, settings=SequencerSettings.from_dict(seq_cfg)
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

    def retrieve_acquisition(self) -> Optional[Dataset]:
        """
        Retrieves the latest acquisition results.

        Returns
        -------
        :
            The acquired data.
        """
        if self._acquisition_manager:
            return self._acquisition_manager.retrieve_acquisition()
        else:
            return None

    def prepare(self, program: Dict[str, dict]) -> None:
        """
        Uploads the waveforms and programs to the sequencers and
        configures all the settings required. Keep in mind that values set directly
        through the driver may be overridden (e.g. the offsets will be set according to
        the specified mixer calibration parameters).

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        """

        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"], "sync_en", False
            )

        acq_duration = {}
        for seq_name, seq_cfg in program["sequencers"].items():
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            settings = SequencerSettings.from_dict(seq_cfg)
            self._configure_sequencer_settings(seq_idx=seq_idx, settings=settings)
            acq_duration[seq_name] = settings.integration_length_acq

        if (acq_metadata := program.get("acq_metadata")) is not None:
            scope_mode_sequencer_and_channel = (
                self._determine_scope_mode_acquisition_sequencer_and_channel(
                    acq_metadata
                )
            )
            self._acquisition_manager = _QRMAcquisitionManager(
                parent=self,
                acquisition_metadata=acq_metadata,
                scope_mode_sequencer_and_channel=scope_mode_sequencer_and_channel,
                acquisition_duration=acq_duration,
                seq_name_to_idx_map=self._seq_name_to_idx_map,
            )
            if scope_mode_sequencer_and_channel is not None:
                self._set_parameter(
                    self.instrument,
                    "scope_acq_sequencer_select",
                    scope_mode_sequencer_and_channel[0],
                )
        else:
            self._acquisition_manager = None

        if (settings_entry := program.get("settings")) is not None:
            module_settings = self._hardware_properties.settings_type.from_dict(
                settings_entry
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

        self._clear_sequencer_acquisition_data()
        self._arm_all_sequencers_in_program(program)

    def _clear_sequencer_acquisition_data(self):
        """Clear all acquisition data."""
        for sequencer_id in range(self._hardware_properties.number_of_sequencers):
            self.instrument.delete_acquisition_data(sequencer=sequencer_id, all=True)

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

        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"], "demod_en_acq", settings.nco_en
        )
        if settings.ttl_acq_auto_bin_incr_en is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "ttl_acq_auto_bin_incr_en",
                settings.ttl_acq_auto_bin_incr_en,
            )
        if settings.ttl_acq_threshold is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "ttl_acq_threshold",
                settings.ttl_acq_threshold,
            )
        if settings.ttl_acq_input_select is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "ttl_acq_input_select",
                settings.ttl_acq_input_select,
            )

    def _determine_scope_mode_acquisition_sequencer_and_channel(
        self, acquisition_metadata: Dict[str, AcquisitionMetadata]
    ) -> Optional[Tuple[int, int]]:
        """
        Finds which sequencer, channel has to perform raw trace acquisitions.
        Raises an error if multiple scope mode acquisitions are present per sequencer.
        Note, that compiler ensures there is at most one scope mode acquisition,
        however the user is able to freely modify the compiler program,
        so we make sure this requirement is still satisfied. See
        :func:`~quantify_scheduler.backends.qblox.compiler_abc.QbloxBaseModule._ensure_single_scope_mode_acquisition_sequencer`.

        Parameters
        ----------
        acquisition_metadata
            The acquisition metadata for each sequencer.

        Returns
        -------
        :
            The sequencer and channel for the trace acquisition, if there is any, otherwise None, None.
        """

        sequencer_and_channel = None
        for (
            sequencer_name,
            current_acquisition_metadata,
        ) in acquisition_metadata.items():
            if current_acquisition_metadata.acq_protocol == "Trace":
                # It's in the format "seq{n}", so we cut it.
                sequencer_id = self._seq_name_to_idx_map[sequencer_name]
                if (
                    sequencer_and_channel is not None
                    and sequencer_and_channel[0] != sequencer_id
                ):
                    single_scope_mode_acquisition_raise(
                        sequencer_0=sequencer_id,
                        sequencer_1=sequencer_and_channel[0],
                        module_name=self.name,
                    )
                # For scope protocol, only one channel makes sense, we only need the first key in dict
                channel = next(iter(current_acquisition_metadata.acq_indices.keys()))
                sequencer_and_channel = (sequencer_id, channel)

        return sequencer_and_channel


class QbloxRFComponent(QbloxInstrumentCoordinatorComponentBase):
    """
    Mix-in for RF-module-specific InstrumentCoordinatorComponent behaviour.
    """

    def _configure_sequencer_settings(
        self, seq_idx: int, settings: SequencerSettings
    ) -> None:
        super()._configure_sequencer_settings(seq_idx, settings)
        # Always set override to False. Markers should be controlled using
        # MarkerConfiguration
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "marker_ovr_en",
            False,
        )


class QCMRFComponent(QbloxRFComponent, QCMComponent):
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


class QRMRFComponent(QbloxRFComponent, QRMComponent):
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
        # configure attenuation
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


class _QRMAcquisitionManager:
    """
    Utility class that handles the acquisitions performed with the QRM.

    An instance of this class is meant to exist only for a single prepare-start-
    retrieve_acquisition cycle to prevent stateful behavior.
    """

    def __init__(
        self,
        parent: QRMComponent,
        acquisition_metadata: Dict[str, AcquisitionMetadata],
        scope_mode_sequencer_and_channel: Optional[Tuple[int, int]],
        acquisition_duration: Dict[int, int],
        seq_name_to_idx_map: Dict[str, int],
    ):
        """
        Constructor for `_QRMAcquisitionManager`.

        Parameters
        ----------
        parent
            Reference to the parent QRM IC component.
        acquisition_metadata
            Provides a summary of the used acquisition protocol, bin mode, acquisition channels,
            acquisition indices per channel, and repetitions, for each sequencer.
        scope_mode_sequencer_and_channel
            The sequencer and channel of the scope mode acquisition if there's any.
        acquisition_duration
            The duration of each acquisition for each sequencer.
        seq_name_to_idx_map
            All available sequencer names to their ids in a dict.
        """
        self.parent: QRMComponent = parent
        self._acquisition_metadata: Dict[
            str, AcquisitionMetadata
        ] = acquisition_metadata

        self._scope_mode_sequencer_and_channel: Optional[
            Tuple[int, int]
        ] = scope_mode_sequencer_and_channel
        self._acq_duration: Dict[str, int] = acquisition_duration
        self._seq_name_to_idx_map = seq_name_to_idx_map

    @property
    def instrument(self):
        """Returns the QRM driver from the parent IC component."""
        return self.parent.instrument

    def retrieve_acquisition(self) -> Dataset:
        """
        Retrieves all the acquisition data in the correct format.

        Returns
        -------
        :
            The acquisitions with the protocols specified in the `acquisition_metadata`.
            Each `xarray.DataArray` in the `xarray.Dataset` corresponds to one `acq_channel`.
            The `acq_channel` is the name of each `xarray.DataArray` in the `xarray.Dataset`.
            Each `xarray.DataArray` is a two-dimensional array, with `acq_index` and `repetition` as
            dimensions.
        """

        protocol_to_function_mapping = {
            "WeightedIntegratedComplex": self._get_integration_data,
            "SSBIntegrationComplex": self._get_integration_amplitude_data,
            "Trace": self._get_scope_data,
            "TriggerCount": self._get_trigger_count_data,
        }
        self._store_scope_acquisition()

        dataset = Dataset()

        for sequencer_name, acquisition_metadata in self._acquisition_metadata.items():
            acquisition_function: Callable = protocol_to_function_mapping[
                acquisition_metadata.acq_protocol
            ]
            # retrieve the raw data from the qrm sequencer module
            acquisitions = self.instrument.get_acquisitions(
                self._seq_name_to_idx_map[sequencer_name]
            )
            for acq_channel, acq_indices in acquisition_metadata.acq_indices.items():
                # the acquisition_function retrieves the right part of the acquisitions
                # data structure returned by the qrm
                formatted_acquisitions = acquisition_function(
                    acq_indices=acq_indices,
                    acquisitions=acquisitions,
                    acq_channel=acq_channel,
                    acq_duration=self._acq_duration[sequencer_name],
                    acquisition_metadata=acquisition_metadata,
                )
                formatted_acquisitions_dataset = Dataset(
                    {acq_channel: formatted_acquisitions}
                )

                check_already_existing_acquisition(
                    new_dataset=formatted_acquisitions_dataset, current_dataset=dataset
                )
                dataset = dataset.merge(formatted_acquisitions_dataset)

        return dataset

    @classmethod
    def _format_acquisitions_data_array_for_scope_and_integration(
        cls,
        acquisition_metadata: AcquisitionMetadata,
        acq_indices: Iterable,
        acquisitions_data: Iterable,
    ) -> DataArray:
        """
        Formats the scope and integration type acquisition data to a `xarray.DataArray`.
        acquisitions_data is expected to be in the following order for the binned acquisitions:
        :code:`[index0_rep0, index1_rep0, index2_rep0, index0_rep1, index1_rep1, index2_rep1, ...]`.
        The data is converted into the following order for the `xarray.DataArray`:
        :code:`[[index0_rep0, index1_rep0, index2_rep0], [index0_rep1, index1_rep1, index2_rep1], ...]`.

        Parameters
        ----------
        acquisition_metadata
            Acquisition metadata.
        acq_indices
            Indices for the acquisition channel.
        acquisitions_data
            Data of the acquisitions.

        Returns
        -------
        :
            Formatted `xarray.DataArray` of the acquisition data by cutting the raw data
            retrieved from the hardware into repetitions, and giving them
            appropriate coordinates in the `xarray.DataArray`.
        """
        appended_repetitions = (
            acquisition_metadata.repetitions
            if acquisition_metadata.bin_mode == BinMode.APPEND
            else 1
        )
        return cls._format_acquisitions_data_array(
            acquisitions_data=np.split(acquisitions_data, appended_repetitions),
            coords_repetition=range(appended_repetitions),
            coords_acq_index=acq_indices,
        )

    @staticmethod
    def _format_acquisitions_data_array(
        acquisitions_data, coords_repetition, coords_acq_index
    ) -> DataArray:
        """
        Formats acquisitions_data to an `xarray.DataArray` by giving them
        the appropriate coordinates: :code:`coords_repetition` as repetition coordinates
        and :code:`coords_acq_index` as acquisition indices.
        It's just a simple wrapper, so that each `xarray.DataArray`
        creation is consistent among protocols.
        """
        return DataArray(
            data=acquisitions_data,
            coords=[coords_repetition, coords_acq_index],
            dims=["repetition", "acq_index"],
        )

    def _store_scope_acquisition(self):
        """
        Calls :code:`store_scope_acquisition` function on the Qblox instrument.
        This will ensure that the correct sequencer will store the scope acquisition
        data on the hardware, so it will be filled out when we call :code:`get_acquisitions`
        on the Qblox instrument's sequencer corresponding to the scope acquisition.
        """
        if self._scope_mode_sequencer_and_channel is None:
            return

        sequencer_index = self._scope_mode_sequencer_and_channel[0]

        if sequencer_index not in self._seq_name_to_idx_map.values():
            raise ValueError(
                f"Attempting to retrieve scope mode data from sequencer "
                f"{sequencer_index}. A QRM only has the following sequencer indices: "
                f"{list(self._seq_name_to_idx_map.values())}."
            )
        acq_channel = self._scope_mode_sequencer_and_channel[1]
        acq_name = self._channel_index_to_channel_name(acq_channel)
        self.instrument.store_scope_acquisition(sequencer_index, acq_name)

    def _get_scope_data(
        self,
        acq_indices: list,
        acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,
        acq_channel: int = 0,
    ) -> DataArray:
        """
        Retrieves the scope mode acquisition associated with an `acq_channel`.

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        acq_channel
            The `acq_channel` from which to get the data.

        Returns
        -------
        :
            The scope mode data.
        """
        if acq_duration < 0 or acq_duration > constants.MAX_SAMPLE_SIZE_ACQUISITIONS:
            raise ValueError(
                "Attempting to retrieve sample of size "
                f"{acq_duration} "
                f"(maximum allowed sample size: "
                f"{constants.MAX_SAMPLE_SIZE_ACQUISITIONS})"
            )
        acq_name = self._channel_index_to_channel_name(acq_channel)
        scope_data = acquisitions[acq_name]["acquisition"]["scope"]
        for path_label in ("path0", "path1"):
            if scope_data[path_label]["out-of-range"]:
                logger.warning(
                    f"The scope mode data of {path_label} of {self.parent.name} with "
                    f"acq_channel={acq_channel}  was out-of-range."
                )
        # NB hardware already divides by avg_count for scope mode
        scope_data_i = np.array(scope_data["path0"]["data"][:acq_duration])
        scope_data_q = np.array(scope_data["path1"]["data"][:acq_duration])

        return self._format_acquisitions_data_array_for_scope_and_integration(
            acquisition_metadata=acquisition_metadata,
            acq_indices=range(acq_duration),
            acquisitions_data=scope_data_i * 1j + scope_data_q,
        )

    def _get_integration_data(
        self,
        acq_indices: list,
        acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,  # pylint: disable=unused-argument
        acq_channel: int = 0,
    ) -> DataArray:
        """
        Retrieves the integrated acquisition data associated with an `acq_channel`.

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Not used in this function.
        acq_channel
            The `acq_channel` from which to get the data.

        Returns
        -------
        :
            The integrated data.
        """

        bin_data = self._get_bin_data(acquisitions, acq_channel)

        i_data, q_data = (
            np.array(bin_data["integration"]["path0"]),
            np.array(bin_data["integration"]["path1"]),
        )

        acquisitions_data = i_data * 1j + q_data
        return self._format_acquisitions_data_array_for_scope_and_integration(
            acquisition_metadata=acquisition_metadata,
            acq_indices=acq_indices,
            acquisitions_data=acquisitions_data,
        )

    def _get_integration_amplitude_data(
        self,
        acq_indices: list,
        acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,
        acq_channel: int = 0,
    ) -> DataArray:
        """
        Gets the integration data but normalized to the integration time (number of
        samples summed). The return value is thus the amplitude of the demodulated
        signal directly and has volt units (i.e. same units as a single sample of the
        integrated signal).

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Duration of the acquisition. This needs to be specified.
        acq_channel
            The `acq_channel` from which to get the data.

        Returns
        -------
        :
            Array containing binned, normalized acquisition data.
        """
        if acq_duration is None:
            raise RuntimeError(
                "Retrieving data failed. Expected the integration length to be defined,"
                " but it is `None`."
            )
        formatted_data = self._get_integration_data(
            acq_indices=acq_indices,
            acquisitions=acquisitions,
            acquisition_metadata=acquisition_metadata,
            acq_duration=acq_duration,
            acq_channel=acq_channel,
        )

        return formatted_data / acq_duration

    def _get_threshold_data(
        self,
        acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,  # pylint: disable=unused-argument
        acq_duration: int,  # pylint: disable=unused-argument
        acq_channel: int = 0,
        acq_index: int = 0,
    ) -> float:
        """
        Retrieves the thresholded acquisition data associated with `acq_channel` and
        `acq_index`.

        Parameters
        ----------
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Not used in this function.
        acq_duration
            Not used in this function.
        acq_channel
            The `acq_channel` from which to get the data.
        acq_index
            The acquisition index.

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

    def _get_trigger_count_data(
        self,
        acq_indices: list,
        acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,  # pylint: disable=unused-argument
        acq_channel: int = 0,
    ) -> DataArray:
        """
        Retrieves the trigger count acquisition data associated with `acq_channel`.

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Not used in this function.
        acq_channel
            The `acq_channel` from which to get the data.

        Returns
        -------
        :
        count
            A list of integers indicating the amount of triggers counted.
        occurence
            For BinMode.AVERAGE a list of integers with the occurence of each trigger count,
            for BinMode.APPEND a list of 1's.
        """
        bin_data = self._get_bin_data(acquisitions, acq_channel)

        if acquisition_metadata.bin_mode == BinMode.AVERAGE:
            bin_count_diff = np.diff(bin_data["avg_cnt"])
            res = {
                count + 1: -v
                for (count, v) in enumerate(bin_count_diff)
                if not (np.isnan(v) or np.isclose(v, 0))
            }
            return self._format_acquisitions_data_array(
                acquisitions_data=[list(res.values())],
                coords_repetition=[0],
                coords_acq_index=list(res.keys()),
            )

        elif acquisition_metadata.bin_mode == BinMode.APPEND:
            counts = list(bin_data["avg_cnt"])
            counts = [0 if np.isnan(x) else x for x in counts]
            return self._format_acquisitions_data_array(
                acquisitions_data=[[1] * len(counts)],
                coords_repetition=[0],
                coords_acq_index=counts,
            )

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
        self._program = {}

        for instrument_module in instrument.modules:
            try:
                icc_class: type = {
                    (True, False): QCMComponent,
                    (True, True): QCMRFComponent,
                    (False, False): QRMComponent,
                    (False, True): QRMRFComponent,
                }[(instrument_module.is_qcm_type, instrument_module.is_rf_type)]
            except KeyError:
                continue

            self._cluster_modules[instrument_module.name] = icc_class(instrument_module)

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
        self._program = copy.deepcopy(options)

        for name, comp_options in self._program.items():
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
        for comp_name, comp in self._cluster_modules.items():
            if comp_name not in self._program:
                continue

            comp_acq = comp.retrieve_acquisition()
            if comp_acq is not None:
                check_already_existing_acquisition(
                    new_dataset=comp_acq, current_dataset=acquisitions
                )
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
