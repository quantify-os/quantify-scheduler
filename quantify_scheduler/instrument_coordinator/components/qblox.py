# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing Qblox InstrumentCoordinator Components."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from math import isnan
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
)
from uuid import uuid4

import numpy as np
from qblox_instruments import (
    Cluster,
    ConfigurationManager,
    SequencerStates,
    SequencerStatus,
)
from qblox_instruments.qcodes_drivers.time import SyncRef
from xarray import DataArray, Dataset

from quantify_core.data.handling import get_datadir
from quantify_core.utilities.general import without
from quantify_scheduler.backends.qblox import constants, driver_version_check
from quantify_scheduler.backends.qblox.enums import (
    ChannelMode,
    LoCalEnum,
    SidebandCalEnum,
    TimetagTraceType,
)
from quantify_scheduler.backends.qblox.helpers import (
    single_scope_mode_acquisition_raise,
)
from quantify_scheduler.backends.qblox.operation_handling.bin_mode_compat import (
    QRM_COMPATIBLE_BIN_MODES,
    QTM_COMPATIBLE_BIN_MODES,
    IncompatibleBinModeError,
)
from quantify_scheduler.backends.qblox.qblox_acq_index_manager import (
    QbloxAcquisitionHardwareMapping,
    QbloxAcquisitionIndex,
)
from quantify_scheduler.backends.types.qblox import (
    AnalogModuleSettings,
    AnalogSequencerSettings,
    BaseModuleSettings,
    ClusterSettings,
    ExternalTriggerSyncSettings,
    RFModuleSettings,
    SequencerSettings,
    TimetagModuleSettings,
    TimetagSequencerSettings,
)
from quantify_scheduler.enums import BinMode, TimeRef, TriggerCondition
from quantify_scheduler.helpers.schedule import (
    _is_acquisition_binned_append,
    _is_acquisition_binned_average,
)
from quantify_scheduler.instrument_coordinator.components import base
from quantify_scheduler.instrument_coordinator.utility import (
    add_acquisition_coords_binned,
    add_acquisition_coords_nonbinned,
    check_already_existing_acquisition,
    lazy_set,
    parameter_value_same_as_cache,
    search_settable_param,
)
from quantify_scheduler.schedules.schedule import AcquisitionChannelsData

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from qblox_instruments.qcodes_drivers.module import Module
    from qcodes.instrument.instrument_base import InstrumentBase

    from quantify_scheduler.backends.types.common import ThresholdedTriggerCountMetadata
    from quantify_scheduler.schedules.schedule import CompiledSchedule

ComponentTypeProperties = namedtuple(
    "ComponentTypeProperties",
    (
        "is_qcm_type",
        "is_qrm_type",
        "is_rf_type",
        "is_qtm_type",
        "is_qrc_type",
    ),
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Prevent unsupported qblox-instruments version from crashing this submodule
driver_version_check.verify_qblox_instruments_version()


@dataclass(frozen=True)
class _StaticHardwareProperties:
    """Dataclass for storing configuration differences across Qblox devices."""

    settings_type: type[BaseModuleSettings]
    """The settings dataclass to use that the hardware needs to configure to."""
    number_of_sequencers: int
    """The number of sequencers the hardware has available."""
    number_of_output_channels: int
    """The number of physical output channels that can be used."""
    number_of_input_channels: int
    """The number of physical input channels that can be used."""
    number_of_scope_acq_channels: int
    """The number of scope acquisition channels."""


@dataclass(frozen=True)
class _StaticAnalogModuleProperties(_StaticHardwareProperties):
    """Dataclass for storing configuration differences across Qblox devices."""

    settings_type: type[AnalogModuleSettings]
    """The settings dataclass to use that the hardware needs to configure to."""
    has_internal_lo: bool
    """Specifies if an internal lo source is available."""


@dataclass(frozen=True)
class _StaticTimetagModuleProperties(_StaticHardwareProperties):
    """Dataclass for storing configuration differences across Qblox devices."""

    settings_type: type[TimetagModuleSettings]
    """The settings dataclass to use that the hardware needs to configure to."""


_QCM_BASEBAND_PROPERTIES = _StaticAnalogModuleProperties(
    settings_type=AnalogModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=6,
    number_of_output_channels=4,
    number_of_input_channels=0,
    number_of_scope_acq_channels=0,
)
_QRM_BASEBAND_PROPERTIES = _StaticAnalogModuleProperties(
    settings_type=AnalogModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=6,
    number_of_output_channels=2,
    number_of_input_channels=2,
    number_of_scope_acq_channels=2,
)
_QCM_RF_PROPERTIES = _StaticAnalogModuleProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=6,
    number_of_output_channels=2,
    number_of_input_channels=0,
    number_of_scope_acq_channels=0,
)
_QRM_RF_PROPERTIES = _StaticAnalogModuleProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=6,
    number_of_output_channels=1,
    number_of_input_channels=1,
    number_of_scope_acq_channels=2,
)
_QRC_PROPERTIES = _StaticAnalogModuleProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=12,
    number_of_output_channels=6,
    number_of_input_channels=2,
    number_of_scope_acq_channels=4,
)
_QTM_PROPERTIES = _StaticTimetagModuleProperties(
    settings_type=TimetagModuleSettings,
    number_of_sequencers=8,
    number_of_output_channels=8,
    number_of_input_channels=8,
    number_of_scope_acq_channels=0,
)


_HardwarePropertiesT_co = TypeVar(
    "_HardwarePropertiesT_co", bound=_StaticHardwareProperties, covariant=True
)


class _ModuleComponentBase(
    base.InstrumentCoordinatorComponentBase, Generic[_HardwarePropertiesT_co]
):
    """Qblox InstrumentCoordinator component base class."""

    _hardware_properties: _HardwarePropertiesT_co

    def __init__(self, instrument: Module) -> None:
        super().__init__(instrument)

        # The base class `InstrumentCoordinatorComponentBase` expects `instrument` to
        # be a subclass of `Instrument`, _This_ class expects `instrument` to be a
        # `Module` (for legacy reasons?), which does not subclass `Instrument` but
        # `InstrumentChannel`, and is therefore not globally findable. Ergo, we store a
        # reference here directly.
        self._instrument_module = instrument

        self._seq_name_to_idx_map = {
            f"seq{idx}": idx for idx in range(self._hardware_properties.number_of_sequencers)
        }

        # TODO make a data model for this and replace the dictionary by a class.
        self._program = {}

        self._nco_frequency_changed: dict[int, bool] = {}
        """
        Private attribute for automatic mixer calibration. The keys are sequencer
        indices. The `prepare` method resets this to an empty dictionary.
        """

    # See the comment on self._instrument_module in __init__. Base class is incorrectly
    # overridden, so we silence pyright.
    @property
    def instrument(self) -> Module:  # type: ignore
        """Returns a reference to the module instrument."""
        return self._instrument_module

    def _set_parameter(
        self,
        instrument: InstrumentBase,
        parameter_name: str,
        val: Any,  # noqa: ANN401, disallow Any as type
    ) -> None:
        """
        Set the parameter directly or using the lazy set.

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
        # TODO: Bias-tee parameters for RTP are not officially supported in
        # qblox-instruments. This hack is at a customer's request.
        try:
            search_settable_param(instrument=instrument, nested_parameter_name=parameter_name)
        except ValueError as e:
            if re.search(r".*(out|marker)[0-3]_bt_config", parameter_name) and val == "bypassed":
                return
            if re.search(
                r".*(out|marker)[0-3]_bt_time_constant",
                parameter_name,
            ):
                return
            raise e
        if self.force_set_parameters():
            instrument.parameters[parameter_name].set(val)
        else:
            lazy_set(instrument, parameter_name, val)

    @property
    def is_running(self) -> bool:
        """
        Finds if any of the sequencers is currently running.

        Returns
        -------
        :
            True if any of the sequencers reports the `SequencerStates.RUNNING` status.

        """
        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            seq_status = self.instrument.get_sequencer_status(seq_idx)
            if seq_status.state is SequencerStates.RUNNING:
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
            state: SequencerStatus = self.instrument.get_sequencer_status(
                sequencer=idx, timeout=timeout_min
            )
            for flag in state.info_flags:
                logger.log(
                    level=logging.INFO,
                    msg=f"[{self.name}|seq{idx}] {flag} - {flag.value}",
                )
            for flag in state.warn_flags:
                logger.log(
                    level=logging.WARNING,
                    msg=f"[{self.name}|seq{idx}] {flag} - {flag.value}",
                )
            for flag in state.err_flags:
                logger.log(
                    level=logging.ERROR,
                    msg=f"[{self.name}|seq{idx}] {flag} - {flag.value}",
                )

    def get_hardware_log(
        self,
        compiled_schedule: CompiledSchedule,
    ) -> dict | None:
        """
        Retrieve the hardware log of the Qblox instrument associated to this component.

        This log does not include the instrument serial number and firmware version.

        Parameters
        ----------
        compiled_schedule
            Compiled schedule to check if this component is referenced in.

        Returns
        -------
        :
            A dict containing the hardware log of the Qblox instrument, in case the
            component was referenced; else None.

        """
        if self.instrument.name not in compiled_schedule.compiled_instructions:
            return None

        return _download_log(_get_configuration_manager(_get_instrument_ip(self)))

    def prepare(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        program: dict[str, dict],
        acq_channels_data: AcquisitionChannelsData,  # noqa: ARG002, unused parameter
        repetitions: int,  # noqa: ARG002, unused parameter
    ) -> None:
        """Store program containing sequencer settings."""
        self._program = program
        self._nco_frequency_changed = {}

    def disable_sync(self) -> None:
        """Disable sync for all sequencers."""
        for idx in range(self._hardware_properties.number_of_sequencers):
            # Prevent hanging on next run if instrument is not used.
            self._set_parameter(self.instrument.sequencers[idx], "sync_en", False)

    def stop(self) -> None:
        """Stops all execution."""
        self.disable_sync()
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

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings) -> None:
        """
        Configures all sequencer-specific settings.

        Parameters
        ----------
        seq_idx
            Index of the sequencer to configure.
        settings
            The settings to configure it to.

        """
        self._set_parameter(self.instrument.sequencers[seq_idx], "sync_en", settings.sync_en)

        self._set_parameter(self.instrument.sequencers[seq_idx], "sequence", settings.sequence)

        for address, trigger_settings in settings.thresholded_acq_trigger_read_settings.items():
            if trigger_settings.thresholded_acq_trigger_invert is not None:
                self._set_parameter(
                    self.instrument.sequencers[seq_idx],
                    f"trigger{address}_threshold_invert",
                    trigger_settings.thresholded_acq_trigger_invert,
                )
            if trigger_settings.thresholded_acq_trigger_count is not None:
                self._set_parameter(
                    self.instrument.sequencers[seq_idx],
                    f"trigger{address}_count_threshold",
                    trigger_settings.thresholded_acq_trigger_count,
                )

    def arm_all_sequencers_in_program(self) -> None:
        """Arm all the sequencers that are part of the program."""
        for seq_name in self._program.get("sequencers", {}):
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
                self.instrument.arm_sequencer(sequencer=seq_idx)

    def start(self) -> None:
        """Clear data, arm sequencers and start sequencers."""
        self.clear_data()
        self.arm_all_sequencers_in_program()
        self._start_armed_sequencers()

    def _start_armed_sequencers(self) -> None:
        """Start execution of the schedule: start armed sequencers."""
        for idx in range(self._hardware_properties.number_of_sequencers):
            state = self.instrument.get_sequencer_status(idx)
            if state.state is SequencerStates.ARMED:
                self.instrument.start_sequencer(idx)

    def clear_data(self) -> None:
        """Clears remaining data on the module. Module type specific function."""
        return None


class _AnalogModuleComponent(_ModuleComponentBase):
    """Qblox InstrumentCoordinator component base class."""

    _hardware_properties: _StaticAnalogModuleProperties

    def __init__(self, instrument: Module) -> None:
        super().__init__(instrument)

        if instrument.is_rf_type is not self._hardware_properties.has_internal_lo:
            raise RuntimeError(
                f"{self.__class__.__name__} not compatible with the "
                "provided instrument. Please confirm whether your device "
                "is a Qblox RF or baseband module (having or not having an "
                "internal LO)."
            )

    @abstractmethod
    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.

        """

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings) -> None:
        """
        Configures all sequencer-specific settings.

        Parameters
        ----------
        seq_idx
            Index of the sequencer to configure.
        settings
            The settings to configure it to.

        """
        assert isinstance(settings, AnalogSequencerSettings)

        if settings.nco_en:
            self._nco_frequency_changed[seq_idx] = not parameter_value_same_as_cache(
                self.instrument.sequencers[seq_idx],
                "nco_freq",
                settings.modulation_freq,
            )
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "nco_freq",
                settings.modulation_freq,
            )
        else:
            # NCO off == no change.
            self._nco_frequency_changed[seq_idx] = False

        self._set_parameter(
            self.instrument.sequencers[seq_idx],
            "offset_awg_path0",
            settings.init_offset_awg_path_I,
        )
        self._set_parameter(
            self.instrument.sequencers[seq_idx],
            "offset_awg_path1",
            settings.init_offset_awg_path_Q,
        )

        self._set_parameter(
            self.instrument.sequencers[seq_idx],
            "gain_awg_path0",
            settings.init_gain_awg_path_I,
        )
        self._set_parameter(
            self.instrument.sequencers[seq_idx],
            "gain_awg_path1",
            settings.init_gain_awg_path_Q,
        )

        channel_map_parameters = self._determine_channel_map_parameters(settings)
        for channel_param, channel_setting in channel_map_parameters.items():
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                channel_param,
                channel_setting,
            )

        super()._configure_sequencer_settings(seq_idx, settings)

    def _determine_channel_map_parameters(
        self, settings: AnalogSequencerSettings
    ) -> dict[str, str]:
        """Returns a dictionary with the channel map parameters for this module."""
        channel_map_parameters = {}
        self._determine_output_channel_map_parameters(settings, channel_map_parameters)

        return channel_map_parameters

    def _determine_output_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: dict[str, str]
    ) -> dict[str, str]:
        """Adds the outputs to the channel map parameters dict."""
        for channel_idx in range(self._hardware_properties.number_of_output_channels):
            param_setting = "off"
            # For baseband, output indices map 1-to-1 to channel map indices
            if (
                len(settings.connected_output_indices) > 0
                and channel_idx in settings.connected_output_indices
                and channel_idx in settings.connected_output_indices
            ):
                if ChannelMode.COMPLEX in settings.channel_name:
                    param_setting = "I" if (channel_idx % 2 == 0) else "Q"
                elif ChannelMode.REAL in settings.channel_name:
                    param_setting = "I"

            channel_map_parameters[f"connect_out{channel_idx}"] = param_setting

        return channel_map_parameters

    def _configure_nco_mixer_calibration(
        self, seq_idx: int, settings: AnalogSequencerSettings
    ) -> None:
        if (
            settings.auto_sideband_cal == SidebandCalEnum.ON_INTERM_FREQ_CHANGE
            and self._nco_frequency_changed[seq_idx]
        ):
            self.instrument.sequencers[seq_idx].sideband_cal()
        else:
            if settings.mixer_corr_phase_offset_degree is not None:
                self._set_parameter(
                    self.instrument.sequencers[seq_idx],
                    "mixer_corr_phase_offset_degree",
                    settings.mixer_corr_phase_offset_degree,
                )
            if settings.mixer_corr_gain_ratio is not None:
                self._set_parameter(
                    self.instrument.sequencers[seq_idx],
                    "mixer_corr_gain_ratio",
                    settings.mixer_corr_gain_ratio,
                )


class _QCMComponent(_AnalogModuleComponent):
    """QCM specific InstrumentCoordinator component."""

    _hardware_properties = _QCM_BASEBAND_PROPERTIES

    def __init__(self, instrument: Module) -> None:
        if not instrument.is_qcm_type:
            raise TypeError(
                f"Trying to create _QCMComponent from non-QCM instrument "
                f'of type "{type(instrument)}".'
            )
        super().__init__(instrument)

    def retrieve_acquisition(self) -> None:
        """
        Retrieves the previous acquisition.

        Returns
        -------
        :
            QCM returns None since the QCM has no acquisition.

        """
        return None

    def prepare(
        self, program: dict[str, dict], acq_channels_data: AcquisitionChannelsData, repetitions: int
    ) -> None:
        """
        Uploads the waveforms and programs to the sequencers.

        All the settings that are required are configured. Keep in mind that
        values set directly through the driver may be overridden (e.g. the
        offsets will be set according to the specified mixer calibration
        parameters).

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        acq_channels_data
            Acquisition channels data.
        repetitions
            Repetitions of the schedule.

        """
        super().prepare(program, acq_channels_data, repetitions)

        if (settings := program.get("settings")) is not None:
            if isinstance(settings, dict):
                settings = self._hardware_properties.settings_type.from_dict(settings)
            self._configure_global_settings(settings)

        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            self._set_parameter(self.instrument.sequencers[seq_idx], "sync_en", False)

        for seq_name, settings in program["sequencers"].items():
            if isinstance(settings, dict):
                sequencer_settings = AnalogSequencerSettings.from_dict(settings)
            else:
                sequencer_settings = settings
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            self._configure_sequencer_settings(seq_idx=seq_idx, settings=sequencer_settings)
            self._configure_nco_mixer_calibration(seq_idx=seq_idx, settings=sequencer_settings)

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings) -> None:
        assert isinstance(settings, AnalogSequencerSettings)
        self._set_parameter(self.instrument.sequencers[seq_idx], "mod_en_awg", settings.nco_en)
        super()._configure_sequencer_settings(seq_idx, settings)

    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.

        """
        assert isinstance(settings, AnalogModuleSettings)
        # configure mixer correction offsets
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(self.instrument, "out0_offset", settings.offset_ch0_path_I)
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(self.instrument, "out1_offset", settings.offset_ch0_path_Q)
        if settings.offset_ch1_path_I is not None:
            self._set_parameter(self.instrument, "out2_offset", settings.offset_ch1_path_I)
        if settings.offset_ch1_path_Q is not None:
            self._set_parameter(self.instrument, "out3_offset", settings.offset_ch1_path_Q)

        for output, dc_settings in enumerate(
            settings.distortion_corrections[: self._hardware_properties.number_of_output_channels]
        ):
            for i in range(4):
                if getattr(dc_settings, f"exp{i}").coeffs is not None:
                    self._set_parameter(
                        self.instrument,
                        f"out{output}_exp{i}_time_constant",
                        getattr(dc_settings, f"exp{i}").coeffs[0],
                    )
                    self._set_parameter(
                        self.instrument,
                        f"out{output}_exp{i}_amplitude",
                        getattr(dc_settings, f"exp{i}").coeffs[1],
                    )
                self._set_parameter(
                    self.instrument,
                    f"out{output}_exp{i}_config",
                    getattr(dc_settings, f"exp{i}").config.value,
                )
                self._set_parameter(
                    self.instrument,
                    f"marker{output}_exp{i}_config",
                    getattr(dc_settings, f"exp{i}").marker_delay.value,
                )
            if dc_settings.fir.coeffs is not None:
                self._set_parameter(
                    self.instrument, f"out{output}_fir_coeffs", dc_settings.fir.coeffs
                )
            self._set_parameter(
                self.instrument, f"out{output}_fir_config", dc_settings.fir.config.value
            )
            self._set_parameter(
                self.instrument,
                f"marker{output}_fir_config",
                dc_settings.fir.marker_delay.value,
            )


class _AnalogReadoutComponent(_AnalogModuleComponent):
    """Qblox InstrumentCoordinator readout component base class."""

    def __init__(self, instrument: Module) -> None:
        if not (instrument.is_qrm_type or instrument.is_qrc_type):
            raise TypeError(
                f"Trying to create _AnalogReadoutComponent from non-QRM or QRC instrument "
                f'of type "{type(instrument)}".'
            )
        super().__init__(instrument)

        self._acquisition_manager: _QRMAcquisitionManager | None = None
        """Holds all the acquisition related logic."""

    def retrieve_acquisition(self) -> Dataset | None:
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

    def prepare(
        self, program: dict[str, dict], acq_channels_data: AcquisitionChannelsData, repetitions: int
    ) -> None:
        """
        Uploads the waveforms and programs to the sequencers.

        All the settings that are required are configured. Keep in mind that
        values set directly through the driver may be overridden (e.g. the
        offsets will be set according to the specified mixer calibration
        parameters).

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        acq_channels_data
            Acquisition channels data.
        repetitions
            Repetitions of the schedule.

        """
        super().prepare(program, acq_channels_data, repetitions)

        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            self._set_parameter(self.instrument.sequencers[seq_idx], "sync_en", False)

        acq_duration = {}
        for seq_name, settings in program["sequencers"].items():
            if isinstance(settings, dict):
                sequencer_settings = AnalogSequencerSettings.from_dict(settings)
            else:
                sequencer_settings = settings
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            self._configure_sequencer_settings(seq_idx=seq_idx, settings=sequencer_settings)
            acq_duration[seq_name] = sequencer_settings.integration_length_acq

        if (acq_hardware_mapping := program.get("acq_hardware_mapping")) is not None:
            scope_mode_sequencer_and_qblox_acq_index = (
                self._determine_scope_mode_acquisition_sequencer_and_qblox_acq_index(
                    acq_channels_data, acq_hardware_mapping
                )
            )
            self._acquisition_manager = _QRMAcquisitionManager(
                parent=self,
                acq_channels_data=acq_channels_data,
                acq_hardware_mapping=acq_hardware_mapping,
                scope_mode_sequencer_and_qblox_acq_index=scope_mode_sequencer_and_qblox_acq_index,
                acquisition_duration=acq_duration,
                seq_name_to_idx_map=self._seq_name_to_idx_map,
                sequencers=program.get("sequencers"),
                repetitions=repetitions,
            )
        else:
            self._acquisition_manager = None

        if (settings := program.get("settings")) is not None:
            if isinstance(settings, dict):
                settings = self._hardware_properties.settings_type.from_dict(settings)
            self._configure_global_settings(settings)

        for path in range(self._hardware_properties.number_of_scope_acq_channels):
            self._set_parameter(self.instrument, f"scope_acq_trigger_mode_path{path}", "sequencer")
            self._set_parameter(self.instrument, f"scope_acq_avg_mode_en_path{path}", True)

    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.

        """
        # configure mixer correction offsets
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(self.instrument, "out0_offset", settings.offset_ch0_path_I)
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(self.instrument, "out1_offset", settings.offset_ch0_path_Q)
        # configure gain
        if settings.in0_gain is not None:
            self._set_parameter(self.instrument, "in0_gain", settings.in0_gain)
        if settings.in1_gain is not None:
            self._set_parameter(self.instrument, "in1_gain", settings.in1_gain)

        for output, dc_settings in enumerate(
            settings.distortion_corrections[: self._hardware_properties.number_of_output_channels]
        ):
            for i in range(4):
                self._set_parameter(
                    self.instrument,
                    f"out{output}_exp{i}_config",
                    getattr(dc_settings, f"exp{i}").config.value,
                )
            self._set_parameter(
                self.instrument, f"out{output}_fir_config", dc_settings.fir.config.value
            )

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings) -> None:
        assert isinstance(settings, AnalogSequencerSettings)
        super()._configure_sequencer_settings(seq_idx, settings)

        if settings.integration_length_acq is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "integration_length_acq",
                settings.integration_length_acq,
            )

        if settings.ttl_acq_auto_bin_incr_en is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "ttl_acq_auto_bin_incr_en",
                settings.ttl_acq_auto_bin_incr_en,
            )
        if settings.ttl_acq_threshold is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "ttl_acq_threshold",
                settings.ttl_acq_threshold,
            )
        if settings.ttl_acq_input_select is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "ttl_acq_input_select",
                settings.ttl_acq_input_select,
            )
        if settings.thresholded_acq_rotation is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "thresholded_acq_rotation",
                settings.thresholded_acq_rotation,
            )
        if settings.thresholded_acq_threshold is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "thresholded_acq_threshold",
                settings.thresholded_acq_threshold,
            )
        if settings.thresholded_acq_trigger_write_address is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "thresholded_acq_trigger_address",
                settings.thresholded_acq_trigger_write_address,
            )
        if settings.thresholded_acq_trigger_write_en is not None:
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "thresholded_acq_trigger_en",
                settings.thresholded_acq_trigger_write_en,
            )
            self._set_parameter(
                self.instrument.sequencers[seq_idx],
                "thresholded_acq_trigger_invert",
                settings.thresholded_acq_trigger_write_invert,
            )

    def _determine_channel_map_parameters(
        self, settings: AnalogSequencerSettings
    ) -> dict[str, str]:
        """Returns a dictionary with the channel map parameters for this module."""
        channel_map_parameters = {}
        self._determine_output_channel_map_parameters(settings, channel_map_parameters)
        self._determine_input_channel_map_parameters(settings, channel_map_parameters)

        return channel_map_parameters

    def _determine_input_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: dict[str, str]
    ) -> dict[str, str]:
        """Adds the inputs to the channel map parameters dict."""
        param_name = {0: "connect_acq_I", 1: "connect_acq_Q"}

        for channel_idx in range(self._hardware_properties.number_of_input_channels):
            if (
                len(settings.connected_input_indices) > 0
                and channel_idx in settings.connected_input_indices
            ):  # For baseband, input indices map 1-to-1 to channel map indices
                param_setting = f"in{channel_idx}"
            else:
                param_setting = "off"

            channel_map_parameters[param_name[channel_idx]] = param_setting

        return channel_map_parameters

    def _determine_scope_mode_acquisition_sequencer_and_qblox_acq_index(
        self,
        acq_channels_data: AcquisitionChannelsData,
        acq_hardware_mapping: dict[str, dict[Hashable, QbloxAcquisitionHardwareMapping]],
    ) -> tuple[int, int] | None:
        """
        Finds the sequencer and qblox_acq_index that performs the raw trace acquisition.

        Raises an error if multiple scope mode acquisitions are present per sequencer.
        Note, that compiler ensures there is at most one scope mode acquisition,
        however the user is able to freely modify the compiler program,
        so we make sure this requirement is still satisfied. See
        :func:`~quantify_scheduler.backends.qblox.analog.AnalogModuleCompiler._ensure_single_scope_mode_acquisition_sequencer`.

        Returns
        -------
        :
            The sequencer and qblox_acq_channel for the trace acquisition, if there is any,
            otherwise None.

        """
        sequencer_and_qblox_acq_index = None
        for acq_channel, acq_channel_data in acq_channels_data.items():
            if acq_channel_data.protocol == "Trace":
                for sequencer_name, sequencer_hardware_mapping in acq_hardware_mapping.items():
                    # It's in the format "seq{n}", so we cut it.
                    sequencer_id = self._seq_name_to_idx_map[sequencer_name]
                    for (
                        hardware_mapping_acq_channel,
                        qblox_acq_index,
                    ) in sequencer_hardware_mapping.items():
                        if acq_channel == hardware_mapping_acq_channel:
                            assert isinstance(qblox_acq_index, int)
                            if (
                                sequencer_and_qblox_acq_index is not None
                                and sequencer_and_qblox_acq_index[0] != sequencer_id
                            ):
                                single_scope_mode_acquisition_raise(
                                    sequencer_0=sequencer_id,
                                    sequencer_1=sequencer_and_qblox_acq_index[0],
                                    module_name=self.name,
                                )
                            sequencer_and_qblox_acq_index = sequencer_id, qblox_acq_index

        return sequencer_and_qblox_acq_index

    def clear_data(self) -> None:
        """Clears remaining data on the module. Module type specific function."""
        if self._acquisition_manager:
            self._acquisition_manager.delete_acquisition_data()


class _QRMComponent(_AnalogReadoutComponent):
    """QRM specific InstrumentCoordinator component."""

    _hardware_properties = _QRM_BASEBAND_PROPERTIES

    def prepare(
        self, program: dict[str, dict], acq_channels_data: AcquisitionChannelsData, repetitions: int
    ) -> None:
        """
        Uploads the waveforms and programs to the sequencers.

        All the settings that are required are configured. Keep in mind that
        values set directly through the driver may be overridden (e.g. the
        offsets will be set according to the specified mixer calibration
        parameters).

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        acq_channels_data
            Acquisition channels data.
        repetitions
            Repetitions of the schedule.

        """
        super().prepare(program, acq_channels_data, repetitions)

        for seq_name, settings in program["sequencers"].items():
            if isinstance(settings, dict):
                sequencer_settings = AnalogSequencerSettings.from_dict(settings)
            else:
                sequencer_settings = settings
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            self._configure_nco_mixer_calibration(seq_idx=seq_idx, settings=sequencer_settings)
        if (acq_hardware_mapping := program.get("acq_hardware_mapping")) is not None:
            scope_mode_sequencer_and_qblox_acq_index = (
                self._determine_scope_mode_acquisition_sequencer_and_qblox_acq_index(
                    acq_channels_data, acq_hardware_mapping
                )
            )
            if scope_mode_sequencer_and_qblox_acq_index is not None:
                self._set_parameter(
                    self.instrument,
                    "scope_acq_sequencer_select",
                    scope_mode_sequencer_and_qblox_acq_index[0],
                )

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings) -> None:
        assert isinstance(settings, AnalogSequencerSettings)
        self._set_parameter(self.instrument.sequencers[seq_idx], "mod_en_awg", settings.nco_en)
        self._set_parameter(self.instrument.sequencers[seq_idx], "demod_en_acq", settings.nco_en)
        super()._configure_sequencer_settings(seq_idx, settings)


class _RFComponent(_AnalogModuleComponent):
    """Mix-in for RF-module-specific InstrumentCoordinatorComponent behaviour."""

    def prepare(
        self, program: dict[str, dict], acq_channels_data: AcquisitionChannelsData, repetitions: int
    ) -> None:
        """
        Uploads the waveforms and programs to the sequencers.

        Overrides the parent method to additionally set LO settings for automatic mixer
        calibration. This must be done _after_ all NCO frequencies have been set.

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        acq_channels_data
            Acquisition channels data.
        repetitions
            Repetitions of the schedule.

        """
        super().prepare(program, acq_channels_data, repetitions)

        lo_idx_to_connected_seq_idx: dict[int, list[int]] = {}
        for seq_name, settings in program["sequencers"].items():
            if isinstance(settings, dict):
                sequencer_settings = AnalogSequencerSettings.from_dict(settings)
            else:
                sequencer_settings = settings
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            for lo_idx in self._get_connected_lo_idx_for_sequencer(
                sequencer_settings=sequencer_settings
            ):
                if lo_idx not in lo_idx_to_connected_seq_idx:
                    lo_idx_to_connected_seq_idx[lo_idx] = []
                lo_idx_to_connected_seq_idx[lo_idx].append(seq_idx)

        if (settings := program.get("settings")) is not None:
            if isinstance(settings, dict):
                settings = self._hardware_properties.settings_type.from_dict(settings)
            assert isinstance(settings, RFModuleSettings)
            self._configure_lo_settings(
                settings=settings,
                lo_idx_to_connected_seq_idx=lo_idx_to_connected_seq_idx,
            )

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings) -> None:
        super()._configure_sequencer_settings(seq_idx, settings)
        # Always set override to False.
        self._set_parameter(
            self.instrument.sequencers[seq_idx],
            "marker_ovr_en",
            False,
        )

    def _determine_output_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: dict[str, str]
    ) -> dict[str, str]:
        """Adds the outputs to the channel map parameters dict."""
        expected_output_indices = {
            i: (2 * i, 2 * i + 1)
            for i in range(self._hardware_properties.number_of_output_channels)
        }

        for channel_idx in range(self._hardware_properties.number_of_output_channels):
            param_setting = "off"
            if (
                ChannelMode.DIGITAL not in settings.channel_name
                and len(settings.connected_output_indices) > 0
                and tuple(settings.connected_output_indices)
                == tuple(expected_output_indices[channel_idx])
            ):
                param_setting = "IQ"

            channel_map_parameters[f"connect_out{channel_idx}"] = param_setting
        return channel_map_parameters

    def _get_connected_lo_idx_for_sequencer(
        self, sequencer_settings: AnalogSequencerSettings
    ) -> list[int]:
        """
        Looks at the connected _output_ ports of the sequencer (if any) to determine
        which LO this sequencer's output is coupled to.
        """
        connected_lo_idx = []
        channel_map_parameters = self._determine_output_channel_map_parameters(
            sequencer_settings, channel_map_parameters={}
        )
        for channel_idx in range(self._hardware_properties.number_of_output_channels):
            if channel_map_parameters.get(f"connect_out{channel_idx}") == "IQ":
                connected_lo_idx.append(channel_idx)
        return connected_lo_idx

    @abstractmethod
    def _configure_lo_settings(
        self,
        settings: RFModuleSettings,
        lo_idx_to_connected_seq_idx: dict[int, list[int]],
    ) -> None:
        """Configure the settings for LO frequency and automatic mixer calibration."""


class _QCMRFComponent(_RFComponent, _QCMComponent):
    """QCM-RF specific InstrumentCoordinator component."""

    _hardware_properties = _QCM_RF_PROPERTIES

    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.

        """
        assert isinstance(settings, RFModuleSettings)
        # configure mixer correction offsets
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(self.instrument, "out0_offset_path0", settings.offset_ch0_path_I)
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(self.instrument, "out0_offset_path1", settings.offset_ch0_path_Q)
        if settings.offset_ch1_path_I is not None:
            self._set_parameter(self.instrument, "out1_offset_path0", settings.offset_ch1_path_I)
        if settings.offset_ch1_path_Q is not None:
            self._set_parameter(self.instrument, "out1_offset_path1", settings.offset_ch1_path_Q)
        # configure attenuation
        if settings.out0_att is not None:
            self._set_parameter(self.instrument, "out0_att", settings.out0_att)
        if settings.out1_att is not None:
            self._set_parameter(self.instrument, "out1_att", settings.out1_att)

    def _configure_lo_settings(
        self,
        settings: RFModuleSettings,
        lo_idx_to_connected_seq_idx: dict[int, list[int]],
    ) -> None:
        """Configure the settings for LO frequency and automatic mixer calibration."""
        lo0_freq_changed = False
        lo1_freq_changed = False
        if settings.lo0_freq is not None:
            lo0_freq_changed = not parameter_value_same_as_cache(
                self.instrument, "out0_lo_freq", settings.lo0_freq
            )
            self._set_parameter(self.instrument, "out0_lo_freq", settings.lo0_freq)
        if settings.lo1_freq is not None:
            lo1_freq_changed = not parameter_value_same_as_cache(
                self.instrument, "out1_lo_freq", settings.lo1_freq
            )
            self._set_parameter(self.instrument, "out1_lo_freq", settings.lo1_freq)

        any_nco_frequencies_changed_lo0 = any(
            self._nco_frequency_changed[seq_idx]
            for seq_idx in lo_idx_to_connected_seq_idx.get(0, [])
        )
        any_nco_frequencies_changed_lo1 = any(
            self._nco_frequency_changed[seq_idx]
            for seq_idx in lo_idx_to_connected_seq_idx.get(1, [])
        )
        if (
            settings.out0_lo_freq_cal_type_default == LoCalEnum.ON_LO_FREQ_CHANGE
            and lo0_freq_changed
        ) or (
            settings.out0_lo_freq_cal_type_default == LoCalEnum.ON_LO_INTERM_FREQ_CHANGE
            and (lo0_freq_changed or any_nco_frequencies_changed_lo0)
        ):
            self.instrument.out0_lo_cal()
        if (
            settings.out1_lo_freq_cal_type_default == LoCalEnum.ON_LO_FREQ_CHANGE
            and lo1_freq_changed
        ) or (
            settings.out1_lo_freq_cal_type_default == LoCalEnum.ON_LO_INTERM_FREQ_CHANGE
            and (lo1_freq_changed or any_nco_frequencies_changed_lo1)
        ):
            self.instrument.out1_lo_cal()


class _QRMRFComponent(_RFComponent, _QRMComponent):
    """QRM-RF specific InstrumentCoordinator component."""

    _hardware_properties = _QRM_RF_PROPERTIES

    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.

        """
        assert isinstance(settings, RFModuleSettings)
        # configure mixer correction offsets
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(self.instrument, "out0_offset_path0", settings.offset_ch0_path_I)
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(self.instrument, "out0_offset_path1", settings.offset_ch0_path_Q)
        # configure attenuation
        if settings.out0_att is not None:
            self._set_parameter(self.instrument, "out0_att", settings.out0_att)
        if settings.in0_att is not None:
            self._set_parameter(self.instrument, "in0_att", settings.in0_att)

    def _configure_lo_settings(
        self,
        settings: RFModuleSettings,
        lo_idx_to_connected_seq_idx: dict[int, list[int]],
    ) -> None:
        """Configure the settings for LO frequency and automatic mixer calibration."""
        lo0_freq_changed = False
        if settings.lo0_freq is not None:
            lo0_freq_changed = not parameter_value_same_as_cache(
                self.instrument, "out0_in0_lo_freq", settings.lo0_freq
            )
            self._set_parameter(self.instrument, "out0_in0_lo_freq", settings.lo0_freq)

        any_nco_frequencies_changed_lo0 = any(
            self._nco_frequency_changed[seq_idx]
            for seq_idx in lo_idx_to_connected_seq_idx.get(0, [])
        )
        if (
            settings.out0_lo_freq_cal_type_default == LoCalEnum.ON_LO_FREQ_CHANGE
            and lo0_freq_changed
        ) or (
            settings.out0_lo_freq_cal_type_default == LoCalEnum.ON_LO_INTERM_FREQ_CHANGE
            and (lo0_freq_changed or any_nco_frequencies_changed_lo0)
        ):
            self.instrument.out0_in0_lo_cal()

    def _determine_input_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: dict[str, str]
    ) -> dict[str, str]:
        """Adds the inputs to the channel map parameters dict."""
        channel_map_parameters["connect_acq"] = (
            "in0" if tuple(settings.connected_input_indices) == (0, 1) else "off"
        )

        return channel_map_parameters


class _QRCComponent(_RFComponent, _AnalogReadoutComponent):
    """QRC specific InstrumentCoordinator component."""

    _hardware_properties = _QRC_PROPERTIES

    def prepare(
        self, program: dict[str, dict], acq_channels_data: AcquisitionChannelsData, repetitions: int
    ) -> None:
        """
        Uploads the waveforms and programs to the sequencers.

        All the settings that are required are configured. Keep in mind that
        values set directly through the driver may be overridden (e.g. the
        offsets will be set according to the specified mixer calibration
        parameters).

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        acq_channels_data
            Acquisition channels data.
        repetitions
            Repetitions of the schedule.

        """
        super().prepare(program, acq_channels_data, repetitions)
        if (acq_hardware_mapping := program.get("acq_hardware_mapping")) is not None:
            scope_mode_sequencer_and_qblox_acq_index = (
                self._determine_scope_mode_acquisition_sequencer_and_qblox_acq_index(
                    acq_channels_data, acq_hardware_mapping
                )
            )
            if scope_mode_sequencer_and_qblox_acq_index is not None:
                scope_mode_sequencer_and_qblox_acq_index = scope_mode_sequencer_and_qblox_acq_index[
                    0
                ]
                # Note, for QRC sometime in the future we might be able to set
                # different sequencers per scope paths; this is why
                # for the beta product we handle QRC differently.
                self._set_parameter(
                    self.instrument,
                    "scope_acq_sequencer_select",
                    scope_mode_sequencer_and_qblox_acq_index,
                )

    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.

        """
        # configure attenuation

        for i in range(6):
            if out_att := getattr(settings, f"out{i}_att") is not None:
                self._set_parameter(self.instrument, f"out{i}_att", out_att)

    def _configure_lo_settings(
        self,
        settings: RFModuleSettings,
        lo_idx_to_connected_seq_idx: dict[int, list[int]],  #  noqa: ARG002
    ) -> None:
        """Configure the settings for the frequency."""
        # For QRC, there are no LO frequencies, only output frequencies.

        for i in range(2):
            if (freq := getattr(settings, f"in{i}_freq")) is not None:
                self._set_parameter(self.instrument, f"in{i}_freq", freq)

        # For the beta QRC product, we do not have a 0, 1 output channel frequencies.
        for i in range(2, 6):
            if (freq := getattr(settings, f"lo{i}_freq")) is not None:
                self._set_parameter(self.instrument, f"out{i}_freq", freq)

    def _determine_input_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: dict[str, str]
    ) -> dict[str, str]:
        """Adds the inputs to the channel map parameters dict."""
        if tuple(settings.connected_input_indices) == (0, 1):
            channel_map_parameters["connect_acq"] = "in0"
        elif tuple(settings.connected_input_indices) == (2, 3):
            channel_map_parameters["connect_acq"] = "in1"
        else:
            channel_map_parameters["connect_acq"] = "off"

        return channel_map_parameters

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings) -> None:
        assert isinstance(settings, AnalogSequencerSettings)
        if not settings.nco_en:
            raise ValueError(f"QRC module '{self.name}' does not support operating without NCO.")
        super()._configure_sequencer_settings(seq_idx, settings)


class _QTMComponent(_ModuleComponentBase):
    """QTM specific InstrumentCoordinator component."""

    _hardware_properties = _QTM_PROPERTIES

    def __init__(self, instrument: Module) -> None:
        if not instrument.is_qtm_type:
            raise TypeError(
                f"Trying to create _QTMComponent from non-QTM instrument "
                f'of type "{type(instrument)}".'
            )
        super().__init__(instrument)

        self._acquisition_manager: _QTMAcquisitionManager | None = None
        """Holds all the acquisition related logic."""

    def retrieve_acquisition(self) -> Dataset | None:
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

    def prepare(
        self, program: dict[str, dict], acq_channels_data: AcquisitionChannelsData, repetitions: int
    ) -> None:
        """
        Uploads the waveforms and programs to the sequencers.

        All the settings that are required are configured. Keep in mind that
        values set directly through the driver may be overridden (e.g. the
        offsets will be set according to the specified mixer calibration
        parameters).

        Parameters
        ----------
        program
            Program to upload to the sequencers.
            Under the key :code:`"sequencer"` you specify the sequencer specific
            options for each sequencer, e.g. :code:`"seq0"`.
            For global settings, the options are under different keys, e.g. :code:`"settings"`.
        acq_channels_data
            Acquisition channels data.
        repetitions
            Repetitions of the schedule.

        """
        super().prepare(program, acq_channels_data, repetitions)

        for seq_idx in range(self._hardware_properties.number_of_sequencers):
            self._set_parameter(self.instrument.sequencers[seq_idx], "sync_en", False)

        trace_acq_duration = {}
        for seq_name, settings in program["sequencers"].items():
            if seq_name in self._seq_name_to_idx_map:
                seq_idx = self._seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer "
                    f'with name "{seq_name}".'
                )

            # 1-1 Sequencer-io_channel coupling,
            # the io_channel settings are inside SequencerSettings
            self._configure_sequencer_settings(seq_idx=seq_idx, settings=settings)
            self._configure_io_channel_settings(seq_idx=seq_idx, settings=settings)
            trace_acq_duration[seq_name] = settings.trace_acq_duration

        if (acq_hardware_mapping := program.get("acq_hardware_mapping")) is not None:
            self._acquisition_manager = _QTMAcquisitionManager(
                parent=self,
                # Ignoring type, because this cannot be solved by a simple assert isinstance.
                acq_channels_data=acq_channels_data,  # type; ignore
                acq_hardware_mapping=acq_hardware_mapping,
                acquisition_duration=trace_acq_duration,
                seq_name_to_idx_map=self._seq_name_to_idx_map,
                repetitions=repetitions,
            )
        else:
            self._acquisition_manager = None

        if (settings := program.get("settings")) is not None:
            assert isinstance(settings, TimetagModuleSettings)
            self._configure_global_settings(settings)

    def _configure_global_settings(self, settings: BaseModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.

        """
        # No global settings yet.

    def _configure_io_channel_settings(
        self, seq_idx: int, settings: TimetagSequencerSettings
    ) -> None:
        """
        Configures all io_channel-specific settings.

        Parameters
        ----------
        seq_idx
            Index of the sequencer to configure.
        settings
            The settings to configure it to.

        """
        # Note: there is no channel mapping in QTM firmware V1 (meaning, each sequencer
        # only connects to its corresponding io channel). The contents of
        # `connected_input_indices` and `connected_output_indices` are validated in
        # `TimetagSequencerSettings`. The code below already assumes there is a channel
        # mapping and does no further validation.
        for channel_idx in settings.connected_input_indices:
            self._set_parameter(
                self.instrument.io_channels[channel_idx],
                "mode",
                "input",
            )
        for channel_idx in settings.connected_output_indices:
            self._set_parameter(
                self.instrument.io_channels[channel_idx],
                "mode",
                "output",
            )

        if settings.analog_threshold is not None:
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "analog_threshold",
                settings.analog_threshold,
            )
        self._set_parameter(
            self.instrument.io_channels[seq_idx],
            "forward_trigger_en",
            False,
        )

        self._set_parameter(
            self.instrument.io_channels[seq_idx],
            "binned_acq_on_invalid_time_delta",
            "record_0",
        )

        if settings.scope_trace_type == TimetagTraceType.SCOPE:
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "scope_trigger_mode",
                "sequencer",
            )
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "scope_mode",
                "scope",
            )
        elif settings.scope_trace_type == TimetagTraceType.TIMETAG:
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "scope_trigger_mode",
                "sequencer",
            )
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "scope_mode",
                "timetags-windowed",
            )
        if settings.time_source is not None:
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "binned_acq_time_source",
                str(settings.time_source),
            )
        if settings.time_ref is not None:
            if settings.time_ref == TimeRef.TIMESTAMP:
                time_ref = "sequencer"
            elif settings.time_ref == TimeRef.PORT:
                time_ref = f"first{settings.time_ref_channel}"
            else:
                time_ref = str(settings.time_ref)
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "binned_acq_time_ref",
                time_ref,
            )

        self._set_parameter(
            self.instrument.io_channels[seq_idx],
            "thresholded_acq_trigger_address_high",
            settings.thresholded_acq_trigger_write_address_high,
        )
        self._set_parameter(
            self.instrument.io_channels[seq_idx],
            "thresholded_acq_trigger_address_mid",
            settings.thresholded_acq_trigger_write_address_mid,
        )
        self._set_parameter(
            self.instrument.io_channels[seq_idx],
            "thresholded_acq_trigger_address_low",
            settings.thresholded_acq_trigger_write_address_low,
        )
        self._set_parameter(
            self.instrument.io_channels[seq_idx],
            "thresholded_acq_trigger_address_invalid",
            settings.thresholded_acq_trigger_write_address_invalid,
        )
        if settings.thresholded_acq_trigger_write_en is not None:
            self._set_parameter(
                self.instrument.io_channels[seq_idx],
                "thresholded_acq_trigger_en",
                settings.thresholded_acq_trigger_write_en,
            )

    def clear_data(self) -> None:
        """Clears remaining data on the module. Module type specific function."""
        if self._acquisition_manager:
            self._acquisition_manager.delete_acquisition_data()


_ReadoutModuleComponentT = Union[_AnalogReadoutComponent, _QTMComponent]


class _AcquisitionManagerBase(ABC):
    """
    Utility class that handles the acquisitions performed with a module.

    An instance of this class is meant to exist only for a single prepare-start-
    retrieve_acquisition cycle to prevent stateful behavior.

    Parameters
    ----------
    parent
        Reference to the parent QRM IC component.
    acq_channels_data
        Provides a summary of the used acquisition protocol, bin mode, acquisition channels,
        acquisition indices per channel, and repetitions.
    acq_hardware_mapping
        Acquisition hardware mapping.
    acquisition_duration
        The duration of each acquisition for each sequencer.
    seq_name_to_idx_map
        All available sequencer names to their ids in a dict.
    repetitions
        How many times the schedule repeats.

    """

    def __init__(
        self,
        parent: _ReadoutModuleComponentT,
        acq_channels_data: AcquisitionChannelsData,
        acq_hardware_mapping: dict[str, dict[Hashable, QbloxAcquisitionHardwareMapping]],
        acquisition_duration: dict[str, int],
        seq_name_to_idx_map: dict[str, int],
        repetitions: int,
    ) -> None:
        self.parent = parent
        self._acq_channels_data = acq_channels_data
        self._acq_hardware_mapping = acq_hardware_mapping

        self._acq_duration = acquisition_duration
        self._seq_name_to_idx_map = seq_name_to_idx_map
        self._repetitions = repetitions

    @property
    def instrument(self) -> Module:
        """Returns the QRM driver from the parent IC component."""
        return self.parent.instrument

    @staticmethod
    @abstractmethod
    def _check_bin_mode_compatible(
        acq_channels_data: AcquisitionChannelsData,
        acq_hardware_mapping: dict[str, dict[Hashable, QbloxAcquisitionHardwareMapping]],
    ) -> None:
        pass

    @property
    @abstractmethod
    def _protocol_to_acq_function_map(self) -> dict[str, Callable]:
        """
        Mapping from acquisition protocol name to the function that processes the raw
        acquisition data.
        """

    def retrieve_acquisition(self) -> Dataset:
        """
        Retrieves all the acquisition data in the correct format.

        Returns
        -------
        :
            The acquisitions with the protocols specified in the `acquisition_metadata`.
            Each `xarray.DataArray` in the `xarray.Dataset` corresponds to one `acq_channel`.
            The ``acq_channel`` is the name of each `xarray.DataArray` in the `xarray.Dataset`.
            Each `xarray.DataArray` is a two-dimensional array, with ``acq_index`` and
            Each `xarray.DataArray` is a two-dimensional array,
            with ``acq_index`` and ``repetition`` as dimensions.

        """
        dataset = Dataset()
        self._check_bin_mode_compatible(self._acq_channels_data, self._acq_hardware_mapping)

        for sequencer_name, seq_hardware_mapping in self._acq_hardware_mapping.items():
            # Retrieve the raw data from the readout sequencer.
            hardware_retrieved_acquisitions = self.instrument.get_acquisitions(
                self._seq_name_to_idx_map[sequencer_name]
            )
            for acq_channel, seq_channel_hardware_mapping in seq_hardware_mapping.items():
                acquisition_function: Callable = self._protocol_to_acq_function_map[
                    self._acq_channels_data[acq_channel].protocol
                ]
                # Verifying whether returned data contains the Qblox acquisition indices properly.
                if isinstance(seq_channel_hardware_mapping, QbloxAcquisitionIndex):
                    qblox_acq_index = seq_channel_hardware_mapping
                    self._assert_acquisition_data_exists(
                        hardware_retrieved_acquisitions, qblox_acq_index, acq_channel
                    )
                else:
                    for qblox_acq_index_bin in seq_channel_hardware_mapping.values():
                        self._assert_acquisition_data_exists(
                            hardware_retrieved_acquisitions, qblox_acq_index_bin.index, acq_channel
                        )
                # The `acquisition_function` formats the the acquisitions
                # returned by the readout sequencer.
                formatted_acquisitions = acquisition_function(
                    hardware_retrieved_acquisitions=hardware_retrieved_acquisitions,
                    acq_channel=acq_channel,
                    seq_channel_hardware_mapping=seq_channel_hardware_mapping,
                    acq_duration=self._acq_duration[sequencer_name],
                    sequencer_name=sequencer_name,
                )
                formatted_acquisitions_dataset = Dataset({acq_channel: formatted_acquisitions})

                check_already_existing_acquisition(
                    new_dataset=formatted_acquisitions_dataset, current_dataset=dataset
                )
                dataset = dataset.merge(formatted_acquisitions_dataset)

        return dataset

    def delete_acquisition_data(self) -> None:
        """
        Delete acquisition data from sequencers that have associated hardware acquisition mapping.

        To be called before starting the sequencers, so that old data does not get retrieved more
        than once.
        """
        for seq_name in self._acq_hardware_mapping:
            self.instrument.delete_acquisition_data(
                sequencer=self._seq_name_to_idx_map[seq_name], all=True
            )

    def _assert_acquisition_data_exists(
        self,
        hardware_retrieved_acquisitions: dict,
        qblox_acq_index: int,
        acq_channel: Hashable,
    ) -> None:
        """Assert that the qblox_acq_index is in the acquisition data."""
        qblox_acq_name = self._qblox_acq_index_to_qblox_acq_name(qblox_acq_index)
        if qblox_acq_name not in hardware_retrieved_acquisitions:
            raise KeyError(
                f"The acquisition data retrieved from the hardware does not contain "
                f"data for acquisition channel {acq_channel} (referred to by Qblox "
                f"acquisition index {qblox_acq_index}).\n"
                f"{hardware_retrieved_acquisitions=}"
            )

    @staticmethod
    def _acq_channel_attrs(
        protocol: str,
    ) -> dict:
        return {"acq_protocol": protocol}

    @classmethod
    def _get_bin_data(cls, hardware_retrieved_acquisitions: dict, qblox_acq_index: int = 0) -> dict:
        """Returns the bin entry of the acquisition data dict."""
        qblox_acq_name = cls._qblox_acq_index_to_qblox_acq_name(qblox_acq_index)
        channel_data = hardware_retrieved_acquisitions[qblox_acq_name]
        if channel_data["index"] != qblox_acq_index:
            raise RuntimeError(
                f"Name does not correspond to a valid acquisition for name {qblox_acq_name}, "
                f"which has index {channel_data['index']}."
            )
        return channel_data["acquisition"]["bins"]

    @staticmethod
    def _qblox_acq_index_to_qblox_acq_name(qblox_acq_index: int) -> str:
        """Returns the name of the acquisition from the qblox_acq_index."""
        return str(qblox_acq_index)

    def _get_binned_data(
        self,
        bin_data_function: Callable[[int, int, ThresholdedTriggerCountMetadata | None], Any],
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
    ) -> DataArray:
        """
        Retrieves the binned acquisition data associated with an `acq_channel`.

        Parameters
        ----------
        bin_data_function
            Function that returns the bin data for the Qblox acquisition index and bin.
        acq_channel
            Acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.

        Returns
        -------
        :
            The integrated data.

        """
        # For binned acquisition protocols, the mapping is of QbloxAcquisitionBinMapping type,
        # guaranteed by the QbloxAcquisitionIndexManager, and coords is a list.
        assert isinstance(seq_channel_hardware_mapping, dict)
        coords = self._acq_channels_data[acq_channel].coords
        assert isinstance(coords, list)

        if _is_acquisition_binned_average(
            self._acq_channels_data[acq_channel].protocol,
            self._acq_channels_data[acq_channel].bin_mode,
        ):
            formatted_data = []
            for _acq_index, qblox_acq_index_bin in seq_channel_hardware_mapping.items():
                formatted_data.append(
                    bin_data_function(
                        qblox_acq_index_bin.index,
                        qblox_acq_index_bin.bin,
                        qblox_acq_index_bin.thresholded_trigger_count_metadata,
                    )
                )
            acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
            data_array = DataArray(
                formatted_data,
                dims=[acq_index_dim_name],
                coords={
                    acq_index_dim_name: list(seq_channel_hardware_mapping.keys()),
                },
                attrs=self._acq_channel_attrs(self._acq_channels_data[acq_channel].protocol),
            )
            add_acquisition_coords_binned(data_array, coords, acq_index_dim_name)
            return data_array
        elif _is_acquisition_binned_append(
            self._acq_channels_data[acq_channel].protocol,
            self._acq_channels_data[acq_channel].bin_mode,
        ):
            # TODO: Optimize how this data is extracted.
            # This might be not the most performant way to extract the data from the bins.
            # In the general case multiple loops and repetitions, we need to
            # extract the data not in the most straightforward way.
            formatted_data = []
            for repetition in range(self._repetitions):
                formatted_data_repetition = []
                for _acq_index, qblox_acq_index_bin in seq_channel_hardware_mapping.items():
                    formatted_data_repetition.append(
                        bin_data_function(
                            qblox_acq_index_bin.index,
                            qblox_acq_index_bin.bin + repetition * qblox_acq_index_bin.stride,
                            qblox_acq_index_bin.thresholded_trigger_count_metadata,
                        )
                    )
                formatted_data.append(formatted_data_repetition)
            acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
            data_array = DataArray(
                formatted_data,
                dims=["repetition", acq_index_dim_name],
                coords={
                    "repetition": list(range(self._repetitions)),
                    acq_index_dim_name: list(seq_channel_hardware_mapping.keys()),
                },
                attrs=self._acq_channel_attrs(self._acq_channels_data[acq_channel].protocol),
            )
            add_acquisition_coords_binned(data_array, coords, acq_index_dim_name)
            return data_array
        else:
            # In principle unreachable due to _check_bin_mode_compatible, but included for
            # completeness.
            raise AssertionError("This should not be reachable due to _check_bin_mode_compatible.")

    def _get_trigger_count_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,  # noqa: ARG002, unused argument
        sequencer_name: str,  # noqa: ARG002, unused argument
        sum_multiply_repetitions: bool,
    ) -> DataArray:
        """
        Retrieves the trigger count acquisition data associated with `acq_channel`.

        Parameters
        ----------
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        sequencer_name
            Sequencer.
        sum_multiply_repetitions
            Multiplies data by repetitions for the SUM bin mode.

        Returns
        -------
        data : xarray.DataArray
            The acquired trigger count data.

        Notes
        -----
        - For BinMode.DISTRIBUTION, `data` contains the distribution of counts.
        - For BinMode.APPEND, `data` contains the raw trigger counts.

        """
        if self._acq_channels_data[acq_channel].bin_mode == BinMode.DISTRIBUTION:
            # For TriggerCount distribution acquisition protocol,
            # the mapping is of QbloxAcquisitionIndex type,
            # guaranteed by the QbloxAcquisitionIndexManager.
            assert isinstance(seq_channel_hardware_mapping, int)

            qblox_acq_index = seq_channel_hardware_mapping
            bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)

            def _convert_from_cumulative(
                cumulative_values: list[int],
            ) -> dict[int, int]:
                """
                Return the distribution of counts from a cumulative distribution.

                Note, the cumulative distribution is in reverse order.
                The cumulative_values list can contain any number of integers and NaNs.
                """
                result = {}

                last_cumulative_value = 0
                for count, current_cumulative_value in reversed(list(enumerate(cumulative_values))):
                    if (not isnan(current_cumulative_value)) and (
                        last_cumulative_value != current_cumulative_value
                    ):
                        result[count + 1] = current_cumulative_value - last_cumulative_value
                        last_cumulative_value = current_cumulative_value

                return result

            result = _convert_from_cumulative(bin_data["avg_cnt"])
            acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
            counts = list(result.keys())[::-1]
            coords = self._acq_channels_data[acq_channel].coords
            assert isinstance(coords, dict)
            data_array = DataArray(
                list(result.values())[::-1],
                dims=[acq_index_dim_name],
                coords={
                    acq_index_dim_name: range(len(counts)),
                    f"counts_{acq_channel}": (acq_index_dim_name, counts),
                },
                attrs=self._acq_channel_attrs(self._acq_channels_data[acq_channel].protocol),
            )
            add_acquisition_coords_nonbinned(data_array, coords, acq_index_dim_name)
            return data_array
        elif self._acq_channels_data[acq_channel].bin_mode in (BinMode.SUM, BinMode.APPEND):

            def _get_bin(
                qblox_acq_index: int,
                qblox_acq_bin: int,
                thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,  # noqa: ARG001, unused argument
            ) -> int:
                bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
                # "avg_cnt" is the key for QRM modules, "count" for QTM modules. Note
                # that QTM modules also return a "avg_cnt" key, that should not be used!
                counts = bin_data["count"] if "count" in bin_data else bin_data["avg_cnt"]
                return (
                    self._repetitions * int(counts[qblox_acq_bin])
                    if (
                        sum_multiply_repetitions
                        and self._acq_channels_data[acq_channel].bin_mode == BinMode.SUM
                    )
                    else int(counts[qblox_acq_bin])
                )

            return self._get_binned_data(
                bin_data_function=_get_bin,
                acq_channel=acq_channel,
                seq_channel_hardware_mapping=seq_channel_hardware_mapping,
            )
        else:
            # In principle unreachable due to _check_bin_mode_compatible, but included for
            # completeness.
            raise AssertionError("This should not be reachable due to _check_bin_mode_compatible.")

    def _get_trigger_count_threshold_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,  # noqa: ARG002, unused argument
        sequencer_name: str,  # noqa: ARG002, unused argument
    ) -> DataArray:
        """
        Gets the integration data but normalized to the integration time.

        The return value is thus the amplitude of the demodulated
        signal directly and has volt units (i.e. same units as a single sample of the
        integrated signal).

        Parameters
        ----------
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        sequencer_name
            Sequencer.

        Returns
        -------
        :
            DataArray containing thresholded acquisition data.

        """

        def _get_bin(
            qblox_acq_index: int,
            qblox_acq_bin: int,
            thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,
        ) -> int:
            assert thresholded_trigger_count_metadata is not None
            bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
            # "avg_cnt" is the key for QRM modules, "count" for QTM modules. Note
            # that QTM modules also return a "avg_cnt" key, that should not be used!
            counts = bin_data["count"] if "count" in bin_data else bin_data["avg_cnt"]

            if np.isnan(counts[qblox_acq_bin]):
                return -1
            # To make the return data similar to thresholded acquisition, it is cast
            # to integers. Note that the hardware returns counts as floats.
            elif (
                thresholded_trigger_count_metadata.condition
                == TriggerCondition.GREATER_THAN_EQUAL_TO
            ):
                return int(
                    round(counts[qblox_acq_bin]) >= thresholded_trigger_count_metadata.threshold
                )
            elif thresholded_trigger_count_metadata.condition == TriggerCondition.LESS_THAN:
                return int(
                    round(counts[qblox_acq_bin]) < thresholded_trigger_count_metadata.threshold
                )
            else:
                raise ValueError(
                    f"Unknown trigger condition {thresholded_trigger_count_metadata.condition}"
                )

        return self._get_binned_data(
            bin_data_function=_get_bin,
            acq_channel=acq_channel,
            seq_channel_hardware_mapping=seq_channel_hardware_mapping,
        )


class _QRMAcquisitionManager(_AcquisitionManagerBase):
    """
    Utility class that handles the acquisitions performed with the QRM and QRC.

    An instance of this class is meant to exist only for a single prepare-start-
    retrieve_acquisition cycle to prevent stateful behavior.

    Parameters
    ----------
    parent
        Reference to the parent QRM or QRC IC component.
    acq_channels_data
        Provides a summary of the used acquisition protocol, bin mode, acquisition channels,
        acquisition indices per channel.
    acq_hardware_mapping
        Acquisition hardware mapping.
    acquisition_duration
        The duration of each acquisition for each sequencer.
    seq_name_to_idx_map
        All available sequencer names to their ids in a dict.
    scope_mode_sequencer_and_qblox_acq_index
        The sequencer and qblox acq_index of the scope mode acquisition if there's any.
    sequencers
        Sequencer data.

    """

    def __init__(
        self,
        parent: _AnalogReadoutComponent,
        acq_channels_data: AcquisitionChannelsData,
        acq_hardware_mapping: dict[str, dict[Hashable, QbloxAcquisitionHardwareMapping]],
        acquisition_duration: dict[str, int],
        seq_name_to_idx_map: dict[str, int],
        repetitions: int,
        scope_mode_sequencer_and_qblox_acq_index: tuple[int, int] | None = None,
        sequencers: dict[str, dict] | None = None,
    ) -> None:
        super().__init__(
            parent=parent,
            acq_channels_data=acq_channels_data,
            acq_hardware_mapping=acq_hardware_mapping,
            acquisition_duration=acquisition_duration,
            seq_name_to_idx_map=seq_name_to_idx_map,
            repetitions=repetitions,
        )
        self._scope_mode_sequencer_and_qblox_acq_index = scope_mode_sequencer_and_qblox_acq_index
        self._sequencers = sequencers

    @property
    def _protocol_to_acq_function_map(self) -> dict[str, Callable]:
        return {
            "WeightedIntegratedSeparated": self._get_integration_weighted_separated_data,
            "NumericalSeparatedWeightedIntegration": self._get_integration_weighted_separated_data,
            "NumericalWeightedIntegration": self._get_integration_real_data,
            "SSBIntegrationComplex": self._get_integration_amplitude_data,
            "ThresholdedAcquisition": self._get_threshold_data,
            "WeightedThresholdedAcquisition": self._get_threshold_data,
            "ThresholdedTriggerCount": self._get_trigger_count_threshold_data,
            "Trace": self._get_scope_data,
            "TriggerCount": partial(self._get_trigger_count_data, sum_multiply_repetitions=False),
        }

    @staticmethod
    def _check_bin_mode_compatible(
        acq_channels_data: AcquisitionChannelsData,
        acq_hardware_mapping: dict[str, dict[Hashable, QbloxAcquisitionHardwareMapping]],
    ) -> None:
        for seq_acq_hardware_mapping in acq_hardware_mapping.values():
            for acq_channel in seq_acq_hardware_mapping:
                if (
                    acq_channels_data[acq_channel].bin_mode
                    not in QRM_COMPATIBLE_BIN_MODES[acq_channels_data[acq_channel].protocol]
                ):
                    raise IncompatibleBinModeError(
                        module_type="QRM",
                        protocol=acq_channels_data[acq_channel].protocol,
                        bin_mode=acq_channels_data[acq_channel].bin_mode,
                    )

    def retrieve_acquisition(self) -> Dataset:
        """
        Retrieves all the acquisition data in the correct format.

        Returns
        -------
        :
            The acquisitions with the protocols specified in the `acq_channels_data`.
            Each `xarray.DataArray` in the `xarray.Dataset` corresponds to one `acq_channel`.
            The ``acq_channel`` is the name of each `xarray.DataArray` in the `xarray.Dataset`.
            Each `xarray.DataArray` is a two-dimensional array,
            with ``acq_index`` and ``repetition`` as dimensions.

        """
        self._store_scope_acquisition()
        return super().retrieve_acquisition()

    def _store_scope_acquisition(self) -> None:
        """
        Calls :code:`store_scope_acquisition` function on the Qblox instrument.

        This will ensure that the correct sequencer will store the scope acquisition
        data on the hardware, so it will be filled out when we call :code:`get_acquisitions`
        on the Qblox instrument's sequencer corresponding to the scope acquisition.
        """
        if self._scope_mode_sequencer_and_qblox_acq_index is None:
            return

        sequencer_index = self._scope_mode_sequencer_and_qblox_acq_index[0]

        if sequencer_index not in self._seq_name_to_idx_map.values():
            raise ValueError(
                f"Attempting to retrieve scope mode data from sequencer "
                f"{sequencer_index}. A QRM only has the following sequencer indices: "
                f"{list(self._seq_name_to_idx_map.values())}."
            )
        qblox_acq_index = self._scope_mode_sequencer_and_qblox_acq_index[1]
        qblox_acq_name = self._qblox_acq_index_to_qblox_acq_name(qblox_acq_index)
        self.instrument.store_scope_acquisition(sequencer_index, qblox_acq_name)

    def _get_scope_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,
        sequencer_name: str,
    ) -> DataArray:
        """
        Retrieves the scope mode acquisition associated with an `acq_channel`.

        Parameters
        ----------
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        sequencer_name
            Sequencer.

        Returns
        -------
        :
            The scope mode data.

        """
        if acq_duration < 0 or acq_duration > constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS:
            raise ValueError(
                "Attempting to retrieve sample of size "
                f"{acq_duration}, but only integer values "
                f"0,...,{constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS} "
                f"are allowed."
            )
        # For scope acquisition protocols, the mapping is of QbloxAcquisitionIndex type,
        # guaranteed by the QbloxAcquisitionIndexManager and coords is a dict.
        assert isinstance(seq_channel_hardware_mapping, int)
        coords = self._acq_channels_data[acq_channel].coords
        assert isinstance(coords, dict)

        qblox_acq_index = seq_channel_hardware_mapping
        qblox_acq_name = self._qblox_acq_index_to_qblox_acq_name(qblox_acq_index)
        scope_data = hardware_retrieved_acquisitions[qblox_acq_name]["acquisition"]["scope"]

        path0 = scope_data["path0"]
        path1 = scope_data["path1"]
        if (self._sequencers is not None) and (
            connected_sequencer := self._sequencers.get(sequencer_name)
        ) is not None:
            connected_input_indices = getattr(connected_sequencer, "connected_input_indices", None)
            if connected_input_indices == (2, 3):
                path0 = scope_data["path2"]
                path1 = scope_data["path3"]
            elif connected_input_indices != (0, 1):
                logger.warning(
                    f"Invalid input indices are connected. Scope data might be invalid. "
                    f"Connected input indices are {connected_input_indices}. "
                    f"Valid indices are (0, 1) and (2, 3)."
                )

        if path0["out-of-range"] or path1["out-of-range"]:
            logger.warning(
                f"The scope mode data of {self.parent.name} with "
                f"acq_channel={acq_channel}  was out-of-range."
            )
        # NB hardware already divides by avg_count for scope mode
        scope_data_i = np.array(path0["data"][:acq_duration])
        scope_data_q = np.array(path1["data"][:acq_duration])

        # If using a dummy setup and there's no scope data set, then return NaN's
        if self.parent.instrument.is_dummy and (len(scope_data_i) == len(scope_data_q) == 0):
            scope_data_i = np.asarray([float("nan")] * acq_duration)
            scope_data_q = np.asarray([float("nan")] * acq_duration)

        acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
        trace_index_dim_name = f"trace_index_{acq_channel}"
        data_array = DataArray(
            data=(scope_data_i + scope_data_q * 1j).reshape((1, -1)),
            dims=[acq_index_dim_name, trace_index_dim_name],
            coords={
                acq_index_dim_name: [0],
                trace_index_dim_name: list(range(acq_duration)),
            },
            attrs=self._acq_channel_attrs(self._acq_channels_data[acq_channel].protocol),
        )
        add_acquisition_coords_nonbinned(data_array, coords, acq_index_dim_name)
        return data_array

    def _get_integration_weighted_separated_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,  # noqa: ARG002, unused argument
        sequencer_name: str,  # noqa: ARG002, unused argument
    ) -> DataArray:
        """

        The return value is thus the amplitude of the demodulated
        signal directly and has volt units (i.e. same units as a single sample of the
        integrated signal).

        Parameters
        ----------
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        sequencer_name
            Sequencer.

        Returns
        -------
        :
            Array containing binned, normalized acquisition data.

        """

        def _get_bin(
            qblox_acq_index: int,
            qblox_acq_bin: int,
            thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,  # noqa: ARG001, unused argument
        ) -> complex:
            bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
            i_data = bin_data["integration"]["path0"][qblox_acq_bin]
            q_data = bin_data["integration"]["path1"][qblox_acq_bin]
            return i_data + q_data * 1j

        return self._get_binned_data(
            bin_data_function=_get_bin,
            acq_channel=acq_channel,
            seq_channel_hardware_mapping=seq_channel_hardware_mapping,
        )

    def _get_integration_amplitude_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,
        sequencer_name: str,  # noqa: ARG002, unused argument
    ) -> DataArray:
        """

        The return value is thus the amplitude of the demodulated
        signal directly and has volt units (i.e. same units as a single sample of the
        integrated signal).

        Parameters
        ----------
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        sequencer_name
            Sequencer.

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

        def _get_bin(
            qblox_acq_index: int,
            qblox_acq_bin: int,
            thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,  # noqa: ARG001, unused argument
        ) -> complex:
            bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
            i_data = bin_data["integration"]["path0"][qblox_acq_bin]
            q_data = bin_data["integration"]["path1"][qblox_acq_bin]
            return (i_data + q_data * 1j) / acq_duration

        return self._get_binned_data(
            bin_data_function=_get_bin,
            acq_channel=acq_channel,
            seq_channel_hardware_mapping=seq_channel_hardware_mapping,
        )

    def _get_integration_real_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,  # noqa: ARG002, unused argument
        sequencer_name: str,  # noqa: ARG002, unused argument
    ) -> DataArray:
        """
        Gets the integration data but normalized to the integration time.

        The return value is thus the amplitude of the demodulated
        signal directly and has volt units (i.e. same units as a single sample of the
        integrated signal).

        Parameters
        ----------
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        sequencer_name
            Sequencer.

        Returns
        -------
        :
            Array containing binned, normalized acquisition data.

        """

        def _get_bin(
            qblox_acq_index: int,
            qblox_acq_bin: int,
            thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,  # noqa: ARG001, unused argument
        ) -> complex:
            bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
            i_data = bin_data["integration"]["path0"][qblox_acq_bin]
            q_data = bin_data["integration"]["path1"][qblox_acq_bin]
            return complex(i_data + q_data)

        return self._get_binned_data(
            bin_data_function=_get_bin,
            acq_channel=acq_channel,
            seq_channel_hardware_mapping=seq_channel_hardware_mapping,
        )

    def _get_threshold_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,
        sequencer_name: str,  # noqa: ARG002, unused argument
    ) -> DataArray:
        """
        Retrieve the thresholded acquisition data associated with ``acq_channel`` and ``acq_index``.

        Parameters
        ----------
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acquisition channel.
        seq_channel_hardware_mapping
            Acquisition hardware mapping for the sequencer and channel.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        sequencer_name
            Sequencer.

        Returns
        -------
        :
            DataArray containing thresholded acquisition data.

        """
        if acq_duration is None:
            raise RuntimeError(
                "Retrieving data failed. Expected the integration length to be defined,"
                " but it is `None`."
            )

        def _get_bin(
            qblox_acq_index: int,
            qblox_acq_bin: int,
            thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,  # noqa: ARG001, unused argument
        ) -> int:
            bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
            n = bin_data["threshold"][qblox_acq_bin]
            return n if not isnan(n) else -1

        return self._get_binned_data(
            bin_data_function=_get_bin,
            acq_channel=acq_channel,
            seq_channel_hardware_mapping=seq_channel_hardware_mapping,
        )


class _QTMAcquisitionManager(_AcquisitionManagerBase):
    """
    Utility class that handles the acquisitions performed with the QTM.

    An instance of this class is meant to exist only for a single prepare-start-
    retrieve_acquisition cycle to prevent stateful behavior.

    Parameters
    ----------
    parent
        Reference to the parent QRM IC component.
    acq_channels_data
        Provides a summary of the used acquisition protocol, bin mode, acquisition channels,
        acquisition indices per channel, and repetitions, for each sequencer.
    acquisition_duration
        The duration of each acquisition for each sequencer.
    seq_name_to_idx_map
        All available sequencer names to their ids in a dict.

    """

    @property
    def _protocol_to_acq_function_map(self) -> dict[str, Callable]:
        return {
            "TriggerCount": partial(self._get_trigger_count_data, sum_multiply_repetitions=True),
            "ThresholdedTriggerCount": self._get_trigger_count_threshold_data,
            "DualThresholdedTriggerCount": partial(
                self._get_trigger_count_data, sum_multiply_repetitions=False
            ),
            "Trace": self._get_digital_trace_data,
            "Timetag": self._get_timetag_data,
            "TimetagTrace": self._get_timetag_trace_data,
        }

    @staticmethod
    def _check_bin_mode_compatible(
        acq_channels_data: AcquisitionChannelsData,
        acq_hardware_mapping: dict[str, dict[Hashable, QbloxAcquisitionHardwareMapping]],
    ) -> None:
        for seq_acq_hardware_mapping in acq_hardware_mapping.values():
            for acq_channel in seq_acq_hardware_mapping:
                if (
                    acq_channels_data[acq_channel].bin_mode
                    not in QTM_COMPATIBLE_BIN_MODES[acq_channels_data[acq_channel].protocol]
                ):
                    raise IncompatibleBinModeError(
                        module_type="QTM",
                        protocol=acq_channels_data[acq_channel].protocol,
                        bin_mode=acq_channels_data[acq_channel].bin_mode,
                    )

    def _get_digital_trace_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,  # noqa: ARG002, unused argument
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,  # noqa: ARG002, unused argument
        acq_duration: int,
        sequencer_name: str,
    ) -> DataArray:
        coords = self._acq_channels_data[acq_channel].coords
        assert isinstance(coords, dict)
        # We ignore the hardware_retrieved_acquisitions,
        # and use data from from the io channel.
        seq_idx = self._seq_name_to_idx_map[sequencer_name]
        scope_data = np.array(self.instrument.io_channels[seq_idx].get_scope_data()[:acq_duration])

        acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
        trace_index_dim_name = f"trace_index_{acq_channel}"
        data_array = DataArray(
            scope_data.reshape((1, -1)),
            dims=[acq_index_dim_name, trace_index_dim_name],
            coords={
                acq_index_dim_name: [0],
                trace_index_dim_name: list(range(acq_duration)),
            },
            attrs=self._acq_channel_attrs(self._acq_channels_data[acq_channel].protocol),
        )
        add_acquisition_coords_nonbinned(data_array, coords, acq_index_dim_name)
        return data_array

    def _get_timetag_trace_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,  # noqa: ARG002, unused argument
        sequencer_name: str,
    ) -> DataArray:
        # TODO: do the same logic as for binned acquisitions
        seq_idx = self._seq_name_to_idx_map[sequencer_name]

        # For TimetagTrace distribution acquisition protocol,
        # the mapping is of QbloxAcquisitionBinMapping type,
        # guaranteed by the QbloxAcquisitionIndexManager,
        # and coords is a list.
        assert isinstance(seq_channel_hardware_mapping, dict)
        coords = self._acq_channels_data[acq_channel].coords
        assert isinstance(coords, list)

        # There is only one Qblox acquisition index used for all
        # acquisitions and bins for this protocol; we only need to retrieve the first.
        qblox_acq_index = next(iter(seq_channel_hardware_mapping.values())).index

        qblox_acq_name = self._qblox_acq_index_to_qblox_acq_name(qblox_acq_index)
        scope_data = self.instrument.io_channels[seq_idx].get_scope_data()
        timetag_traces = self._split_timetag_trace_data_per_window(
            timetags=hardware_retrieved_acquisitions[qblox_acq_name]["acquisition"]["bins"][
                "timedelta"
            ],
            scope_data=scope_data,
        )

        # Turn the inhomogeneous 2D array into a rectangular matrix compatible with
        # xarray. Pad with NaN to make it rectangular.
        max_timetag_traces = max(len(t) for t in timetag_traces)
        rect_array = np.empty(
            [
                self._repetitions,
                len(self._acq_channels_data[acq_channel].coords),
                max_timetag_traces,
            ],
            dtype=np.float64,
        )
        rect_array[:] = np.nan

        for repetition in range(self._repetitions):
            for i, (_acq_index, qblox_acq_index_bin) in enumerate(
                seq_channel_hardware_mapping.items()
            ):
                timetag_traces_bin = timetag_traces[
                    qblox_acq_index_bin.bin + repetition * qblox_acq_index_bin.stride
                ]
                rect_array[repetition][i][0 : len(timetag_traces_bin)] = timetag_traces_bin
        acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
        trace_index_dim_name = f"trace_index_{acq_channel}"
        data_array = DataArray(
            rect_array,
            dims=["repetition", acq_index_dim_name, trace_index_dim_name],
            coords={
                "repetition": list(range(self._repetitions)),
                acq_index_dim_name: list(seq_channel_hardware_mapping.keys()),
                trace_index_dim_name: list(range(max_timetag_traces)),
            },
            attrs=self._acq_channel_attrs(self._acq_channels_data[acq_channel].protocol),
        )
        add_acquisition_coords_binned(data_array, coords, acq_index_dim_name)
        return data_array

    def _split_timetag_trace_data_per_window(
        self,
        timetags: list[int],
        scope_data: list[tuple[str, int]],
    ) -> list[list[float]]:
        """
        Split the long array of ``scope_data`` on acquisition windows.

        The scope_data is formatted like [[TYPE, TIME],[TYPE,TIME],...], where TYPE is one of
        "OPEN", "RISE", "CLOSE". The TIME is absolute (cluster system time).

        Each acquisition window starts with "OPEN" and ends with "CLOSE". This method
        uses that information to divide the long ``scope_data`` array up into smaller
        arrays for each acquisition window.

        Furthermore, the ``timetags`` list contains the *relative* timetags of the
        *first* pulse recorded in each window. This data is used to calculate the
        relative timetags for all timetags in the trace.
        """
        parsed_scope_data: list[list[int]] = []
        for event, time in scope_data:
            if event == "OPEN":
                parsed_scope_data.append([])
            elif event == "CLOSE":
                pass
            else:
                parsed_scope_data[-1].append(time)

        timetag_traces = []
        for ref_timetag, timetags_in_window in zip(timetags, parsed_scope_data):
            timetag_traces.append(
                [
                    (ref_timetag + event - timetags_in_window[0]) / 2048
                    for event in timetags_in_window
                ]
            )

        return timetag_traces

    def _get_timetag_data(
        self,
        *,
        hardware_retrieved_acquisitions: dict,
        acq_channel: str,
        seq_channel_hardware_mapping: QbloxAcquisitionHardwareMapping,
        acq_duration: int,  # noqa: ARG002, unused argument
        sequencer_name: str,  # noqa: ARG002, unused argument
    ) -> DataArray:
        def _get_bin(
            qblox_acq_index: int,
            qblox_acq_bin: int,
            thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,  # noqa: ARG001, unused argument
        ) -> int:
            bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
            return bin_data["timedelta"][qblox_acq_bin] / 2048

        return self._get_binned_data(
            bin_data_function=_get_bin,
            acq_channel=acq_channel,
            seq_channel_hardware_mapping=seq_channel_hardware_mapping,
        )


class ClusterComponent(base.InstrumentCoordinatorComponentBase):
    """
    Class that represents an instrument coordinator component for a Qblox cluster.

    New instances of the ClusterComponent will automatically add installed
    modules using name `"<cluster_name>_module<slot>"`.

    Parameters
    ----------
    instrument
        Reference to the cluster driver object.

    """

    @dataclass
    class _Program:
        module_programs: dict[str, Any]
        settings: ClusterSettings

    def __init__(self, instrument: Cluster) -> None:
        super().__init__(instrument)
        self._cluster_modules: dict[str, base.InstrumentCoordinatorComponentBase] = {}
        self._program: ClusterComponent._Program | None = None

        # Important: a tuple with only False may not occur as a key, because new
        # unsupported module types may return False on all is_..._type functions.
        # is_qcm_type, is_qrm_type, is_rf_type, is_qtm_type, is_qrc_type
        module_type_map: dict[ComponentTypeProperties, type[_ModuleComponentBase]] = {
            ComponentTypeProperties(True, False, False, False, False): _QCMComponent,
            ComponentTypeProperties(True, False, True, False, False): _QCMRFComponent,
            ComponentTypeProperties(False, True, False, False, False): _QRMComponent,
            ComponentTypeProperties(False, True, True, False, False): _QRMRFComponent,
            ComponentTypeProperties(False, False, True, False, True): _QRCComponent,
            ComponentTypeProperties(False, False, False, True, False): _QTMComponent,
        }
        for instrument_module in instrument.modules:
            try:
                icc_class = module_type_map[
                    ComponentTypeProperties(
                        instrument_module.is_qcm_type,
                        instrument_module.is_qrm_type,
                        instrument_module.is_rf_type,
                        getattr(instrument_module, "is_qtm_type", False),
                        getattr(instrument_module, "is_qrc_type", False),
                    )
                ]
            except KeyError:
                continue

            self._cluster_modules[instrument_module.name] = icc_class(instrument_module)

    @property
    def is_running(self) -> bool:
        """Returns true if any of the modules are currently running."""
        return any(comp.is_running for comp in self._cluster_modules.values())

    def _set_parameter(
        self,
        instrument: InstrumentBase,
        parameter_name: str,
        val: Any,  # noqa: ANN401, disallow Any as type
    ) -> None:
        """
        Set the parameter directly or using the lazy set.

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

    def start(self) -> None:
        """Starts all the modules in the cluster."""
        # Disarming all sequencers, to make sure the last
        # `self.instrument.start_sequencer` only starts sequencers
        # which are explicitly armed by the subsequent calls.
        self.instrument.stop_sequencer()

        if self._program is None:
            # Nothing to start
            return

        # Arming all sequencers in the program.
        for comp_name, comp in self._cluster_modules.items():
            if comp_name in self._program.module_programs:
                comp.clear_data()
                comp.arm_all_sequencers_in_program()

        # Starts all sequencers in the cluster, time efficiently.
        if self._program.settings.sync_on_external_trigger is not None:
            self._sync_on_external_trigger(self._program.settings.sync_on_external_trigger)

        self.instrument.start_sequencer()

    def _sync_on_external_trigger(self, settings: ExternalTriggerSyncSettings) -> None:
        module_idx = settings.slot - 1
        channel_idx = settings.channel - 1
        if module_idx != -1:  # if not CMM
            module: Module = self.instrument.modules[module_idx]
            if not module.is_qtm_type:
                raise ValueError(
                    f"Invalid module type {module.module_type} for sync_on_external_trigger. Must "
                    "be one of MM or QTM."
                )
            # See ClusterCompiler._validate_external_trigger_sync, which validates that these
            # settings do not conflict, if they already exist in a QTMComponent.
            self._set_parameter(
                self.instrument.modules[module_idx].io_channels[channel_idx],
                "mode",
                "input",
            )
            if settings.input_threshold is None:
                raise ValueError(
                    "Using `sync_on_external_trigger` with a QTM module, but there was no input "
                    "threshold specified in either `hardware_options.digitization_thresholds` or "
                    "`sync_on_external_trigger.input_threshold`."
                )
            self._set_parameter(
                self.instrument.modules[module_idx].io_channels[channel_idx],
                "analog_threshold",
                settings.input_threshold,
            )
        sync_ref = SyncRef.ON if settings.sync_to_ref_clock else SyncRef.OFF
        self.instrument.time.sync_ext_trigger(
            slot=settings.slot,
            channel=settings.channel,
            trigger_timestamp=settings.trigger_timestamp,
            timeout=settings.timeout,
            format=settings.format,
            edge_polarity=settings.edge_polarity,
            sync_ref=sync_ref,
        )

    def stop(self) -> None:
        """Stops all the modules in the cluster."""
        for comp in self._cluster_modules.values():
            comp.disable_sync()
        # Stops all sequencers in the cluster, time efficiently.
        self.instrument.stop_sequencer()

    def prepare(
        self,
        program: dict[str, dict | ClusterSettings],
    ) -> None:
        """
        Prepares the cluster component for execution of a schedule.

        Parameters
        ----------
        program
            The compiled instructions to configure the cluster to.
        acq_channels_data
            Acquisition channels data for acquisition mapping.
        repetitions
            Repetitions of the schedule.

        """
        settings = program.get("settings", {})
        if not isinstance(settings, ClusterSettings):
            cluster_settings = ClusterSettings.from_dict(settings)
        else:
            cluster_settings = settings

        self._set_parameter(self.instrument, "reference_source", cluster_settings.reference_source)

        # See QTFY-848.
        # Removing acq_channels_data and repetitions from program,
        # and later (in a subsequent change) we will use it to process acquisition data.
        acq_channels_data = program.get("acq_channels_data", AcquisitionChannelsData())
        repetitions = program.get("repetitions", 1)
        self._program = ClusterComponent._Program(
            module_programs=without(program, ["settings", "acq_channels_data", "repetitions"]),
            settings=cluster_settings,
        )
        for name, comp_options in self._program.module_programs.items():
            if name not in self._cluster_modules:
                raise KeyError(
                    f"Attempting to prepare module {name} of cluster {self.name}, while"
                    f" module has not been added to the cluster component."
                )
            # See QTFY-848.
            # InstrumentCoordinatorComponentBase.prepare expects only one argument,
            # but we intend to change that in the future, so we can safely ignore the
            # warning of reportCallIssue.
            self._cluster_modules[name].prepare(comp_options, acq_channels_data, repetitions)  # pyright: ignore[reportCallIssue]

    def retrieve_acquisition(self) -> Dataset | None:
        """
        Retrieves all the data from the instruments.

        Returns
        -------
        :
            The acquired data or ``None`` if no acquisitions have been performed.

        """
        if self._program is None:
            # No acquisitions
            return

        acquisitions = Dataset()
        for comp_name, comp in self._cluster_modules.items():
            if comp_name not in self._program.module_programs:
                continue

            comp_acq = comp.retrieve_acquisition()
            if comp_acq is not None:
                check_already_existing_acquisition(
                    new_dataset=comp_acq, current_dataset=acquisitions
                )
                acquisitions = acquisitions.merge(comp_acq)

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

    def get_hardware_log(
        self,
        compiled_schedule: CompiledSchedule,
    ) -> dict | None:
        """
        Retrieve the hardware log of the Cluster Management Module and associated modules.

        This log includes the module serial numbers and
        firmware version.

        Parameters
        ----------
        compiled_schedule
            Compiled schedule to check if this cluster is referenced in (and if so,
            which specific modules are referenced in).

        Returns
        -------
        :
            A dict containing the hardware log of the cluster, in case the
            component was referenced; else None.

        """
        cluster = self.instrument
        if cluster.name not in compiled_schedule.compiled_instructions:
            return None

        cluster_ip = _get_instrument_ip(self)
        hardware_log = {
            f"{cluster.name}_cmm": _download_log(
                config_manager=_get_configuration_manager(cluster_ip),
                is_cluster=True,
            ),
            f"{cluster.name}_idn": str(cluster.get_idn()),
            f"{cluster.name}_mods_info": str(cluster._get_mods_info()),
        }

        for module in cluster.modules:
            if module.name in compiled_schedule.compiled_instructions[cluster.name]:
                # Cannot fetch log from module.get_hardware_log here since modules are
                # not InstrumentCoordinator components when using a cluster
                module_ip = f"{cluster_ip}/{module.slot_idx}"
                hardware_log[module.name] = _download_log(_get_configuration_manager(module_ip))

        return hardware_log


def _get_instrument_ip(component: base.InstrumentCoordinatorComponentBase) -> str:
    ip_config = component.instrument.get_ip_config()

    if ip_config == "0":
        raise ValueError(
            f"Instrument '{component.instrument.name}' returned {ip_config=}."
            f"Please make sure the physical instrument is connected and has a valid ip."
        )

    instrument_ip = ip_config
    if "/" in instrument_ip:
        instrument_ip = instrument_ip.split("/")[0]

    return instrument_ip


def _get_configuration_manager(instrument_ip: str) -> ConfigurationManager:
    try:
        config_manager = ConfigurationManager(instrument_ip)
    except RuntimeError as error:
        new_message = f"{error}\nNote: qblox-instruments might have changed ip formatting."
        raise type(error)(new_message) from None
    return config_manager


def _download_log(
    config_manager: ConfigurationManager,
    is_cluster: bool | None = False,
) -> dict:
    hardware_log = {}

    sources = ["app", "system"]
    if is_cluster:
        sources.append("cfg_man")

    for source in sources:
        # uuid prevents unwanted deletion if file already exists
        temp_log_file_name = os.path.join(get_datadir(), f"{source}_{uuid4()}")
        config_manager.download_log(source=source, fmt="txt", file=temp_log_file_name)
        if os.path.isfile(temp_log_file_name):
            with open(temp_log_file_name, encoding="utf-8", errors="replace") as file:
                log = file.read()
            os.remove(temp_log_file_name)
            hardware_log[f"{source}_log"] = log
        else:
            raise RuntimeError(
                f"`ConfigurationManager.download_log` did not create a `{source}` file."
            )

    return hardware_log
