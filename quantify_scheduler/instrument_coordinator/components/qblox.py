# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing Qblox InstrumentCoordinator Components."""
from __future__ import annotations

import copy
import logging
import os
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from math import isnan
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4
import re

import numpy as np
from qblox_instruments import (
    Cluster,
    ConfigurationManager,
    SequencerStates,
    SequencerStatus,
)
from quantify_core.data.handling import get_datadir
from xarray import DataArray, Dataset

from quantify_scheduler.backends.qblox import constants, driver_version_check
from quantify_scheduler.backends.qblox.enums import ChannelMode
from quantify_scheduler.backends.qblox.helpers import (
    single_scope_mode_acquisition_raise,
)
from quantify_scheduler.backends.types.qblox import (
    AnalogModuleSettings,
    AnalogSequencerSettings,
    RFModuleSettings,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.instrument_coordinator.components import base
from quantify_scheduler.instrument_coordinator.utility import (
    check_already_existing_acquisition,
    lazy_set,
    search_settable_param,
)

if TYPE_CHECKING:
    from qblox_instruments.qcodes_drivers.module import Module
    from qblox_instruments.qcodes_drivers.sequencer import Sequencer
    from quantify_scheduler.schedules.schedule import (
        AcquisitionMetadata,
        CompiledSchedule,
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Prevent unsupported qblox-instruments version from crashing this submodule
driver_version_check.verify_qblox_instruments_version()


@dataclass(frozen=True)
class _StaticHardwareProperties:
    """Dataclass for storing configuration differences across Qblox devices."""

    settings_type: Type[AnalogModuleSettings]
    """The settings dataclass to use that the hardware needs to configure to."""
    has_internal_lo: bool
    """Specifies if an internal lo source is available."""
    number_of_sequencers: int
    """The number of sequencers the hardware has available."""
    number_of_output_channels: int
    """The number of physical output channels that can be used."""
    number_of_input_channels: int
    """"The number of physical input channels that can be used."""


_QCM_BASEBAND_PROPERTIES = _StaticHardwareProperties(
    settings_type=AnalogModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QCM,
    number_of_output_channels=4,
    number_of_input_channels=0,
)
_QRM_BASEBAND_PROPERTIES = _StaticHardwareProperties(
    settings_type=AnalogModuleSettings,
    has_internal_lo=False,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QRM,
    number_of_output_channels=2,
    number_of_input_channels=2,
)
_QCM_RF_PROPERTIES = _StaticHardwareProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QCM,
    number_of_output_channels=2,
    number_of_input_channels=0,
)
_QRM_RF_PROPERTIES = _StaticHardwareProperties(
    settings_type=RFModuleSettings,
    has_internal_lo=True,
    number_of_sequencers=constants.NUMBER_OF_SEQUENCERS_QRM,
    number_of_output_channels=1,
    number_of_input_channels=1,
)


class _ModuleComponentBase(base.InstrumentCoordinatorComponentBase):
    """Qblox InstrumentCoordinator component base class."""

    def __init__(self, instrument: Module) -> None:
        super().__init__(instrument)

        self._instrument_module = instrument

        if instrument.is_rf_type is not self._hardware_properties.has_internal_lo:
            raise RuntimeError(
                "_ModuleComponentBase not compatible with the "
                "provided instrument. Please confirm whether your device "
                "is a Qblox RF or baseband module (having or not having an "
                "internal LO)."
            )

        self._seq_name_to_idx_map = {
            f"seq{idx}": idx
            for idx in range(self._hardware_properties.number_of_sequencers)
        }

        self._program = {}

    # Necessary to override the `instrument` attr from `InstrumentCoordinatorComponentBase`,
    # `Module` is a qcodes `InstrumentModule` subclass
    @property
    def instrument(self) -> Module:
        """Returns a reference to the module instrument."""
        return self._instrument_module

    def _set_parameter(
        self,
        instrument: Union[Module, Sequencer],
        parameter_name: str,
        val: Any,
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
        # TODO: these qcodes parameters already exist in the development branch
        # of qblox-instruments, but will be released in 0.13.0 when RTP is
        # officially supported. Until then, catching the value error is needed.
        try:
            search_settable_param(
                instrument=instrument, nested_parameter_name=parameter_name
            )
        except ValueError as e:
            if (
                re.search(
                    r".*(out|marker)[0-9]_(exp|bt|fir)[0-9]?_config", parameter_name
                )
                and val == "bypassed"
            ):
                return
            if re.search(
                r".*(out|marker)[0-9]_(exp|bt|fir)[0-9]?_(time_constant|amplitude|coeffs)",
                parameter_name,
            ):
                return
            raise e
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
        if self.instrument.name not in compiled_schedule.compiled_instructions.keys():
            return None

        return _download_log(_get_configuration_manager(_get_instrument_ip(self)))

    def prepare(self, program: Dict[str, dict]) -> None:
        """Store program containing sequencer settings."""
        self._program = program

    def disable_sync(self) -> None:
        """Disable sync for all sequencers."""
        for idx in range(self._hardware_properties.number_of_sequencers):
            # Prevent hanging on next run if instrument is not used.
            self._set_parameter(self.instrument[f"sequencer{idx}"], "sync_en", False)

    def stop(self) -> None:
        """Stops all execution."""
        self.disable_sync()
        self.instrument.stop_sequencer()

    @abstractmethod
    def _configure_global_settings(self, settings: AnalogModuleSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """

    def _configure_sequencer_settings(
        self, seq_idx: int, settings: AnalogSequencerSettings
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
            settings.init_offset_awg_path_I,
        )
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "offset_awg_path1",
            settings.init_offset_awg_path_Q,
        )

        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "gain_awg_path0",
            settings.init_gain_awg_path_I,
        )
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "gain_awg_path1",
            settings.init_gain_awg_path_Q,
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

        channel_map_parameters = self._determine_channel_map_parameters(settings)
        for channel_param, channel_setting in channel_map_parameters.items():
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                channel_param,
                channel_setting,
            )

        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"], "sequence", settings.sequence
        )

    def _determine_channel_map_parameters(
        self, settings: AnalogSequencerSettings
    ) -> Dict[str, str]:
        """Returns a dictionary with the channel map parameters for this module."""
        channel_map_parameters = {}
        self._determine_output_channel_map_parameters(settings, channel_map_parameters)

        return channel_map_parameters

    def _determine_output_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: Dict[str, str]
    ) -> Dict[str, str]:
        """Adds the outputs to the channel map parameters dict."""
        for channel_idx in range(self._hardware_properties.number_of_output_channels):
            param_setting = "off"
            if (
                len(settings.connected_output_indices) > 0
                and channel_idx in settings.connected_output_indices
            ):  # For baseband, output indices map 1-to-1 to channel map indices
                if channel_idx in settings.connected_output_indices:
                    if ChannelMode.COMPLEX in settings.channel_name:
                        param_setting = ["I", "Q", "I", "Q"][channel_idx]
                    elif ChannelMode.REAL in settings.channel_name:
                        param_setting = "I"

            channel_map_parameters[f"connect_out{channel_idx}"] = param_setting

        return channel_map_parameters

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

    def _start_armed_sequencers(self):
        """Start execution of the schedule: start armed sequencers."""
        for idx in range(self._hardware_properties.number_of_sequencers):
            state = self.instrument.get_sequencer_status(idx)
            if state.state is SequencerStates.ARMED:
                self.instrument.start_sequencer(idx)

    def clear_data(self) -> None:
        """Clears remaining data on the module. Module type specific function."""
        return None

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


class _QCMComponent(_ModuleComponentBase):
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

    def prepare(self, program: Dict[str, dict]) -> None:
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
        """
        super().prepare(program)

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
                seq_idx=seq_idx, settings=AnalogSequencerSettings.from_dict(seq_cfg)
            )

    def _configure_global_settings(self, settings: AnalogModuleSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        # configure mixer correction offsets
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(
                self.instrument, "out0_offset", settings.offset_ch0_path_I
            )
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(
                self.instrument, "out1_offset", settings.offset_ch0_path_Q
            )
        if settings.offset_ch1_path_I is not None:
            self._set_parameter(
                self.instrument, "out2_offset", settings.offset_ch1_path_I
            )
        if settings.offset_ch1_path_Q is not None:
            self._set_parameter(
                self.instrument, "out3_offset", settings.offset_ch1_path_Q
            )

        for output, dc_settings in enumerate(
            settings.distortion_corrections[
                : self._hardware_properties.number_of_output_channels
            ]
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


class _QRMComponent(_ModuleComponentBase):
    """QRM specific InstrumentCoordinator component."""

    _hardware_properties = _QRM_BASEBAND_PROPERTIES

    def __init__(self, instrument: Module) -> None:
        if not instrument.is_qrm_type:
            raise TypeError(
                f"Trying to create _QRMComponent from non-QRM instrument "
                f'of type "{type(instrument)}".'
            )
        super().__init__(instrument)

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
        """
        super().prepare(program)

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

            settings = AnalogSequencerSettings.from_dict(seq_cfg)
            self._configure_sequencer_settings(seq_idx=seq_idx, settings=settings)
            acq_duration[seq_name] = settings.integration_length_acq

        if (acq_metadata := program.get("acq_metadata")) is not None:
            scope_mode_sequencer_and_qblox_acq_index = (
                self._determine_scope_mode_acquisition_sequencer_and_qblox_acq_index(
                    acq_metadata
                )
            )
            self._acquisition_manager = _QRMAcquisitionManager(
                parent=self,
                acquisition_metadata=acq_metadata,
                scope_mode_sequencer_and_qblox_acq_index=scope_mode_sequencer_and_qblox_acq_index,
                acquisition_duration=acq_duration,
                seq_name_to_idx_map=self._seq_name_to_idx_map,
            )
            if scope_mode_sequencer_and_qblox_acq_index is not None:
                self._set_parameter(
                    self.instrument,
                    "scope_acq_sequencer_select",
                    scope_mode_sequencer_and_qblox_acq_index[0],
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

    def _configure_global_settings(self, settings: AnalogModuleSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        # configure mixer correction offsets
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(
                self.instrument, "out0_offset", settings.offset_ch0_path_I
            )
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(
                self.instrument, "out1_offset", settings.offset_ch0_path_Q
            )
        # configure gain
        if settings.in0_gain is not None:
            self._set_parameter(self.instrument, "in0_gain", settings.in0_gain)
        if settings.in1_gain is not None:
            self._set_parameter(self.instrument, "in1_gain", settings.in1_gain)

        for output, dc_settings in enumerate(
            settings.distortion_corrections[
                : self._hardware_properties.number_of_output_channels
            ]
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

    def _configure_sequencer_settings(
        self, seq_idx: int, settings: AnalogSequencerSettings
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
        if settings.thresholded_acq_rotation is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "thresholded_acq_rotation",
                settings.thresholded_acq_rotation,
            )
        if settings.thresholded_acq_threshold is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "thresholded_acq_threshold",
                settings.thresholded_acq_threshold,
            )
        if settings.thresholded_acq_trigger_address is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "thresholded_acq_trigger_address",
                settings.thresholded_acq_trigger_address,
            )
        if settings.thresholded_acq_trigger_en is not None:
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "thresholded_acq_trigger_en",
                settings.thresholded_acq_trigger_en,
            )
            self._set_parameter(
                self.instrument[f"sequencer{seq_idx}"],
                "thresholded_acq_trigger_invert",
                settings.thresholded_acq_trigger_invert,
            )

    def _determine_channel_map_parameters(
        self, settings: AnalogSequencerSettings
    ) -> Dict[str, str]:
        """Returns a dictionary with the channel map parameters for this module."""
        channel_map_parameters = {}
        self._determine_output_channel_map_parameters(settings, channel_map_parameters)
        self._determine_input_channel_map_parameters(settings, channel_map_parameters)

        return channel_map_parameters

    def _determine_input_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: Dict[str, str]
    ) -> Dict[str, str]:
        """Adds the inputs to the channel map parameters dict."""
        param_name = {0: "connect_acq_I", 1: "connect_acq_Q"}

        for channel_idx in range(self._hardware_properties.number_of_input_channels):
            param_setting = (
                f"in{channel_idx}"
                if "output" in settings.channel_name
                and ChannelMode.DIGITAL not in settings.channel_name
                else "off"
            )
            if (
                len(settings.connected_input_indices) > 0
                and channel_idx in settings.connected_input_indices
            ):  # For baseband, input indices map 1-to-1 to channel map indices
                param_setting = f"in{channel_idx}"

            channel_map_parameters[param_name[channel_idx]] = param_setting

        return channel_map_parameters

    def _determine_scope_mode_acquisition_sequencer_and_qblox_acq_index(
        self, acquisition_metadata: Dict[str, AcquisitionMetadata]
    ) -> Optional[Tuple[int, int]]:
        """
        Finds the sequencer and qblox_acq_index that performs the raw trace acquisition.

        Raises an error if multiple scope mode acquisitions are present per sequencer.
        Note, that compiler ensures there is at most one scope mode acquisition,
        however the user is able to freely modify the compiler program,
        so we make sure this requirement is still satisfied. See
        :func:`~quantify_scheduler.backends.qblox.analog.AnalogModuleCompiler._ensure_single_scope_mode_acquisition_sequencer`.

        Parameters
        ----------
        acquisition_metadata
            The acquisition metadata for each sequencer.

        Returns
        -------
        :
            The sequencer and qblox_acq_channel for the trace acquisition, if there is any, otherwise None, None.
        """
        sequencer_and_qblox_acq_index = None
        for (
            sequencer_name,
            current_acquisition_metadata,
        ) in acquisition_metadata.items():
            if current_acquisition_metadata.acq_protocol == "Trace":
                # It's in the format "seq{n}", so we cut it.
                sequencer_id = self._seq_name_to_idx_map[sequencer_name]
                if (
                    sequencer_and_qblox_acq_index is not None
                    and sequencer_and_qblox_acq_index[0] != sequencer_id
                ):
                    single_scope_mode_acquisition_raise(
                        sequencer_0=sequencer_id,
                        sequencer_1=sequencer_and_qblox_acq_index[0],
                        module_name=self.name,
                    )
                # For scope protocol, only one channel makes sense, we only need the first key in dict
                qblox_acq_index = next(
                    iter(current_acquisition_metadata.acq_channels_metadata.keys())
                )
                sequencer_and_qblox_acq_index = (sequencer_id, qblox_acq_index)

        return sequencer_and_qblox_acq_index

    def clear_data(self) -> None:
        """Clears remaining data on the module. Module type specific function."""
        for sequencer_id in range(self._hardware_properties.number_of_sequencers):
            self.instrument.delete_acquisition_data(sequencer=sequencer_id, all=True)


class _RFComponent(_ModuleComponentBase):
    """Mix-in for RF-module-specific InstrumentCoordinatorComponent behaviour."""

    def _configure_sequencer_settings(
        self, seq_idx: int, settings: AnalogSequencerSettings
    ) -> None:
        super()._configure_sequencer_settings(seq_idx, settings)
        # Always set override to False.
        self._set_parameter(
            self.instrument[f"sequencer{seq_idx}"],
            "marker_ovr_en",
            False,
        )

    def _determine_output_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: Dict[str, str]
    ) -> Dict[str, str]:
        """Adds the outputs to the channel map parameters dict."""
        expected_output_indices = {0: (0, 1), 1: (2, 3)}

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


class _QCMRFComponent(_RFComponent, _QCMComponent):
    """QCM-RF specific InstrumentCoordinator component."""

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
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path0", settings.offset_ch0_path_I
            )
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path1", settings.offset_ch0_path_Q
            )
        if settings.offset_ch1_path_I is not None:
            self._set_parameter(
                self.instrument, "out1_offset_path0", settings.offset_ch1_path_I
            )
        if settings.offset_ch1_path_Q is not None:
            self._set_parameter(
                self.instrument, "out1_offset_path1", settings.offset_ch1_path_Q
            )
        # configure attenuation
        if settings.out0_att is not None:
            self._set_parameter(self.instrument, "out0_att", settings.out0_att)
        if settings.out1_att is not None:
            self._set_parameter(self.instrument, "out1_att", settings.out1_att)


class _QRMRFComponent(_RFComponent, _QRMComponent):
    """QRM-RF specific InstrumentCoordinator component."""

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
        if settings.offset_ch0_path_I is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path0", settings.offset_ch0_path_I
            )
        if settings.offset_ch0_path_Q is not None:
            self._set_parameter(
                self.instrument, "out0_offset_path1", settings.offset_ch0_path_Q
            )
        # configure attenuation
        if settings.out0_att is not None:
            self._set_parameter(self.instrument, "out0_att", settings.out0_att)
        if settings.in0_att is not None:
            self._set_parameter(self.instrument, "in0_att", settings.in0_att)

    def _determine_input_channel_map_parameters(
        self, settings: AnalogSequencerSettings, channel_map_parameters: Dict[str, str]
    ) -> Dict[str, str]:
        """Adds the inputs to the channel map parameters dict."""
        channel_map_parameters["connect_acq"] = (
            "in0" if tuple(settings.connected_input_indices) == (0, 1) else "off"
        )
        if (
            "output" in settings.channel_name
            and ChannelMode.DIGITAL not in settings.channel_name
        ):
            channel_map_parameters["connect_acq"] = "in0"

        return channel_map_parameters


class _QRMAcquisitionManager:
    """
    Utility class that handles the acquisitions performed with the QRM.

    An instance of this class is meant to exist only for a single prepare-start-
    retrieve_acquisition cycle to prevent stateful behavior.

    Parameters
    ----------
    parent
        Reference to the parent QRM IC component.
    acquisition_metadata
        Provides a summary of the used acquisition protocol, bin mode, acquisition channels,
        acquisition indices per channel, and repetitions, for each sequencer.
    scope_mode_sequencer_and_qblox_acq_index
        The sequencer and qblox acq_index of the scope mode acquisition if there's any.
    acquisition_duration
        The duration of each acquisition for each sequencer.
    seq_name_to_idx_map
        All available sequencer names to their ids in a dict.
    """

    def __init__(
        self,
        parent: _QRMComponent,
        acquisition_metadata: Dict[str, AcquisitionMetadata],
        scope_mode_sequencer_and_qblox_acq_index: Optional[Tuple[int, int]],
        acquisition_duration: Dict[int, int],
        seq_name_to_idx_map: Dict[str, int],
    ):
        self.parent: _QRMComponent = parent
        self._acquisition_metadata: Dict[str, AcquisitionMetadata] = (
            acquisition_metadata
        )

        self._scope_mode_sequencer_and_qblox_acq_index: Optional[Tuple[int, int]] = (
            scope_mode_sequencer_and_qblox_acq_index
        )
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
            The ``acq_channel`` is the name of each `xarray.DataArray` in the `xarray.Dataset`.
            Each `xarray.DataArray` is a two-dimensional array, with ``acq_index`` and ``repetition`` as
            dimensions.
        """
        protocol_to_function_mapping = {
            "WeightedIntegratedSeparated": partial(
                self._get_integration_data, separated=True
            ),
            "NumericalSeparatedWeightedIntegration": partial(
                self._get_integration_data, separated=True
            ),
            "NumericalWeightedIntegration": partial(
                self._get_integration_data, separated=False
            ),
            "SSBIntegrationComplex": self._get_integration_amplitude_data,
            "ThresholdedAcquisition": self._get_threshold_data,
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
            hardware_retrieved_acquisitions = self.instrument.get_acquisitions(
                self._seq_name_to_idx_map[sequencer_name]
            )
            for (
                qblox_acq_index,
                acq_channel_metadata,
            ) in acquisition_metadata.acq_channels_metadata.items():
                acq_channel: Hashable = acq_channel_metadata.acq_channel
                acq_indices: list[int] = acq_channel_metadata.acq_indices
                # the acquisition_function retrieves the right part of the acquisitions
                # data structure returned by the qrm
                formatted_acquisitions = acquisition_function(
                    acq_indices=acq_indices,
                    hardware_retrieved_acquisitions=hardware_retrieved_acquisitions,
                    acquisition_metadata=acquisition_metadata,
                    acq_duration=self._acq_duration[sequencer_name],
                    qblox_acq_index=qblox_acq_index,
                    acq_channel=acq_channel,
                )
                formatted_acquisitions_dataset = Dataset(
                    {acq_channel: formatted_acquisitions}
                )

                check_already_existing_acquisition(
                    new_dataset=formatted_acquisitions_dataset, current_dataset=dataset
                )
                dataset = dataset.merge(formatted_acquisitions_dataset)

        return dataset

    def _store_scope_acquisition(self):
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

    @staticmethod
    def _acq_channel_attrs(
        protocol: str,
    ) -> dict:
        return {"acq_protocol": protocol}

    def _get_scope_data(
        self,
        acq_indices: list,
        hardware_retrieved_acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,
        qblox_acq_index: int,
        acq_channel: Hashable,
    ) -> DataArray:
        """
        Retrieves the scope mode acquisition associated with an `acq_channel`.

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        qblox_acq_index
            The Qblox acquisition index from which to get the data.
        acq_channel
            The acquisition channel.

        Returns
        -------
        :
            The scope mode data.
        """
        if acquisition_metadata.bin_mode != BinMode.AVERAGE:
            raise RuntimeError(
                f"{acquisition_metadata.acq_protocol} acquisition protocol does not"
                f"support bin mode {acquisition_metadata.bin_mode}"
            )
        if (
            acq_duration < 0
            or acq_duration > constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS
        ):
            raise ValueError(
                "Attempting to retrieve sample of size "
                f"{acq_duration}, but only integer values "
                f"0,...,{constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS} "
                f"are allowed."
            )
        qblox_acq_name = self._qblox_acq_index_to_qblox_acq_name(qblox_acq_index)
        scope_data = hardware_retrieved_acquisitions[qblox_acq_name]["acquisition"][
            "scope"
        ]
        for path_label in ("path0", "path1"):
            if scope_data[path_label]["out-of-range"]:
                logger.warning(
                    f"The scope mode data of {path_label} of {self.parent.name} with "
                    f"acq_channel={acq_channel}  was out-of-range."
                )
        # NB hardware already divides by avg_count for scope mode
        scope_data_i = np.array(scope_data["path0"]["data"][:acq_duration])
        scope_data_q = np.array(scope_data["path1"]["data"][:acq_duration])

        acq_index_dim_name = f"acq_index_{acq_channel}"
        trace_index_dim_name = f"trace_index_{acq_channel}"
        return DataArray(
            (scope_data_i + scope_data_q * 1j).reshape((1, -1)),
            dims=[acq_index_dim_name, trace_index_dim_name],
            coords={
                acq_index_dim_name: acq_indices,
                trace_index_dim_name: list(range(acq_duration)),
            },
            attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
        )

    def _get_integration_data(
        self,
        acq_indices: list,
        hardware_retrieved_acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,
        qblox_acq_index: int,
        acq_channel: Hashable,
        multiplier: float = 1,
        separated: bool = True,
    ) -> DataArray:
        """
        Retrieves the integrated acquisition data associated with an `acq_channel`.

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        qblox_acq_index
            The Qblox acquisition index from which to get the data.
        acq_channel
            The acquisition channel.
        multiplier
            Multiplies the data with this number.
        separated
            True: return I and Q data separately
            False: return I+Q in the real part and 0 in the imaginary part

        Returns
        -------
        :
            The integrated data.
        """
        bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
        i_data = np.array(bin_data["integration"]["path0"])
        q_data = np.array(bin_data["integration"]["path1"])
        if not separated:
            i_data = i_data + q_data
            q_data = np.zeros_like(q_data)
        acquisitions_data = multiplier * (i_data + q_data * 1j)
        acq_index_dim_name = f"acq_index_{acq_channel}"

        if acquisition_metadata.bin_mode == BinMode.AVERAGE:
            return DataArray(
                acquisitions_data.reshape((len(acq_indices),)),
                dims=[acq_index_dim_name],
                coords={acq_index_dim_name: acq_indices},
                attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
            )
        elif acquisition_metadata.bin_mode == BinMode.APPEND:
            if (
                acquisition_metadata.repetitions * len(acq_indices)
                == acquisitions_data.size
            ):
                acq_data = acquisitions_data.reshape(
                    (acquisition_metadata.repetitions, len(acq_indices))
                )
                return DataArray(
                    acq_data,
                    dims=["repetition", acq_index_dim_name],
                    coords={acq_index_dim_name: acq_indices},
                    attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
                )

            # There is control flow containing measurements, skip reshaping
            else:
                warnings.warn(
                    "The format of acquisition data of looped measurements in APPEND mode"
                    " will change in quantify-scheduler>=0.18.0",
                    FutureWarning,
                )
                acq_data = acquisitions_data.reshape(
                    (acquisition_metadata.repetitions, -1)
                )
                return DataArray(
                    acq_data,
                    dims=["repetition", "loop_repetition"],
                    coords=None,
                    attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
                )
        else:
            raise RuntimeError(
                f"{acquisition_metadata.acq_protocol} acquisition protocol does not"
                f" support bin mode {acquisition_metadata.bin_mode}."
            )

    def _get_integration_amplitude_data(
        self,
        acq_indices: list,
        hardware_retrieved_acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,
        qblox_acq_index: int,
        acq_channel: Hashable,
    ) -> DataArray:
        """
        Gets the integration data but normalized to the integration time.

        The return value is thus the amplitude of the demodulated
        signal directly and has volt units (i.e. same units as a single sample of the
        integrated signal).

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        qblox_acq_index
            The Qblox acquisition index from which to get the data.
        acq_channel
            The acquisition channel.

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
            hardware_retrieved_acquisitions=hardware_retrieved_acquisitions,
            acquisition_metadata=acquisition_metadata,
            acq_duration=acq_duration,
            qblox_acq_index=qblox_acq_index,
            acq_channel=acq_channel,
            multiplier=1 / acq_duration,
        )

        return formatted_data

    def _get_threshold_data(
        self,
        acq_indices: list,
        hardware_retrieved_acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,
        qblox_acq_index: int,
        acq_channel: Hashable,
    ) -> DataArray:
        """
        Retrieve the thresholded acquisition data associated with ``acq_channel`` and ``acq_index``.

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        qblox_acq_index
            The Qblox acquisition index from which to get the data.
        acq_channel
            The acquisition channel.

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
        bin_data = self._get_bin_data(
            hardware_retrieved_acquisitions=hardware_retrieved_acquisitions,
            qblox_acq_index=qblox_acq_index,
        )

        acq_index_dim_name = f"acq_index_{acq_channel}"

        if acquisition_metadata.bin_mode == BinMode.AVERAGE:
            acquisitions_data = np.array(bin_data["threshold"])
            return DataArray(
                acquisitions_data.reshape((len(acq_indices),)),
                dims=[acq_index_dim_name],
                coords={acq_index_dim_name: acq_indices},
                attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
            )
        elif acquisition_metadata.bin_mode == BinMode.APPEND:
            acquisitions_data = np.array(
                bin_data["threshold"], dtype=acquisition_metadata.acq_return_type
            )
            return DataArray(
                acquisitions_data.reshape(
                    (acquisition_metadata.repetitions, len(acq_indices))
                ),
                dims=["repetition", acq_index_dim_name],
                coords={acq_index_dim_name: acq_indices},
                attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
            )
        else:
            raise RuntimeError(
                f"{acquisition_metadata.acq_protocol} acquisition protocol does not"
                f" support bin mode {acquisition_metadata.bin_mode}."
            )

    def _get_trigger_count_data(
        self,
        acq_indices: list,
        hardware_retrieved_acquisitions: dict,
        acquisition_metadata: AcquisitionMetadata,
        acq_duration: int,
        qblox_acq_index: int,
        acq_channel: Hashable,
    ) -> DataArray:
        """
        Retrieves the trigger count acquisition data associated with `acq_channel`.

        Parameters
        ----------
        acq_indices
            Acquisition indices.
        hardware_retrieved_acquisitions
            The acquisitions dict as returned by the sequencer.
        acquisition_metadata
            Acquisition metadata.
        acq_duration
            Desired maximum number of samples for the scope acquisition.
        qblox_acq_index
            The Qblox acquisition index from which to get the data.
        acq_channel
            The acquisition channel.

        Returns
        -------
        data : xarray.DataArray
            The acquired trigger count data.

        Notes
        -----
        - For BinMode.AVERAGE, `data` contains the distribution of counts.
        - For BinMode.APPEND, `data` contains the raw trigger counts.
        """
        bin_data = self._get_bin_data(hardware_retrieved_acquisitions, qblox_acq_index)
        acq_index_dim_name = f"acq_index_{acq_channel}"

        if acquisition_metadata.bin_mode == BinMode.AVERAGE:

            def _convert_from_cumulative(cumulative_values):
                """
                Return the distribution of counts from a cumulative distribution.

                Note, the cumulative distribution is in reverse order.
                The cumulative_values list can contain any number of integers and NaNs.
                """
                cumulative_values = list(enumerate(cumulative_values))

                result = {}

                last_cumulative_value = 0
                for count, current_cumulative_value in reversed(cumulative_values):
                    if (not isnan(current_cumulative_value)) and (
                        last_cumulative_value != current_cumulative_value
                    ):
                        result[count + 1] = (
                            current_cumulative_value - last_cumulative_value
                        )
                        last_cumulative_value = current_cumulative_value

                return result

            result = _convert_from_cumulative(bin_data["avg_cnt"])
            return DataArray(
                [list(result.values())[::-1]],
                dims=["repetition", "counts"],
                coords={"repetition": [0], "counts": list(result.keys())[::-1]},
                attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
            )
        elif acquisition_metadata.bin_mode == BinMode.APPEND:
            counts = np.array(bin_data["avg_cnt"]).astype(int)
            return DataArray(
                [counts],
                dims=["repetition", acq_index_dim_name],
                coords={"repetition": [0], acq_index_dim_name: range(len(counts))},
                attrs=self._acq_channel_attrs(acquisition_metadata.acq_protocol),
            )
        else:
            raise RuntimeError(
                f"{acquisition_metadata.acq_protocol} acquisition protocol does not"
                f"support bin mode {acquisition_metadata.bin_mode}"
            )

    @staticmethod
    def _qblox_acq_index_to_qblox_acq_name(qblox_acq_index: int) -> str:
        """Returns the name of the acquisition from the qblox_acq_index."""
        return str(qblox_acq_index)

    @classmethod
    def _get_bin_data(
        cls, hardware_retrieved_acquisitions: dict, qblox_acq_index: int = 0
    ) -> dict:
        """Returns the bin entry of the acquisition data dict."""
        qblox_acq_name = cls._qblox_acq_index_to_qblox_acq_name(qblox_acq_index)
        channel_data = hardware_retrieved_acquisitions[qblox_acq_name]
        if channel_data["index"] != qblox_acq_index:
            raise RuntimeError(
                f"Name does not correspond to a valid acquisition for name {qblox_acq_name}, "
                f'which has index {channel_data["index"]}.'
            )
        return channel_data["acquisition"]["bins"]


_ClusterModule = Union[_QCMComponent, _QRMComponent, _QCMRFComponent, _QRMRFComponent]
"""Type that combines all the possible modules for a cluster."""


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

    def __init__(self, instrument: Cluster) -> None:
        super().__init__(instrument)
        self._cluster_modules: Dict[str, _ClusterModule] = {}
        self._program = {}

        for instrument_module in instrument.modules:
            try:
                icc_class: type = {
                    (True, False): _QCMComponent,
                    (True, True): _QCMRFComponent,
                    (False, False): _QRMComponent,
                    (False, True): _QRMRFComponent,
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
        # Disarming all sequencers, to make sure the last
        # `self.instrument.start_sequencer` only starts sequencers
        # which are explicitly armed by the subsequent calls.
        self.instrument.stop_sequencer()

        # Arming all sequencers in the program.
        for comp_name, comp in self._cluster_modules.items():
            if comp_name in self._program:
                comp.clear_data()
                comp.arm_all_sequencers_in_program()

        # Starts all sequencers in the cluster, time efficiently.
        self.instrument.start_sequencer()

    def stop(self) -> None:
        """Stops all the modules in the cluster."""
        for comp in self._cluster_modules.values():
            comp.disable_sync()
        # Stops all sequencers in the cluster, time efficiently.
        self.instrument.stop_sequencer()

    def _configure_cmm_settings(self, settings: Dict[str, Any]):
        """
        Set all the settings of the Cluster Management Module.

        These setting have been
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
        if cluster.name not in compiled_schedule.compiled_instructions.keys():
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
                hardware_log[module.name] = _download_log(
                    _get_configuration_manager(module_ip)
                )

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
        new_message = (
            f"{error}\nNote: qblox-instruments might have changed ip formatting."
        )
        raise type(error)(new_message)
    return config_manager


def _download_log(
    config_manager: ConfigurationManager,
    is_cluster: Optional[bool] = False,
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
            with open(
                temp_log_file_name, "r", encoding="utf-8", errors="replace"
            ) as file:
                log = file.read()
            os.remove(temp_log_file_name)
            hardware_log[f"{source}_log"] = log
        else:
            raise RuntimeError(
                f"`ConfigurationManager.download_log` did not create a `{source}`"
                f" file."
            )

    return hardware_log
