# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing Qblox InstrumentCoordinator Components."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Callable, Union
from collections import namedtuple

import logging

import copy
from abc import abstractmethod

import numpy as np
from pulsar_qcm import pulsar_qcm
from pulsar_qrm import pulsar_qrm
from qcodes.instrument.base import Instrument
from quantify_scheduler.instrument_coordinator.components import base
from quantify_scheduler.backends.types.qblox import PulsarSettings, SequencerSettings
from quantify_scheduler.backends.qblox.constants import (
    NUMBER_OF_SEQUENCERS_QCM,
    NUMBER_OF_SEQUENCERS_QRM,
)

logger = logging.getLogger(__name__)


class PulsarInstrumentCoordinatorComponent(base.InstrumentCoordinatorComponentBase):
    """Qblox Pulsar InstrumentCoordinator component base class."""

    def __init__(self, instrument: Instrument, **kwargs) -> None:
        """Create a new instance of PulsarInstrumentCoordinatorComponent base class."""
        super().__init__(instrument, **kwargs)

    @property
    @abstractmethod
    def _number_of_sequencers(self) -> int:
        """The number of sequencers this pulsar has."""

    @property
    def is_running(self) -> bool:
        """
        Finds if any of the sequencers is currently running.

        Returns
        -------
        :
            True if any of the sequencers reports the "RUNNING" status.
        """
        for seq_idx in range(self._number_of_sequencers):
            seq_state = self.instrument.get_sequencer_state(seq_idx)
            if seq_state["status"] == "RUNNING":
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
        for idx in range(self._number_of_sequencers):
            self.instrument.get_sequencer_state(idx, timeout_min)

    def start(self) -> None:
        """
        Starts execution of the schedule.
        """
        for idx in range(self._number_of_sequencers):
            state = self.instrument.get_sequencer_state(idx)
            if state["status"] == "ARMED":
                self.instrument.start_sequencer(idx)

    def stop(self) -> None:
        """
        Stops all execution.
        """
        self.instrument.stop_sequencer()

    def _configure_global_settings(self, settings: PulsarSettings) -> None:
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        self.instrument.set("reference_source", settings.ref)

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
        self.instrument.set(f"sequencer{seq_idx}_sync_en", settings.sync_en)
        self.instrument.set(
            f"sequencer{seq_idx}_offset_awg_path0", settings.awg_offset_path_0
        )
        self.instrument.set(
            f"sequencer{seq_idx}_offset_awg_path1", settings.awg_offset_path_1
        )

        nco_en: bool = settings.nco_en
        self.instrument.set(f"sequencer{seq_idx}_mod_en_awg", nco_en)
        if nco_en:
            self.instrument.set(
                f"sequencer{seq_idx}_nco_freq", settings.modulation_freq
            )


# pylint: disable=too-many-ancestors
class PulsarQCMComponent(PulsarInstrumentCoordinatorComponent):
    """
    Pulsar QCM specific InstrumentCoordinator component.
    """

    _number_of_sequencers: int = NUMBER_OF_SEQUENCERS_QCM
    """Specifies the amount of sequencers available to this QCM."""

    def __init__(self, instrument: pulsar_qcm.pulsar_qcm_qcodes, **kwargs) -> None:
        """Create a new instance of PulsarQCMComponent."""
        assert isinstance(instrument, pulsar_qcm.pulsar_qcm_qcodes)
        super().__init__(instrument, **kwargs)

    @property
    def instrument(self) -> pulsar_qcm.pulsar_qcm_qcodes:
        return super().instrument

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
        seq_name_to_idx_map = {
            f"seq{idx}": idx for idx in range(self._number_of_sequencers)
        }
        if "settings" in program:
            settings_entry = program.pop("settings")
            pulsar_settings = PulsarSettings.from_dict(settings_entry)
            self._configure_global_settings(pulsar_settings)

        for seq_name, seq_cfg in program.items():
            if seq_name in seq_name_to_idx_map:
                seq_idx = seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer with"
                    f' name "{seq_name}".'
                )
            if "settings" in seq_cfg:
                seq_settings = SequencerSettings.from_dict(seq_cfg["settings"])
                self._configure_sequencer_settings(
                    seq_idx=seq_idx, settings=seq_settings
                )

            self.instrument.set(
                f"sequencer{seq_idx}_waveforms_and_program", seq_cfg["seq_fn"]
            )

            self.instrument.arm_sequencer(sequencer=seq_idx)


# pylint: disable=too-many-ancestors
class PulsarQRMComponent(PulsarInstrumentCoordinatorComponent):
    """
    Pulsar QRM specific InstrumentCoordinator component.
    """

    _number_of_sequencers: int = NUMBER_OF_SEQUENCERS_QRM

    def __init__(self, instrument: pulsar_qrm.pulsar_qrm_qcodes, **kwargs) -> None:
        """Create a new instance of PulsarQRMComponent."""
        assert isinstance(instrument, pulsar_qrm.pulsar_qrm_qcodes)
        self._acquisition_manager: Optional[_QRMAcquisitionManager] = None
        """Holds all the acquisition related logic."""
        super().__init__(instrument, **kwargs)

    @property
    def instrument(self) -> pulsar_qrm.pulsar_qrm_qcodes:
        return super().instrument

    def retrieve_acquisition(self) -> Union[Dict[Tuple[int, int], Any], None]:
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
        seq_name_to_idx_map = {
            f"seq{idx}": idx for idx in range(self._number_of_sequencers)
        }
        if "acq_mapping" in program:  # Resets everything to do with acquisition.
            acq_mapping = program.pop("acq_mapping")
            self._acquisition_manager = _QRMAcquisitionManager(
                self, self._number_of_sequencers, acquisition_mapping=acq_mapping
            )
        else:
            self._acquisition_manager = None

        if "settings" in program:
            settings_entry = program.pop("settings")
            pulsar_settings = PulsarSettings.from_dict(settings_entry)
            if self._acquisition_manager is not None:
                self._acquisition_manager.scope_mode_sequencer = (
                    pulsar_settings.scope_mode_sequencer
                )
            self._configure_global_settings(pulsar_settings)

        for path in [0, 1]:
            self.instrument.set(f"scope_acq_trigger_mode_path{path}", "sequencer")
            self.instrument.set(f"scope_acq_avg_mode_en_path{path}", True)

        for seq_name, seq_cfg in program.items():
            if seq_name in seq_name_to_idx_map:
                seq_idx = seq_name_to_idx_map[seq_name]
            else:
                raise KeyError(
                    f"Invalid program. Attempting to access non-existing sequencer with"
                    f' name "{seq_name}".'
                )
            if "settings" in seq_cfg:
                seq_settings = SequencerSettings.from_dict(seq_cfg["settings"])
                self._configure_sequencer_settings(
                    seq_idx=seq_idx, settings=seq_settings
                )

            self.instrument.set(
                f"sequencer{seq_idx}_waveforms_and_program", seq_cfg["seq_fn"]
            )

            self.instrument.arm_sequencer(sequencer=seq_idx)

    def _configure_sequencer_settings(
        self, seq_idx: int, settings: SequencerSettings
    ) -> None:
        super()._configure_sequencer_settings(seq_idx, settings)
        self.instrument.set(
            f"sequencer{seq_idx}_integration_length_acq",
            settings.integration_length_acq,
        )
        self.instrument.set(f"sequencer{seq_idx}_demod_en_acq", settings.nco_en)


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
        parent: PulsarQRMComponent,
        number_of_sequencers: int,
        acquisition_mapping: Dict[Tuple[int, int], Tuple[str, str]],
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
        """
        self.parent: PulsarQRMComponent = parent
        self.number_of_sequencers: int = number_of_sequencers
        self.acquisition_mapping: Dict[
            Tuple[int, int], Tuple[str, str]
        ] = acquisition_mapping
        self.scope_mode_sequencer: Optional[str] = None
        self.seq_name_to_idx_map = {
            f"seq{idx}": idx for idx in range(number_of_sequencers)
        }

    @property
    def instrument(self):
        """Returns the QRM driver from the parent IC component."""
        return self.parent.instrument

    def retrieve_acquisition(self) -> Dict[Tuple[int, int], Any]:
        """
        Retrieves all the acquisition data in the correct format.

        Returns
        -------
        :
            The acquisitions with the protocols specified in the `acq_mapping` as a
            `dict` with the `(acq_channel, acq_index)` as keys.
        """
        protocol_to_function_mapping = {
            # Implicitly covers SSBIntegrationComplex too
            "weighted_integrated_complex": self._get_integration_data,
            "trace": self._get_scope_data,
            # NB thresholded protocol is still missing since there is nothing in
            # the acquisition library for it yet.
        }
        self._store_scope_acquisition()

        formatted_acquisitions: Dict[AcquisitionIndexing, Any] = dict()
        for seq_idx in range(self.number_of_sequencers):
            acquisitions = self.instrument.get_acquisitions(seq_idx)
            for acq_channel, acq_index in self.acquisition_mapping.keys():
                if self._get_sequencer_index(acq_channel, acq_index) != seq_idx:
                    continue

                protocol = self._get_protocol(acq_channel, acq_index)
                acquisition_function: Callable = protocol_to_function_mapping[protocol]

                formatted_acquisitions[
                    AcquisitionIndexing(acq_channel=acq_channel, acq_index=acq_index)
                ] = acquisition_function(acquisitions, acq_channel, acq_index)
        return formatted_acquisitions

    def _store_scope_acquisition(self):
        sequencer_index = self.seq_name_to_idx_map.get(self.scope_mode_sequencer)
        if sequencer_index is None:
            return

        if sequencer_index > self.number_of_sequencers:
            raise ValueError(
                f"Attempting to retrieve scope mode data from sequencer "
                f"{sequencer_index}. QRM has only "
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

    def _get_sequencer_index(self, acq_channel, acq_index) -> str:
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
                ch_and_idx: AcquisitionIndexing = key
        return ch_and_idx

    def _get_scope_data(
        self, acquisitions: dict, acq_channel: int = 0, acq_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the scope mode acquisition associated with `acq_channel` and
        `acq_index`.

        Parameters
        ----------
        acq_channel
            The acq_channel to get the scope mode acquisition for.
        acq_index
            The acq_index to get the scope mode acquisition for.

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
                    f"acq_channel={acq_channel} and acq_index={acq_index} was "
                    f"out-of-range."
                )
        # NB hardware already divides by avg_count for scope mode
        scope_data_i = scope_data["path0"]["data"]
        scope_data_q = scope_data["path1"]["data"]
        return scope_data_i, scope_data_q

    def _get_integration_data(
        self, acquisitions: dict, acq_channel: int = 0, acq_index: int = 0
    ) -> Tuple[float, float]:
        """
        Retrieves the integrated acquisition data associated with `acq_channel` and
        `acq_index`.

        Parameters
        ----------
        acquisitions
            The acquisitions dict as returned by the sequencer.
        acq_channel
            The acq_channel to get integrated acquisition data for.
        acq_index
            The acq_index to get the integrated acquisition data for.

        Returns
        -------
        i_data
            The integrated data for path0.
        q_data
            The integrated data for path1.
        """
        bin_data = self._get_bin_data(acquisitions, acq_channel)
        i_data, q_data = (
            bin_data["integration"]["path0"],
            bin_data["integration"]["path1"],
        )
        if acq_index > len(i_data):
            raise ValueError(
                f"Attempting to access acq_index {acq_index} on "
                f"{self.parent.name} but only {len(i_data)} values found "
                f"in acquisition data."
            )
        return i_data[acq_index], q_data[acq_index]

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
