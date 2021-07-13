# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing Qblox InstrumentCoordinator Components."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from dataclasses import dataclass
import logging

import copy
from typing_extensions import Literal
from collections import namedtuple

import numpy as np
from pulsar_qcm import pulsar_qcm
from pulsar_qrm import pulsar_qrm
from qcodes.instrument.base import Instrument
from quantify_scheduler.instrument_coordinator.components import base
from quantify_scheduler.helpers.waveforms import modulate_waveform
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

    @abstracmethod
    @property
    def _number_of_sequencers(self) -> int:
        """"""

    @property
    def is_running(self) -> bool:
        raise False

    def wait_done(self, timeout_sec: int = 10) -> None:
        timeout_min = timeout_sec // 60
        if timeout_min == 0:
            timeout_min = 1
        for idx in range(self.number_of_sequencers):
            state = self.instrument.get_sequencer_state(idx, timeout_min)

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

    def _configure_global_settings(self, settings: PulsarSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        self.instrument.set("reference_source", settings.ref)

    def _configure_sequencer_settings(self, seq_idx: int, settings: SequencerSettings):
        """
        Configures all sequencer specific settings.

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

    @property
    def is_running(self) -> bool:
        return False

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
        Makes the devices in the InstrumentCoordinator ready for execution of a
        schedule.

        This involves uploading the waveforms and programs to the sequencers as well as
        configuring all the settings required. Keep in mind that values set directly
        through the driver may be overridden (e.g. the offsets will be set according to
        the specified mixer calibration parameters).

        Parameters
        ----------
        options
            Program to upload to the sequencers. The key is a sequencer, e.g.,
            :code:`"seq0"`, or :code:`"settings"`,
            the value is the global settings dict or a sequencer specific configuration.
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
        super().__init__(instrument, **kwargs)

    @property
    def instrument(self) -> pulsar_qrm.pulsar_qrm_qcodes:
        return super().instrument

    @property
    def is_running(self) -> bool:
        return False

    # pylint: disable=arguments-differ
    def retrieve_acquisition(self, acq_channel: int = 0, acq_index: int = 0) -> Any:
        """
        Retrieves the latest acquisition results.

        Parameters
        ----------
        acq_channel
            TODO
        acq_index
            TODO

        Returns
        -------
        :
            The acquired data.
        """
        acquisition_function = self._get_integration_data
        return acquisition_function(acq_channel, acq_index)

    def _get_scope_data(
        self, acq_channel: int = 0, acq_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        seq_name_to_idx_map = {
            f"seq{idx}": idx for idx in range(self._number_of_sequencers)
        }
        sequencer_index = seq_name_to_idx_map.get(self._settings.scope_mode_sequencer)
        if sequencer_index is None:
            raise ValueError(
                f"Attempting to retrieve scope mode data, while no "
                f"sequencer has been assigned to perform this in the "
                f"compilation."
            )
        if sequencer_index > self._number_of_sequencers:
            raise ValueError(
                f"Attempting to retrieve scope mode data from sequencer "
                f"{sequencer_index}, even though the QRM only has "
                f"{self._number_of_sequencers} sequencers."
            )
        acq_name = _channel_index_to_channel_name(acq_channel)
        self.instrument.store_scope_acquisition(sequencer_index, acq_name)
        acquisitions = self.instrument.get_acquisitions(sequencer_index)
        scope_data = acquisitions[acq_name]["acquisition"]["scope"]
        for path_label in ("path0", "path1"):
            if scope_data[path_label]["out-of-range"]:
                logger.warning(
                    f"The scope mode data of {path_label} of sequencer "
                    f"{sequencer_index} of {self.name} is out-of-range."
                )
        scope_data_i = scope_data["path0"]["data"]
        scope_data_q = scope_data["path1"]["data"]
        return scope_data_i, scope_data_q

    def _get_bin_data(self, acq_channel: int = 0) -> dict:
        acquisitions = self.instrument.get_acquisitions(0)
        acq_name = _channel_index_to_channel_name(acq_channel)
        channel_data = acquisitions[acq_name]
        if channel_data["index"] != acq_channel:
            raise RuntimeError(
                f"Name does not correspond to a valid acquisition for name {acq_name}, "
                f'which has index {channel_data["index"]}.'
            )
        return channel_data["acquisition"]["bins"]

    def _get_integration_data(
        self, acq_channel: int = 0, acq_index: int = 0
    ) -> Tuple[float, float]:
        bin_data = self._get_bin_data(acq_channel)
        i_data, q_data = (
            bin_data["integration"]["path0"],
            bin_data["integration"]["path1"],
        )
        if acq_index > len(i_data):
            raise ValueError(
                f"Attempting to access acq_index {acq_index} on "
                f"{self.name} but only {len(i_data)} values found "
                f"in acquisition data."
            )
        avg_count = bin_data["avg_count"][acq_index]
        return i_data[acq_index] / avg_count, q_data[acq_index] / avg_count

    def _get_threshold_data(self, acq_channel: int = 0, acq_index: int = 0):
        bin_data = self._get_bin_data(acq_channel)
        i_data, q_data = (
            bin_data["threshold"]["path0"],
            bin_data["threshold"]["path1"],
        )
        if acq_index > len(i_data):
            raise ValueError(
                f"Attempting to access acq_index {acq_index} on "
                f"{self.name} but only {len(i_data)} values found "
                f"in acquisition data."
            )
        return i_data[acq_index], q_data[acq_index]

    def _acquire_ssb_integration_complex(
        self,
        i_trace: np.ndarray,
        q_trace: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Performs the required transformation to obtain a
        single phasor from the acquired I and Q traces
        in software.

        Parameters
        ----------
        i_trace
            The data of the acquisition from the I path.
        q_trace
            The data of the acquisition from the Q path.

        Returns
        -------
        :
            The static phasor extracted from the data.
        """
        interm_freq = self._acq_settings.modulation_freq
        demod_trace_complex = _demodulate_trace(interm_freq, i_trace, q_trace)
        i_demod, q_demod = demod_trace_complex.real, demod_trace_complex.imag

        return np.average(i_demod), np.average(q_demod)

    def prepare(self, options: Dict[str, dict]) -> None:
        """
        Makes the devices in the InstrumentCoordinator ready for execution of a
        schedule.

        This involves uploading the waveforms and programs to the sequencers as well as
        configuring all the settings required. Keep in mind that values set directly
        through the driver may be overridden (e.g. the offsets will be set according to
        the specified mixer calibration parameters).

        Parameters
        ----------
        options
            Program to upload to the sequencers. The key is a sequencer or "settings",
            the value is the global settings dict or a sequencer specific configuration.
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

            for path in [0, 1]:
                self.instrument.set(
                    f"sequencer{seq_idx}_trigger_mode_acq_path{path}", "sequencer"
                )
                self.instrument.set(
                    f"sequencer{seq_idx}_avg_mode_en_acq_path{path}", True
                )

            self.instrument.set(
                f"sequencer{seq_idx}_waveforms_and_program", seq_cfg["seq_fn"]
            )

            self.instrument.arm_sequencer(sequencer=seq_idx)


# ----------------- Utility -----------------
def _channel_index_to_channel_name(index: int) -> str:
    return str(index)


def _demodulate_trace(
    demod_freq: float,
    trace_i: np.ndarray,
    trace_q: np.ndarray,
    sampling_rate: float = 1e9,
) -> np.ndarray:
    """
    Digital demodulation of traces.

    Parameters
    ----------
    demod_freq
        Frequency to use for demodulation.
    trace_i
        I data to demodulate.
    trace_q
        Q data to demodulate.
    sampling_rate
        Sampling rate of the data (Hz).

    Returns
    -------
    :
        The demodulated data.
    """

    complex_signal = trace_i + 1.0j * trace_q
    complex_signal -= np.average(complex_signal)

    tbase = np.arange(0, len(complex_signal), 1) / sampling_rate

    return modulate_waveform(t=tbase, envelope=complex_signal, freq=-demod_freq)
