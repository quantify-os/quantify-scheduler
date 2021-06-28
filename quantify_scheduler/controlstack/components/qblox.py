# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing Qblox ControlStack Components."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from dataclasses import dataclass
import logging

import copy
from typing_extensions import Literal

import numpy as np
from pulsar_qcm import pulsar_qcm
from pulsar_qrm import pulsar_qrm
from quantify_scheduler.controlstack.components import base
from quantify_scheduler.helpers.waveforms import modulate_waveform
from quantify_scheduler.backends.types.qblox import PulsarSettings, SequencerSettings
from quantify_scheduler.backends.qblox.constants import (
    NUMBER_OF_SEQUENCERS_QCM,
    NUMBER_OF_SEQUENCERS_QRM,
)

logger = logging.getLogger(__name__)

# pylint: disable=too-many-ancestors
class PulsarQCMComponent(
    pulsar_qcm.pulsar_qcm_qcodes, base.AbstractControlStackComponent
):
    """
    Pulsar QCM specific control stack component.
    """

    number_of_sequencers: int = NUMBER_OF_SEQUENCERS_QCM
    """Specifies the amount of sequencers available to this QCM."""

    def __init__(self, name, host, port=5025, debug=0) -> None:
        if host == "dummy":
            debug = 1
            transport_inst = pulsar_qcm.pulsar_dummy_transport(
                pulsar_qcm.pulsar_qcm_ifc._get_sequencer_cfg_format()
            )
            self._dummy_instr = True
        else:
            transport_inst = pulsar_qcm.ip_transport(host=host, port=port)
            self._dummy_instr = False
        super().__init__(name, transport_inst, debug=debug)

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
        Makes the devices in the control stack ready for execution of a schedule.

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
            f"seq{idx}": idx for idx in range(self.number_of_sequencers)
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

            self.set(f"sequencer{seq_idx}_waveforms_and_program", seq_cfg["seq_fn"])

            self.arm_sequencer(sequencer=seq_idx)

    def start(self) -> None:
        """
        Starts execution of the schedule.
        """
        self.start_sequencer()

    def stop(self) -> None:
        """
        Stops all execution.
        """
        self.stop_sequencer()

    def _configure_global_settings(self, settings: PulsarSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        self.set("reference_source", settings.ref)

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
        self.set(f"sequencer{seq_idx}_sync_en", settings.sync_en)
        self.set(f"sequencer{seq_idx}_offset_awg_path0", settings.awg_offset_path_0)
        self.set(f"sequencer{seq_idx}_offset_awg_path1", settings.awg_offset_path_1)

        nco_en: bool = settings.nco_en
        self.set(f"sequencer{seq_idx}_mod_en_awg", nco_en)
        if nco_en:
            self.set(f"sequencer{seq_idx}_nco_freq", settings.modulation_freq)

    def wait_done(self, timeout_sec: int = 10) -> None:
        pass


# pylint: disable=too-many-ancestors
class PulsarQRMComponent(
    pulsar_qrm.pulsar_qrm_qcodes, base.AbstractControlStackComponent
):
    """
    Pulsar QRM specific stack component.
    """

    number_of_sequencers: int = NUMBER_OF_SEQUENCERS_QRM

    def __init__(self, name, host, port=5025, debug=0) -> None:
        if host == "dummy":
            debug = 1
            transport_inst = pulsar_qcm.pulsar_dummy_transport(
                pulsar_qrm.pulsar_qrm_ifc._get_sequencer_cfg_format()
            )
            self._dummy_instr = True
        else:
            transport_inst = pulsar_qrm.ip_transport(host=host, port=port)
            self._dummy_instr = False
        super().__init__(name, transport_inst, debug=debug)

        self._acq_settings = None

    @property
    def is_running(self) -> bool:
        return False

    # pylint: disable=arguments-differ
    def retrieve_acquisition(self, num_of_samples: int = 2 ** 16) -> Any:
        """
        Retrieves the latest acquisition results.

        Parameters
        ----------
        num_of_samples
            Unsigned integer representing the number of data points to acquire.

        Returns
        -------
        :
            The acquired data.
        """

        msmt_id: str = "msmt_00000"

        self.delete_acquisitions(sequencer=0)
        if not self._dummy_instr:
            self.get_sequencer_state(0, 10, 0.01)
            self.get_acquisition_state(sequencer=0, timeout=10, timeout_poll_res=0.1)

        self.store_acquisition(0, msmt_id, num_of_samples)
        acq: Dict[str, dict] = self.get_acquisitions(0)

        hardware_averages: int = self._acq_settings.hardware_averages
        duration: int = self._acq_settings.duration_ns

        path_labels = ("path_0", "path_1")
        traces = [None] * len(path_labels)
        for path_idx, label in enumerate(path_labels):
            if acq[msmt_id][label]["out-of-range"]:
                logger.warning(f"ADC out-of-range of {self.name} on {label}.")

            traces[path_idx] = (
                np.array(acq[msmt_id][label]["data"][:duration]) / hardware_averages
            )
        i_trace, q_trace = traces

        acq_processing_func: Callable[..., Any] = {
            "SSBIntegrationComplex": self._acquire_ssb_integration_complex,
        }.get(self._acq_settings.acq_mode, None)

        if acq_processing_func is not None:
            i_trace, q_trace = acq_processing_func(i_trace, q_trace)

        return i_trace, q_trace

    def _acquire_ssb_integration_complex(
        self,
        i_trace: np.ndarray,
        q_trace: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Performs the required transformation to obtain a
        single phasor from the acquired I and Q traces.

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
        Makes the devices in the control stack ready for execution of a schedule.

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
            f"seq{idx}": idx for idx in range(self.number_of_sequencers)
        }
        acq_settings = _AcquisitionSettings()
        if "settings" in program:
            settings_entry = program.pop("settings")
            pulsar_settings = PulsarSettings.from_dict(settings_entry)
            self._configure_global_settings(pulsar_settings)

            acq_settings.hardware_averages = pulsar_settings.hardware_averages
            acq_settings.acq_mode = pulsar_settings.acq_mode

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
                acq_settings.modulation_freq = seq_settings.modulation_freq
                acq_settings.duration_ns = seq_settings.duration

            for path in [0, 1]:
                self.set(f"sequencer{seq_idx}_trigger_mode_acq_path{path}", "sequencer")
                self.set(f"sequencer{seq_idx}_avg_mode_en_acq_path{path}", True)

            self._acq_settings = acq_settings

            self.set(f"sequencer{seq_idx}_waveforms_and_program", seq_cfg["seq_fn"])

            self.arm_sequencer(sequencer=seq_idx)

    def _configure_global_settings(self, settings: PulsarSettings):
        self.set("reference_source", settings.ref)

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
        self.set(f"sequencer{seq_idx}_sync_en", settings.sync_en)
        self.set(f"sequencer{seq_idx}_offset_awg_path0", settings.awg_offset_path_0)
        self.set(f"sequencer{seq_idx}_offset_awg_path1", settings.awg_offset_path_1)

        nco_en: bool = settings.nco_en
        self.set(f"sequencer{seq_idx}_mod_en_awg", nco_en)
        if nco_en:
            self.set(f"sequencer{seq_idx}_nco_freq", settings.modulation_freq)

    def start(self) -> None:
        """
        Starts execution of the schedule.
        """
        self.start_sequencer()

    def stop(self) -> None:
        """
        Stops all execution.
        """
        self.stop_sequencer()

    def wait_done(self, timeout_sec: int = 10) -> None:
        pass


# ----------------- Utility -----------------


@dataclass
class _AcquisitionSettings:
    """Holds all information required to perform and process the acquisition."""

    duration_ns: int = 0
    """Duration of the acquisition trace."""
    acq_mode: Literal["raw_trace", "SSBIntegrationComplex"] = "SSBIntegrationComplex"
    """
    Current mode of the acquisition to use. This effectively specifies the data
    processing function.
    """
    hardware_averages: int = 1
    """The number of hardware averages to use."""
    modulation_freq: float = 0
    """The modulation frequency used. Used for digital demodulation."""


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
