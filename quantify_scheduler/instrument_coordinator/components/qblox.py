# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing Qblox InstrumentCoordinator Components."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from dataclasses import dataclass
import logging

import copy
from typing_extensions import Literal

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

    @property
    def is_running(self) -> bool:
        raise NotImplementedError()


# pylint: disable=too-many-ancestors
class PulsarQCMComponent(PulsarInstrumentCoordinatorComponent):
    """
    Pulsar QCM specific InstrumentCoordinator component.
    """

    number_of_sequencers: int = NUMBER_OF_SEQUENCERS_QCM
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

            self.instrument.set(
                f"sequencer{seq_idx}_waveforms_and_program", seq_cfg["seq_fn"]
            )

            self.instrument.arm_sequencer(sequencer=seq_idx)

    def start(self) -> None:
        """
        Starts execution of the schedule.
        """
        for seq_idx in [0, 1]:
            state = self.instrument.get_sequencer_state(seq_idx)
            if state["status"] == "ARMED":
                self.instrument.start_sequencer(seq_idx)

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

    def wait_done(self, timeout_sec: int = 10) -> None:
        pass


# pylint: disable=too-many-ancestors
class PulsarQRMComponent(PulsarInstrumentCoordinatorComponent):
    """
    Pulsar QRM specific InstrumentCoordinator component.
    """

    number_of_sequencers: int = NUMBER_OF_SEQUENCERS_QRM

    def __init__(self, instrument: pulsar_qrm.pulsar_qrm_qcodes, **kwargs) -> None:
        """Create a new instance of PulsarQRMComponent."""
        assert isinstance(instrument, pulsar_qrm.pulsar_qrm_qcodes)
        super().__init__(instrument, **kwargs)
        self._acq_settings: Optional[_AcquisitionSettings] = None

    @property
    def instrument(self) -> pulsar_qrm.pulsar_qrm_qcodes:
        return super().instrument

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

        self.instrument.delete_acquisitions(sequencer=0)
        if not isinstance(self.instrument, pulsar_qrm.pulsar_qrm_dummy):
            self.instrument.get_sequencer_state(0, 10, 0.01)
            self.instrument.get_acquisition_state(
                sequencer=0, timeout=10, timeout_poll_res=0.1
            )

        self.instrument.store_acquisition(0, msmt_id, num_of_samples)
        acq: Dict[str, dict] = self.instrument.get_acquisitions(0)

        hardware_averages: int = self._acq_settings.hardware_averages
        duration: int = self._acq_settings.duration_ns

        path_labels = ("path_0", "path_1")
        traces = [None] * len(path_labels)
        for path_idx, label in enumerate(path_labels):
            if acq[msmt_id][label]["out-of-range"]:
                logger.warning(
                    f"ADC out-of-range of {self.instrument.name} on {label}."
                )

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
        self, i_trace: np.ndarray, q_trace: np.ndarray,
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
                self.instrument.set(
                    f"sequencer{seq_idx}_trigger_mode_acq_path{path}", "sequencer"
                )
                self.instrument.set(
                    f"sequencer{seq_idx}_avg_mode_en_acq_path{path}", True
                )

            self._acq_settings = acq_settings

            self.instrument.set(
                f"sequencer{seq_idx}_waveforms_and_program", seq_cfg["seq_fn"]
            )

            self.instrument.arm_sequencer(sequencer=seq_idx)

    def _configure_global_settings(self, settings: PulsarSettings):
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

    def start(self) -> None:
        """
        Starts execution of the schedule.
        """
        state = self.instrument.get_sequencer_state(0)
        if state["status"] == "ARMED":
            self.instrument.start_sequencer(0)

    def stop(self) -> None:
        """
        Stops all execution.
        """
        self.instrument.stop_sequencer()

    def wait_done(self, timeout_sec: int = 10) -> None:
        pass


class PulsarQCMRFComponent(PulsarQCMComponent):
    """
    Pulsar QCM-RF specific control stack component.
    """

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

        nco_en: bool = settings.nco_en
        self.set(f"sequencer{seq_idx}_mod_en_awg", nco_en)
        if nco_en:
            self.set(f"sequencer{seq_idx}_nco_freq", settings.modulation_freq)

    def _configure_global_settings(self, settings: PulsarSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        super()._configure_global_settings(settings)
        if settings.lo0_freq:
            self.set("lo0_freq", settings.lo0_freq)
        if settings.lo1_freq:
            self.set("lo1_freq", settings.lo1_freq)

        if settings.offset_I_ch0:
            self.set("offset_I_ch0", settings.offset_I_ch0)
        if settings.offset_Q_ch0:
            self.set("offset_Q_ch0", settings.offset_Q_ch0)
        if settings.offset_I_ch1:
            self.set("offset_I_ch1", settings.offset_I_ch1)
        if settings.offset_Q_ch1:
            self.set("offset_Q_ch1", settings.offset_Q_ch1)


class PulsarQRMRFComponent(PulsarQRMComponent):
    """
    Pulsar QRM-RF specific stack component.
    """


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

        nco_en: bool = settings.nco_en
        self.set(f"sequencer{seq_idx}_mod_en_awg", nco_en)
        if nco_en:
            self.set(f"sequencer{seq_idx}_nco_freq", settings.modulation_freq)

    def _configure_global_settings(self, settings: PulsarSettings):
        """
        Configures all settings that are set globally for the whole instrument.

        Parameters
        ----------
        settings
            The settings to configure it to.
        """
        super()._configure_global_settings(settings)
        if settings.lo0_freq:
            self.set("lo0_freq", settings.lo0_freq)
        if settings.lo1_freq:
            self.set("lo1_freq", settings.lo1_freq)
        # if settings.offset_I_ch0:
        #     self.set("offset_I_ch0", settings.offset_I_ch0)
        # if settings.offset_Q_ch0:
        #     self.set("offset_Q_ch0", settings.offset_Q_ch0)


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
