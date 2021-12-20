# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing Zurich Instruments InstrumentCoordinator Components."""
# pylint: disable=useless-super-delegation
# pylint: disable=too-many-arguments
# pylint: disable=too-many-ancestors

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from quantify_core.data import handling
from zhinst import qcodes

from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst.settings import ZISerializeSettings
from quantify_scheduler.instrument_coordinator.components import base

if TYPE_CHECKING:
    from zhinst.qcodes.base import ZIBaseInstrument

    from quantify_scheduler.backends.zhinst.settings import ZISettings
    from quantify_scheduler.backends.zhinst_backend import ZIDeviceConfig


logger = logging.getLogger(__name__)


def convert_to_instrument_coordinator_format(acquisition_results, n_acquisitions: int):
    """
    Converts the acquisition results format of the UHFQA component to
    the format required by InstrumentCoordinator.
    Converts from `Dict[int, np.ndarray]` to `Dict[Tuple[int, int], Any]`.
    """
    reformatted_results: Dict[Tuple[int, int], Any] = dict()
    for acq_channel in acquisition_results:
        results_array = acquisition_results.get(acq_channel)
        # this case corresponds to a trace acquisition
        if n_acquisitions == 1 and len(results_array) > 1:
            reformatted_results[(acq_channel, 0)] = (
                np.real(results_array),
                np.imag(results_array),
            )
        else:
            for i, complex_value in enumerate(results_array):
                separated_value = (np.real(complex_value), np.imag(complex_value))
                reformatted_results[(acq_channel, i)] = separated_value
    return reformatted_results


class ZIInstrumentCoordinatorComponent(base.InstrumentCoordinatorComponentBase):
    """Zurich Instruments InstrumentCoordinator component base class."""

    def __init__(self, instrument: ZIBaseInstrument, **kwargs) -> None:
        """Create a new instance of ZIInstrumentCoordinatorComponent."""
        super().__init__(instrument, **kwargs)
        self.zi_device_config: Optional[ZIDeviceConfig] = None
        self.zi_settings: Optional[ZISettings] = None
        self._data_path: Path = Path(".")

    @property
    def is_running(self) -> bool:
        raise NotImplementedError()

    # pylint: disable=arguments-differ
    def prepare(self, zi_device_config: ZIDeviceConfig) -> bool:
        """
        Prepare the InstrumentCoordinator component with configuration
        required to arm the instrument.

        The preparation is skipped when the new zi_device_config is the same as that
        from the previous time prepare was called. This saves significant time overhead.

        Parameters
        ----------
        zi_device_config :
            The ZI instrument configuration. See the link for details of the
            configuration format.

        Returns
        -------
        :
            A boolean indicating if the ZI component was configured in this call.
        """
        self.zi_device_config = zi_device_config

        new_zi_settings = zi_device_config.settings_builder.build()
        old_zi_settings = self.zi_settings

        if new_zi_settings == old_zi_settings:
            logger.info(
                f"{self.name}: device config and settings "
                + "are identical! Compilation skipped."
            )
            return False

        logger.info(f"Configuring {self.name}.")
        # if the settings are not identical, update the attributes of the
        # ic component and apply the settings to the hardware.
        self.zi_settings = new_zi_settings

        # Writes settings to filestorage
        self._data_path = Path(handling.get_datadir())
        self.zi_settings.serialize(
            self._data_path,
            ZISerializeSettings(
                self.name, self.instrument._serial, self.instrument._type
            ),
        )

        # Upload settings, seqc and waveforms
        self.zi_settings.apply(self.instrument)

        return True

    def retrieve_acquisition(self) -> Any:
        return None


class HDAWGInstrumentCoordinatorComponent(ZIInstrumentCoordinatorComponent):
    """Zurich Instruments HDAWG InstrumentCoordinator Component class."""

    def __init__(self, instrument: qcodes.HDAWG, **kwargs) -> None:
        """Create a new instance of HDAWGInstrumentCoordinatorComponent."""
        assert isinstance(instrument, qcodes.HDAWG)
        super().__init__(instrument, **kwargs)

    @property
    def instrument(self) -> qcodes.HDAWG:
        return super().instrument

    @property
    def is_running(self) -> bool:
        return any(
            self.get_awg(awg_index).is_running
            for awg_index in self.zi_settings.awg_indexes
        )

    def get_awg(self, index: int) -> qcodes.hdawg.AWG:
        """
        Returns the AWG by index.

        Parameters
        ----------
        index :
            The awg index.

        Returns
        -------
        :
            The HDAWG AWG instance.
        """
        return self.instrument.awgs[index]

    def start(self) -> None:
        """Starts all HDAWG AWG(s) in reversed order by index."""
        for awg_index in reversed(self.zi_settings.awg_indexes):
            self.get_awg(awg_index).run()

    def stop(self) -> None:
        """Stops all HDAWG AWG(s) in order by index."""
        for awg_index in self.zi_settings.awg_indexes:
            self.get_awg(awg_index).stop()

    def retrieve_acquisition(self) -> Any:
        return None

    def wait_done(self, timeout_sec: int = 10) -> None:
        for awg_index in reversed(self.zi_settings.awg_indexes):
            self.get_awg(awg_index).wait_done(timeout_sec)


class UHFQAInstrumentCoordinatorComponent(ZIInstrumentCoordinatorComponent):
    """Zurich Instruments UHFQA InstrumentCoordinator Component class."""

    def __init__(self, instrument: qcodes.UHFQA, **kwargs) -> None:
        """Create a new instance of UHFQAInstrumentCoordinatorComponent."""
        assert isinstance(instrument, qcodes.UHFQA)
        super().__init__(instrument, **kwargs)

    @property
    def instrument(self) -> qcodes.UHFQA:
        return super().instrument

    @property
    def is_running(self) -> bool:
        return self.instrument.awg.is_running

    def start(self) -> None:
        self.instrument.awg.run()

    def stop(self) -> None:
        self.instrument.awg.stop()

    def prepare(self, zi_device_config: ZIDeviceConfig) -> bool:
        """
        Prepares the component with configurations
        required to arm the instrument.

        After this step is complete, the waveform file is uploaded
        to the LabOne WebServer.

        Parameters
        ----------
        zi_device_config :
            The ZI instrument configuration. See the link for details of the
            configuration format.

        Returns
        -------
        :
            A boolean indicating if the ZI component was configured in this call.
        """
        # always start by resetting the counters and stopping the AWG
        self.instrument.qas[0].result.enable(0)
        self.instrument.awg.stop()

        self.instrument.qas[0].result.reset(1)
        self.instrument.qas[0].result.enable(1)

        try:
            # if settings where identical, no configuration is needed.
            configure = super().prepare(zi_device_config)
            if configure is False:
                return False

        # pylint: disable=broad-except
        # the exception being raised is "Upload failed", but the ZI backend raises it
        # as a general exception.
        except Exception as e:
            # whenever a new UHF device is used for the first time,
            # certain waveform files will not exist. The lines below copy files so
            # that it is possible to read from that location.
            # this line of code should only be logging a warning the very first time
            # a new setup is used, and then resolve auto.
            logger.warning(e)
            configure = True

        self._data_path = Path(handling.get_datadir())
        # Copy the UHFQA waveforms to the waves directory
        # This is required before compilation.

        # N.B. note this copies waves that were written during compilation, but are not
        # contained in the zi_device_config that is passed as an argument here.
        waves_path: Path = zi_helpers.get_waves_directory(self.instrument.awg)
        wave_files = list(self._data_path.glob(f"{self.name}*.csv"))
        for file in wave_files:
            shutil.copy2(str(file), str(waves_path))

        # prepare twice to resolve issue with waveform memory not being updated
        # correctly. In practice, we see that integration weights update correctly, but
        # the waveforms in pulses do not. This problem is not fully understood, but this
        # resolves the issue at a minor overhead.

        if configure:
            # Upload settings, seqc and waveforms
            self.zi_settings.apply(self.instrument)
        return True

    def retrieve_acquisition(self) -> Dict[int, np.ndarray]:
        if self.zi_device_config is None:
            raise RuntimeError("Undefined device config, first prepare UHFQA!")

        acq_config = self.zi_device_config.acq_config

        acq_channel_results: Dict[int, np.ndarray] = dict()
        for acq_channel, resolve in acq_config.resolvers.items():
            acq_channel_results[acq_channel] = resolve(uhfqa=self.instrument)

        reformatted_results = convert_to_instrument_coordinator_format(
            acq_channel_results, n_acquisitions=acq_config.n_acquisitions
        )

        return reformatted_results

    def wait_done(self, timeout_sec: int = 10) -> None:
        self.instrument.awg.wait_done(timeout_sec)
