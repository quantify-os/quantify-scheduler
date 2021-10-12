# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing Zurich Instruments InstrumentCoordinator Components."""
# pylint: disable=useless-super-delegation
# pylint: disable=too-many-arguments
# pylint: disable=too-many-ancestors

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from zhinst import qcodes
from quantify_core.data import handling
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.instrument_coordinator.components import base
from quantify_scheduler.backends.zhinst.settings import ZISerializeSettings

if TYPE_CHECKING:
    from zhinst.qcodes.base import ZIBaseInstrument
    from quantify_scheduler.backends.zhinst_backend import ZIDeviceConfig
    from quantify_scheduler.backends.zhinst.settings import ZISettings


def convert_to_instrument_coordinator_format(acquisition_results):
    """
    Converts the acquisition results format of the UHFQA component to
    the format required by InstrumentCoordinator.
    Converts from `Dict[int, np.ndarray]` to `Dict[Tuple[int, int], Any]`.
    """
    reformatted_results: Dict[Tuple[int, int], Any] = dict()
    for acq_channel in acquisition_results:
        results_array = acquisition_results.get(acq_channel)
        for i, complex_value in enumerate(results_array):
            separated_value = (np.real(complex_value), np.imag(complex_value))
            reformatted_results[(acq_channel, i)] = separated_value
    return reformatted_results


class ZIInstrumentCoordinatorComponent(base.InstrumentCoordinatorComponentBase):
    """Zurich Instruments InstrumentCoordinator component base class."""

    def __init__(self, instrument: ZIBaseInstrument, **kwargs) -> None:
        """Create a new instance of ZIInstrumentCoordinatorComponent."""
        super().__init__(instrument, **kwargs)
        self.device_config: Optional[ZIDeviceConfig] = None
        self.zi_settings: Optional[ZISettings] = None
        self._data_path: Path = Path(".")

    @property
    def is_running(self) -> bool:
        raise NotImplementedError()

    def prepare(self, options: ZIDeviceConfig) -> None:
        """
        Prepare the InstrumentCoordinator component with configuration
        required to arm the instrument.

        Parameters
        ----------
        options :
            The ZI instrument configuration.
        """
        self.device_config = options

        self.zi_settings = self.device_config.settings_builder.build()

        print(f"{self.device_config=}")
        print(f"{self.zi_settings=}")

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

    def prepare(self, options: ZIDeviceConfig) -> None:
        print(f"hdawg_config: {options=}")
        super().prepare(options)

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

    def prepare(self, options: ZIDeviceConfig) -> None:
        """
        Prepares the component with configurations
        required to arm the instrument.
        After this step is complete, the waveform file is uploaded
        to the LabOne WebServer.
        """

        super().prepare(options)
        self._data_path = Path(handling.get_datadir())

        # Copy the UHFQA waveforms to the waves directory
        # This is required before compilation.
        waves_path: Path = zi_helpers.get_waves_directory(self.instrument.awg)
        wave_files = list(self._data_path.glob(f"{self.name}*.csv"))
        for file in wave_files:
            shutil.copy2(str(file), str(waves_path))
        print(f"uhfqa_config: {options=}")

    def retrieve_acquisition(self) -> Dict[int, np.ndarray]:
        if self.device_config is None:
            raise RuntimeError("Undefined device config, first prepare UHFQA!")

        acq_config = self.device_config.acq_config

        acq_channel_results: Dict[int, np.ndarray] = dict()
        for acq_channel, resolve in acq_config.resolvers.items():
            acq_channel_results[acq_channel] = resolve(uhfqa=self.instrument)

        reformatted_results = convert_to_instrument_coordinator_format(
            acq_channel_results
        )

        return reformatted_results

    def wait_done(self, timeout_sec: int = 10) -> None:
        self.instrument.awg.wait_done(timeout_sec)
