# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing Zurich Instruments ControlStack Components."""
# pylint: disable=useless-super-delegation
# pylint: disable=too-many-arguments
# pylint: disable=too-many-ancestors

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, TYPE_CHECKING, Any, Optional

from zhinst import qcodes
from quantify_core.data import handling
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.controlstack.components import base

if TYPE_CHECKING:
    import numpy as np
    from zhinst.qcodes.base import ZIBaseInstrument
    from quantify_scheduler.backends.zhinst_backend import ZIDeviceConfig
    from quantify_scheduler.backends.zhinst.settings import ZISettings


class ZIControlStackComponent(base.ControlStackComponentBase):
    """Zurich Instruments ControlStack component base class."""

    def __init__(self, instrument: ZIBaseInstrument, **kwargs) -> None:
        """Create a new instance of ZIControlStackComponent."""
        super().__init__(instrument, **kwargs)
        self.device_config: Optional[ZIDeviceConfig] = None
        self.zi_settings: Optional[ZISettings] = None
        self._data_path: Path = Path(".")

    @property
    def is_running(self) -> bool:
        raise NotImplementedError()

    def prepare(self, options: ZIDeviceConfig) -> None:
        """
        Prepare the ControlStack component with configuration
        required to arm the instrument.

        Parameters
        ----------
        options :
            The ZI instrument configuration.
        """
        self.device_config = options

        self.zi_settings = self.device_config.settings_builder.build()

        # Writes settings to filestorage
        self._data_path = Path(handling.get_datadir())
        self.zi_settings.serialize(self._data_path, self.instrument)

        # Upload settings, seqc and waveforms
        self.zi_settings.apply(self.instrument)

    def retrieve_acquisition(self) -> Any:
        return None


class HDAWGControlStackComponent(ZIControlStackComponent):
    """Zurich Instruments HDAWG ControlStack Component class."""

    def __init__(self, instrument: qcodes.HDAWG, **kwargs) -> None:
        """Create a new instance of HDAWGControlStackComponent."""
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
        super().prepare(options)

    def retrieve_acquisition(self) -> Any:
        return None

    def wait_done(self, timeout_sec: int = 10) -> None:
        for awg_index in reversed(self.zi_settings.awg_indexes):
            self.get_awg(awg_index).wait_done(timeout_sec)


class UHFQAControlStackComponent(ZIControlStackComponent):
    """Zurich Instruments UHFQA ControlStack Component class."""

    def __init__(self, instrument: qcodes.UHFQA, **kwargs) -> None:
        """Create a new instance of UHFQAControlStackComponent."""
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
        self._data_path = Path(handling.get_datadir())

        # Copy the UHFQA waveforms to the waves directory
        # This is required before compilation.
        waves_path: Path = zi_helpers.get_waves_directory(self.instrument.awg)
        wave_files = self._data_path.cwd().glob(f"{self.instrument.name}*.csv")
        for file in wave_files:
            shutil.copy2(str(file), str(waves_path))

        super().prepare(options)

    def retrieve_acquisition(self) -> Dict[int, np.ndarray]:
        if self.device_config is None:
            raise RuntimeError("Undefined device config, first prepare UHFQA!")

        acq_config = self.device_config.acq_config

        acq_channel_results: Dict[int, np.ndarray] = dict()
        for acq_channel, resolve in acq_config.resolvers.items():
            acq_channel_results[acq_channel] = resolve(uhfqa=self)

        return acq_channel_results

    def wait_done(self, timeout_sec: int = 10) -> None:
        self.instrument.awg.wait_done(timeout_sec)
