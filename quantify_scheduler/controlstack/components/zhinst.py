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
    from quantify_scheduler.backends.zhinst_backend import ZIDeviceConfig
    from quantify_scheduler.backends.zhinst.settings import ZISettings
    import numpy as np


class ZIControlStackComponent(base.AbstractControlStackComponent):
    """Zurich Instruments ControlStack component base class."""

    def __init__(self) -> None:
        """Create a new instance on ZIControlStackComponent."""
        super().__init__()
        self.device_config: Optional[ZIDeviceConfig] = None
        self.zi_settings: Optional[ZISettings] = None
        self._data_path: Path = Path(".")

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
        self.zi_settings.serialize(self._data_path, self)

        # Upload settings, seqc and waveforms
        self.zi_settings.apply(self)

    def retrieve_acquisition(self) -> Any:
        return None


class HDAWGControlStackComponent(qcodes.HDAWG, ZIControlStackComponent):
    """Zurich Instruments HDAWG ControlStack Component class."""

    def __init__(
        self,
        name: str,
        serial: str,
        interface: str = "1gbe",
        host: str = "localhost",
        port: int = 8004,
        api: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(  # pylint: disable=too-many-function-args
            name, serial, interface, host, port, api, **kwargs
        )

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
        return self.awgs[index]

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


class UHFQAControlStackComponent(qcodes.UHFQA, ZIControlStackComponent):
    """Zurich Instruments UHFQA ControlStack Component class."""

    def __init__(
        self,
        name: str,
        serial: str,
        interface: str = "1gbe",
        host: str = "localhost",
        port: int = 8004,
        api: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(  # pylint: disable=too-many-function-args
            name, serial, interface, host, port, api, **kwargs
        )

    @property
    def is_running(self) -> bool:
        return self.awg.is_running

    def start(self) -> None:
        self.awg.run()

    def stop(self) -> None:
        self.awg.stop()

    def prepare(self, options: ZIDeviceConfig) -> None:
        self._data_path = Path(handling.get_datadir())

        # Copy the UHFQA waveforms to the waves directory
        # This is required before compilation.
        waves_path: Path = zi_helpers.get_waves_directory(self.awg)
        wave_files = self._data_path.cwd().glob(f"{self.name}*.csv")
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
        self.awg.wait_done(timeout_sec)
