# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing Zurich Instruments InstrumentCoordinator Components."""


from __future__ import annotations

from quantify_scheduler.compatibility_check import check_zhinst_compatibility

check_zhinst_compatibility()

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable

import xarray
from zhinst import qcodes

from quantify_core.data import handling
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst.settings import ZISerializeSettings
from quantify_scheduler.enums import BinMode
from quantify_scheduler.instrument_coordinator.components import base

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from zhinst.qcodes.base import ZIBaseInstrument

    from quantify_scheduler.backends.zhinst.settings import ZISettings
    from quantify_scheduler.backends.zhinst_backend import ZIDeviceConfig
    from quantify_scheduler.schedules.schedule import CompiledSchedule


logger = logging.getLogger(__name__)


class AcquisitionProtocolNotSupportedError(NotImplementedError):
    pass


class ZIInstrumentCoordinatorComponent(base.InstrumentCoordinatorComponentBase):
    """Zurich Instruments InstrumentCoordinator component base class."""

    def __init__(
        self,
        instrument: ZIBaseInstrument,
        **kwargs: Any,  # noqa: ANN401 # Need to fix that in the parent class first.
    ) -> None:
        super().__init__(instrument, **kwargs)
        self.zi_device_config: ZIDeviceConfig | None = None
        self.zi_settings: ZISettings | None = None
        self._data_path: Path = Path(".")

    @property
    def is_running(self) -> bool:
        raise NotImplementedError()

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

        logger.info(f"Configuring {self.instrument.name}.")
        # if the settings are not identical, update the attributes of the
        # ic component and apply the settings to the hardware.
        self.zi_settings = new_zi_settings

        # Writes settings to filestorage
        self._data_path = Path(handling.get_datadir())
        self.zi_settings.serialize(
            self._data_path,
            ZISerializeSettings(
                self.instrument.name, self.instrument._serial, self.instrument._type
            ),
        )

        # Upload settings, seqc and waveforms
        self.zi_settings.apply(self.instrument)

        return True

    def retrieve_acquisition(self) -> xarray.Dataset | None:
        return None


class HDAWGInstrumentCoordinatorComponent(ZIInstrumentCoordinatorComponent):
    """Zurich Instruments HDAWG InstrumentCoordinator Component class."""

    def __init__(
        self,
        instrument: qcodes.HDAWG,
        **kwargs: Any,  # noqa: ANN401 # Need to fix that in the parent class first.
    ) -> None:
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

    def retrieve_acquisition(self) -> None:
        return None

    def wait_done(self, timeout_sec: int = 10) -> None:
        for awg_index in reversed(self.zi_settings.awg_indexes):
            self.get_awg(awg_index).wait_done(timeout_sec)

    def get_hardware_log(
        self,
        compiled_schedule: CompiledSchedule,  # noqa: ARG002
    ) -> dict | None:
        pass


class UHFQAInstrumentCoordinatorComponent(ZIInstrumentCoordinatorComponent):
    """Zurich Instruments UHFQA InstrumentCoordinator Component class."""

    def __init__(
        self,
        instrument: qcodes.UHFQA,
        **kwargs: Any,  # noqa: ANN401  # Need to fix that in the parent class first.
    ) -> None:
        if not isinstance(instrument, qcodes.UHFQA):
            raise ValueError("`instrument` must be an instance of UHFQA.")
        super().__init__(instrument, **kwargs)

    @property
    def instrument(self) -> qcodes.UHFQA:
        if not isinstance((instrument := super().instrument), qcodes.UHFQA):
            raise ValueError("`self.instrument` must be an instance of UHFQA.")
        return instrument

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
        wave_files = list(self._data_path.glob(f"{self.instrument.name}*.csv"))
        for file in wave_files:
            shutil.copy2(str(file), str(waves_path))

        # prepare twice to resolve issue with waveform memory not being updated
        # correctly. In practice, we see that integration weights update correctly, but
        # the waveforms in pulses do not. This problem is not fully understood, but this
        # resolves the issue at a minor overhead.

        if configure and self.zi_settings:
            # Upload settings, seqc and waveforms
            self.zi_settings.apply(self.instrument)
        return True

    def retrieve_acquisition(self) -> xarray.Dataset:
        if self.zi_device_config is None:
            raise RuntimeError("Undefined device config, first prepare UHFQA!")

        acq_config = self.zi_device_config.acq_config
        if acq_config is None:
            raise RuntimeError(
                "Attempting to retrieve acquisition from an instrument coordinator"
                " component that was not prepared. Execute"
                " UHFQAInstrumentCoordinatorComponent.prepare(zi_device_config) first."
            )

        # acq_channel_results: Dict[int, np.ndarray] = dict()
        acq_channel_results: list[dict[Hashable, xarray.DataArray]] = []
        for acq_channel, resolve in acq_config.resolvers.items():
            data: NDArray = resolve(uhfqa=self.instrument)
            acq_protocol = acq_config.acq_protocols[acq_channel]
            if acq_protocol == "Trace" and acq_config.bin_mode == BinMode.AVERAGE:
                acq_channel_results.append(
                    {
                        acq_channel: xarray.DataArray(
                            data.reshape((1, -1)),
                            dims=(
                                f"acq_index_{acq_channel}",
                                f"trace_index_{acq_channel}",
                            ),
                            attrs={"acq_protocol": acq_protocol},
                        )
                    }
                )
            elif (
                acq_protocol
                in (
                    "SSBIntegrationComplex",
                    "WeightedIntegratedSeparated",
                    "NumericalSeparatedWeightedIntegration",
                    "NumericalWeightedIntegration",
                )
                and acq_config.bin_mode == BinMode.AVERAGE
            ):
                acq_channel_results.append(
                    {
                        acq_channel: xarray.DataArray(
                            # Sanity check: data size must be equal to n_acquisitions
                            data.reshape((acq_config.n_acquisitions,)),
                            dims=(f"acq_index_{acq_channel}",),
                            attrs={"acq_protocol": acq_protocol},
                        )
                    }
                )
            elif (
                acq_protocol
                in (
                    "SSBIntegrationComplex",
                    "WeightedIntegratedSeparated",
                    "NumericalSeparatedWeightedIntegration",
                    "NumericalWeightedIntegration",
                )
                and acq_config.bin_mode == BinMode.APPEND
            ):
                acq_channel_results.append(
                    {
                        acq_channel: xarray.DataArray(
                            data.reshape((-1, acq_config.n_acquisitions)),
                            dims=("repetition", f"acq_index_{acq_channel}"),
                            attrs={"acq_protocol": acq_protocol},
                        )
                    }
                )
            else:
                raise AcquisitionProtocolNotSupportedError(
                    f"Acquisition protocol {acq_protocol} with bin mode"
                    f" {acq_config.bin_mode} is not supproted by the backend."
                )

        return xarray.merge(acq_channel_results, compat="no_conflicts")

    def wait_done(self, timeout_sec: int = 10) -> None:
        self.instrument.awg.wait_done(timeout_sec)

    def get_hardware_log(
        self,
        compiled_schedule: CompiledSchedule,  # noqa: ARG002
    ) -> dict | None:
        pass
