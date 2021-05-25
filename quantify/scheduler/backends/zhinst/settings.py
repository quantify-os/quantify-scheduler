# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Settings builder for Zurich Instruments."""
from __future__ import annotations
from functools import partial

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from zhinst.qcodes import base

from quantify.scheduler.backends.types import zhinst as zi_types
from quantify.scheduler.backends.zhinst import helpers as zi_helpers


@dataclass
class ZISetting:
    """Zurich Instruments Settings record type."""

    node: str
    value: Any
    apply_fn: Callable

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the key-value pair as a dictionary.

        Returns
        -------
        :
        """
        return {self.node: self.value}

    def apply(self, instrument: base.ZIBaseInstrument):
        """
        Applies settings to the Instrument.

        Parameters
        ----------
        instrument :
        """
        self.apply_fn(instrument=instrument, node=self.node, value=self.value)


class ZISettings:
    """Zurich Instruments settings result class."""

    def __init__(
        self,
        instrument: base.ZIBaseInstrument,
        daq_settings: List[ZISetting],
        awg_settings: List[Tuple[int, ZISetting]],
    ):
        self._instrument = instrument
        self._daq_settings = daq_settings
        self._awg_settings = awg_settings

        # Prefix all nodes with the instrument's serial.
        for setting in self._daq_settings:
            setting.node = f"/{instrument._serial}/{setting.node}"

        self._awg_indexes = [awg_index for (awg_index, _) in self._awg_settings]

    @property
    def awg_indexes(self) -> List[int]:
        """Returns a list of enabled AWG indexes."""
        return self._awg_indexes

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the ZISettings as a dictionary.

        Returns
        -------
        :
        """
        collection = dict()
        for setting in self._daq_settings:
            collection = {**collection, **setting.as_dict()}

        for (_, setting) in self._awg_settings:
            collection = {**collection, **setting.as_dict()}

        return collection

    def apply(self) -> None:
        """Apply all settings to the instrument."""
        for setting in self._daq_settings:
            setting.apply(self._instrument)

        for (_, setting) in self._awg_settings:
            setting.apply(self._instrument)

    def serialize(self, root: Path) -> Path:
        """
        Serializes the settings to file storage.

        Returns
        -------
        :
        """
        collection = dict()
        _tmp_daq_list = deepcopy(self._daq_settings)
        _tmp_awg_list = deepcopy(self._awg_settings)

        for setting in _tmp_daq_list:
            if "waveform/waves" in setting.node:
                nodes = setting.node.split("/")
                awg_index = nodes[3]
                wave_index = nodes[-1]
                name = (
                    f"{self._instrument._serial}_"
                    + f"awg{awg_index}_wave{wave_index}.csv"
                )
                file_path = root / name

                columns = 2
                waveform_data = np.reshape(
                    setting.value, (int(len(setting.value) / columns), -1)
                )
                np.savetxt(file_path, waveform_data, delimiter=";")

                setting.value = str(file_path)
            elif "commandtable/data" in setting.node:
                awg_index = setting.node.split("/")[3]
                name = f"{self._instrument._serial}_awg{awg_index}.json"
                file_path = root / name
                file_path.touch()
                file_path.write_text(json.dumps(setting.value))

                setting.value = str(file_path)

            collection = {**collection, **setting.as_dict()}

        for (awg_index, setting) in _tmp_awg_list:

            if "compiler/sourcestring" in setting.node:
                if "compiler/sourcestring" not in collection:
                    collection["compiler/sourcestring"] = list()

                name = f"{self._instrument._serial}_awg{awg_index}.seqc"
                file_path = root / name
                file_path.touch()
                file_path.write_text(setting.value)

                setting.value = str(file_path)
                collection["compiler/sourcestring"].append(setting.value)

        file_path = root / f"{self._instrument._serial}_settings.json"
        file_path.touch()
        file_path.write_text(json.dumps(collection))

        return file_path


class ZISettingsBuilder:
    # pylint: disable=too-many-public-methods
    """
    The Zurich Instruments Settings builder class.

    This class provides an API for settings that
    are configured in the zhinst backend. The ZISettings
    class is the resulting set that holds settings.
    """

    _daq_settings: Dict[str, ZISetting]
    _awg_settings: Dict[str, Tuple[int, ZISetting]]

    def __init__(self):
        """Creates a new instance of ZISettingsBuilder"""
        self._daq_settings = dict()
        self._awg_settings = dict()

    def _set_daq(self, setting: ZISetting) -> ZISettingsBuilder:
        """
        Sets an daq module setting.

        Parameters
        ----------
        setting :

        Returns
        -------
        :
        """
        self._daq_settings[setting.node] = setting
        return self

    def _set_awg(self, awg_index: int, setting: ZISetting) -> ZISettingsBuilder:
        """
        Sets an awg module setting.

        Parameters
        ----------
        awg_index :
        setting :

        Returns
        -------
        :
        """
        self._awg_settings[f"{awg_index}/{setting.node}"] = (awg_index, setting)
        return self

    def with_defaults(
        self, defaults: List[Tuple[str, Union[str, int]]]
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments default settings.

        Parameters
        ----------
        defaults :

        Returns
        -------
        :
        """
        for (node, value) in defaults:
            self._set_daq(ZISetting(node, value, zi_helpers.set_value))
        return self

    def with_wave_vector(
        self, awg_index: int, wave_index: int, vector: Union[List, str]
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments waveform vector setting
        by index for an awg by index.

        Parameters
        ----------
        awg_index :
        wave_index :
        vector :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                f"awgs/{awg_index:d}/waveform/waves/{wave_index:d}",
                vector,
                zi_helpers.set_vector,
            )
        )

    def with_commandtable_data(
        self, awg_index: int, json_data: Union[Dict[str, Any], str]
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments CommandTable
        json vector setting to the awg by index.

        Parameters
        ----------
        awg_index :
        json_data :

        Returns
        -------
        :
        """
        if not isinstance(json_data, str):
            json_data = json.dumps(json_data)

        return self._set_daq(
            ZISetting(
                f"awgs/{awg_index:d}/commandtable/data",
                str(json_data),
                zi_helpers.set_vector,
            )
        )

    def with_awg_time(self, awg_index: int, clock_rate_index: int) -> ZISettingsBuilder:
        """
        Adds the Instruments clock rate frequency
        setting.

        See ZI instrument user manual
            /DEV..../AWGS/n/TIME

        Parameters
        ----------
        awg_index :
        clock_rate_index :

        Returns
        -------
        :
        """
        assert clock_rate_index < 14

        return self._set_daq(
            ZISetting(
                f"awgs/{awg_index:d}/time", clock_rate_index, zi_helpers.set_value
            )
        )

    def with_qas_delay(self, delay: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS delay.

        Parameters
        ----------
        delay :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/delay",
                delay,
                zi_helpers.set_value,
            )
        )

    def with_qas_result_enable(self, enabled: bool) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor result
        enable setting.

        Parameters
        ----------
        enabled :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/result/enable",
                int(enabled),
                zi_helpers.set_value,
            )
        )

    def with_qas_result_length(self, n_samples: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor result
        length setting.

        Parameters
        ----------
        n_samples :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/result/length",
                n_samples,
                zi_helpers.set_value,
            )
        )

    def with_qas_result_averages(self, n_averages: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor result
        averages setting.

        Parameters
        ----------
        n_averages :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/result/averages",
                n_averages,
                zi_helpers.set_value,
            )
        )

    def with_qas_result_mode(self, mode: zi_types.QasResultMode) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor result
        mode setting.

        Parameters
        ----------
        mode :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/result/mode",
                mode.value,
                zi_helpers.set_value,
            )
        )

    def with_qas_result_source(
        self, mode: zi_types.QasResultSource
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor result
        source setting.

        Parameters
        ----------
        mode :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/result/source",
                mode.value,
                zi_helpers.set_value,
            )
        )

    def with_qas_integration_length(self, n_samples: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor integration
        length setting.

        Parameters
        ----------
        n_samples :

        Returns
        -------
        :
        """
        assert n_samples <= 4096
        return self._set_daq(
            ZISetting(
                "qas/0/integration/length",
                n_samples,
                zi_helpers.set_value,
            )
        )

    def with_qas_integration_mode(
        self,
        mode: zi_types.QasIntegrationMode,
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor integration
        mode setting.

        Parameters
        ----------
        mode :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/integration/mode",
                mode.value,
                zi_helpers.set_value,
            )
        )

    def with_qas_integration_weights(
        self,
        channels: Union[int, List[int]],
        weights_i: List[int],
        weights_q: List[int],
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor integration
        weights setting.

        Parameters
        ----------
        channels :
        weights_i :
        weights_q :

        Returns
        -------
        :
        """
        assert len(weights_i) <= 4096
        assert len(weights_q) <= 4096

        node = "qas/0/integration/weights/"
        channels_list = [channels] if isinstance(channels, int) else channels
        for channel_index in channels_list:
            self._set_daq(
                ZISetting(
                    f"{node}{channel_index}/real",
                    np.array(weights_i),
                    zi_helpers.set_vector,
                )
            )
            self._set_daq(
                ZISetting(
                    f"{node}{channel_index}/imag",
                    np.array(weights_q),
                    zi_helpers.set_vector,
                )
            )
        return self

    def with_qas_monitor_enable(self, enabled: bool) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor enable setting.

        Parameters
        ----------
        enabled :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/monitor/enable",
                int(enabled),
                zi_helpers.set_value,
            )
        )

    def with_qas_monitor_length(self, n_samples: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor length setting.

        Parameters
        ----------
        n_samples :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/monitor/length",
                n_samples,
                zi_helpers.set_value,
            )
        )

    def with_qas_monitor_averages(self, n_averages: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor averages setting.

        Parameters
        ----------
        n_averages :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/monitor/averages",
                n_averages,
                zi_helpers.set_value,
            )
        )

    def with_qas_rotations(
        self, channels: Union[int, List[int]], degrees: int
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS rotation setting.

        Parameters
        ----------
        channels :
        degrees :

        Returns
        -------
        :
        """
        complex_value = np.exp(1j * np.deg2rad(degrees))
        channels_list = [channels] if isinstance(channels, int) else channels
        for channel_index in channels_list:
            self._set_daq(
                ZISetting(
                    f"qas/0/rotations/{channel_index}",
                    complex_value,
                    zi_helpers.set_value,
                )
            )
        return self

    def with_system_channelgrouping(self, channelgrouping: int) -> ZISettingsBuilder:
        """
        Adds the Instruments channelgrouping
        setting.

        Parameters
        ----------
        channelgrouping :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "system/awg/channelgrouping",
                channelgrouping,
                zi_helpers.set_value,
            )
        )

    def with_sigouts(
        self, awg_index: int, outputs: Tuple[int, int]
    ) -> ZISettingsBuilder:
        """
        Adds the channel sigouts setting
        for the Instruments awg by index.

        Parameters
        ----------
        awg_index :
        outputs :

        Returns
        -------
        :
        """
        onoff_0, onoff_1 = outputs
        channel_0 = awg_index * 2
        channel_1 = (awg_index * 2) + 1
        self._set_daq(
            ZISetting(f"sigouts/{channel_0:d}/on", onoff_0, zi_helpers.set_value)
        )
        return self._set_daq(
            ZISetting(f"sigouts/{channel_1:d}/on", onoff_1, zi_helpers.set_value)
        )

    def with_compiler_sourcestring(
        self, awg_index: int, seqc: str
    ) -> ZISettingsBuilder:
        """
        Adds the sequencer compiler sourcestring
        setting for the Instruments awg by index.

        Parameters
        ----------
        awg_index :
        seqc :

        Returns
        -------
        :
        """
        return self._set_awg(
            awg_index,
            ZISetting(
                "compiler/sourcestring",
                seqc,
                partial(zi_helpers.set_awg_value, awg_index=awg_index),
            ),
        )

    def build(self, instrument: base.ZIBaseInstrument) -> ZISettings:
        """
        Builds the ZISettings class.

        Parameters
        ----------
        instrument :

        Returns
        -------
        :
        """
        return ZISettings(
            instrument, self._daq_settings.values(), self._awg_settings.values()
        )
