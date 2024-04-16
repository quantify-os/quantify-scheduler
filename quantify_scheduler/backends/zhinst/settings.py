# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Settings builder for Zurich Instruments."""
from __future__ import annotations

import dataclasses
import itertools
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import numpy as np
from zhinst.qcodes import base

from quantify_scheduler.backends.types import zhinst as zi_types
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.helpers.collections import make_hash


# same as backends.zhinst_backend.NUM_UHFQA_READOUT_CHANNELS
# copied here to avoid a circular import
NUM_UHFQA_READOUT_CHANNELS = 10


@dataclasses.dataclass(frozen=True)
class ZISerializeSettings:
    """
    Serialization data container to decouple filenames from
    instrument names during the serialization.
    """

    name: str
    _serial: str
    _type: str


@dataclasses.dataclass
class ZISetting:
    """Zurich Instruments Settings record type."""

    node: str
    value: Any
    apply_fn: Callable[[base.ZIBaseInstrument, str, Any], None]

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
    """
    A collection of AWG and DAQ settings for a Zurich Instruments device.

    Parameters
    ----------
    daq_settings :
        The data acquisition node settings.
    awg_settings :
        The AWG(s) node settings.
    """

    def __init__(
        self,
        daq_settings: List[ZISetting],
        awg_settings: Dict[int, ZISetting],
    ):
        self._daq_settings: List[ZISetting] = daq_settings
        self._awg_settings: Dict[int, ZISetting] = awg_settings
        self._awg_indexes = list(self._awg_settings.keys())

    def __eq__(self, other):
        self_dict = self.as_dict()
        if not isinstance(other, ZISettings):
            return False
        other_dict = other.as_dict()
        settings_equal = make_hash(self_dict) == make_hash(other_dict)
        return settings_equal

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
        settings_dict: Dict[str, Any] = dict()
        for setting in self._daq_settings:
            settings_dict = {**settings_dict, **setting.as_dict()}

        for awg_index, setting in self._awg_settings.items():
            # need to explicitly initialize an empty dict as different awgs can set a
            # setting related to the same node and we do not want to overwrite the key.
            if setting.node not in settings_dict:
                settings_dict[setting.node] = {}
            settings_dict[setting.node][awg_index] = setting.value

        return settings_dict

    def apply(self, instrument: base.ZIBaseInstrument) -> None:
        """Apply all settings to the instrument."""
        for _, setting in self._awg_settings.items():
            setting.apply(instrument)

        def sort_by_fn(setting: ZISetting):
            """Returns ZISetting callable apply function as a sorter."""
            return setting.apply_fn

        for apply_fn, group in itertools.groupby(self._daq_settings, sort_by_fn):
            if apply_fn is zi_helpers.set_value:
                values: List[Tuple[str, Any]] = list()
                for setting in group:
                    node = f"/{instrument._serial}/{setting.node}"
                    values.append((node, setting.value))
                zi_helpers.set_values(instrument, values)
            else:
                for setting in group:
                    # Call apply_fn by property to avoid a deepcopy of all settings.
                    node = f"/{instrument._serial}/{setting.node}"
                    setting.apply_fn(
                        instrument=instrument,
                        node=node,
                        value=setting.value,
                    )

    def serialize(self, root: Path, options: ZISerializeSettings) -> Path:
        """
        Serializes the ZISerializeSettings to file storage.
        The parent '{options.name}_settings.json' file contains references to all
        child files.

        While settings are stored in JSON the waveforms are stored in CSV.

        Parameters
        ----------
        root :
            The root path to serialized files.
        options :
            The serialization options to associate these settings.

        Returns
        -------
        :
            The path to the parent JSON file.
        """
        collection = {
            "name": options.name,
            "serial": options._serial,
            "type": options._type,
        }
        # Copy the settings to avoid modifying the original values.
        _tmp_daq_list = list(map(dataclasses.replace, self._daq_settings))
        _tmp_awg_list = deepcopy(self._awg_settings)

        for setting in _tmp_daq_list:
            if "waveform/waves" in setting.node:
                nodes = setting.node.split("/")
                awg_index = int(nodes[1])
                wave_index = int(nodes[-1])
                name = f"{options.name}_awg{awg_index}_wave{wave_index}.csv"
                file_path = root / name

                columns = 2
                waveform_data = np.reshape(
                    setting.value, (len(setting.value) // columns, -1)
                )
                ############################################################
                # WARNING: For saving waveform in csv format, the data
                # MUST be in floating point type, and NOT int16 (as is required)
                # when using the Zhinst-toolkit.helpers.Waveform class.
                # Hotfix applied to rescale.
                ############################################################
                waveform_data = waveform_data / (2**15 - 1)
                np.savetxt(file_path, waveform_data, delimiter=";")

                setting.value = str(file_path)
            elif "commandtable/data" in setting.node:
                awg_index = setting.node.split("/")[1]
                name = f"{options.name}_awg{awg_index}.json"
                file_path = root / name
                file_path.touch()
                file_path.write_text(json.dumps(setting.value))

                setting.value = str(file_path)
            elif "integration/weights" in setting.node:
                setting.value = setting.value.tolist()
            elif "rotations/" in setting.node:
                setting.value = str(setting.value).replace(" ", "")

            collection = {**collection, **setting.as_dict()}

        for awg_index, setting in _tmp_awg_list.items():
            if setting.node == "compiler/sourcestring":
                if "compiler/sourcestring" not in collection:
                    collection["compiler/sourcestring"] = dict()

                name = f"{options.name}_awg{awg_index}.seqc"
                file_path = root / name
                file_path.touch()
                file_path.write_text(setting.value)

                setting.value = str(file_path)
                collection["compiler/sourcestring"][str(awg_index)] = setting.value

        file_path = root / f"{options.name}_settings.json"
        file_path.touch()
        file_path.write_text(json.dumps(collection))

        return file_path

    @classmethod
    def deserialize(cls, settings_path: Path) -> ZISettingsBuilder:
        """
        Deserializes the JSON settings for Zurich Instruments in to the
        :class:`.ZISettingsBuilder`.

        Parameters
        ----------
        settings_path :
            The path to the parent JSON file.

        Returns
        -------
        :
            The ZISettingsBuilder containing all the deserialized settings.

        Raises
        ------
        ValueError
            If the settings_path does not end with '_settings.json'.
        """
        if not settings_path.name.endswith("_settings.json"):
            raise ValueError(
                "Invalid value for param 'settings_path' "
                "provide path to '{instrument}_settings.json'"
            )

        settings_data: Dict[str, Any] = json.loads(settings_path.read_text())
        settings_data.pop("name")
        settings_data.pop("serial")
        device_type_str: str = settings_data.pop("type")
        device_type = zi_types.DeviceType(device_type_str.upper())
        builder = ZISettingsBuilder()

        for node in settings_data:
            value = settings_data[node]

            if "waveform/waves" in node:
                nodes = node.split("/")
                awg_index = int(nodes[1])
                wave_index = int(nodes[-1])
                waveform_data = np.loadtxt(value, delimiter=";")
                if device_type == zi_types.DeviceType.UHFQA:
                    builder.with_csv_wave_vector(awg_index, wave_index, waveform_data)
                else:
                    builder.with_wave_vector(awg_index, wave_index, waveform_data)
            elif "commandtable/data" in node:
                awg_index = int(node.split("/")[1])
                json_str = Path(value).read_text()
                json_data = json.loads(json_str)
                builder.with_commandtable_data(awg_index, json_data)
            elif "integration/weights" in node:
                integration_weights = np.array(value)
                parts = node.split("/")
                channel_index = int(parts[4])
                if node.endswith("real"):
                    builder.with_qas_integration_weights_real(
                        channel_index, integration_weights
                    )
                elif node.endswith("imag"):
                    builder.with_qas_integration_weights_imag(
                        channel_index, integration_weights
                    )
            elif "rotations/" in node:
                channel_index = int(node.split("/")[-1])
                complex_value = complex(value.replace(" ", ""))
                builder.with_qas_rotations(channel_index, complex_value)
            elif node == "compiler/sourcestring":
                seqc_per_awg = cast(Dict[int, str], value)

                for awg_index, seqc_file in seqc_per_awg.items():
                    seqc_path = Path(seqc_file)
                    seqc_content = seqc_path.read_text()
                    builder.with_compiler_sourcestring(int(awg_index), seqc_content)
            else:
                builder._daq_settings.append(
                    ZISetting(node, value, zi_helpers.set_value)
                )

        return builder


class ZISettingsBuilder:
    """
    The Zurich Instruments Settings builder class.

    This class provides an API for settings that are configured in the zhinst backend.
    The ZISettings class is the resulting set that holds settings.

    This class exist because configuring these settings requires logic in how the
    settings are configured using the zurich instruments API.

    .. tip::

        Build the settings using :meth:`~.build` and then view them as a dictionary
        using :meth:`ZISettings.as_dict` to see what settings will be configured.

    """

    _daq_settings: List[ZISetting]
    _awg_settings: List[Tuple[str, Tuple[int, ZISetting]]]

    def __init__(self):
        self._daq_settings = list()
        self._awg_settings = list()

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
        self._daq_settings.append(setting)
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
        self._awg_settings.append((setting.node, (awg_index, setting)))
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
        for node, value in defaults:
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

    def with_csv_wave_vector(
        self, awg_index: int, wave_index: int, vector: Union[List, str]
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments waveform vector setting
        by index for an awg by index.

        This equivalent to ``with_wave_vector`` only it does
        not upload the setting to the node, because
        for loading waveforms using a CSV file this is
        not required.

        Parameters
        ----------
        awg_index :
        wave_index :
        vector :

        Returns
        -------
        :
        """

        def void(instrument: base.ZIBaseInstrument, node, value):
            pass

        return self._set_daq(
            ZISetting(
                f"awgs/{awg_index:d}/waveform/waves/{wave_index:d}",
                vector,
                void,
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

    def with_qas_result_reset(self, value: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Result reset setting.

        Parameters
        ----------
        value :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/result/reset",
                value,
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

    def with_qas_integration_weights_real(
        self,
        channels: Union[int, List[int]],
        real: Union[List[int], np.ndarray],
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor integration real weights setting.

        Parameters
        ----------
        channels :
        real :

        Returns
        -------
        :

        Raises
        ------
        ValueError
            If a channel used is larger than 9.
        """
        assert len(real) <= 4096

        node = "qas/0/integration/weights/"
        channels_list = [channels] if isinstance(channels, int) else channels
        for channel_index in channels_list:
            if channel_index >= NUM_UHFQA_READOUT_CHANNELS:
                raise ValueError(
                    f"channel_index = {channel_index}: the UHFQA supports up to "
                    f"{NUM_UHFQA_READOUT_CHANNELS} integration weigths."
                )

            self._set_daq(
                ZISetting(
                    f"{node}{channel_index}/real",
                    np.array(real),
                    zi_helpers.set_vector,
                )
            )
        return self

    def with_qas_integration_weights_imag(
        self,
        channels: Union[int, List[int]],
        imag: Union[List[int], np.ndarray],
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor integration imaginary weights setting.

        Parameters
        ----------
        channels :
        imag :

        Returns
        -------
        :

        Raises
        ------
        ValueError
            If a channel used is larger than 9.
        """
        assert len(imag) <= 4096

        node = "qas/0/integration/weights/"
        channels_list = [channels] if isinstance(channels, int) else channels
        for channel_index in channels_list:
            if channel_index >= NUM_UHFQA_READOUT_CHANNELS:
                raise ValueError(
                    f"channel_index = {channel_index}: the UHFQA supports up to "
                    f"{NUM_UHFQA_READOUT_CHANNELS} integration weigths."
                )

            self._set_daq(
                ZISetting(
                    f"{node}{channel_index}/imag",
                    np.array(imag),
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

    def with_qas_monitor_reset(self, value: int) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS Monitor reset setting.

        Parameters
        ----------
        value :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                "qas/0/monitor/reset",
                value,
                zi_helpers.set_value,
            )
        )

    def with_qas_rotations(
        self, channels: Union[int, List[int]], value: Union[int, complex]
    ) -> ZISettingsBuilder:
        """
        Adds the Instruments QAS rotation setting.

        Parameters
        ----------
        channels :
        value :
            Number of degrees or a complex value.

        Returns
        -------
        :
        """
        if isinstance(value, int):
            value = np.exp(1j * np.deg2rad(value))

        channels_list = [channels] if isinstance(channels, int) else channels
        for channel_index in channels_list:
            self._set_daq(
                ZISetting(
                    f"qas/0/rotations/{channel_index}",
                    value,
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

    def with_sigout_offset(
        self, channel_index: int, offset_in_millivolts: float
    ) -> ZISettingsBuilder:
        """
        Adds the channel sigout offset
        setting in volts.

        Parameters
        ----------
        channel_index :
        offset_in_millivolts :

        Returns
        -------
        :
        """
        return self._set_daq(
            ZISetting(
                f"sigouts/{channel_index:d}/offset",
                offset_in_millivolts,
                zi_helpers.set_value,
            )
        )

    def with_gain(self, awg_index: int, gain: Tuple[float, float]) -> ZISettingsBuilder:
        """
        Adds the gain settings
        for the Instruments awg by index.

        Parameters
        ----------
        awg_index :
        gain :
            The gain values for output 1 and 2.

        Returns
        -------
        :
        """
        gain1, gain2 = gain

        assert gain1 >= -1 <= 1
        assert gain2 >= -1 <= 1

        self._set_daq(
            ZISetting(
                f"awgs/{awg_index:d}/outputs/0/gains/0", gain1, zi_helpers.set_value
            )
        )
        return self._set_daq(
            ZISetting(
                f"awgs/{awg_index:d}/outputs/1/gains/1", gain2, zi_helpers.set_value
            )
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
        waveforms_dict :

        Returns
        -------
        :
        """
        return self._set_awg(
            awg_index,
            ZISetting(
                "compiler/sourcestring",
                seqc,
                partial(
                    zi_helpers.set_and_compile_awg_seqc,
                    awg_index=awg_index,
                ),
            ),
        )

    def build(self) -> ZISettings:
        """
        Builds the ZISettings class.

        Returns
        -------
        :
        """
        # return ZISettings(self._daq_settings, dict(self._awg_settings).values())

        # extract the awg_index from the settings tuples.
        awg_settings_dict = {}
        for awg_setting in self._awg_settings:
            # this particular indexing is very specific to the data format of the
            # awg settings and could be improved/refactored, N.B. hard to debug!
            awg_index = awg_setting[1][0]
            setting = awg_setting[1][1]
            awg_settings_dict[awg_index] = setting

        return ZISettings(
            daq_settings=self._daq_settings,
            awg_settings=awg_settings_dict,
        )
