# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Helpers for Zurich Instruments."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from zhinst.qcodes import base
from zhinst import qcodes


logger = logging.getLogger()


def get_value(instrument: base.ZIBaseInstrument, node: str) -> str:
    """
    Gets the value of a ZI node.

    Parameters
    ----------
    instrument :
    node :

    Returns
    -------
    :
        The node value.
    """
    if not node.startswith(f"/{instrument._serial}"):
        node = f"/{instrument._serial}/{node}"
    logger.debug(node)
    return instrument._controller._get(node)


def set_value(
    instrument: base.ZIBaseInstrument,
    node: str,
    value,
) -> None:
    """
    Sets the value of a ZI node.

    Parameters
    ----------
    instrument :
        The instrument.
    path :
        The node path.
    value :
        The new node value.
    """
    if not node.startswith(f"/{instrument._serial}"):
        node = f"/{instrument._serial}/{node}"
    logger.debug(node)
    instrument._controller._set(node, value)


def set_vector(
    instrument: base.ZIBaseInstrument,
    node: str,
    value: Union[List, str],
) -> None:
    """
    Sets the vector value of a ZI node.

    Parameters
    ----------
    instrument :
        The instrument.
    awg_index :
        The awg to configure.
    node :
        The node path.
    value :
        The new node vector value.
    """
    if not node.startswith(f"/{instrument._serial}"):
        node = f"/{instrument._serial}/{node}"
    logger.debug(node)
    instrument._controller._controller._connection._daq.setVector(node, value)


def set_awg_value(
    instrument: base.ZIBaseInstrument,
    awg_index: int,
    node: str,
    value: Union[int, str],
) -> None:
    """
    Sets the vector value of a ZI node.

    Parameters
    ----------
    instrument :
        The instrument.
    awg_index :
        The awg to configure.
    node :
        The node path.
    vector :
        The new node vector value.
    """
    logger.debug(node)

    awgs = [instrument.awg] if not hasattr(instrument, "awgs") else instrument.awgs
    awgs[awg_index]._awg._module.set(node, value)


def set_wave_vector(
    instrument: base.ZIBaseInstrument,
    awg_index: int,
    wave_index: int,
    vector: Union[List, str],
) -> None:
    """
    Sets the command table wave vector for an awg of an instrument.

    Parameters
    ----------
    instrument :
        The instrument.
    awg_index :
        The index of an AWG
    wave_index :
        The wave index.
    vector :
        The vector value.
    """
    path: str = f"awgs/{awg_index:d}/waveform/waves/{wave_index:d}"
    set_vector(instrument, path, vector)


def set_commandtable_data(
    instrument: base.ZIBaseInstrument,
    awg_index: int,
    json_data: Union[Dict[str, Any], str],
) -> None:
    """
    Sets the commandtable JSON for an AWG.

    Parameters
    ----------
    instrument :
        The instrument
    awg_index :
        The awg index.
    json_data :
        The json data.
    """
    if not isinstance(json_data, str):
        json_data = json.dumps(json_data)

    path = f"awgs/{awg_index:d}/commandtable/data"
    set_vector(instrument, path, str(json_data))


def get_directory(awg: qcodes.hdawg.AWG) -> Path:
    """
    Returns the LabOne directory of an AWG.

    Parameters
    ----------
    awg :
        The HDAWG AWG object.

    Returns
    -------
    :
        The path of this directory.
    """
    return Path(awg._awg._module.get_string("directory"))


def get_src_directory(awg: qcodes.hdawg.AWG) -> Path:
    """
    Returns the source directory of an AWG.

    Parameters
    ----------
    awg :
        The HDAWG AWG object.

    Returns
    -------
    :
        The path to the source directory.
    """
    return get_directory(awg).joinpath("awg", "src")


def get_waves_directory(awg: qcodes.hdawg.AWG) -> Path:
    """
    Returns the waves directory of an AWG.

    Parameters
    ----------
    awg :
        The HDAWG AWG object.

    Returns
    -------
    :
        The path to the waves directory.
    """
    return get_directory(awg).joinpath("awg", "waves")


def write_seqc_file(awg: qcodes.hdawg.AWG, contents: str, filename: str) -> Path:
    """
    Writes the contents of to the source directory
    of LabOne.

    Parameters
    ----------
    awg :
        The HDAWG AWG instance.
    contents :
        The content to write.
    filename :
        The name of the file.

    Returns
    -------
    :
        Returns the path which was written.
    """
    path = get_src_directory(awg).joinpath(filename)
    path.write_text(contents)

    return path


def get_waveform_table(
    pulse_ids: List[int], pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]]
) -> Dict[int, int]:
    """
    Returns a dictionary that contains the locations of
    pulses in the AWG waveform table.

    Parameters
    ----------
    pulse_ids :
        The list of pulse ids.
    pulseid_pulseinfo_dict :
        The info lookup dictionary.

    Returns
    -------
    :
        The waveform table dictionary.
    """
    waveform_table: Dict[int, int] = dict()
    index = 0
    for pulse_id in pulse_ids:
        if pulse_id in waveform_table:
            # Skip duplicate pulses.
            continue

        pulse_info = pulseid_pulseinfo_dict[pulse_id]
        if pulse_info["port"] is None:
            # Skip pulses without a port. Such as the IdlePulse.
            continue

        waveform_table[pulse_id] = index
        index += 1

    return waveform_table


def get_readout_channel_bitmask(readout_channels_count: int) -> str:
    """
    Returns a bitmask to enable readout channels.
    The bitmask can be used to turn on QA for
    induvidual channels in startQAResult.

    Parameters
    ----------
    readout_channels_count :
        The amount of readout channels to enable.
        Maximum readout channels for UHFQA is 10.

    Returns
    -------
    :
        The channel bitmask.
    """
    assert readout_channels_count <= 10

    mask: int = 0
    for i in range(readout_channels_count):
        mask += 1 << i

    bitmask = format(mask, "b").zfill(10)

    return f"0b{bitmask}"


def get_clock_rates(base_clock: float) -> Dict[int, int]:
    """
    Returns the allowed clock rate values.
    See zhinst User manuals, section /DEV..../AWGS/n/TIME

    Parameters
    ----------
    base_clock :
        The Instruments base clock rate.
    Returns
    -------
    Dict[int, int]
        The node and clock rate values.
    """
    return dict(
        map(
            lambda i: (i, int(base_clock))
            if i == 0
            else (i, int(base_clock / pow(2, i))),
            range(14),
        )
    )
