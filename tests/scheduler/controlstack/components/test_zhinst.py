# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
from __future__ import annotations

from pathlib import Path
from unittest.mock import call

import numpy as np
import pytest
from qcodes.instrument import channel
from zhinst import qcodes
from zhinst import toolkit as tk
from quantify.scheduler.backends.zhinst import helpers as zi_helpers
from quantify.scheduler.backends.zhinst import settings
from quantify.scheduler.backends.zhinst_backend import (
    ZIAcquisitionConfig,
    ZIDeviceConfig,
)
from quantify.scheduler.controlstack.components import zhinst
from quantify.scheduler.types import Schedule


@pytest.fixture
def make_hdawg(mocker):
    instrument = None

    def _make_hdawg(name: str, serial: str) -> zhinst.HDAWGControlStackComponent:
        mocker.patch.object(
            tk,
            "HDAWG",
        )
        mocker.patch("qcodes.instrument.Instrument.record_instance")
        mocker.patch.object(qcodes.HDAWG, "connect_message")
        awg = mocker.Mock(spec=qcodes.hdawg.AWG)
        mocker.patch.object(
            qcodes.hdawg,
            "AWG",
            return_value=awg,
        )
        channel_list = mocker.Mock(spec=channel.ChannelList)
        mocker.patch.object(qcodes.hdawg, "ChannelList", return_value=channel_list)
        instrument = zhinst.HDAWGControlStackComponent(name=name, serial=serial)
        instrument.awgs = list(
            map(mocker.create_autospec(qcodes.hdawg.AWG, instance=True), range(4))
        )
        return instrument

    yield _make_hdawg

    if not instrument is None:
        instrument.close()


@pytest.fixture
def make_uhfqa(mocker):
    instrument = None

    def _make_uhfqa(name: str, serial: str) -> zhinst.UHFQAControlStackComponent:
        awg = mocker.create_autospec(qcodes.uhfqa.AWG, instance=True)
        mocker.patch.object(
            tk,
            "UHFQA",
        )
        mocker.patch("qcodes.instrument.Instrument.record_instance")
        mocker.patch.object(qcodes.UHFQA, "connect_message")
        mocker.patch.object(qcodes.UHFQA, "add_parameter")
        mocker.patch.object(
            qcodes.uhfqa,
            "AWG",
            return_value=awg,
        )

        instrument = zhinst.UHFQAControlStackComponent(name=name, serial=serial)
        instrument.awg = awg

        return instrument

    yield _make_uhfqa

    if not instrument is None:
        instrument.close()


def test_initialize_hdawg(make_hdawg):
    make_hdawg("hdawg0", "dev1234")


def test_hdawg_start(mocker, make_hdawg):
    # Arrange
    hdawg: zhinst.HDAWGControlStackComponent = make_hdawg("hdawg0", "dev1234")
    get_awg_spy = mocker.patch.object(hdawg, "get_awg", wraps=hdawg.get_awg)
    hdawg.zi_settings = settings.ZISettings(
        hdawg,
        list(),
        [
            (0, mocker.Mock()),
            (1, mocker.Mock()),
            (2, mocker.Mock()),
            (3, mocker.Mock()),
        ],
    )

    # Act
    hdawg.start()

    # Assert
    assert get_awg_spy.call_args_list == [
        call(3),
        call(2),
        call(1),
        call(0),
    ]
    for i in range(4):
        hdawg.get_awg(i).run.assert_called()


def test_hdawg_stop(mocker, make_hdawg):
    # Arrange
    hdawg: zhinst.HDAWGControlStackComponent = make_hdawg("hdawg0", "dev1234")
    get_awg_spy = mocker.patch.object(hdawg, "get_awg", wraps=hdawg.get_awg)
    hdawg.zi_settings = settings.ZISettings(
        hdawg,
        list(),
        [
            (0, mocker.Mock()),
            (1, mocker.Mock()),
            (2, mocker.Mock()),
            (3, mocker.Mock()),
        ],
    )

    # Act
    hdawg.stop()

    # Assert
    assert get_awg_spy.call_args_list == [
        call(0),
        call(1),
        call(2),
        call(3),
    ]
    for i in range(4):
        hdawg.get_awg(i).stop.assert_called()


def test_hdawg_prepare(mocker, make_hdawg):
    # Arrange
    hdawg: zhinst.HDAWGControlStackComponent = make_hdawg("hdawg0", "dev1234")
    config = ZIDeviceConfig(
        "hdawg0", Schedule("test"), settings.ZISettingsBuilder(), None
    )
    serialize = mocker.patch.object(settings.ZISettings, "serialize")
    apply = mocker.patch.object(settings.ZISettings, "apply")
    mocker.patch("quantify.data.handling.get_datadir", return_value=".")

    # Act
    hdawg.prepare(config)

    # Assert
    serialize.assert_called_with(Path("."))
    apply.assert_called()


def test_hdawg_retrieve_acquisition(make_hdawg):
    # Arrange
    hdawg: zhinst.HDAWGControlStackComponent = make_hdawg("hdawg0", "dev1234")

    # Act
    acq_result = hdawg.retrieve_acquisition()

    # Assert
    assert acq_result is None


def test_initialize_uhfqa(make_uhfqa):
    make_uhfqa("uhfqa0", "dev1234")


def test_uhfqa_start(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAControlStackComponent = make_uhfqa("uhfqa0", "dev1234")
    uhfqa.zi_settings = settings.ZISettings(
        uhfqa,
        list(),
        [
            (0, mocker.Mock()),
        ],
    )

    # Act
    uhfqa.start()

    # Assert
    uhfqa.awg.run.assert_called()


def test_uhfqa_stop(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAControlStackComponent = make_uhfqa("uhfqa0", "dev1234")
    uhfqa.zi_settings = settings.ZISettings(
        uhfqa,
        list(),
        [
            (0, mocker.Mock()),
        ],
    )

    # Act
    uhfqa.stop()

    # Assert
    uhfqa.awg.stop.assert_called()


def test_uhfqa_prepare(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAControlStackComponent = make_uhfqa("uhfqa0", "dev1234")
    config = ZIDeviceConfig(
        "hdawg0", Schedule("test"), settings.ZISettingsBuilder(), None
    )
    serialize = mocker.patch.object(settings.ZISettings, "serialize")
    apply = mocker.patch.object(settings.ZISettings, "apply")
    mocker.patch("quantify.data.handling.get_datadir", return_value=".")

    mocker.patch.object(zi_helpers, "get_waves_directory", return_value=Path("waves/"))
    mocker.patch.object(Path, "glob", return_value=["uhfqa0_awg0.csv"])
    copy2 = mocker.patch("shutil.copy2")

    # Act
    uhfqa.prepare(config)

    # Assert
    serialize.assert_called_with(Path("."))
    apply.assert_called()
    copy2.assert_called_with("uhfqa0_awg0.csv", "waves")


def test_uhfqa_retrieve_acquisition(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAControlStackComponent = make_uhfqa("uhfqa0", "dev1234")
    expected_data = np.ones(64)

    def resolver(uhfqa):  # pylint: disable=unused-argument
        return expected_data

    config = ZIDeviceConfig(
        "hdawg0",
        Schedule("test"),
        settings.ZISettingsBuilder(),
        ZIAcquisitionConfig(1, {0: resolver}),
    )
    mocker.patch.object(settings.ZISettings, "serialize")
    mocker.patch.object(settings.ZISettings, "apply")
    mocker.patch("quantify.data.handling.get_datadir", return_value=".")

    mocker.patch.object(zi_helpers, "get_waves_directory", return_value=Path("waves/"))
    mocker.patch.object(Path, "glob", return_value=[])

    uhfqa.prepare(config)

    # Act
    acq_result = uhfqa.retrieve_acquisition()

    # Assert
    assert not acq_result is None
    assert 0 in acq_result
    assert (acq_result[0] == expected_data).all()
