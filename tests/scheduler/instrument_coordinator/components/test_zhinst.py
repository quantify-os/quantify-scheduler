# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import call

import numpy as np
import pytest
from zhinst import qcodes

from quantify_scheduler import enums
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst import settings
from quantify_scheduler.backends.zhinst_backend import (
    ZIAcquisitionConfig,
    ZIDeviceConfig,
)
from quantify_scheduler.instrument_coordinator.components import zhinst


@pytest.fixture
def make_hdawg(mocker):
    def _make_hdawg(
        name: str, serial: str
    ) -> zhinst.HDAWGInstrumentCoordinatorComponent:
        mocker.patch("qcodes.instrument.Instrument.record_instance")
        hdawg: qcodes.HDAWG = mocker.create_autospec(qcodes.HDAWG, instance=True)
        hdawg.name = name
        hdawg._serial = serial
        hdawg.awgs = [None] * 4
        for i in range(4):
            hdawg.awgs[i] = mocker.create_autospec(qcodes.hdawg.AWG, instance=True)

        component = zhinst.HDAWGInstrumentCoordinatorComponent(hdawg)
        mocker.patch.object(component.instrument_ref, "get_instr", return_value=hdawg)

        return component

    yield _make_hdawg


@pytest.fixture
def make_uhfqa(mocker):
    def _make_uhfqa(
        name: str, serial: str
    ) -> zhinst.HDAWGInstrumentCoordinatorComponent:
        mocker.patch("qcodes.instrument.Instrument.record_instance")
        uhfqa: qcodes.UHFQA = mocker.create_autospec(qcodes.UHFQA, instance=True)
        uhfqa.name = name
        uhfqa._serial = serial
        uhfqa.awg = mocker.create_autospec(qcodes.uhfqa.AWG, instance=True)
        # the quantum analyzer setup "qas"
        uhfqa.qas = [None] * 1
        uhfqa.qas[0] = mocker.create_autospec(None, instance=True)

        component = zhinst.UHFQAInstrumentCoordinatorComponent(uhfqa)
        mocker.patch.object(component.instrument_ref, "get_instr", return_value=uhfqa)

        return component

    yield _make_uhfqa


def test_initialize_hdawg(make_hdawg):
    make_hdawg("hdawg0", "dev1234")


def test_hdawg_start(mocker, make_hdawg):
    # Arrange
    hdawg: zhinst.HDAWGInstrumentCoordinatorComponent = make_hdawg("hdawg0", "dev1234")
    get_awg_spy = mocker.patch.object(hdawg, "get_awg", wraps=hdawg.get_awg)
    hdawg.zi_settings = settings.ZISettings(
        list(),
        {
            0: mocker.Mock(),
            1: mocker.Mock(),
            2: mocker.Mock(),
            3: mocker.Mock(),
        },
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
    hdawg: zhinst.HDAWGInstrumentCoordinatorComponent = make_hdawg("hdawg0", "dev1234")
    get_awg_spy = mocker.patch.object(hdawg, "get_awg", wraps=hdawg.get_awg)
    hdawg.zi_settings = settings.ZISettings(
        list(),
        {
            0: mocker.Mock(),
            1: mocker.Mock(),
            2: mocker.Mock(),
            3: mocker.Mock(),
        },
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
    hdawg: zhinst.HDAWGInstrumentCoordinatorComponent = make_hdawg("hdawg0", "dev1234")
    config = ZIDeviceConfig("hdawg0", settings.ZISettingsBuilder(), None)
    serialize = mocker.patch.object(settings.ZISettings, "serialize")
    apply = mocker.patch.object(settings.ZISettings, "apply")
    mocker.patch("quantify_core.data.handling.get_datadir", return_value=".")

    # Act
    hdawg.prepare(config)

    # Assert
    hdawg_serialize_settings = settings.ZISerializeSettings(
        f"{hdawg.instrument.name}", hdawg.instrument._serial, hdawg.instrument._type
    )
    serialize.assert_called_with(Path("."), hdawg_serialize_settings)
    apply.assert_called_with(hdawg.instrument)


def test_hdawg_retrieve_acquisition(make_hdawg):
    # Arrange
    hdawg: zhinst.HDAWGInstrumentCoordinatorComponent = make_hdawg("hdawg0", "dev1234")

    # Act
    acq_result = hdawg.retrieve_acquisition()

    # Assert
    assert acq_result is None


def test_hdawg_wait_done(mocker, make_hdawg):
    # Arrange
    hdawg: zhinst.HDAWGInstrumentCoordinatorComponent = make_hdawg("hdawg0", "dev1234")
    get_awg_spy = mocker.patch.object(hdawg, "get_awg", wraps=hdawg.get_awg)
    hdawg.zi_settings = settings.ZISettings(
        list(),
        {
            0: mocker.Mock(),
            1: mocker.Mock(),
            2: mocker.Mock(),
            3: mocker.Mock(),
        },
    )
    timeout: int = 20

    # Act
    hdawg.wait_done(timeout)

    # Assert
    assert get_awg_spy.call_args_list == [
        call(3),
        call(2),
        call(1),
        call(0),
    ]
    for i in range(4):
        hdawg.get_awg(i).wait_done.assert_called_with(timeout)


def test_initialize_uhfqa(make_uhfqa):
    make_uhfqa("uhfqa0", "dev1234")


def test_uhfqa_start(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAInstrumentCoordinatorComponent = make_uhfqa("uhfqa0", "dev1234")
    uhfqa.zi_settings = settings.ZISettings(
        list(),
        {
            0: mocker.Mock(),
        },
    )

    # Act
    uhfqa.start()

    # Assert
    uhfqa.instrument.awg.run.assert_called()


def test_uhfqa_stop(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAInstrumentCoordinatorComponent = make_uhfqa("uhfqa0", "dev1234")
    uhfqa.zi_settings = settings.ZISettings(
        list(),
        {
            0: mocker.Mock(),
        },
    )

    # Act
    uhfqa.stop()

    # Assert
    uhfqa.instrument.awg.stop.assert_called()


def test_uhfqa_prepare(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAInstrumentCoordinatorComponent = make_uhfqa("uhfqa0", "dev1234")
    config = ZIDeviceConfig("hdawg0", settings.ZISettingsBuilder(), None)
    serialize = mocker.patch.object(settings.ZISettings, "serialize")
    apply = mocker.patch.object(settings.ZISettings, "apply")
    mocker.patch("quantify_core.data.handling.get_datadir", return_value=".")

    mocker.patch.object(zi_helpers, "get_waves_directory", return_value=Path("waves/"))
    mocker.patch.object(Path, "glob", return_value=["uhfqa0_awg0.csv"])
    copy2 = mocker.patch("shutil.copy2")

    # Act
    uhfqa.prepare(config)

    # Assert
    uhfqa_serialize_settings = settings.ZISerializeSettings(
        f"{uhfqa.instrument.name}", uhfqa.instrument._serial, uhfqa.instrument._type
    )
    serialize.assert_called_with(Path("."), uhfqa_serialize_settings)
    apply.assert_called_with(uhfqa.instrument)
    copy2.assert_called_with("uhfqa0_awg0.csv", "waves")


@pytest.mark.parametrize("bin_mode", [enums.BinMode.AVERAGE, enums.BinMode.APPEND])
def test_uhfqa_retrieve_acquisition(mocker, make_uhfqa, bin_mode):
    # Arrange
    uhfqa: zhinst.UHFQAInstrumentCoordinatorComponent = make_uhfqa("uhfqa0", "dev1234")
    expected_data = np.ones(64)

    def resolver(uhfqa):  # pylint: disable=unused-argument
        return expected_data

    config = ZIDeviceConfig(
        "hdawg0",
        settings.ZISettingsBuilder(),
        ZIAcquisitionConfig(
            n_acquisitions=1, resolvers={0: resolver}, bin_mode=bin_mode
        ),
    )
    mocker.patch.object(settings.ZISettings, "serialize")
    mocker.patch.object(settings.ZISettings, "apply")
    mocker.patch("quantify_core.data.handling.get_datadir", return_value=".")

    mocker.patch.object(zi_helpers, "get_waves_directory", return_value=Path("waves/"))
    mocker.patch.object(Path, "glob", return_value=[])

    uhfqa.prepare(config)

    # Act
    acq_result = uhfqa.retrieve_acquisition()

    expected_acq_result: Dict[Tuple[int, int], Any] = dict()
    expected_acq_result[(0, 0)] = (expected_data, np.zeros(expected_data.shape))

    # Assert
    assert not acq_result is None
    assert (0, 0) in acq_result

    for key in acq_result:
        np.testing.assert_array_almost_equal(
            acq_result[key][0], expected_acq_result[key][0]
        )
        np.testing.assert_array_almost_equal(
            acq_result[key][1], expected_acq_result[key][1]
        )


def test_uhfqa_wait_done(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAInstrumentCoordinatorComponent = make_uhfqa("uhfqa0", "dev1234")

    wait_done = mocker.patch.object(uhfqa.instrument.awg, "wait_done")
    timeout: int = 20

    # Act
    uhfqa.wait_done(timeout)

    # Assert
    wait_done.assert_called_with(timeout)
