# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch


from __future__ import annotations

from pathlib import Path
from unittest.mock import call

import numpy as np
import pytest
import xarray as xr
from zhinst import qcodes

from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst import settings
from quantify_scheduler.backends.zhinst_backend import (
    ZIAcquisitionConfig,
    ZIDeviceConfig,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.instrument_coordinator.components import zhinst


@pytest.fixture
def make_hdawg(mocker):
    def _make_hdawg(name: str, serial: str) -> zhinst.HDAWGInstrumentCoordinatorComponent:
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
    def _make_uhfqa(name: str, serial: str) -> zhinst.HDAWGInstrumentCoordinatorComponent:
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


@pytest.fixture(
    params=[
        ("Trace", BinMode.AVERAGE),
        ("SSBIntegrationComplex", BinMode.AVERAGE),
        ("SSBIntegrationComplex", BinMode.APPEND),
    ]
)
def acquisition_test_data(request):
    acq_protocol, bin_mode = request.param

    if acq_protocol == "Trace" and bin_mode == BinMode.AVERAGE:
        expected_data = np.ones((1, 64), dtype=np.complex128)
        expected_result = xr.Dataset(
            {
                0: (
                    ["acq_index_0", "trace_index_0"],
                    expected_data,
                    {"acq_protocol": acq_protocol},
                )
            }
        )
    elif acq_protocol == "SSBIntegrationComplex" and bin_mode == BinMode.AVERAGE:
        expected_data = np.ones((1,), dtype=np.complex128)
        expected_result = xr.Dataset(
            {0: (["acq_index_0"], expected_data, {"acq_protocol": acq_protocol})}
        )
    elif acq_protocol == "SSBIntegrationComplex" and bin_mode == BinMode.APPEND:
        expected_data = np.ones((64, 1), dtype=np.complex128)
        expected_result = xr.Dataset(
            {
                0: (
                    ["repetition", "acq_index_0"],
                    expected_data,
                    {"acq_protocol": acq_protocol},
                )
            }
        )
    else:
        raise RuntimeError("Unknown protocol")

    def resolver(uhfqa):
        return expected_data

    return acq_protocol, bin_mode, resolver, expected_result


def test_uhfqa_retrieve_acquisition(mocker, make_uhfqa, acquisition_test_data):
    # Arrange
    acq_protocol, bin_mode, resolver, expected_result = acquisition_test_data
    uhfqa: zhinst.UHFQAInstrumentCoordinatorComponent = make_uhfqa("uhfqa0", "dev1234")

    config = ZIDeviceConfig(
        "hdawg0",
        settings.ZISettingsBuilder(),
        ZIAcquisitionConfig(
            n_acquisitions=1,
            resolvers={0: resolver},
            bin_mode=bin_mode,
            acq_protocols={0: acq_protocol},
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
    xr.testing.assert_identical(acq_result, expected_result)


def test_uhfqa_wait_done(mocker, make_uhfqa):
    # Arrange
    uhfqa: zhinst.UHFQAInstrumentCoordinatorComponent = make_uhfqa("uhfqa0", "dev1234")

    wait_done = mocker.patch.object(uhfqa.instrument.awg, "wait_done")
    timeout: int = 20

    # Act
    uhfqa.wait_done(timeout)

    # Assert
    wait_done.assert_called_with(timeout)
