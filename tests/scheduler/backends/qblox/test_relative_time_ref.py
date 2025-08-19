from unittest.mock import Mock

import numpy as np
import pytest

from quantify_scheduler.backends.qblox.instrument_compilers import QTMCompiler
from quantify_scheduler.backends.qblox.operation_handling.acquisitions import (
    TimetagAcquisitionStrategy,
)
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler
from quantify_scheduler.backends.qblox_backend import ChannelPath, _SequencerCompilationConfig
from quantify_scheduler.backends.types.common import ModulationFrequencies
from quantify_scheduler.backends.types.qblox import (
    DigitalChannelDescription,
    OpInfo,
    QbloxHardwareOptions,
    QTMDescription,
    SequencerOptions,
    StaticTimetagModuleProperties,
)
from quantify_scheduler.enums import BinMode, TimeRef, TimeSource


def test_set_time_ref_channel_success():
    portclock_to_path = {
        "some:other_port-some.other_clock": ChannelPath.from_path(
            "cluster0.module10.digital_input_0"
        )
    }
    op_infos = {
        ("some:other_port", "some.other_clock"): [
            OpInfo(
                name="foo",
                data={
                    "port": "some:other_port",
                    "clock": "some.other_clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_ref": TimeRef.START,
                    "time_ref_port": None,
                },
                timing=0,
            )
        ],
        ("some:port", "some.clock"): [
            OpInfo(
                name="foo",
                data={
                    "port": "some:port",
                    "clock": "some.clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_ref": TimeRef.PORT,
                    "time_ref_port": "some:other_port",
                },
                timing=0,
            )
        ],
    }
    QTMCompiler._set_time_ref_channel(op_infos, portclock_to_path)

    assert op_infos[("some:port", "some.clock")][0].data["time_ref_channel"] == 0


def test_set_time_ref_channel_port_not_found():
    portclock_to_path = {}
    op_infos = {
        ("some:other_port", "some.other_clock"): [
            OpInfo(
                name="foo",
                data={
                    "port": "some:other_port",
                    "clock": "some.other_clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_ref": TimeRef.START,
                    "time_ref_port": None,
                },
                timing=0,
            )
        ],
        ("some:port", "some.clock"): [
            OpInfo(
                name="foo",
                data={
                    "port": "some:port",
                    "clock": "some.clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_ref": TimeRef.PORT,
                    "time_ref_port": "some:nonexisting_port",
                },
                timing=0,
            )
        ],
    }

    info = op_infos[("some:port", "some.clock")][0]
    with pytest.raises(
        ValueError,
        match=f"Found no channels connected to time_ref_port={info.data['time_ref_port']} "
        f"on the same module as the acquisition port={info.data['port']}",
    ):
        QTMCompiler._set_time_ref_channel(op_infos, portclock_to_path)


def test_set_time_ref_channel_too_many_matching_ports():
    portclock_to_path = {
        "some:other_port-some.other_clock": ChannelPath.from_path(
            "cluster0.module10.digital_input_0"
        ),
        "some:other_port-some.yet_another_clock": ChannelPath.from_path(
            "cluster0.module10.digital_input_1"
        ),
        "some:port-some.clock": ChannelPath.from_path("cluster0.module10.digital_input_2"),
    }
    op_infos = {
        ("some:other_port", "some.other_clock"): [
            OpInfo(
                name="foo",
                data={
                    "port": "some:other_port",
                    "clock": "some.other_clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_ref": TimeRef.START,
                    "time_ref_port": None,
                },
                timing=0,
            )
        ],
        ("some:other_port", "some.yet_another_clock"): [
            OpInfo(
                name="foo",
                data={
                    "port": "some:other_port",
                    "clock": "some.yet_another_clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_ref": TimeRef.START,
                    "time_ref_port": None,
                },
                timing=0,
            )
        ],
        ("some:port", "some.clock"): [
            OpInfo(
                name="foo",
                data={
                    "port": "some:port",
                    "clock": "some.clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_ref": TimeRef.PORT,
                    "time_ref_port": "some:other_port",
                },
                timing=0,
            )
        ],
    }

    info = op_infos[("some:port", "some.clock")][0]
    with pytest.raises(
        ValueError,
        match=f"Found multiple channels connected to time_ref_port={info.data['time_ref_port']}",
    ):
        QTMCompiler._set_time_ref_channel(op_infos, portclock_to_path)


def test_prepare_acq_settings():
    qtm_seq_compiler = TimetagSequencerCompiler(
        parent=Mock(),
        index=0,
        static_hw_properties=StaticTimetagModuleProperties(instrument_type="QTM"),
        sequencer_cfg=_SequencerCompilationConfig(
            hardware_description=DigitalChannelDescription(),
            sequencer_options=SequencerOptions(),
            portclock="some:port-some.clock",
            channel_name="digital_input_0",
            channel_name_measure=None,
            latency_correction=0.0,
            distortion_correction=None,
            lo_name=None,
            modulation_frequencies=ModulationFrequencies(),
            mixer_corrections=None,
            digitization_thresholds=None,
        ),
    )

    time_ref_channel = 5

    acquisitions = [
        TimetagAcquisitionStrategy(
            OpInfo(
                name="foo",
                data={
                    "port": "some:port",
                    "clock": "some.clock",
                    "acq_channel": 0,
                    "bin_mode": BinMode.APPEND,
                    "time_source": TimeSource.FIRST,
                    "time_ref": TimeRef.PORT,
                    "time_ref_port": "some:other_port",
                    "time_ref_channel": time_ref_channel,
                    "protocol": "Timetag",
                    "duration": 4e-6,
                },
                timing=0,
            )
        )
    ]
    qtm_seq_compiler._prepare_acq_settings(
        acquisitions=acquisitions,  # type: ignore
    )

    assert qtm_seq_compiler._settings.time_ref_channel == time_ref_channel
