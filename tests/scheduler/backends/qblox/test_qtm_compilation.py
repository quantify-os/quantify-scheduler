# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the QTM."""
from unittest.mock import Mock

from quantify_scheduler.backends.qblox.instrument_compilers import TimetagModuleCompiler
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler


def test_construct_sequencer_compilers():
    test_module = TimetagModuleCompiler(
        parent=Mock(),
        name="cluster0_module1",
        total_play_time=100e-9,
        instrument_cfg={
            "instrument_type": "QTM",
            "digital_output_0": {
                "portclock_configs": [
                    {
                        "port": "q0:switch",
                        "clock": "digital",
                    }
                ]
            },
            "digital_input_1": {
                "portclock_configs": [
                    {
                        "port": "q0:readout",
                        "clock": "digital",
                    }
                ]
            },
        },
    )

    test_module._op_infos = {
        ("q0:switch", "digital"): [Mock()],
        ("q0:readout", "digital"): [Mock()],
    }

    test_module._construct_all_sequencer_compilers()
    seq_keys = list(test_module.sequencers.keys())

    assert len(seq_keys) == 2
    assert isinstance(test_module.sequencers[seq_keys[0]], TimetagSequencerCompiler)
    assert isinstance(test_module.sequencers[seq_keys[1]], TimetagSequencerCompiler)
