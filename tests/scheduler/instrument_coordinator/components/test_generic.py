# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals
from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List
from unittest.mock import ANY, call

import numpy as np
import pytest

from quantify_scheduler.instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)


@pytest.fixture
def typical_zi_hardware_map() -> Dict[str, Any]:
    return json.loads(
        """
        {
            "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
            "mode": "calibration",
            "generic_devices": {
                     "lo_mw_q0": {"frequency": 7.5e9, "power": 16, "status": false},
                     "lo_ro_q0": {"frequency": null, "power": 13, "status": false},
                     "lo_spec_q0": {"frequency": null, "power": null, "status": false}
                   },
            "devices": [
                {
                    "name": "hdawg0",
                    "type": "HDAWG4",
                    "clock_select": 0,
                    "ref": "int",
                    "channelgrouping": 0,
                    "channel_0": {
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "mode": "complex",
                        "modulation": {"type": "premod", "interm_freq": 0},
                        "local_oscillator": "lo_mw_q0",
                        "clock_frequency": 6e9,
                        "line_trigger_delay": 191e-9,
                        "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                        "gain1": 1,
                        "gain2": 1,
                        "latency": 12e-9,
                        "mixer_corrections": {
                            "amp_ratio": 0.950,
                            "phase_error": 90,
                            "dc_offset_I": -0.5420,
                            "dc_offset_Q": -0.3280
                        }
                    }
                },
                {
                    "name": "uhfqa0",
                    "type": "UHFQA",
                    "ref": "ext",
                    "channel_0": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "mode": "real",
                        "modulation": {"type": "premod", "interm_freq": 100e6},
                        "local_oscillator": "lo_ro_q0",
                        "clock_frequency": 6e9,
                        "triggers": [2]
                    }
                }
            ]
        }
        """
    )
