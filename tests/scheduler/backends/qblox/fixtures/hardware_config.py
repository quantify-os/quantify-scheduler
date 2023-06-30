from typing import Any, Dict, Generator

import pytest

from quantify_scheduler.resources import BasebandClockResource
from quantify_scheduler.schemas.examples import utils

QBLOX_HARDWARE_COMPILATION_CONFIG = utils.load_json_example_scheme(
    "qblox_hardware_compilation_config.json"
)


@pytest.fixture
def hardware_compilation_config_qblox_example() -> (
    Generator[Dict[str, Any], None, None]
):
    yield dict(QBLOX_HARDWARE_COMPILATION_CONFIG)


@pytest.fixture
def hardware_cfg_latency_corrections_invalid():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        # None is not a valid key for the latency corrections
        "latency_corrections": {"q0:mw-q0.01": 2e-8, "q1:mw-q1.01": None},
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:mw",
                            "clock": "q1.01",
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_qcm_rf():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:mw",
                            "clock": "q1.01",
                            "interm_freq": 50e6,
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_cluster_and_pulsar_latency_corrections():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "latency_corrections": {"q0:mw-q0.01": 2e-8, "q1:mw-q1.01": -5e-9},
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "portclock_configs": [{"port": "q0:mw", "clock": "q0.01"}],
            },
        },
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:mw",
                            "clock": "q1.01",
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_pulsar():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "lo_name": "lo0",
                "dc_mixer_offset_I": 0.1234,
                "dc_mixer_offset_Q": -1.337,
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                        "mixer_amp_ratio": 0.9998,
                        "mixer_phase_error_deg": -4.1,
                    }
                ],
            },
            "complex_output_1": {
                "lo_name": "lo1",
                "dc_mixer_offset_I": 0.2345,
                "dc_mixer_offset_Q": 1.337,
                "portclock_configs": [
                    {"port": "q1:mw", "clock": "q1.01", "interm_freq": None}
                ],
            },
        },
        "qcm1": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "dc_mixer_offset_I": -0.045,
                "dc_mixer_offset_Q": -0.035,
                "portclock_configs": [
                    {
                        "mixer_amp_ratio": 0.9996,
                        "mixer_phase_error_deg": -3.9,
                        "port": "q2:mw",
                        "clock": "q2.01",
                        "interm_freq": 6.33e9,
                    }
                ],
            },
        },
        "qrm0": {
            "instrument_type": "Pulsar_QRM",
            "ref": "external",
            "complex_output_0": {
                "lo_name": "lo1",
                "dc_mixer_offset_I": -0.054,
                "dc_mixer_offset_Q": -0.034,
                "input_gain_I": 2,
                "input_gain_Q": 3,
                "portclock_configs": [
                    {
                        "mixer_amp_ratio": 0.9997,
                        "mixer_phase_error_deg": -4.0,
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "interm_freq": None,
                    },
                    {
                        "mixer_amp_ratio": 0.9997,
                        "mixer_phase_error_deg": -4.0,
                        "port": "q0:res",
                        "clock": "q0.multiplex",
                        "interm_freq": None,
                    },
                ],
            },
        },
        "qrm1": {
            "instrument_type": "Pulsar_QRM",
            "ref": "external",
            "complex_output_0": {
                "portclock_configs": [{"port": "q1:res", "clock": "q1.ro"}],
            },
        },
        "qrm2": {
            "instrument_type": "Pulsar_QRM",
            "ref": "external",
            "real_output_0": {
                "input_gain_0": 1,
                "portclock_configs": [
                    {
                        "mixer_amp_ratio": 0.9997,
                        "mixer_phase_error_deg": -4.0,
                        "port": "q0:fl",
                        "clock": "cl0.baseband",
                        "interm_freq": None,
                    }
                ],
            },
            "real_output_1": {
                "input_gain_1": 3,
                "portclock_configs": [
                    {
                        "mixer_amp_ratio": 0.9997,
                        "mixer_phase_error_deg": -4.0,
                        "port": "q1:fl",
                        "clock": "cl0.baseband",
                        "interm_freq": None,
                    }
                ],
            },
        },
        "qrm3": {
            "instrument_type": "Pulsar_QRM",
            "ref": "external",
            "complex_output_0": {
                "lo_freq": 7.2e9,
                "dc_mixer_offset_I": -0.046,
                "dc_mixer_offset_Q": -0.036,
                "portclock_configs": [
                    {
                        "mixer_amp_ratio": 0.9999,
                        "mixer_phase_error_deg": -3.8,
                        "port": "q2:res",
                        "clock": "q2.ro",
                        "interm_freq": None,
                    }
                ],
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": 7.2e9, "power": 1},
    }


@pytest.fixture
def hardware_cfg_pulsar_rf():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm_rf0": {
            "instrument_type": "Pulsar_QCM_RF",
            "ref": "internal",
            "complex_output_0": {
                "dc_mixer_offset_I": -0.045,
                "dc_mixer_offset_Q": -0.035,
                "portclock_configs": [
                    {
                        "mixer_amp_ratio": 0.9996,
                        "mixer_phase_error_deg": -3.9,
                        "port": "q2:mw",
                        "clock": "q2.01",
                        "interm_freq": 50e6,
                    }
                ],
            },
            "complex_output_1": {
                "lo_freq": 5e9,
                "portclock_configs": [
                    {"port": "q3:mw", "clock": "q3.01", "interm_freq": None}
                ],
            },
        },
        "qrm_rf0": {
            "instrument_type": "Pulsar_QRM_RF",
            "ref": "external",
            "complex_output_0": {
                "lo_freq": 7.2e9,
                "dc_mixer_offset_I": -0.046,
                "dc_mixer_offset_Q": -0.036,
                "portclock_configs": [
                    {
                        "mixer_amp_ratio": 0.9999,
                        "mixer_phase_error_deg": -3.8,
                        "port": "q2:res",
                        "clock": "q2.ro",
                        "interm_freq": None,
                    }
                ],
            },
        },
        "qrm_rf1": {
            "instrument_type": "Pulsar_QRM_RF",
            "ref": "external",
            "complex_output_0": {
                "portclock_configs": [
                    {"port": "q3:res", "clock": "q3.ro", "interm_freq": 100e6}
                ]
            },
        },
    }


@pytest.fixture
def hardware_cfg_pulsar_qcm(add_lo1):
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "mix_lo": True,
                "lo_name": "lo0",
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": True,
                        "interm_freq": 50e6,
                    }
                ],
            },
            "complex_output_1": {
                "lo_name": "lo1" if add_lo1 else None,
                "portclock_configs": [{"port": "q1:mw", "clock": "q1.01"}],
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": 4.8e9, "power": 1},
    }


@pytest.fixture
def hardware_cfg_pulsar_qcm_real_mode(
    instruction_generated_pulses_enabled,
):  # pylint: disable=line-too-long
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "real_output_0": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_1",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,  # noqa: E501
                    },
                ],
            },
            "real_output_1": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_2",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,  # noqa: E501
                    }
                ],
            },
            "real_output_2": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_3",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,  # noqa: E501
                    }
                ],
            },
            "real_output_3": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_4",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,  # noqa: E501
                    }
                ],
            },
        },
    }


@pytest.fixture
def hardware_cfg_pulsar_qcm_multiplexing():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "lo_name": "lo0",
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q1:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q2:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q3:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q4:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                ],
            },
            "complex_output_1": {
                "portclock_configs": [{"port": "q1:mw", "clock": "q1.01"}],
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
    }


@pytest.fixture
def hardware_cfg_pulsar_qcm_two_qubit_gate():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "portclock_configs": [
                    {"port": f"{qubit}:fl", "clock": clock}
                    for qubit in ["q2", "q3"]
                    for clock in [BasebandClockResource.IDENTITY, f"{qubit}.01"]
                ]
            },
        },
    }
