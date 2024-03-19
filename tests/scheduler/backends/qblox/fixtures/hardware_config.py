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
def hardware_cfg_rf():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module2": {
                "instrument_type": "QCM_RF",
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
            "cluster0_module4": {
                "instrument_type": "QRM_RF",
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
        },
    }


@pytest.fixture
def hardware_cfg_rf_two_clusters():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster1": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster1_module1": {
                "instrument_type": "QCM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q2:mw",
                            "clock": "q2.01",
                            "interm_freq": 50e6,
                        },
                    ],
                },
            },
            "cluster1_module2": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q2:res",
                            "clock": "q2.ro",
                            "interm_freq": 300e6,
                        }
                    ],
                },
            },
        },
        "cluster2": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster2_module1": {
                "instrument_type": "QCM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q3:mw",
                            "clock": "q3.01",
                            "interm_freq": 50e6,
                        },
                    ],
                },
            },
            "cluster2_module2": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q3:res",
                            "clock": "q3.ro",
                            "interm_freq": 300e6,
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_trigger_count():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module3": {
                "instrument_type": "QRM",
                "real_input_0": {
                    "lo_name": "laser_red",
                    "mix_lo": False,
                    "portclock_configs": [
                        {
                            "port": "qe0:optical_readout",
                            "clock": "qe0.ge0",
                            "interm_freq": 50e6,
                            "ttl_acq_threshold": 0.5,
                        },
                    ],
                },
                "real_output_0": {
                    "portclock_configs": [
                        {
                            "port": "qe0:optical_control",
                            "clock": "qe0.ge0",
                            "interm_freq": 0,
                        }
                    ],
                },
            },
        },
        "laser_red": {
            "instrument_type": "LocalOscillator",
            "frequency": None,
            "power": 1,
        },
    }


@pytest.fixture
def hardware_cfg_cluster_latency_corrections():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "latency_corrections": {"q0:mw-q0.01": 2e-8, "q1:mw-q1.01": -5e-9},
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q0:mw",
                            "clock": "q0.01",
                        }
                    ],
                },
                "complex_output_1": {
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
def hardware_cfg_cluster():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
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
            "cluster0_module2": {
                "instrument_type": "QCM",
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
            "cluster0_module3": {
                "instrument_type": "QRM",
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
            "cluster0_module4": {
                "instrument_type": "QRM",
                "complex_output_0": {
                    "portclock_configs": [{"port": "q1:res", "clock": "q1.ro"}],
                },
            },
            "cluster0_module5": {
                "instrument_type": "QRM",
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
            "cluster0_module6": {
                "instrument_type": "QRM",
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
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": 7.2e9, "power": 1},
    }


@pytest.fixture
def hardware_cfg_cluster_test_component():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "dc_mixer_offset_I": -0.045,
                    "dc_mixer_offset_Q": -0.035,
                    "portclock_configs": [
                        {
                            "mixer_amp_ratio": 0.9996,
                            "mixer_phase_error_deg": -3.9,
                            "port": "q0:mw",
                            "clock": "q0.01",
                            "interm_freq": 6.33e9,
                        }
                    ],
                },
            },
            "cluster0_module3": {
                "instrument_type": "QRM",
                "complex_output_0": {
                    "portclock_configs": [{"port": "q0:res", "clock": "q0.ro"}],
                },
            },
            "cluster0_module2": {
                "instrument_type": "QCM_RF",
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
            "cluster0_module4": {
                "instrument_type": "QRM_RF",
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
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": 7.2e9, "power": 1},
    }


@pytest.fixture
def hardware_cfg_qcm(add_lo1):
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "mix_lo": True,
                    "lo_name": "lo0",
                    "portclock_configs": [
                        {
                            "port": "q0:mw",
                            "clock": "cl0.baseband",
                            "interm_freq": 50e6,
                        }
                    ],
                },
                "complex_output_1": {
                    "lo_name": "lo1" if add_lo1 else None,
                    "portclock_configs": [{"port": "q1:mw", "clock": "q1.01"}],
                },
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": 4.8e9, "power": 1},
    }


@pytest.fixture
def hardware_cfg_real_mode():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "real_output_0": {
                    "portclock_configs": [
                        {
                            "port": "dummy_port_1",
                            "clock": "cl0.baseband",
                        },
                    ],
                },
                "real_output_1": {
                    "portclock_configs": [
                        {
                            "port": "dummy_port_2",
                            "clock": "cl0.baseband",
                        }
                    ],
                },
                "real_output_2": {
                    "portclock_configs": [
                        {
                            "port": "dummy_port_3",
                            "clock": "cl0.baseband",
                        }
                    ],
                },
                "real_output_3": {
                    "portclock_configs": [
                        {
                            "port": "dummy_port_4",
                            "clock": "cl0.baseband",
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_qcm_multiplexing():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
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
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
    }


@pytest.fixture
def hardware_cfg_qcm_two_qubit_gate():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "portclock_configs": [
                        {"port": f"{qubit}:fl", "clock": clock}
                        for qubit in ["q2", "q3"]
                        for clock in [BasebandClockResource.IDENTITY, f"{qubit}.01"]
                    ]
                },
            },
        },
    }
