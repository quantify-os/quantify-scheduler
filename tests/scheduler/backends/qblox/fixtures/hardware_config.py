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
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            # None is not a valid key for the latency corrections
            "latency_corrections": {"q0:mw-q0.01": 2e-08, "q1:mw-q1.01": None}
        },
        "connectivity": {"graph": [["cluster0.module1.complex_output_0", "q1:mw"]]},
    }


@pytest.fixture
def hardware_cfg_qcm_rf():
    yield {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q1:mw-q1.01": {"interm_freq": 50000000.0}}
        },
        "connectivity": {"graph": [["cluster0.module1.complex_output_0", "q1:mw"]]},
    }


@pytest.fixture
def hardware_cfg_rf():
    yield {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "2": {"instrument_type": "QCM_RF"},
                    "4": {"instrument_type": "QRM_RF"},
                },
                "ref": "internal",
            }
        },
        "hardware_options": {
            "mixer_corrections": {
                "q2:mw-q2.01": {
                    "dc_offset_i": -0.045,
                    "dc_offset_q": -0.035,
                    "amp_ratio": 0.9996,
                    "phase_error": -3.9,
                },
                "q2:res-q2.ro": {
                    "dc_offset_i": -0.046,
                    "dc_offset_q": -0.036,
                    "amp_ratio": 0.9999,
                    "phase_error": -3.8,
                },
            },
            "modulation_frequencies": {
                "q2:mw-q2.01": {"interm_freq": 50000000.0},
                "q3:mw-q3.01": {"lo_freq": 5000000000.0, "interm_freq": None},
                "q2:res-q2.ro": {"lo_freq": 7200000000.0, "interm_freq": None},
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module2.complex_output_0", "q2:mw"],
                ["cluster0.module2.complex_output_1", "q3:mw"],
                ["cluster0.module4.complex_output_0", "q2:res"],
            ]
        },
    }


@pytest.fixture
def hardware_cfg_rf_legacy():
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
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster1": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {"instrument_type": "QCM_RF"},
                    "2": {"instrument_type": "QRM_RF"},
                },
                "ref": "internal",
            },
            "cluster2": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {"instrument_type": "QCM_RF"},
                    "2": {"instrument_type": "QRM_RF"},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q2:mw-q2.01": {"interm_freq": 50000000.0},
                "q2:res-q2.ro": {"interm_freq": 300000000.0},
                "q3:mw-q3.01": {"interm_freq": 50000000.0},
                "q3:res-q3.ro": {"interm_freq": 300000000.0},
            }
        },
        "connectivity": {
            "graph": [
                ["cluster1.module1.complex_output_0", "q2:mw"],
                ["cluster1.module2.complex_output_0", "q2:res"],
                ["cluster2.module1.complex_output_0", "q3:mw"],
                ["cluster2.module2.complex_output_0", "q3:res"],
            ]
        },
    }


@pytest.fixture
def hardware_cfg_rf_two_clusters_legacy():
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
def hardware_cfg_trigger_count_legacy():
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
def hardware_cfg_cluster_latency_corrections_legacy():
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
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                    "2": {"instrument_type": "QCM"},
                    "3": {"instrument_type": "QRM"},
                    "4": {"instrument_type": "QRM"},
                    "5": {"instrument_type": "QRM"},
                    "6": {"instrument_type": "QRM"},
                },
                "ref": "internal",
            },
            "iq_mixer_lo0": {"instrument_type": "IQMixer"},
            "iq_mixer_lo1": {"instrument_type": "IQMixer"},
            "iq_mixer_lo2": {"instrument_type": "IQMixer"},
            "lo0": {"instrument_type": "LocalOscillator", "power": 1},
            "lo1": {"instrument_type": "LocalOscillator", "power": 1},
            "lo2": {"instrument_type": "LocalOscillator", "power": 1},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:mw-q0.01": {"lo_freq": None, "interm_freq": 50000000.0},
                "q1:mw-q1.01": {"lo_freq": 7200000000.0, "interm_freq": None},
                "q2:mw-q2.01": {"interm_freq": 6330000000.0},
                "q0:res-q0.ro": {"lo_freq": 7200000000.0, "interm_freq": None},
                "q0:res-q0.multiplex": {"lo_freq": 7200000000.0, "interm_freq": None},
                "q2:res-q2.ro": {"lo_freq": 7200000000.0, "interm_freq": None},
            },
            "mixer_corrections": {
                "q0:mw-q0.01": {
                    "dc_offset_i": 0.1234,
                    "dc_offset_q": -1.337,
                    "amp_ratio": 0.9998,
                    "phase_error": -4.1,
                },
                "q1:mw-q1.01": {"dc_offset_i": 0.2345, "dc_offset_q": 1.337},
                "q2:mw-q2.01": {
                    "dc_offset_i": -0.045,
                    "dc_offset_q": -0.035,
                    "amp_ratio": 0.9996,
                    "phase_error": -3.9,
                },
                "q0:res-q0.ro": {
                    "dc_offset_i": -0.054,
                    "dc_offset_q": -0.034,
                    "amp_ratio": 0.9997,
                    "phase_error": -4.0,
                },
                "q0:res-q0.multiplex": {
                    "dc_offset_i": -0.054,
                    "dc_offset_q": -0.034,
                    "amp_ratio": 0.9997,
                    "phase_error": -4.0,
                },
                "q2:res-q2.ro": {
                    "dc_offset_i": -0.046,
                    "dc_offset_q": -0.036,
                    "amp_ratio": 0.9999,
                    "phase_error": -3.8,
                },
            },
            "input_gain": {
                "q0:res-q0.ro": {"gain_I": 2, "gain_Q": 3},
                "q0:res-q0.multiplex": {"gain_I": 2, "gain_Q": 3},
                "q0:fl-cl0.baseband": 1,
                "q1:fl-cl0.baseband": 3,
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "iq_mixer_lo0.if"],
                ["lo0.output", "iq_mixer_lo0.lo"],
                ["iq_mixer_lo0.rf", "q0:mw"],
                ["cluster0.module1.complex_output_1", "iq_mixer_lo1.if"],
                ["lo1.output", "iq_mixer_lo1.lo"],
                ["iq_mixer_lo1.rf", "q1:mw"],
                ["cluster0.module2.complex_output_0", "q2:mw"],
                ["cluster0.module3.complex_output_0", "iq_mixer_lo2.if"],
                ["lo2.output", "iq_mixer_lo2.lo"],
                ["iq_mixer_lo2.rf", "q0:res"],
                ["cluster0.module4.complex_output_0", "q1:res"],
                ["cluster0.module5.real_output_0", "q0:fl"],
                ["cluster0.module5.real_output_1", "q1:fl"],
                ["cluster0.module6.complex_output_0", "q2:res"],
            ]
        },
    }


@pytest.fixture
def hardware_cfg_cluster_legacy():
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
                    "lo_name": "lo2",
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
        "lo2": {"instrument_type": "LocalOscillator", "frequency": 7.2e9, "power": 1},
    }


@pytest.fixture
def hardware_cfg_cluster_test_component():
    yield {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                    "3": {"instrument_type": "QRM"},
                    "2": {"instrument_type": "QCM_RF"},
                    "4": {"instrument_type": "QRM_RF"},
                },
                "ref": "internal",
            },
            "lo0": {"instrument_type": "LocalOscillator", "power": 1},
            "lo1": {"instrument_type": "LocalOscillator", "power": 1},
        },
        "hardware_options": {
            "mixer_corrections": {
                "q0:mw-q0.01": {
                    "dc_offset_i": -0.045,
                    "dc_offset_q": -0.035,
                    "amp_ratio": 0.9996,
                    "phase_error": -3.9,
                },
                "q2:mw-q2.01": {
                    "dc_offset_i": -0.045,
                    "dc_offset_q": -0.035,
                    "amp_ratio": 0.9996,
                    "phase_error": -3.9,
                },
                "q2:res-q2.ro": {
                    "dc_offset_i": -0.046,
                    "dc_offset_q": -0.036,
                    "amp_ratio": 0.9999,
                    "phase_error": -3.8,
                },
            },
            "modulation_frequencies": {
                "q0:mw-q0.01": {"interm_freq": 6330000000.0},
                "q2:mw-q2.01": {"interm_freq": 50000000.0},
                "q3:mw-q3.01": {"lo_freq": 5000000000.0, "interm_freq": None},
                "q2:res-q2.ro": {"lo_freq": 7200000000.0, "interm_freq": None},
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:mw"],
                ["cluster0.module3.complex_output_0", "q0:res"],
                ["cluster0.module2.complex_output_0", "q2:mw"],
                ["cluster0.module2.complex_output_1", "q3:mw"],
                ["cluster0.module4.complex_output_0", "q2:res"],
            ]
        },
    }


@pytest.fixture
def hardware_cfg_qcm_legacy():
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
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
    }


@pytest.fixture
def hardware_cfg_qcm():
    yield {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM"}},
                "ref": "internal",
            },
            "iq_mixer_lo0": {"instrument_type": "IQMixer"},
            "lo0": {"instrument_type": "LocalOscillator", "power": 1},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:mw-cl0.baseband": {"lo_freq": None, "interm_freq": 50000000.0}
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "iq_mixer_lo0.if"],
                ["lo0.output", "iq_mixer_lo0.lo"],
                ["iq_mixer_lo0.rf", "q0:mw"],
            ]
        },
    }


@pytest.fixture
def hardware_cfg_real_mode_legacy():
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
                            "port": "q0:fl",
                            "clock": "cl0.baseband",
                        },
                    ],
                },
                "real_output_1": {
                    "portclock_configs": [
                        {
                            "port": "q1:fl",
                            "clock": "cl0.baseband",
                        }
                    ],
                },
                "real_output_2": {
                    "portclock_configs": [
                        {
                            "port": "q2:fl",
                            "clock": "cl0.baseband",
                        }
                    ],
                },
                "real_output_3": {
                    "portclock_configs": [
                        {
                            "port": "q3:fl",
                            "clock": "cl0.baseband",
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_real_mode():
    yield {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ["cluster0.module1.real_output_0", "q0:fl"],
                ["cluster0.module1.real_output_1", "q1:fl"],
                ["cluster0.module1.real_output_2", "q2:fl"],
                ["cluster0.module1.real_output_3", "q3:fl"],
            ]
        },
    }


@pytest.fixture
def hardware_cfg_qcm_multiplexing():
    yield {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM"}},
                "ref": "internal",
            },
            "iq_mixer_lo0": {"instrument_type": "IQMixer"},
            "lo0": {"instrument_type": "LocalOscillator", "power": 1},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:mw-q0.01": {"lo_freq": None, "interm_freq": 50000000.0},
                "q1:mw-q0.01": {"lo_freq": None, "interm_freq": 50000000.0},
                "q2:mw-q0.01": {"lo_freq": None, "interm_freq": 50000000.0},
                "q3:mw-q0.01": {"lo_freq": None, "interm_freq": 50000000.0},
                "q4:mw-q0.01": {"lo_freq": None, "interm_freq": 50000000.0},
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "iq_mixer_lo0.if"],
                ["lo0.output", "iq_mixer_lo0.lo"],
                ["iq_mixer_lo0.rf", "q0:mw"],
                ["iq_mixer_lo0.rf", "q1:mw"],
                ["iq_mixer_lo0.rf", "q2:mw"],
                ["iq_mixer_lo0.rf", "q3:mw"],
                ["iq_mixer_lo0.rf", "q4:mw"],
            ]
        },
    }
