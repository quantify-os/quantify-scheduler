import re
from copy import deepcopy

import networkx as nx
import pytest
from pydantic import ValidationError

from quantify_scheduler import ClockResource, Schedule, SerialCompiler
from quantify_scheduler.backends.qblox_backend import (
    ChannelPath,
    QbloxHardwareCompilationConfig,
    _ClusterCompilationConfig,
    _QCMCompilationConfig,
)
from quantify_scheduler.backends.types.qblox import QbloxHardwareDistortionCorrection
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.operations import Measure, SquarePulse, X


def test_invalid_channel_names_connectivity(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_compilation_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                    },
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ["cluster0.module1.wrong_key", "q0:res"],
            ]
        },
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_compilation_config)

    with pytest.raises(ValueError, match="Invalid channel name"):
        _ = QbloxHardwareCompilationConfig(**hardware_compilation_config)


def test_missing_module_in_description_raises(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_compilation_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "2": {
                        "instrument_type": "QCM",
                    },
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:res"],
            ]
        },
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_compilation_config)

    with pytest.raises(KeyError, match="not found in the hardware description"):
        quantum_device.generate_compilation_config()


def test_channel_as_both_input_and_output_qtm():
    hardware_compilation_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QTM",
                    },
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_0", "q0:switch"),
                ("cluster0.module1.digital_input_0", "q0:switch"),
            ]
        },
    }

    quantum_device = QuantumDevice("quantum_device")
    quantum_device.hardware_config(hardware_compilation_config)

    schedule = Schedule("test channel names")

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="The configuration for the QTM module contains channel names with port "
        "numbers that are assigned as both input and output. This is not "
        "allowed. Conflicting channel names:\ndigital_input_0\ndigital_output_0",
    ):
        compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


# Using the old-style / legacy hardware config dict is deprecated
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_warn_mix_lo_false():
    hardware_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module3": {
                "instrument_type": "QRM",
                "real_input_0": {
                    "lo_name": "red_laser",
                    "mix_lo": False,
                    "portclock_configs": [
                        {
                            "port": "qe0:optical_readout",
                            "clock": "qe0.ge0",
                        },
                    ],
                },
            },
        },
        "red_laser": {
            "instrument_type": "LocalOscillator",
            "frequency": None,
            "power": 1,
        },
    }

    with pytest.warns(UserWarning) as warn:
        _ = QbloxHardwareCompilationConfig.model_validate(hardware_config)

    assert (
        "Using `mix_lo=False` in channels coupled to lasers might cause undefined behavior."
        in str(warn[1].message)
    )


def test_channel_path():
    # Test channel path when "input" path is added as `channel_name_measure`
    channel_path = ChannelPath.from_path("cluster0.module1.complex_output_0")
    assert channel_path.cluster_name == "cluster0"
    assert channel_path.module_name == "module1"
    assert channel_path.channel_name == "complex_output_0"
    assert channel_path.module_idx == 1
    assert channel_path.channel_name_measure is None

    channel_path.add_channel_name_measure("complex_input_0")
    assert channel_path.channel_name == "complex_output_0"
    assert channel_path.channel_name_measure == ["complex_input_0"]

    # Test channel path when "output" path is added as `channel_name_measure`
    channel_path = ChannelPath.from_path("cluster0.module1.complex_input_0")
    channel_path.add_channel_name_measure("complex_output_0")
    assert channel_path.channel_name == "complex_output_0"
    assert channel_path.channel_name_measure == ["complex_input_0"]

    # Test two `channel_name_measure`
    channel_path = ChannelPath.from_path("cluster0.module1.real_output_0")
    channel_path.add_channel_name_measure("real_input_0")
    channel_path.add_channel_name_measure("real_input_1")
    assert channel_path.channel_name == "real_output_0"
    assert channel_path.channel_name_measure == ["real_input_0", "real_input_1"]


@pytest.mark.parametrize(
    "graph",
    [
        [
            ["cluster0.module1.complex_output_0", "q5:res"],
            ["cluster0.module2.complex_input_0", "q5:res"],
        ],
        [
            ["cluster0.module1.complex_output_0", "q5:res"],
            ["cluster1.module1.complex_input_0", "q5:res"],
        ],
    ],
)
def test_channel_name_measure_no_same_module_error(
    graph,
):
    hardware_config = {
        "version": "0.2",
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": "QRM"},
                    2: {"instrument_type": "QRM"},
                },
                "ref": "internal",
            },
            "cluster1": {
                "instrument_type": "Cluster",
                "modules": {1: {"instrument_type": "QRM"}},
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {"graph": graph},
    }

    q5 = BasicTransmonElement("q5")

    q5.rxy.amp180(0.213)
    q5.clock_freqs.f01(4.33e8)
    q5.clock_freqs.f12(6.09e9)
    q5.clock_freqs.readout(4.5e8)
    q5.measure.acq_delay(100e-9)

    schedule = Schedule("test_channel_measure")
    schedule.add(Measure("q5"))

    quantum_device = QuantumDevice("basic_transmon_quantum_device")
    quantum_device.add_element(q5)
    quantum_device.hardware_config(hardware_config)

    with pytest.raises(ValueError) as error:
        _ = SerialCompiler(name="compiler").compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )
    assert "Provided channel names" in error.exconly()


@pytest.mark.parametrize(
    "module_type, channel_name, channel_name_measure, error_message",
    [
        (
            "QCM",
            "complex_output_0",
            "complex_output_1",
            "two channel names",
        ),
        ("QRM", "complex_output_0", "digital_output_0", "not of the same mode"),
        (
            "QRM",
            "digital_output_0",
            ["real_input_0", "real_input_1"],
            "incorrect combination of three",
        ),
        (
            "QCM_RF",
            "complex_output_0",
            "digital_output_0",
            "Repeated portclocks are forbidden",
        ),
        ("QRM_RF", "complex_output_0", "digital_output_0", "of the same mode"),
        ("QTM", "digital_output_0", "digital_input_1", "not implemented"),
    ],
)
def test_channel_name_measure_invalid_combinations(
    module_type, channel_name, channel_name_measure, error_message
):
    if isinstance(channel_name_measure, list):
        channel_name_measure_1, channel_name_measure_2 = channel_name_measure
    else:
        channel_name_measure_1, channel_name_measure_2 = channel_name_measure, None

    hardware_config = {
        "version": "0.2",
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": module_type},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                [f"cluster0.module1.{channel_name}", "q5:res"],
                [f"cluster0.module1.{channel_name_measure_1}", "q5:res"],
            ]
        },
    }

    if isinstance(channel_name_measure, list):
        hardware_config["connectivity"]["graph"].append(
            [f"cluster0.module1.{channel_name_measure_2}", "q5:res"]
        )

    q5 = BasicTransmonElement("q5")

    q5.rxy.amp180(0.213)
    q5.clock_freqs.f01(4.33e8)
    q5.clock_freqs.f12(6.09e9)
    q5.clock_freqs.readout(4.5e8)
    q5.measure.acq_delay(100e-9)

    schedule = Schedule("test_channel_measure")
    if "QRM" in module_type:
        schedule.add(Measure("q5"))
    else:
        schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q5:res", clock="q5.ro"))

    quantum_device = QuantumDevice("basic_transmon_quantum_device")
    quantum_device.add_element(q5)
    quantum_device.hardware_config(hardware_config)

    error_type = NotImplementedError if module_type == "QTM" else ValueError

    with pytest.raises(error_type) as error:
        _ = SerialCompiler(name="compiler").compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )
    assert error_message in error.exconly()


@pytest.mark.parametrize(
    "instrument_type, first_channel_name, second_channel_name, result_channel_name_measure",
    [
        (
            "QCM",
            "complex_output_0",
            None,
            None,
        ),
        (
            "QRM",
            "complex_output_0",
            None,
            ["complex_input_0"],
        ),
        (
            "QRM",
            "real_output_0",
            None,
            ["real_input_0", "real_input_1"],
        ),
        (
            "QRM",
            "real_output_0",
            "real_input_0",
            ["real_input_1"],
        ),
        (
            "QRM",
            "real_output_0",
            "real_input_1",
            ["real_input_0"],
        ),
        ("QRM", "real_input_0", None, None),
        (
            "QCM_RF",
            "complex_output_0",
            None,
            None,
        ),
        (
            "QRM_RF",
            "complex_output_0",
            None,
            ["complex_input_0"],
        ),
    ],
)
def test_add_support_input_channel_names(
    instrument_type,
    first_channel_name,
    second_channel_name,
    result_channel_name_measure,
):
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": instrument_type},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                [f"cluster0.module1.{first_channel_name}", "q0:res"],
            ]
        },
    }

    portclocks_used = {("q0:res", "q0.ro")}

    if second_channel_name is not None:
        hardware_config["connectivity"]["graph"].append(
            [f"cluster0.module1.{second_channel_name}", "q1:res"]
        )
        portclocks_used = {("q0:res", "q0.ro"), ("q1:res", "q1.ro")}

    module1_config = (
        QbloxHardwareCompilationConfig.model_validate(hardware_config)
        ._extract_instrument_compilation_configs(portclocks_used)["cluster0"]
        ._extract_module_compilation_configs()[1]
    )

    assert (
        module1_config.portclock_to_path["q0:res-q0.ro"].channel_name_measure
        == result_channel_name_measure
    )


def test_hardware_compilation_config_versioning():
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                    },
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:mw"],
            ]
        },
    }

    v10_config = QbloxHardwareCompilationConfig.model_validate(hardware_config)
    assert v10_config.version == "0.1"

    hardware_config["version"] = "Some unacceptable version"
    with pytest.raises(ValueError) as error:
        _ = QbloxHardwareCompilationConfig.model_validate(hardware_config)

    assert "Unknown hardware config version" in error.exconly()

    hardware_config["version"] = "0.2"
    v11_config = QbloxHardwareCompilationConfig.model_validate(hardware_config)
    assert v11_config.version == "0.2"


# Transmon-specific config
def test_extract_instrument_compilation_configs_cluster():
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                        "complex_output_0": {"marker_debug_mode_enable": True},
                    },
                },
            },
            "cluster1": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QRM_RF"},
                    "2": {"instrument_type": "QCM_RF"},
                },
            },
            "cluster2": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                },
            },
        },
        "hardware_options": {
            "latency_corrections": {
                "q4:mw-q4.01": 8e-9,
            },
            "modulation_frequencies": {
                "q4:mw-q4.01": {
                    "interm_freq": 200e6,
                    "lo_freq": None,
                },
                "q7:res-q7.ro": {"interm_freq": 52e6},
            },
            "mixer_corrections": {
                "q4:mw-q4.01": {"amp_ratio": 0.9999, "phase_error": -4.2},
            },
            "input_att": {"q7:res-q7.ro": 12},
            "sequencer_options": {"qe0:optical_readout-qe0.ge0": {"ttl_acq_threshold": 0.5}},
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q4:mw"],
                ["cluster1.module1.complex_input_0", "q7:res"],
            ]
        },
    }

    hardware_config = QbloxHardwareCompilationConfig.model_validate(hardware_config)

    portclocks_used = {
        ("q4:mw", "q4.01"),
        ("q7:res", "q7.ro"),
    }

    instrument_configs = hardware_config._extract_instrument_compilation_configs(portclocks_used)

    assert list(instrument_configs.keys()) == [
        "cluster0",
        "cluster1",
    ]

    cluster0 = instrument_configs["cluster0"]
    cluster1 = instrument_configs["cluster1"]

    assert cluster0.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "Cluster",
        "ref": "internal",
        "modules": {
            1: {
                "instrument_type": "QCM",
                "complex_output_0": {"marker_debug_mode_enable": True},
            },
        },
    }

    assert cluster0.hardware_options.model_dump(exclude_unset=True) == {
        "latency_corrections": {"q4:mw-q4.01": 8e-09},
        "modulation_frequencies": {
            "q4:mw-q4.01": {"interm_freq": 200000000.0, "lo_freq": None},
        },
        "mixer_corrections": {
            "q4:mw-q4.01": {
                "amp_ratio": 0.9999,
                "phase_error": -4.2,
            },
        },
    }

    assert cluster0.portclock_to_path == {
        ("q4:mw-q4.01"): ChannelPath.from_path("cluster0.module1.complex_output_0"),
    }

    assert cluster0.lo_to_path == {}

    assert cluster1.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "Cluster",
        "ref": "internal",
        "modules": {1: {"instrument_type": "QRM_RF"}},
    }

    assert cluster1.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {"q7:res-q7.ro": {"interm_freq": 52000000.0}},
        "input_att": {"q7:res-q7.ro": 12},
    }

    assert cluster1.portclock_to_path == {
        ("q7:res-q7.ro"): ChannelPath.from_path("cluster1.module1.complex_input_0")
    }
    assert cluster1.lo_to_path == {}


def test_extract_instrument_compilation_configs_lo():

    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "3": {"instrument_type": "QRM"},
                },
            },
            "lo1": {"instrument_type": "LocalOscillator", "power": 1},
            "iq_mixer_lo1": {"instrument_type": "IQMixer"},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7.2e9},
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_output_0", "iq_mixer_lo1.if"],
                ["lo1.output", "iq_mixer_lo1.lo"],
                ["iq_mixer_lo1.rf", "q4:res"],
            ]
        },
    }

    hardware_config = QbloxHardwareCompilationConfig.model_validate(hardware_config)

    portclocks_used = {("q4:res", "q4.ro")}

    instrument_configs = hardware_config._extract_instrument_compilation_configs(portclocks_used)

    assert list(instrument_configs.keys()) == ["cluster0", "lo1"]

    cluster0 = instrument_configs["cluster0"]
    lo1 = instrument_configs["lo1"]

    assert cluster0.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {
            "q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7200000000.0},
        },
    }

    assert cluster0.lo_to_path == {
        "lo1": ChannelPath.from_path("cluster0.module3.complex_output_0"),
    }

    assert lo1.model_dump() == {
        "hardware_description": {
            "instrument_type": "LocalOscillator",
            "instrument_name": "lo1",
            "generic_icc_name": None,
            "frequency_param": "frequency",
            "power_param": "power",
            "power": 1,
        },
        "frequency": 7200000000.0,
    }


def test_extract_module_compilation_configs():
    cluster_compilation_config = {
        "hardware_description": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "sequence_to_file": False,
            "modules": {
                2: {
                    "instrument_type": "QCM_RF",
                },
                3: {
                    "instrument_type": "QRM",
                },
            },
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:mw-q0.01": {"interm_freq": 50000000.0, "lo_freq": None},
                "q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7200000000.0},
            },
            "mixer_corrections": {
                "q4:res-q4.ro": {
                    "dc_offset_i": -0.054,
                    "dc_offset_q": -0.034,
                    "amp_ratio": 0.9997,
                    "phase_error": -4.0,
                }
            },
            "input_gain": {"q4:res-q4.ro": {"gain_I": 2, "gain_Q": 3}},
            "output_att": {"q0:mw-q0.01": 4},
        },
        "portclock_to_path": {
            "q0:mw-q0.01": ChannelPath.from_path("cluster0.module2.complex_output_0"),
            "q4:res-q4.ro": ChannelPath.from_path("cluster0.module3.complex_output_0"),
        },
        "lo_to_path": {"lo1": ChannelPath.from_path("cluster0.module3.complex_output_0")},
        "parent_config_version": "0.2",
    }

    cluster_compilation_config = _ClusterCompilationConfig.model_validate(
        cluster_compilation_config
    )
    module_configs = cluster_compilation_config._extract_module_compilation_configs()

    assert list(module_configs.keys()) == [2, 3]

    module2 = module_configs[2]
    module3 = module_configs[3]

    assert module2.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "QCM_RF"
    }
    assert module2.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {"q0:mw-q0.01": {"interm_freq": 50000000.0, "lo_freq": None}},
        "output_att": {"q0:mw-q0.01": 4},
    }

    assert module2.portclock_to_path == {
        ("q0:mw-q0.01"): ChannelPath.from_path("cluster0.module2.complex_output_0")
    }
    assert module2.lo_to_path == {}

    assert module3.hardware_description.model_dump(exclude_unset=True) == {"instrument_type": "QRM"}
    assert module3.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {"q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7200000000.0}},
        "mixer_corrections": {
            "q4:res-q4.ro": {
                "dc_offset_i": -0.054,
                "dc_offset_q": -0.034,
                "amp_ratio": 0.9997,
                "phase_error": -4.0,
            }
        },
        "input_gain": {"q4:res-q4.ro": {"gain_I": 2, "gain_Q": 3}},
    }

    assert module3.portclock_to_path == {
        ("q4:res-q4.ro"): ChannelPath.from_path("cluster0.module3.complex_output_0")
    }
    assert module3.lo_to_path == {"lo1": ChannelPath.from_path("cluster0.module3.complex_output_0")}


def test_extract_sequencer_compilation_configs():
    module_compilation_config = {
        "hardware_description": {
            "instrument_type": "QCM",
            "complex_output_0": {
                "marker_debug_mode_enable": True,
            },
        },
        "hardware_options": {
            "latency_corrections": {"q0:mw-q0.01": 8e-09},
            "distortion_corrections": {
                "q0:mw-q0.01": {
                    "filter_func": "scipy.signal.lfilter",
                    "input_var_name": "x",
                    "kwargs": {"b": [0, 0.25, 0.5], "a": [1]},
                    "clipping_values": [-2.5, 2.5],
                    "sampling_rate": 1000000000.0,
                }
            },
            "modulation_frequencies": {
                "q0:mw-q0.01": {"interm_freq": None, "lo_freq": 7800000000.0},
                "q1:mw-q1.01": {"interm_freq": 50000000.0, "lo_freq": None},
            },
            "mixer_corrections": {
                "q0:mw-q0.01": {
                    "amp_ratio": 0.9999,
                    "phase_error": -4.2,
                }
            },
            "sequencer_options": {
                "q0:mw-q0.01": {
                    "ttl_acq_threshold": 0.5,
                }
            },
        },
        "portclock_to_path": {
            "q1:mw-q1.01": ChannelPath.from_path("cluster0.module1.complex_output_1"),
            "q0:mw-q0.01": ChannelPath.from_path("cluster0.module1.complex_output_0"),
        },
        "lo_to_path": {"lo0": ChannelPath.from_path("cluster0.module1.complex_output_0")},
        "parent_config_version": "0.2",
    }

    module_compilation_config = _QCMCompilationConfig.model_validate(module_compilation_config)

    sequencer_configs = module_compilation_config._extract_sequencer_compilation_configs()

    assert list(sequencer_configs.keys()) == [0, 1]

    assert sequencer_configs[0].model_dump(exclude_unset=True) == {
        "allow_off_grid_nco_ops": None,
        "sequencer_options": {"ttl_acq_threshold": 0.5},
        "hardware_description": {"marker_debug_mode_enable": True},
        "portclock": "q0:mw-q0.01",
        "channel_name": "complex_output_0",
        "channel_name_measure": None,
        "latency_correction": 8e-09,
        "distortion_correction": {
            "filter_func": "scipy.signal.lfilter",
            "input_var_name": "x",
            "kwargs": {"b": [0, 0.25, 0.5], "a": [1]},
            "clipping_values": [-2.5, 2.5],
            "sampling_rate": 1e9,
        },
        "lo_name": "lo0",
        "modulation_frequencies": {"interm_freq": None, "lo_freq": 7800000000.0},
        "mixer_corrections": {"amp_ratio": 0.9999, "phase_error": -4.2},
        "digitization_thresholds": None,
    }

    assert sequencer_configs[1].model_dump(exclude_unset=True) == {
        "allow_off_grid_nco_ops": None,
        "sequencer_options": {},
        "hardware_description": {},
        "portclock": "q1:mw-q1.01",
        "channel_name": "complex_output_1",
        "channel_name_measure": None,
        "latency_correction": 0.0,
        "distortion_correction": None,
        "lo_name": None,
        "modulation_frequencies": {"interm_freq": 50000000.0, "lo_freq": None},
        "mixer_corrections": None,
        "digitization_thresholds": None,
    }


# NV-center-specific config (this is a sanity check because of common `optical_control`` port)
def test_extract_instrument_compilation_configs_nv_center(
    qblox_hardware_config_nv_center,
):
    hardware_config = QbloxHardwareCompilationConfig.model_validate(
        deepcopy(qblox_hardware_config_nv_center)
    )

    portclocks_used = {
        ("qe0:optical_control", "qe0.ge1"),
        ("qe0:optical_control", "qe0.ionization"),
        ("qe0:optical_control", "qe0.ge0"),
    }

    instrument_configs = hardware_config._extract_instrument_compilation_configs(portclocks_used)

    assert list(instrument_configs.keys()) == [
        "cluster0",
        "red_laser",
        "spinpump_laser",
        "green_laser",
    ]

    cluster0 = instrument_configs["cluster0"]

    assert cluster0.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "Cluster",
        "ref": "internal",
        "modules": {2: {"instrument_type": "QCM"}},
    }

    assert cluster0.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {
            "qe0:optical_control-qe0.ge1": {
                "interm_freq": 200000000.0,
                "lo_freq": None,
            },
            "qe0:optical_control-qe0.ionization": {
                "interm_freq": 200000000.0,
                "lo_freq": None,
            },
            "qe0:optical_control-qe0.ge0": {
                "interm_freq": 200000000.0,
                "lo_freq": None,
            },
        }
    }
    assert cluster0.portclock_to_path == {
        ("qe0:optical_control-qe0.ge1"): ChannelPath.from_path("cluster0.module2.real_output_0"),
        ("qe0:optical_control-qe0.ionization"): ChannelPath.from_path(
            "cluster0.module2.real_output_1"
        ),
        ("qe0:optical_control-qe0.ge0"): ChannelPath.from_path("cluster0.module2.real_output_2"),
    }

    assert cluster0.lo_to_path == {
        "spinpump_laser": ChannelPath.from_path("cluster0.module2.real_output_0"),
        "green_laser": ChannelPath.from_path("cluster0.module2.real_output_1"),
        "red_laser": ChannelPath.from_path("cluster0.module2.real_output_2"),
    }


def test_external_lo_not_present_raises(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q4"))
    sched.add(Measure("q4"))

    compile_config = deepcopy(compile_config_basic_transmon_qblox_hardware)

    # Change to non-existent LO:
    compile_config.hardware_compilation_config.connectivity.graph = nx.relabel_nodes(
        compile_config.hardware_compilation_config.connectivity.graph,
        {"lo0.output": "non_existent_lo.output"},
    )

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Could not find local oscillator device for 'iq_mixer_lo0', which is "
            "connected to cluster module port 'cluster0.module1.complex_output_0' and "
            "port 'q4:mw' in the connectivity. Did find unidentified nodes "
            "['non_existent_lo.output']. Make sure these are specified in the hardware description."
        ),
    ):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(sched, config=compile_config)


def test_assign_attenuation_invalid_raises(
    mock_setup_basic_transmon_with_standard_params, hardware_cfg_qcm_rf
):
    """
    Test that setting a float value (that is not close to an int) raises an error.
    """
    sched = Schedule("Single Gate Experiment")
    sched.add(X("q1"))

    hardware_cfg = deepcopy(hardware_cfg_qcm_rf)
    hardware_cfg["hardware_options"]["output_att"] = {"q1:mw-q0.01": 10.3}

    mock_setup_basic_transmon_with_standard_params["quantum_device"].hardware_config(hardware_cfg)
    with pytest.raises(
        ValueError, match="Input should be a valid integer, got a number with a fractional part"
    ):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=mock_setup_basic_transmon_with_standard_params[
                "quantum_device"
            ].generate_compilation_config(),
        )


# Setting latency corrections in the hardware config is deprecated
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_apply_latency_corrections_hardware_config_invalid_raises(
    mock_setup_basic_transmon, hardware_cfg_latency_corrections_invalid
):
    """
    This test function checks that:
    Providing an invalid latency correction specification raises an exception
    when compiling.
    """

    sched = Schedule("Single Gate Experiment on Two Qubits")
    sched.add(X("q0"))
    sched.add(
        SquarePulse(port="q1:mw", clock="q1.01", amp=0.25, duration=12e-9),
        ref_pt="start",
    )
    sched.add_resources([ClockResource("q0.01", freq=5e9), ClockResource("q1.01", freq=5e9)])

    hardware_cfg = deepcopy(hardware_cfg_latency_corrections_invalid)
    hardware_cfg["hardware_options"]["latency_corrections"]["q1:mw-q1.01"] = None
    mock_setup_basic_transmon["quantum_device"].hardware_config(hardware_cfg)
    with pytest.raises(ValidationError, match="Input should be a valid number"):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=mock_setup_basic_transmon["quantum_device"].generate_compilation_config(),
        )


@pytest.mark.parametrize(
    "portclock, distortion_correction, error_msg",
    [
        (
            "q0:mw-q0.01",
            QbloxHardwareDistortionCorrection(
                exp1_coeffs=[2000, -0.1],
                fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
            ),
            "two corrections are required",
        ),
        (
            "q1:mw-q1.01",
            [
                QbloxHardwareDistortionCorrection(
                    exp1_coeffs=[200, -0.1],
                    fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                ),
                QbloxHardwareDistortionCorrection(
                    exp1_coeffs=[20, -0.1],
                    fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                ),
            ],
            "one correction is required",
        ),
    ],
)
def test_validate_hardware_distortion_corrections_mode(portclock, distortion_correction, error_msg):
    quantum_device = QuantumDevice("qblox_distortions_device")
    compiler = SerialCompiler(name="compiler")

    hardware_compilation_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                },
            },
        },
        "hardware_options": {
            "distortion_corrections": {
                portclock: distortion_correction,
            },
        },
        "connectivity": {
            "graph": [
                ("cluster0.module1.complex_output_0", "q0:mw"),
                ("cluster0.module1.real_output_2", "q1:mw"),
            ]
        },
    }

    port, clock = portclock.split("-")

    sched = Schedule("Qblox hardware distortion corrections test", repetitions=1)
    sched.add_resource(ClockResource(name=clock, freq=5e6))
    sched.add(
        SquarePulse(
            amp=0.1,
            port=port,
            duration=200e-9,
            clock=clock,
            t0=0,
        )
    )

    quantum_device.hardware_config(hardware_compilation_cfg)

    with pytest.raises(ValueError) as error:
        sched = compiler.compile(
            schedule=sched,
            config=quantum_device.generate_compilation_config(),
        )

    assert error_msg in error.value.args[0]
