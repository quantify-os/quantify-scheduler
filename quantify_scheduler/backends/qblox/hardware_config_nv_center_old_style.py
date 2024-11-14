# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Example old-style Qblox hardware config dictionary of a nv-center setup for legacy support."""

from quantify_scheduler.backends.qblox.enums import LoCalEnum, SidebandCalEnum

hardware_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster0": {
        "ref": "internal",
        "sequence_to_file": False,
        "instrument_type": "Cluster",
        "cluster0_module1": {
            "instrument_type": "QCM_RF",
            "sequence_to_file": False,
            "complex_output_0": {
                "lo_freq": None,
                "auto_lo_cal": LoCalEnum.OFF,
                "dc_mixer_offset_I": 0.0,
                "dc_mixer_offset_Q": 0.0,
                "portclock_configs": [
                    {
                        "port": "qe0:mw",
                        "clock": "qe0.spec",
                        "interm_freq": 200000000.0,
                        "auto_sideband_cal": SidebandCalEnum.OFF,
                        "mixer_amp_ratio": 0.9999,
                        "mixer_phase_error_deg": -4.2,
                    }
                ],
            },
        },
        "cluster0_module2": {
            "instrument_type": "QCM",
            "sequence_to_file": False,
            "real_output_0": {
                "lo_name": "spinpump_laser",
                "portclock_configs": [
                    {
                        "port": "qe0:optical_control",
                        "clock": "qe0.ge1",
                        "interm_freq": 200e6,
                    }
                ],
            },
            "real_output_1": {
                "lo_name": "green_laser",
                "portclock_configs": [
                    {
                        "port": "qe0:optical_control",
                        "clock": "qe0.ionization",
                        "interm_freq": 200e6,
                    }
                ],
            },
            "real_output_2": {
                "lo_name": "red_laser",
                "portclock_configs": [
                    {
                        "port": "qe0:optical_control",
                        "clock": "qe0.ge0",
                        "interm_freq": 200e6,
                    }
                ],
            },
        },
        "cluster0_module4": {
            "instrument_type": "QRM",
            "sequence_to_file": False,
            "real_output_0": {
                "portclock_configs": [
                    {
                        "port": "qe0:optical_readout",
                        "clock": "qe0.ge0",
                        "interm_freq": 0.0,
                        "ttl_acq_threshold": 0.5,
                        "init_offset_awg_path_I": 0.0,
                        "init_offset_awg_path_Q": 0.0,
                        "init_gain_awg_path_I": 1.0,
                        "init_gain_awg_path_Q": 1.0,
                        "qasm_hook_func": None,
                    }
                ]
            },
            "real_input_0": {
                "portclock_configs": [{"port": "qe0:optical_readout", "clock": "qe0.ge0"}]
            },
        },
        "cluster0_module5": {
            "instrument_type": "QTM",
            "sequence_to_file": False,
            "digital_output_0": {
                "portclock_configs": [{"port": "qe1:switch", "clock": "digital"}],
            },
            "digital_input_4": {
                "portclock_configs": [
                    {
                        "port": "qe1:optical_readout",
                        "clock": "qe1.ge0",
                        "in_threshold_primary": 0.5,
                    }
                ],
            },
        },
        "cluster0_module7": {
            "instrument_type": "QCM",
            "sequence_to_file": False,
            "real_output_2": {
                "lo_name": "red_laser2",
                "portclock_configs": [
                    {
                        "port": "qe1:optical_control",
                        "clock": "qe1.ge0",
                        "interm_freq": 200e6,
                    }
                ],
            },
        },
    },
    "red_laser": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
    "red_laser2": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
    "spinpump_laser": {
        "instrument_type": "LocalOscillator",
        "frequency": None,
        "power": 1,
    },
    "green_laser": {
        "instrument_type": "LocalOscillator",
        "frequency": None,
        "power": 1,
    },
}
