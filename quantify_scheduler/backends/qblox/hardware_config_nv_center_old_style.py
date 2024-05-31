# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Example old-style Qblox hardware config dictionary of a nv-center setup for legacy support."""

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
                "dc_mixer_offset_I": 0.0,
                "dc_mixer_offset_Q": 0.0,
                "portclock_configs": [
                    {
                        "port": "qe0:mw",
                        "clock": "qe0.spec",
                        "interm_freq": 200000000.0,
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
        },
    },
    "red_laser": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
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
