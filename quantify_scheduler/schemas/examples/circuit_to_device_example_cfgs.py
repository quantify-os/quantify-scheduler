# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

example_transmon_cfg = {
    "backend": "quantify_scheduler.backends.circuit_to_device"
    + ".compile_circuit_to_device",
    "clocks": {
        "q0.01": 6020000000.0,
        "q0.ro": 7040000000.0,
        "q1.01": 5020000000.0,
        "q1.ro": 6900000000.0,
    },
    "elements": {
        "q0": {
            "reset": {
                "factory_func": "quantify_scheduler.operations.pulse_library.IdlePulse",
                "factory_kwargs": {"duration": 0.0002},
            },
            "Rxy": {
                "factory_func": "quantify_scheduler.operations."
                + "pulse_factories.rxy_drag_pulse",
                "gate_info_factory_kwargs": ["theta", "phi"],
                "factory_kwargs": {
                    "amp180": 0.32,
                    "motzoi": 0.45,
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "duration": 2e-08,
                },
            },
            "Z": {
                "factory_func": "quantify_scheduler.operations."
                + "pulse_library.SoftSquarePulse",
                "factory_kwargs": {
                    "amp": 0.23,
                    "duration": 4e-09,
                    "port": "q0:fl",
                    "clock": "cl0.baseband",
                },
            },
            "measure": {
                "factory_func": "quantify_scheduler.operations."
                + "measurement_factories.dispersive_measurement",
                "gate_info_factory_kwargs": ["acq_index", "bin_mode"],
                "factory_kwargs": {
                    "port": "q0:res",
                    "clock": "q0.ro",
                    "pulse_type": "SquarePulse",
                    "pulse_amp": 0.25,
                    "pulse_duration": 1.6e-07,
                    "acq_delay": 1.2e-07,
                    "acq_duration": 3e-07,
                    "acq_protocol": "SSBIntegrationComplex",
                    "acq_channel": 0,
                },
            },
        },
        "q1": {
            "reset": {
                "factory_func": "quantify_scheduler.operations.pulse_library.IdlePulse",
                "factory_kwargs": {"duration": 0.0002},
            },
            "Rxy": {
                "factory_func": "quantify_scheduler.operations."
                + "pulse_factories.rxy_drag_pulse",
                "gate_info_factory_kwargs": ["theta", "phi"],
                "factory_kwargs": {
                    "amp180": 0.4,
                    "motzoi": 0.25,
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "duration": 2e-08,
                },
            },
            "measure": {
                "factory_func": "quantify_scheduler.operations."
                + "measurement_factories.dispersive_measurement",
                "gate_info_factory_kwargs": ["acq_index", "bin_mode"],
                "factory_kwargs": {
                    "port": "q1:res",
                    "clock": "q1.ro",
                    "pulse_type": "SquarePulse",
                    "pulse_amp": 0.21,
                    "pulse_duration": 1.6e-07,
                    "acq_delay": 1.2e-07,
                    "acq_duration": 3e-07,
                    "acq_protocol": "SSBIntegrationComplex",
                    "acq_channel": 1,
                },
            },
        },
    },
    "edges": {
        "q0-q1": {
            "CZ": {
                "factory_func": "quantify_scheduler.operations."
                + "pulse_library.SuddenNetZeroPulse",
                "factory_kwargs": {
                    "port": "q0:fl",
                    "clock": "cl0.baseband",
                    "amp_A": 0.5,
                    "amp_B": 0.4,
                    "net_zero_A_scale": 0.95,
                    "t_pulse": 2e-08,
                    "t_phi": 2e-09,
                    "t_integral_correction": 1e-08,
                },
            }
        }
    },
}
