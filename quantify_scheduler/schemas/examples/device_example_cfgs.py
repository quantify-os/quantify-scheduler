# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains example device config for transmons."""

example_transmon_cfg = {
    "compilation_passes": [
        {
            "name": "circuit_to_device",
            "compilation_func": "quantify_scheduler.backends.circuit_to_device.compile_circuit_to_device_with_config_validation",  # noqa: E501, line too long
        }
    ],
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
                "factory_func": "quantify_scheduler.operations." + "pulse_factories.rxy_drag_pulse",
                "gate_info_factory_kwargs": ["theta", "phi"],
                "factory_kwargs": {
                    "amp180": 0.32,
                    "motzoi": 0.45,
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "duration": 2e-08,
                },
            },
            "Rz": {
                "factory_func": "quantify_scheduler.operations." + "pulse_factories.phase_shift",
                "gate_info_factory_kwargs": ["theta"],
                "factory_kwargs": {"clock": "q0.01"},
            },
            "H": {
                "factory_func": "quantify_scheduler.operations."
                + "composite_factories.hadamard_as_y90z",
                "factory_kwargs": {"qubit": "q0"},
            },
            "measure": {
                "factory_func": "quantify_scheduler.operations."
                + "measurement_factories.dispersive_measurement_transmon",
                "gate_info_factory_kwargs": [
                    "acq_channel_override",
                    "acq_index",
                    "bin_mode",
                    "acq_protocol",
                ],
                "factory_kwargs": {
                    "port": "q0:res",
                    "clock": "q0.ro",
                    "pulse_type": "SquarePulse",
                    "pulse_amp": 0.25,
                    "pulse_duration": 1.6e-07,
                    "acq_delay": 1.2e-07,
                    "acq_duration": 3e-07,
                    "acq_channel": 0,
                    "acq_rotation": 0,
                    "acq_threshold": 0,
                    "freq": None,
                },
            },
        },
        "q1": {
            "reset": {
                "factory_func": "quantify_scheduler.operations.pulse_library.IdlePulse",
                "factory_kwargs": {"duration": 0.0002},
            },
            "Rxy": {
                "factory_func": "quantify_scheduler.operations." + "pulse_factories.rxy_drag_pulse",
                "gate_info_factory_kwargs": ["theta", "phi"],
                "factory_kwargs": {
                    "amp180": 0.4,
                    "motzoi": 0.25,
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "duration": 2e-08,
                },
            },
            "Rz": {
                "factory_func": "quantify_scheduler.operations." + "pulse_factories.phase_shift",
                "gate_info_factory_kwargs": ["theta"],
                "factory_kwargs": {"clock": "q1.01"},
            },
            "measure": {
                "factory_func": "quantify_scheduler.operations."
                + "measurement_factories.dispersive_measurement_transmon",
                "gate_info_factory_kwargs": [
                    "acq_channel_override",
                    "acq_index",
                    "bin_mode",
                    "acq_protocol",
                ],
                "factory_kwargs": {
                    "port": "q1:res",
                    "clock": "q1.ro",
                    "pulse_type": "SquarePulse",
                    "pulse_amp": 0.21,
                    "pulse_duration": 1.6e-07,
                    "acq_delay": 1.2e-07,
                    "acq_duration": 3e-07,
                    "acq_channel": 1,
                    "acq_rotation": 0,
                    "acq_threshold": 0,
                    "freq": None,
                },
            },
        },
    },
    "edges": {
        "q0_q1": {
            "CZ": {
                "factory_func": "quantify_scheduler.operations."
                + "pulse_factories.composite_square_pulse",
                "factory_kwargs": {
                    "square_port": "q0:fl",
                    "square_clock": "cl0.baseband",
                    "square_amp": 0.5,
                    "square_duration": 2e-08,
                    "virt_z_parent_qubit_phase": 44,
                    "virt_z_parent_qubit_clock": "q0.01",
                    "virt_z_child_qubit_phase": 63,
                    "virt_z_child_qubit_clock": "q1.01",
                },
            }
        }
    },
}
