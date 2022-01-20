example_transmon_cfg = {
    "backend": "quantify_scheduler.compilation.add_pulse_information",
    "clocks": {
        "q0.01": 6020000000.0,
        "q0.ro": 7040000000.0,
        "q1.01": 5020000000.0,
        "q1.ro": 6900000000.0,
    },
    "qubits": {
        "q0": {
            # example of a pulse with one-to-one mapping (directly use Pulse class)
            "reset": {
                "generator_func": "quantify_scheduler.operations.pulse_library.IdlePulse",
                "duration": 200e-6,
            },
            # example of a pulse with a parametrized mapping, use a constructor class.
            "Rxy": {
                "generator_func": "quantify_scheduler.operations.pulse_generators.rxy_drag_pulse",
                "gate_info_generator_kwargs": [
                    "theta",
                    "phi",
                ],  # the keys from the gate info to pass to the generator
                "amp180": 0.6,
                "motzoi": 0.45,
                "port": "q0:mw",
                "clock": "q0.01",
                "duration": 20e-9,
            },
            # here we add a Z pulse using a square flux pulse to detune the qubit. Alternatively
            "Z": {
                "generator_func": "quantify_scheduler.operations.pulse_library.SoftSquarePulse",
                "amp": 0.23,
                "duration": 4e-9,
                "port": "q0:fl",
                "clock": "cl0.baseband",
            },
            # the measurement also has a parametrized mapping, and uses a constructor class.
            "measure": {
                "generator_func": "quantify_scheduler.operations.measurement_generators.dispersive_measurement",
                "gate_info_generator_kwargs": ["acq_index", "bin_mode"],
                "port": "q0:ro",
                "clock": "q0.ro",
                "pulse_type": "SquarePulse",
                "pulse_amp": 0.0005,
                "pulse_duration": 160e-9,
                "acq_delay": 120e-9,
                "acq_duration": 300e-9,
                "acq_protocol": "SSBIntegrationComplex",
                "acq_channel": 0,  # channel corresponding to this qubit
            },
        },
        "q1": {
            "reset": {
                "generator_func": "quantify_scheduler.operations.pulse_library.IdlePulse",
                "duration": 200e-6,
            },
            "Rxy": {
                "generator_func": "quantify_scheduler.operations.pulse_generators.rxy_drag_pulse",
                "gate_info_generator_kwargs": [
                    "theta",
                    "phi",
                ],  # the keys from the gate info to pass to the generator
                "amp180": 0.8,
                "motzoi": 0.25,
                "port": "q1:mw",
                "clock": "q1.01",
                "duration": 20e-9,
            },
            "measure": {
                "generator_func": "quantify_scheduler.operations.measurement_generators.dispersive_measurement",
                "gate_info_generator_kwargs": ["acq_index", "bin_mode"],
                "port": "q0:ro",
                "clock": "q0.ro",
                "pulse_type": "SquarePulse",
                "pulse_amp": 0.9,
                "pulse_duration": 160e-9,
                "acq_delay": 120e-9,
                "acq_duration": 300e-9,
                "acq_protocol": "SSBIntegrationComplex",
                "acq_channel": 0,  # channel corresponding to this qubit
            },
        },
    },
    "edges": {
        "q0-q1": {
            "CZ": {
                "generator_func": "quantify_scheduler.operations.pulse_library.SuddenNetZeroPulse",
                "port": "q0:fl",
                "clock": "cl0.baseband",
                "amp_A": 0.5,
                "amp_B": 0.4,
                "net_zero_A_scale": 0.95,
                "t_pulse": 20e-9,
                "t_phi": 2e-9,
                "t_integral_correction": 10e-9,
            },
        },
    },
}
