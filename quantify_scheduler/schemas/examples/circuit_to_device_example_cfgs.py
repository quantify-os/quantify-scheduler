ex_transmon_cfg = {
    "backend": "quantify_scheduler.compilation.add_pulse_information",
    "qubits": {
        "q0": {
            # example of a pulse with one-to-one mapping (directly use Pulse class)
            "Reset": {
                "generator_func": "quantify_scheduler.operations.pulse_library.IdlePulse",
                "duration": 200e-6,
            },
            # example of a pulse with a parametrized mapping, use a constructor class.
            "Rxy": {
                "generator_func": "quantify_scheduler.operations.pulse_generators.DRAG",
                "gate_info_generator_kwargs": [
                    "theta",
                    "phi",
                ],  # the keys from the gate info to pass to the generator
                "amp180": 0.23,
                "motzoi": -0.02,
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
                "generator_func": "quantify_scheduler.operations.pulse_generators.DispersiveMeasurement",
                "gate_info_generator_kwargs": ["acq_index", "bin_mode"],
                "port": "q0:ro",
                "clock": "q0.ro",
                "pulse_type": "Square",
                "pulse_amp": 0.2,
                "pulse_duration": 2e-6,
                "acq_channel": 0,  # channel corresponding to this qubit
            },
        },
        "edges": {
            "q0-q1": {
                "CZ": {},
            }
        },
    },
}
