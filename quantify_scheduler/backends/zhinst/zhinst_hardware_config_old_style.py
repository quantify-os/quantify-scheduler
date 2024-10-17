# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Example old-style Zurich Instruments hardware config dictionary for legacy support."""

hardware_config = {
    "backend": "quantify_scheduler.backends.zhinst_backend.compile_backend",
    "latency_corrections": {
        "q0:mw-q0.01": 95e-9,
        "q1:mw-q1.01": 95e-9,
        "q0:res-q0.ro": -95e-9,
        "q1:res-q1.ro": -95e-9,
    },
    "distortion_corrections": {
        "q0:fl-cl0.baseband": {
            "filter_func": "scipy.signal.lfilter",
            "input_var_name": "x",
            "kwargs": {"b": [0, 0.25, 0.5], "a": [1]},
            "clipping_values": [-2.5, 2.5],
            "sampling_rate": 1e9,
        }
    },
    "local_oscillators": [
        {
            "unique_name": "lo0_ch1",
            "instrument_name": "lo0",
            "frequency": {"ch1.frequency": None},
            "frequency_param": "ch1.frequency",
            "power": {"power": 13},
        },
        {
            "unique_name": "lo0_ch2",
            "instrument_name": "lo0",
            "frequency": {"ch2.frequency": None},
            "frequency_param": "ch2.frequency",
            "power": {"ch2.power": 10},
        },
        {
            "unique_name": "lo1",
            "instrument_name": "lo1",
            "frequency": {"frequency": None},
            "frequency_param": "frequency",
            "power": {"power": 16},
        },
    ],
    "devices": [
        {
            "name": "ic_hdawg0",
            "type": "HDAWG8",
            "clock_select": 0,
            "ref": "int",
            "channelgrouping": 0,
            "channel_0": {
                "port": "q0:mw",
                "clock": "q0.01",
                "mode": "complex",
                "modulation": {"type": "premod", "interm_freq": -100000000.0},
                "local_oscillator": "lo0_ch1",
                "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                "gain1": 1.0,
                "gain2": 1.0,
                "mixer_corrections": {
                    "amp_ratio": 0.95,
                    "phase_error": 0.07,
                    "dc_offset_i": -0.0542,
                    "dc_offset_q": -0.0328,
                },
                "trigger": None,
            },
            "channel_1": {
                "port": "q1:mw",
                "clock": "q1.01",
                "mode": "complex",
                "modulation": {"type": "premod", "interm_freq": -100000000.0},
                "local_oscillator": "lo0_ch2",
                "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                "gain1": 1.0,
                "gain2": 1.0,
                "mixer_corrections": {
                    "amp_ratio": 0.95,
                    "phase_error": 0.07,
                    "dc_offset_i": 0.042,
                    "dc_offset_q": 0.028,
                },
                "trigger": None,
            },
            "channel_2": {
                "port": "q2:mw",
                "clock": "q2.01",
                "mode": "complex",
                "modulation": {"type": "premod", "interm_freq": -100000000.0},
                "local_oscillator": "lo0_ch2",
                "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                "gain1": 1.0,
                "gain2": 1.0,
                "mixer_corrections": {
                    "amp_ratio": 0.95,
                    "phase_error": 0.07,
                    "dc_offset_i": 0.042,
                    "dc_offset_q": 0.028,
                },
                "trigger": None,
            },
            "channel_3": {
                "port": "q3:mw",
                "clock": "q3.01",
                "mode": "complex",
                "modulation": {"type": "premod", "interm_freq": -100000000.0},
                "local_oscillator": "lo0_ch2",
                "markers": ["AWG_MARKER1", "AWG_MARKER2"],
                "gain1": 1.0,
                "gain2": 1.0,
                "mixer_corrections": {
                    "amp_ratio": 0.95,
                    "phase_error": 0.07,
                    "dc_offset_i": 0.042,
                    "dc_offset_q": 0.028,
                },
                "trigger": None,
            },
        },
        {
            "name": "ic_uhfqa0",
            "type": "UHFQA",
            "ref": "ext",
            "channel_0": {
                "port": "q0:res",
                "clock": "q0.ro",
                "mode": "real",
                "modulation": {"type": "premod", "interm_freq": 200000000.0},
                "local_oscillator": "lo1",
                "trigger": 2,
                "markers": [],
            },
        },
    ],
}
