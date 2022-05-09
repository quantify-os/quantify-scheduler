# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Distortion corrections for hardware compilation."""
import numpy as np
from typing import Any, Dict

from quantify_scheduler.backends.qblox.helpers import generate_waveform_data
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.operations.pulse_library import NumericalPulse


def correct_waveform(
    pulse_data: Dict[str, Any], sampling_rate: int, correction_cfg: Dict[str, Any]
) -> NumericalPulse:

    waveform_data = generate_waveform_data(
        data_dict=pulse_data,
        sampling_rate=sampling_rate,
    )

    # TODO: check for keys explicitly and raise KeyError?
    filter_func = import_python_object_from_string(correction_cfg["filter_func"])

    kwargs = {
        correction_cfg["input_var_name"]: waveform_data,
        **correction_cfg["kwargs"],
    }
    corrected_waveform_data = filter_func(**kwargs)

    substitute_pulse = NumericalPulse(
        samples=corrected_waveform_data,
        t_samples=np.linspace(
            start=0,
            stop=pulse_data["duration"],
            num=corrected_waveform_data.size,
        ),
        port=pulse_data["port"],
        clock=pulse_data["clock"],
        t0=pulse_data["t0"],
    )

    return substitute_pulse
