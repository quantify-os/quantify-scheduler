# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pulse corrections for hardware compilation."""
import logging
from typing import Any, Dict

import numpy as np

from quantify_scheduler import Schedule
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.helpers import generate_waveform_data
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.operations.pulse_library import NumericalPulse


logger = logging.getLogger(__name__)


def correct_pulse(
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

    corrected_pulse = NumericalPulse(
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

    return corrected_pulse


def apply_distortion_corrections(
    schedule: Schedule, hardware_cfg: Dict[str, Any]
) -> Schedule:
    """
    TODO: add general description of functionality

    For waveform components in need of correcting (indicated via their port & clock) we
    are *only* replacing the dict in "pulse_info" associated to the waveform component

    This means that we can have a combination of corrected (i.e., pre-sampled) and
    uncorrected waveform components in the same operation

    Also, we are not updating the "operation_repr" key, used to reference the operation
    from the schedulable

    Parameters
    ----------
    schedule
    hardware_cfg

    Returns
    -------

    """

    distortion_corrections_key = "distortion_corrections"
    if distortion_corrections_key not in hardware_cfg:
        logging.info(f'No "{distortion_corrections_key}" supplied')
        return schedule

    for operation_repr in schedule.operations.keys():
        substitute_operation = None

        for pulse_info_idx, pulse_data in enumerate(
            schedule.operations[operation_repr].data["pulse_info"]
        ):
            portclock_key = f"{pulse_data['port']}-{pulse_data['clock']}"

            if portclock_key in hardware_cfg[distortion_corrections_key]:
                correction_cfg = hardware_cfg[distortion_corrections_key][portclock_key]

                corrected_pulse = correct_pulse(
                    pulse_data=pulse_data,
                    sampling_rate=constants.SAMPLING_RATE,
                    correction_cfg=correction_cfg,
                )

                schedule.operations[operation_repr].data["pulse_info"][
                    pulse_info_idx
                ] = corrected_pulse.data["pulse_info"][0]

                if pulse_info_idx == 0:
                    substitute_operation = corrected_pulse

        # Convert to operation type of first entry in pulse_info,
        # required as first entry in pulse_info is used to generate signature in __str__
        if substitute_operation is not None:
            substitute_operation.data["pulse_info"] = schedule.operations[
                operation_repr
            ].data["pulse_info"]
            schedule.operations[operation_repr] = substitute_operation

    return schedule
