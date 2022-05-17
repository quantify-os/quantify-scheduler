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


def _correct_pulse(
    pulse_data: Dict[str, Any],
    sampling_rate: int,
    filter_func_name: str,
    input_var_name: str,
    kwargs_dict: Dict[str, Any],
) -> NumericalPulse:
    """
    TODO: docstring
    """

    waveform_data = generate_waveform_data(
        data_dict=pulse_data,
        sampling_rate=sampling_rate,
    )

    filter_func = import_python_object_from_string(filter_func_name)
    kwargs = {input_var_name: waveform_data, **kwargs_dict}
    corrected_waveform_data = filter_func(**kwargs)

    if corrected_waveform_data.size == 1:  # Interpolation requires two sample points
        corrected_waveform_data = np.append(
            corrected_waveform_data, corrected_waveform_data[-1]
        )

    corrected_pulse = NumericalPulse(
        samples=corrected_waveform_data,
        t_samples=np.linspace(
            start=0, stop=pulse_data["duration"], num=corrected_waveform_data.size
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
    Apply distortion corrections as defined in the hardware configuration file

    .. code-block::
        "distortion_corrections": {
            "q0:fl-cl0.baseband": {
                "filter_func": "scipy.signal.lfilter",
                "input_var_name": "x",
                "kwargs": {
                    "b": [0.0, 0.5, 1.0],
                    "a": 1
                }
            }
        }

    For waveforms in need of correcting (indicated by their port-clock combination) we
    are *only* replacing the dict in "pulse_info" associated to that specific waveform

    This means that we can have a combination of corrected (i.e., pre-sampled) and
    uncorrected waveforms in the same operation

    Also, we are not updating the "operation_repr" key, used to reference the operation
    from the schedulable

    Parameters
    ----------
    schedule
        The schedule that contains operations that are to be distortion corrected.
    hardware_cfg
        The hardware configuration of the setup.

    Returns
    -------
    :
        The schedule with distortion corrected operations.

    Raises
    ------
    KeyError
        This exception is raised when elements are missing in distortion correction
        config for a port-clock combination.
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

                filter_func_name = correction_cfg.get("filter_func", None)
                input_var_name = correction_cfg.get("input_var_name", None)
                kwargs_dict = correction_cfg.get("kwargs", None)

                if None in (filter_func_name, input_var_name, kwargs_dict):
                    raise KeyError(
                        f"One or more elements missing in distortion correction config "
                        f'for "{portclock_key}"\n\n'
                        f'"filter_func: {filter_func_name}"\n'
                        f'"input_var_name: {input_var_name}"\n'
                        f'"kwargs: {kwargs_dict}"'
                    )

                corrected_pulse = _correct_pulse(
                    pulse_data=pulse_data,
                    sampling_rate=constants.SAMPLING_RATE,
                    filter_func_name=filter_func_name,
                    input_var_name=input_var_name,
                    kwargs_dict=kwargs_dict,
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
