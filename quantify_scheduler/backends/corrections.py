# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pulse and acquisition corrections for hardware compilation."""
import logging
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
from quantify_scheduler import Schedule
from quantify_scheduler.backends.graph_compilation import HardwareOption
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.helpers import generate_waveform_data
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.operations.pulse_library import NumericalPulse

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class LatencyCorrections(HardwareOption):
    """
    A datastructure containing the information required to correct for latencies
    on signals specified by port-clock combinations.

    Note, if the port-clock combination of a signal is not specified in the latency
    corrections, no correction will be applied.

    Parameters
    ----------
    latencies
        A dictionary specifying the latencies to be corrected for.
        Keys are port-clocks combinations specifying the signal for which latency
        should be corrected, e.g., `port=q0:mw` and `clock=q0.01` will have the
        string ``"q0:mw-q0.01"`` as a key.
        Values are latencies of the signal to be corrected for in seconds, e.g.,
        if a signal has a latency of 120e-9, the signal will be shifted by
        -120 ns to correct for this latency.
    """

    latencies: Dict[str, float]


def determine_relative_latencies(hardware_cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Generates the latency configuration dict for all port-clock combinations that are present in
    the hardware_cfg. This is done by first setting unspecified latencies to zero, and then
    subtracting the minimum latency from all latencies.
    """

    def _extract_port_clocks(hardware_cfg: Dict[str, Any]) -> Generator:
        """
        Extracts all port-clock combinations that are present in a hardware configuration.
        Based on: https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
        """

        if hasattr(hardware_cfg, "items"):
            for k, v in hardware_cfg.items():
                if k == "port":
                    port_clock = f'{hardware_cfg["port"]}-{hardware_cfg["clock"]}'
                    yield port_clock

                elif isinstance(v, dict):
                    for port_clock in _extract_port_clocks(v):
                        yield port_clock

                elif isinstance(v, list):
                    for d in v:
                        for port_clock in _extract_port_clocks(d):
                            yield port_clock

    if (raw_latency_dict := hardware_cfg.get("latency_corrections")) is None:
        return {}

    port_clocks = _extract_port_clocks(hardware_cfg=hardware_cfg)

    latency_dict = {}
    for port_clock in port_clocks:
        # Set unspecified latencies to zero to avoid ending up with negative latencies
        # after subtracting minimum
        latency_dict[port_clock] = raw_latency_dict.get(port_clock, 0)

    # Subtract lowest value to ensure minimal latency is used and offset the latencies
    # to be relative to the minimum. Note that this supports negative delays (which is
    # useful for calibrating)
    minimum_of_latencies = min(latency_dict.values())
    for port_clock, latency_at_port_clock in latency_dict.items():
        latency_dict[port_clock] = latency_at_port_clock - minimum_of_latencies

    return latency_dict


def distortion_correct_pulse(  # pylint: disable=too-many-arguments
    pulse_data: Dict[str, Any],
    sampling_rate: int,
    filter_func_name: str,
    input_var_name: str,
    kwargs_dict: Dict[str, Any],
    clipping_values: Optional[Tuple[float]] = None,
) -> NumericalPulse:
    """
    Sample pulse and apply filter function to the sample to distortion correct it.

    Parameters
    ----------
    pulse_data
        Definition of the pulse.
    sampling_rate
        The sampling rate used to generate the time axis values.
    filter_func_name
        The filter function path of the dynamically loaded filter function.
        Example: ``"scipy.signal.lfilter"``.
    input_var_name
        The input variable name of the dynamically loaded filter function, most likely:
        ``"x"``.
    kwargs_dict
        Dictionary containing kwargs for the dynamically loaded filter function.
        Example: ``{"b": [0.0, 0.5, 1.0], "a": 1}``.
    clipping_values
        Min and max value to which the corrected pulse will be clipped, depending on
        allowed output values for the instrument.

    Returns
    -------
    :
        The sampled, distortion corrected pulse wrapped in a ``NumericalPulse``.
    """

    waveform_data = generate_waveform_data(
        data_dict=pulse_data,
        sampling_rate=sampling_rate,
    )

    filter_func = import_python_object_from_string(filter_func_name)
    kwargs = {input_var_name: waveform_data, **kwargs_dict}
    corrected_waveform_data = filter_func(**kwargs)

    if clipping_values is not None and len(clipping_values) == 2:
        corrected_waveform_data = np.clip(
            corrected_waveform_data, clipping_values[0], clipping_values[1]
        )

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
    Apply distortion corrections to operations in the schedule.
    Defined via the hardware configuration file, example:

    .. code-block::

        "distortion_corrections": {
            "q0:fl-cl0.baseband": {
                "filter_func": "scipy.signal.lfilter",
                "input_var_name": "x",
                "kwargs": {
                    "b": [0.0, 0.5, 1.0],
                    "a": [1]
                },
                "clipping_values": [-2.5, 2.5]
            }
        }

    Clipping values are the boundaries to which the corrected pulses will be clipped,
    upon exceeding, these are optional to supply.

    For pulses in need of correcting (indicated by their port-clock combination) we are
    **only** replacing the dict in ``"pulse_info"`` associated to that specific
    pulse. This means that we can have a combination of corrected (i.e., pre-sampled)
    and uncorrected pulses in the same operation.

    Note that we are **not** updating the ``"operation_repr"`` key, used to reference
    the operation from the schedulable.

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
        when elements are missing in distortion correction config for a port-clock
        combination.
    KeyError
        when clipping values are supplied but not two values exactly, min and max.
    """

    distortion_corrections_key = "distortion_corrections"
    if distortion_corrections_key not in hardware_cfg:
        logging.debug(f'No key "{distortion_corrections_key}" supplied in hardware_cfg')
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
                clipping_values = correction_cfg.get("clipping_values", None)

                if None in (filter_func_name, input_var_name, kwargs_dict):
                    raise KeyError(
                        f"One or more elements missing in distortion correction config "
                        f'for "{portclock_key}"\n\n'
                        f'"filter_func": {filter_func_name}\n'
                        f'"input_var_name": {input_var_name}\n'
                        f'"kwargs": {kwargs_dict}'
                    )

                if clipping_values and len(clipping_values) != 2:
                    raise KeyError(
                        f'Clipping values for "{portclock_key}" should contain two '
                        "values, min and max.\n"
                        f'"clipping_values": {clipping_values}'
                    )

                corrected_pulse = distortion_correct_pulse(
                    pulse_data=pulse_data,
                    sampling_rate=constants.SAMPLING_RATE,
                    filter_func_name=filter_func_name,
                    input_var_name=input_var_name,
                    kwargs_dict=kwargs_dict,
                    clipping_values=clipping_values,
                )

                schedule.operations[operation_repr].data["pulse_info"][
                    pulse_info_idx
                ] = corrected_pulse.data["pulse_info"][0]

                if pulse_info_idx == 0:
                    substitute_operation = corrected_pulse

        # Convert to operation-type of first entry in pulse_info,
        # required as first entry in pulse_info is used to generate signature in __str__
        if substitute_operation is not None:
            substitute_operation.data["pulse_info"] = schedule.operations[
                operation_repr
            ].data["pulse_info"]
            schedule.operations[operation_repr] = substitute_operation

    return schedule
