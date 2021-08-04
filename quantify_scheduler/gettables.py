# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
r"""
Module containing :class:`~quantify_core.measurement.Gettable`\s for use with
quantify-scheduler.

.. warning::

    The gettable module is expected to change significantly as the
    acquisition protocols (#36 and #80) get fully supported by the scheduler.
    Currently different Gettables are required for different acquisition modes.
    The intent is to have one generic `ScheduleGettable`.
    Expect breaking changes.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, List, Optional, Union

import numpy as np
from qcodes import Parameter
from qcodes.instrument.base import Instrument

from quantify_scheduler import types
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class ScheduleVectorAcqGettable:
    """
    Generic gettable for a quantify schedule using vector (I,Q) acquisition. Can be
    set to return either static (demodulated) I and Q values or magnitude and phase.

    The gettable evaluates the parameters passed as `schedule_kwargs`, then generates
    the `Schedule` using the `schedule_function`, this is then compiled and finally
    executed by the
    :class:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator`.
    """  # pylint: disable=line-too-long

    # pylint: disable=too-many-arguments
    # pylint: disable=line-too-long
    def __init__(
        self,
        device: Instrument,
        schedule_function: Callable[..., types.Schedule],
        schedule_kwargs: Dict[str, Any],
        device_cfg: Dict[str, Any],
        hardware_cfg: Dict[str, Any],
        instr_coord: InstrumentCoordinator,
        channels_and_indices: Optional[List[Tuple[int, int]]] = None,
        real_imag: bool = True,
        hardware_averages: int = 1024,
    ):
        """
        Create a new instance of ScheduleVectorAcqGettable which is used to do I and Q
        acquisition.

        Parameters
        ----------
        device
            The qcodes instrument.
        schedule_function
            A function which returns a :class:`~quantify_scheduler.types.Schedule`.
        schedule_kwargs
            The schedule function keyword arguments, when a value in this dictionary is
            a :class:`~qcodes.instrument.parameter.Parameter`, this parameter will be
            evaluated every time :code:`.get()` is called before being passed to the
            :code:`schedule_function`.
        device_cfg
            The device configuration dictionary.
        hardware_cfg
            The hardware configuration dictionary.
        instr_coord
            An instance of
            :class:`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator`.
        channels_and_indices
            List containing all the acquisition channels and indices to retrieve, the
            channels and indices are provided as tuples. If None, (0,0) is used.
        real_imag
            If true, the gettable returns I, Q values. Otherwise, magnitude and phase
            (degrees) are returned.
        hardware_averages
            The number of hardware averages.
        """  # pylint: disable=line-too-long
        if real_imag:
            self.name = ["I", "Q"]
            self.label = ["Voltage I", "Voltage Q"]
            self.unit = ["V", "V"]
        else:
            self.name = ["magn", "phase"]
            self.label = ["Magnitude", "Phase"]
            self.unit = ["V", "deg"]

        self.batched = False

        self.schedule_function = schedule_function
        self.schedule_kwargs = schedule_kwargs

        self.device_cfg = device_cfg
        self.mapping_cfg = hardware_cfg
        self.instr_coord = instr_coord
        self.channels_and_indices = channels_and_indices

        self.hardware_averages = hardware_averages
        self.real_imag = real_imag
        self.device = device

        self._evaluated_sched_kwargs = {}
        self._config = {}

    def get(self) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """
        Start the experimental sequence and retrieve acquisition data.

        Returns
        -------
        :
            The acquired I/Q voltage signal as a complex number,
            split into a tuple of floats: either real/imaginary parts or
            magnitude/phase, depending on whether :code:`real_imag` is :code:`True`.
        """
        self._evaluated_sched_kwargs = _evaluate_parameter_dict(self.schedule_kwargs)

        # generate a schedule using the evaluated keyword arguments dict
        sched = self.schedule_function(**self._evaluated_sched_kwargs)
        # compile and assign to attributes for debugging purposes
        sched.repetitions = self.hardware_averages
        self._config = qcompile(
            schedule=sched,
            device_cfg=self.device_cfg,
            hardware_mapping=self.mapping_cfg,
        )

        self.instr_coord.schedule_kwargs = self.schedule_kwargs
        self.instr_coord.device = self.device

        # Upload the schedule and configure the instrument coordinator
        self.instr_coord.prepare(self._config)

        # Run experiment
        self.instr_coord.start()

        # TODO instr_coord components need to be awaited # pylint: disable=fixme

        if self.channels_and_indices is None:
            acq_channel_and_index = (0, 0)
            i_val, q_val = self.instr_coord.retrieve_acquisition()[
                acq_channel_and_index
            ]
            if self.real_imag:
                return i_val, q_val

            return _iq_to_mag_phase(i_val, q_val)

        # implicit else:
        formatted_acq = list()
        acquisition = self.instr_coord.retrieve_acquisition()
        for acq_channel, acq_index in self.channels_and_indices:
            this_acq = acquisition[(acq_channel, acq_index)]
            if not self.real_imag:
                this_acq = _iq_to_mag_phase(*this_acq)
            formatted_acq.append(this_acq)
        return formatted_acq


def _iq_to_mag_phase(i_val: float, q_val: float):
    s21: complex = i_val + 1j * q_val
    return np.abs(s21), np.angle(s21, deg=True)


def _evaluate_parameter_dict(parameters: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Loop over the keys and values in a dict and replaces parameters with their current
    value.

    Parameters
    ----------
    parameters
        A dictionary containing a mix of
        :class:`~qcodes.instrument.parameter.Parameter`\s and normal values.

    Returns
    -------
    :
        The `parameters` dictionary, but with the parameters replaced by their current
        value.
    """
    evaluated_parameters = dict()

    for key, val in parameters.items():
        if isinstance(val, Parameter):
            # evaluate the parameter
            evaluated_parameters[key] = val.get()
        else:
            evaluated_parameters[key] = val

    return evaluated_parameters
