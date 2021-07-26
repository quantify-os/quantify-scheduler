# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
r"""
Module containing :class:`~quantify_core.measurement.Gettable`\s for use with
quantify-scheduler.

Schedule gettables are set up to f


.. warning::

    The gettable module is expected to change significantly as the
    acquisition protocols (#36 and #80) get fully supported by the scheduler.
    Currently different Gettables are required for different acquisition modes.
    The intent is to have one generic `ScheduleGettable`.
    Expect breaking changes.


"""

from typing import Any, Callable, Dict, Tuple

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
    executed by the `instrument_coordinator`.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=line-too-long
    def __init__(
        self,
        device: Instrument,
        schedule_function: Callable[..., types.Schedule],
        schedule_kwargs: Dict[str, Any],
        device_cfg: Dict[str, Any],
        hardware_cfg: Dict[str, Any],
        ins_coord: InstrumentCoordinator,
        acq_instr: str,
        real_imag: bool = True,
        hardware_averages: int = 1024,
    ):
        """
        Create a new instance of ScheduleVectorAcqGettable which is used to do I and Q
        acquisition.

        Parameters
        ----------
        device :
            The qcodes instrument.
        schedule_function :
            A function which returns a :class:`~quantify_scheduler.types.Schedule`.
        schedule_kwargs :
            The schedule function keyword arguments, when a value in this dictionary is
            a :class:`~qcodes.instrument.parameter.Parameter`, this parameter will be
            evaluated every time .get is called before being passed to the
            schedule_function.
        device_cfg :
            The device configuration dictionary.
        hardware_cfg :
            The hardware configuration dictionary.
        coordinator :
            An instance of
            :class:`~quantify_scheduler.instrument_coordinator.InstrumentCoordinator`.
        real_imag :
            If true, the gettable returns I, Q values. Otherwise, magnitude and phase
            are returned.
        acq_instr : str
            Name of the instrument that is used to perform the acquisition.
        hardware_averages :
            The number of hardware averages, by default 1024
        """
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
        self.ins_coord = ins_coord

        self.hardware_averages = hardware_averages
        self.acq_instr = acq_instr
        self.real_imag = real_imag
        self.device = device

        self._evaluated_sched_kwargs = {}
        self._config = {}

    def get(self) -> Tuple[float]:
        """
        Start the experimental sequence and retrieve acquisition data

        Returns
        -----------
        :
            The acquired I/Q volatage signal as a complex number,
            split into a tuple of floats: either real/imaginary parts or
            magnitude/phase, depending on whether :code:`real_imag` is :code:`True`
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

        self.ins_coord.acq_instr = self.acq_instr
        self.ins_coord.schedule_kwargs = self.schedule_kwargs
        self.ins_coord.device = self.device

        # Upload the schedule and configure the control stack
        self.ins_coord.prepare(self._config)

        # Run experiment and retrieve data
        self.ins_coord.start()

        # TODO ins_coord components need to be awaited

        # TODO Why index on 'acq_instr'. There can be multiple acquisition instruments.
        # This function should rather return the result of 'retrieve_acquisition' than
        # doing extra additions to the data.
        # This will not work because it needs to also know the acq_index
        # { 'uhfqa0': { [acq_index]: [0,1,1,...] } }
        i_val, q_val = self.ins_coord.retrieve_acquisition()[self.acq_instr]

        s21: np.ndarray = i_val + 1j * q_val
        if self.real_imag:
            return (np.real(s21), np.imag(s21))

        return np.abs(s21), np.angle(s21, deg=True)


def _evaluate_parameter_dict(parameters: Dict[str, Any]):
    """
    Loop over the keys and values in a dict and replaces parameters with their current
    value.

    Parameters
    ----------
    parameters:
        A dictionary containing a mix of Parameters and normal values.

    Returns
    -------
    evaluated_parameters:
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
