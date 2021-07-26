# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Module containing :class:`~quantify_core.measurement.Gettable`\s for use with
quantify-scheduler.

Schedule gettables are set up to f


.. warning:

    The gettable module is expected to change significantly as the
    acquisition protocols (#80) get fully supported by the scheduler.
    Currently different Gettables are required for different acquisition modes.
    The intent is to have one generic `ScheduleGettable`.

    Expect breaking changes.


"""

from typing import Callable, Dict, Any

import numpy as np

from qcodes import Parameter

from quantify_scheduler.compilation import qcompile
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator


class ScheduleVectorAcqGettable:
    """
    Generic gettable for a quantify schedule using vector (I,Q) acquisition. Can be
    set to return either static (demodulated) I and Q values or magnitude and phase.

    The gettable evaluates the parameters the parameters passed as
    `schedule_kwargs`, then generates the `Schedule` using the `schedule_function`,
    this is then compiled and finally executed by the `instrument_coordinator`.

    Examples
    --------

    .. code-block:

        from qcodes import ManualParameter

        from quantify_scheduler.schedules.timedomain_schedules import t1_sched

        tau = ManualParameter('tau', initial_value=1e-6, unit='s', label='Time')
        sched_kwargs = {'qubit': 'q0', 'times': tau}

        gettable = ScheduleVectorAcqGettable(t1_sched, sched_kwargs,
                                            DEVICE_CFG, MAPPING_CFG,
                                            instrument_coordinator=ic, real_imag=True,
                                            acq_instr='qrm0', repetitions=10_000
                                            )
    """

    def __init__(
        self,
        schedule_function: Callable,
        schedule_kwargs: Dict[str, Any],
        device_cfg: Dict[str, Any],
        mapping_cfg: Dict[str, Any],
        instrument_coordinator: InstrumentCoordinator,
        acq_instr: str,
        repetitions: int,
        real_imag: bool = True,
    ):
        """
        Instantiates the gettable.

        Parameters
        ----------
        schedule_function:
            A reference to the function that generates the schedule.
        schedule_kwargs:
            A dictionary containing all the arguments to pass to the schedule generation
            function.
        device_cfg:
            The device configuration dictionary.
        mapping_cfg:
            The hardware configuration dictionary.
        instrument_coordinator:
            The control stack object that will execute the schedule.
        real_imag:
            If true, the gettable returns I, Q values. Otherwise, magnitude and phase
            are returned.
        acq_instr:
            Name of the instrument that is used to perform the acquisition.
        repetitions:
            Value of the `Schedule.repetitions` property.
        """
        if real_imag:
            self.name = ["I", "Q"]
            self.label = ["Voltage I", "Voltage Q"]
            self.unit = ["V", "V"]
        else:
            self.name = ["magn", "phase"]
            self.label = ["Magnitude", "Phase"]
            self.unit = ["V", "deg"]

        self.schedule_function = schedule_function
        self.schedule_kwargs = schedule_kwargs

        self.device_cfg = device_cfg
        self.mapping_cfg = mapping_cfg
        self.instrument_coordinator = instrument_coordinator

        self.acq_instr = acq_instr
        self._real_imag = real_imag
        self.repetitions = repetitions

    def get(self):
        """
        Executes the schedule on hardware using the control stack and returns the data
        obtained.

        Returns
        -------
        :
            The data returned by the control stack.
        """
        evaluated_sched_kwargs = _evaluate_parameter_dict(self.schedule_kwargs)
        # generate a schedule using the evaluated keyword arguments dict
        sched = self.schedule_function(**evaluated_sched_kwargs)

        sched.repetitions = self.repetitions
        config = qcompile(
            schedule=sched,
            device_cfg=self.device_cfg,
            hardware_mapping=self.mapping_cfg,
        )

        # Upload the schedule and configure the control stack
        self.instrument_coordinator.prepare(config)
        # Run experiment and retrieve data
        self.instrument_coordinator.start()
        i_val, q_val = self.instrument_coordinator.retrieve_acquisition()[
            self.acq_instr
        ]

        # complex conjugate of data is taken because of issue #103
        S21 = np.conj(i_val + 1j * q_val)
        if self._real_imag:
            return S21.real, S21.imag
        return np.abs(S21), np.angle(S21, deg=True)


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
