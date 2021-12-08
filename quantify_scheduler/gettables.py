# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
r"""
Module containing :class:`~quantify_core.measurement.types.Gettable`\s for use with
quantify-scheduler.

.. warning::

    The gettable module is expected to change significantly as the
    acquisition protocols (#36 and #80) get fully supported by the scheduler.
    Currently different Gettables are required for different acquisition modes.
    The intent is to have one generic ``ScheduleGettable``.
    Expect breaking changes.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
from qcodes import Parameter

from quantify_scheduler import Schedule
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_schedule,
)


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class ScheduleGettableSingleChannel:
    """
    Generic gettable for a quantify schedule using vector (I,Q) acquisition. Can be
    set to return either static (demodulated) I and Q values or magnitude and phase.

    The gettable evaluates the parameters passed as `schedule_kwargs`, then generates
    the :class:`quantify_scheduler.schedules.schedule.Schedule` using the
    `schedule_function`, this is then compiled and finally executed by the
    :class:`~.InstrumentCoordinator`.
    """  # pylint: disable=line-too-long

    # pylint: disable=too-many-arguments
    # pylint: disable=line-too-long
    def __init__(
        self,
        quantum_device: QuantumDevice,
        schedule_function: Callable[..., Schedule],
        schedule_kwargs: Dict[str, Any],
        real_imag: bool = True,
        batched: bool = False,
        max_batch_size: int = 1024,
        always_initialize=True,
    ):
        """
        Create a new instance of ScheduleGettableSingleChannel which is used to do I and Q
        acquisition.

        Parameters
        ----------
        quantum_device
            The qcodes instrument representing the quantum device under test (DUT)
            containing quantum device properties and setup configuration information.
        schedule_function
            A function which returns a
            :class:`quantify_scheduler.schedules.schedule.Schedule`. The
            function is required to have the `repetitions` keyword argument.
        schedule_kwargs
            The schedule function keyword arguments, when a value in this dictionary is
            a :class:`~qcodes.instrument.parameter.Parameter`, this parameter will be
            evaluated every time :code:`.get()` is called before being passed to the
            :code:`schedule_function`.
        real_imag
            If true, the gettable returns I, Q values. Otherwise, magnitude and phase
            (degrees) are returned.
        batched
            Used to indicate if the experiment is performed in batches or in an
            iterative fashion.
        max_batch_size:
            Determines the maximum number of points to acquire when acquiring in batched
            mode. Can be used to split up a program in parts if required due to hardware
            constraints.
        always_initialize:
            If True, then reinitialize the schedule on each invocation of `get`. If
            False, then only initialize the first invocation of `get`.
        """

        self.always_initialize = always_initialize
        self.is_initialized = False
        self._compiled_schedule = None

        self.real_imag = real_imag
        if self.real_imag:
            self.name = ["I", "Q"]
            self.label = ["Voltage I", "Voltage Q"]
            self.unit = ["V", "V"]
        else:
            self.name = ["magn", "phase"]
            self.label = ["Magnitude", "Phase"]
            self.unit = ["V", "deg"]

        self.batched = batched
        self.batch_size = max_batch_size

        # schedule arguments
        self.schedule_function = schedule_function
        self.schedule_kwargs = schedule_kwargs
        self._evaluated_sched_kwargs = {}

        # the quantum device object containing setup configuration information
        self.quantum_device = quantum_device

    def __call__(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """Acquire and return data"""
        return self.get()

    def initialize(self):
        """
        This generates the schedule and uploads the compiled instructions to the
        hardware using the instrument coordinator.
        """
        self._evaluated_sched_kwargs = _evaluate_parameter_dict(self.schedule_kwargs)

        # generate a schedule using the evaluated keyword arguments dict
        sched = self.schedule_function(
            **self._evaluated_sched_kwargs,
            repetitions=self.quantum_device.cfg_sched_repetitions(),
        )

        self._compiled_schedule = qcompile(
            schedule=sched,
            device_cfg=self.quantum_device.generate_device_config(),
            hardware_cfg=self.quantum_device.generate_hardware_config(),
        )

        instr_coordinator = self.quantum_device.instr_instrument_coordinator.get_instr()
        instr_coordinator.prepare(self._compiled_schedule)

        self.is_initialized = True

    @property
    def compiled_schedule(self) -> Schedule:
        """Return the schedule used in this class"""
        return self._compiled_schedule

    def get(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Start the experimental sequence and retrieve acquisition data.

        Returns
        -------
        :
            The acquired I/Q voltage signal as a complex number,
            split into a tuple of floats: either real/imaginary parts or
            magnitude/phase, depending on whether :code:`real_imag` is :code:`True`.
        """
        if self.always_initialize:
            self.initialize()
        else:
            if not self.is_initialized:
                self.initialize()

        instr_coordinator = self.quantum_device.instr_instrument_coordinator.get_instr()

        instr_coordinator.start()
        acquired_data = instr_coordinator.retrieve_acquisition()
        instr_coordinator.stop()

        result = self.process_acquired_data(acquired_data)
        return result

    def process_acquired_data(
        self, acquired_data
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Reshapes the data as returned from the instrument coordinator into the form
        accepted by the measurement control.
        """
        # pylint: disable=fixme
        # FIXME: this reshaping should happen inside the instrument coordinator
        # blocked by quantify-core#187, and quantify-core#233

        # retrieve the acquisition results
        # FIXME: acq_metadata should be an attribute of the schedule, see also #192
        acq_metadata = extract_acquisition_metadata_from_schedule(
            self.compiled_schedule
        )

        if (
            acq_metadata.bin_mode == BinMode.AVERAGE
            and acq_metadata.acq_protocol == "trace"
        ):
            dataset = {}
            for acq_channel, acq_indices in acq_metadata.acq_indices.items():
                dataset[acq_channel] = np.zeros(len(acq_indices), dtype=complex)
                for acq_idx in acq_indices:
                    val = acquired_data[(acq_channel, acq_idx)]
                    dataset[acq_channel] = val[0] + 1j * val[1]

        elif acq_metadata.bin_mode == BinMode.AVERAGE:

            dataset = {}
            for acq_channel, acq_indices in acq_metadata.acq_indices.items():
                dataset[acq_channel] = np.zeros(len(acq_indices), dtype=complex)
                for acq_idx in acq_indices:
                    val = acquired_data[(acq_channel, acq_idx)]
                    dataset[acq_channel][acq_idx] = val[0] + 1j * val[1]

            # This gettable only supports one acquisition channel
            if len(dataset.keys()) != 1:
                raise ValueError(
                    "Expected a single channel in the retrieved acquisitions "
                    f"dataset.keys={dataset.keys()}."
                )

        elif acq_metadata.bin_mode == BinMode.APPEND:
            dataset = {}
            for acq_channel, acq_indices in acq_metadata.acq_indices.items():
                dataset[acq_channel] = np.zeros(
                    len(acq_indices) * self.compiled_schedule.repetitions, dtype=complex
                )
                acq_stride = len(acq_indices)
                for acq_idx in acq_indices:
                    vals = acquired_data[(acq_channel, acq_idx)]
                    dataset[acq_channel][acq_idx::acq_stride] = vals[0] + 1j * vals[1]

        else:
            raise NotImplementedError(
                f"Acquisition protocol {acq_metadata.acq_protocol} with bin"
                f" mode {acq_metadata.bin_mode} is not supported."
            )

        # Reshaping of the data before returning

        # N.B. this only works if there is a single channel i.e., len(dataset.keys())==1
        for vals in dataset.values():
            if self.batched is False:
                # for iterative mode, we expect only a single value.
                assert (len(vals)) == 1

            # N.B. if there would be multiple channels the return would be outside
            # of this for loop
            if self.real_imag:
                return vals.real, vals.imag
            # implicit else
            return np.abs(vals), np.angle(vals, deg=True)


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

    Raises
    ------
    TypeError
        If a parameter returns None

    """
    evaluated_parameters = dict()

    for key, val in parameters.items():
        if isinstance(val, Parameter):
            # evaluate the parameter
            evaluated_parameters[key] = val.get()
            # verify that the parameter has a value, a missing value typically indicates
            # that it was not initialized.
            if evaluated_parameters[key] is None:
                raise TypeError(
                    f"{key}: parameter {val} returns None. "
                    "It is possible this parameter was not configured correctly."
                )

        else:
            evaluated_parameters[key] = val

    return evaluated_parameters
