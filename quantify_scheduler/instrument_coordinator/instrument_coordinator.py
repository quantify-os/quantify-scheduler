# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the main InstrumentCoordinator Component."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from qcodes.instrument import base as qcodes_base
from qcodes.instrument import parameter
from qcodes.utils import validators

from quantify_scheduler import CompiledSchedule
from quantify_scheduler.instrument_coordinator.components import base


class InstrumentCoordinator(qcodes_base.Instrument):
    """
    The :class:`~.InstrumentCoordinator` serves as the central interface of the hardware
    abstraction layer. It provides a standardized interface to execute Schedules on
    control hardware.

    The :class:`~.InstrumentCoordinator` has two main functionalities exposed to the
    user, the ability to configure its
    :mod:`~.instrument_coordinator.components`
    representing physical instruments, and the ability to execute experiments.


    .. admonition:: Executing a schedule using the instrument coordinator
        :class: dropdown

        To execute a :class:`~.Schedule` , one needs to first
        compile a schedule and then configure all the instrument coordinator components
        using :meth:`~.InstrumentCoordinator.prepare`.
        After starting the experiment, the results can be retrieved using
        :meth:`~.InstrumentCoordinator.retrieve_acquisition`.

        .. code-block::

            from quantify_scheduler.compilation import qcompile

            my_sched: Schedule = ...  # a Schedule describing the experiment to perform
            device_config: dict = ...  # a config file describing the quantum device
            hardware_config: dict = ...  # a config file describing the connection to the hardware
            compiled_sched: CompiledSchedule = qcompile(my_sched, device_config, hardware_config)

            instrument_coordinator.prepare(compiled_sched)
            instrument_coordinator.start()
            dataset = instrument_coordinator.retrieve_acquisition()

    .. admonition:: Adding components to the instrument coordinator
        :class: dropdown

        In order to distribute compiled instructions and execute an experiment,
        the instrument coordinator needs to have references to the individual
        instrument coordinator components. They can be added using
        :meth:`~.InstrumentCoordinator.add_component`.

        .. code-block::

            instrument_coordinator.add_component(qcm_component)

    """  # pylint: disable=line-too-long

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.add_parameter(
            "components",
            initial_value=[],
            parameter_class=parameter.ManualParameter,
            vals=validators.Lists(validators.Strings()),
            docstring="A list containing the names of all components that"
            " are part of this InstrumentCoordinator.",
        )

        self.add_parameter(
            "timeout",
            unit="s",
            initial_value=60,
            vals=validators.Numbers(min_value=0),
            parameter_class=parameter.ManualParameter,
            docstring="The timeout used for waiting for the experiment to complete "
            "when retrieving acquisitions.",
        )
        self._last_schedule = None

    @property
    def last_schedule(self) -> CompiledSchedule:
        """
        Returns the last schedule used to prepare the instrument coordinator.

        This feature is intended to aid users in debugging.
        """
        if self._last_schedule is None:
            raise ValueError(
                f"No {CompiledSchedule.__name__} was handled by the instrument "
                "coordinator. Try calling the .prepare() method with a Schedule."
            )
        return self._last_schedule

    @property
    def is_running(self) -> bool:
        """
        Returns if any of the :class:`.InstrumentCoordinator` components is running.

        Returns
        -------
        :
            The :class:`.InstrumentCoordinator`'s running state.
        """
        return any(
            self.find_instrument(c_name).is_running is True
            for c_name in self.components()
        )

    def get_component(self, name: str) -> base.InstrumentCoordinatorComponentBase:
        """
        Returns the InstrumentCoordinator component by name.

        Parameters
        ----------
        name
            The component name.

        Returns
        -------
        :
            The component.

        Raises
        ------
        KeyError
            If key ``name`` is not present in ``self.components``.
        """
        if name in self.components():
            return self.find_instrument(name)
        raise KeyError(f"'{name}' is not a component of {self.name}!")

    def add_component(
        self,
        component: base.InstrumentCoordinatorComponentBase,
    ) -> None:
        """
        Adds a component to the components collection.

        Parameters
        ----------
        component
            The component to add.

        Raises
        ------
        ValueError
            If a component with a duplicated name is added to the collection.
        TypeError
            If :code:`component` is not an instance of the base component.
        """
        if component.name in self.components():
            raise ValueError(f"'{component.name}' has already been added!")

        if not isinstance(component, base.InstrumentCoordinatorComponentBase):
            raise TypeError(
                f"{repr(component)} is not "
                f"{base.__name__}.{base.InstrumentCoordinatorComponentBase.__name__}."
            )

        self.components().append(component.name)  # list gets updated in place

    def remove_component(self, name: str) -> None:
        """
        Removes a component by name.

        Parameters
        ----------
        name
            The component name.
        """

        self.components().remove(name)  # list gets updated in place

    def prepare(
        self,
        compiled_schedule: CompiledSchedule,
    ) -> None:
        """
        Prepares each component for execution of a schedule.

        It attempts to configure all instrument coordinator components for which
        compiled instructions, typically consisting of a combination of sequence
        programs, waveforms and other instrument settings, are available in the
        compiled schedule.


        Parameters
        ----------
        compiled_schedule
            A schedule containing the information required to execute the program.

        Raises
        ------
        KeyError
            If the compiled schedule contains instructions for a component
            absent in the instrument coordinator.
        TypeError
            If the schedule provided is not a valid :class:`.CompiledSchedule`.
        """
        if not CompiledSchedule.is_valid(compiled_schedule):
            raise TypeError(
                f"{compiled_schedule} is not a valid {CompiledSchedule.__name__}"
            )

        # Adds a reference to the last prepared schedule this can be accessed through
        # the self.last_schedule property.
        self._last_schedule = compiled_schedule

        compiled_instructions = compiled_schedule["compiled_instructions"]
        # Compiled instructions are expected to follow the structure of a dict
        # with keys corresponding to instrument names (InstrumentCoordinatorComponent's)
        # and values containing instructions in the format specific to that type
        # of hardware. See also the specification in the CompiledSchedule class.
        for instrument_name, args in compiled_instructions.items():
            self.get_component(instrument_name).prepare(args)

    def start(self) -> None:
        """
        Start all of the components.

        The components are started in the order in which they were added.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            instrument.start()

    def stop(self) -> None:
        """
        Stops all components.

        The components are stopped in the order in which they were added.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            instrument.stop()

    def retrieve_acquisition(self) -> Dict[Tuple[int, int], Any]:
        """
        Retrieves the latest acquisition results of the components
        with acquisition capabilities.

        Returns
        -------
        :
            The acquisition data per component.
        """
        # pylint: disable=fixme
        # FIXME: update the description of the return type of the instrument coordinator

        self.wait_done(timeout_sec=self.timeout())

        # Temporary. Will probably be replaced by an xarray object
        # See quantify-core#187, quantify-core#233, quantify-scheduler#36
        acquisitions: Dict[Tuple[int, int], Any] = {}
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            acqs = instrument.retrieve_acquisition()
            if acqs is not None:
                acquisitions.update(acqs)
        return acquisitions

    def wait_done(self, timeout_sec: int = 10) -> None:
        """
        Awaits each component until it has stopped running or
        until it has exceeded the amount of time to run.

        The timeout in seconds specifies the allowed amount of time to run before
        it times out.

        Parameters
        ----------
        timeout_sec
            The maximum amount of time in seconds before a timeout.
        """
        for instr_name in self.components():
            instrument = self.find_instrument(instr_name)
            self.get_component(instrument.name).wait_done(timeout_sec)


def _convert_acquisition_data_format(raw_results):
    acquisition_dict = {}
    for (channel, i) in raw_results.keys():
        if channel not in acquisition_dict.keys():
            acquisition_dict[channel] = []
        acquisition_dict[channel].append(raw_results.get((channel, i)))
    acquisitions_list = []
    for channel in acquisition_dict:
        acquisitions_list.append(np.array(acquisition_dict.get(channel)))
    return acquisitions_list


class ZIInstrumentCoordinator(InstrumentCoordinator):
    """
    This class is a hack and extension to the :class:`.InstrumentCoordinator`, which is
    introduced to support the quirks when using the ZI backend
    during the acquisition of results.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.add_parameter(
            "timeout_reacquire",
            unit="",
            initial_value=True,
            vals=validators.Bool(),
            parameter_class=parameter.ManualParameter,
            docstring="Turns on reacquisition in case " "of timeouts.",
        )

        self.add_parameter(
            "max_num_reacquisitions",
            unit="",
            initial_value=5,
            vals=validators.Numbers(min_value=0, max_value=50),
            parameter_class=parameter.ManualParameter,
            docstring="The number of retries to retrieve acquisitions in case "
            "of timeouts.",
        )

        self._last_acquisition = None
        self._num_reacquisitions = 0

    def _compare_reacquire(self, raw_results):
        reacquire = True
        results_list = _convert_acquisition_data_format(raw_results)

        if self._last_acquisition is not None:
            last_acquisition_list = _convert_acquisition_data_format(
                self._last_acquisition
            )
            difference_np_array = np.linalg.norm(
                np.array(results_list[0]) - np.array(last_acquisition_list[0])
            )
            if difference_np_array > 0.0:
                reacquire = False
                self._num_reacquisitions = 0
        return reacquire

    def retrieve_acquisition(self):
        """
        Retrieves the latest acquisition results of the components
        with acquisition capabilities.

        Returns
        -------
        :
            The acquisition data per component.
        """
        # pylint: disable=fixme
        # FIXME: update the description of the return type of the instrument coordinator

        raw_acq_results = super().retrieve_acquisition()
        if self.timeout_reacquire():
            reacquire = self._compare_reacquire(raw_acq_results)
            self._last_acquisition = raw_acq_results
            self._num_reacquisitions = 0

            while reacquire:
                self._num_reacquisitions += 1
                print(
                    "Re-running acquisition "
                    + f"{self._num_reacquisitions}/{self.max_num_reacquisitions()}"
                )
                if self._num_reacquisitions >= self.max_num_reacquisitions():
                    raise RuntimeError(
                        "Exceeded number of reacquisitions:"
                        + f"{self._num_reacquisitions}/"
                        + f"{self.max_num_reacquisitions()}"
                    )
                self.start()
                self.wait_done()
                # Acquire results
                raw_acq_results = super().retrieve_acquisition()
                reacquire = self._compare_reacquire(raw_acq_results)
                self._last_acquisition = raw_acq_results

        return raw_acq_results
