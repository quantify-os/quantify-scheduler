# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing the core concepts of the scheduler."""
from __future__ import annotations

import json
from abc import ABC
from collections import UserDict
from copy import deepcopy
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Literal

from quantify_scheduler import enums, json_utils, resources
from quantify_scheduler.json_utils import JSONSchemaValMixin
from quantify_scheduler.operations.operation import Operation

if TYPE_CHECKING:
    from quantify_scheduler.resources import Resource
    import plotly.graph_objects as go


# pylint: disable=too-many-ancestors
class ScheduleBase(JSONSchemaValMixin, UserDict, ABC):
    """
    The :class:`~.ScheduleBase` is a data structure that is at
    the core of the Quantify-scheduler and describes when what operations are applied
    where.

    The :class:`~.ScheduleBase` is a collection of
    :class:`quantify_scheduler.operations.operation.Operation` objects and timing
    constraints that define relations between the operations.

    The schedule data structure is based on a dictionary.
    This dictionary contains:

    - operation_dict - a hash table containing the unique
        :class:`quantify_scheduler.operations.operation.Operation` s added to the
        schedule.
    - schedulables - a dictionary of all timing constraints added
        between operations.

    The :class:`~.Schedule` provides an API to create schedules.
    The :class:`~.CompiledSchedule` represents a schedule after
    it has been compiled for execution on a backend.


    The :class:`~.Schedule` contains information on the
    :attr:`~.ScheduleBase.operations` and
    :attr:`~.ScheduleBase.schedulables`.
    The :attr:`~.ScheduleBase.operations` is a dictionary of all
    unique operations used in the schedule and contain the information on *what*
    operation to apply *where*.
    The :attr:`~.ScheduleBase.schedulables` is a dictionary of
    Schedulables describing timing constraints between operations, i.e. when to apply
    an operation.


    **JSON schema of a valid Schedule**

    .. jsonschema:: ../schemas/schedule.json

    """

    @property
    def name(self) -> str:
        """Returns the name of the schedule."""
        return self.data["name"]

    @property
    def repetitions(self) -> int:
        """
        Returns the amount of times this Schedule will be repeated.

        Returns
        -------
        :
            The repetitions count.
        """
        return self.data["repetitions"]

    @repetitions.setter
    def repetitions(self, value: int):
        assert value > 0
        self.data["repetitions"] = int(value)

    @property
    def operations(self) -> Dict[str, Operation]:
        """
        A dictionary of all unique operations used in the schedule.
        This specifies information on *what* operation to apply *where*.

        The keys correspond to the :attr:`~.Operation.hash` and values are instances
        of :class:`quantify_scheduler.operations.operation.Operation`.
        """
        return self.data["operation_dict"]

    @property
    def schedulables(self) -> Dict[str, Any]:
        """
        A list of schedulables describing the timing of operations.

        A schedulable uses timing constraints to constrain the operation in time by
        specifying the time (:code:`"rel_time"`) between a reference operation and the
        added operation. The time can be specified with respect to a reference point
        (:code:`"ref_pt"') on the reference operation (:code:`"ref_op"`) and a reference
        point on the next added operation (:code:`"ref_pt_new"').
        A reference point can be either the "start", "center", or "end" of an
        operation. The reference operation (:code:`"ref_op"`) is specified using its
        label property.

        Each item in the list represents a timing constraint and is a dictionary with
        the following keys:

        .. code-block::

            ['label', 'rel_time', 'ref_op', 'ref_pt_new', 'ref_pt', 'operation_repr']

        The label is used as a unique identifier that can be used as a reference for
        other operations, the operation_repr refers to the string representation of a
        operation in :attr:`~.ScheduleBase.operations`.

        .. note::

            timing constraints are not intended to be modified directly.
            Instead use the :meth:`~.Schedule.add`

        """
        return self.data["schedulables"]

    @property
    def resources(self) -> Dict[str, Resource]:
        """
        A dictionary containing resources. Keys are names (str),
        values are instances of :class:`~quantify_scheduler.resources.Resource`.
        """
        return self.data["resource_dict"]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} "{self.data["name"]}" containing '
            f'({len(self.data["operation_dict"])}) '
            f'{len(self.data["schedulables"])}  (unique) operations.'
        )

    def to_json(self) -> str:
        """
        Converts the Schedule data structure to a JSON string.

        Returns
        -------
        :
            The json string result.
        """
        return json.dumps(self.data, cls=json_utils.ScheduleJSONEncoder)

    @classmethod
    def from_json(cls, data: str) -> Schedule:
        """
        Converts the JSON data to a Schedule.

        Parameters
        ----------
        data
            The JSON data.

        Returns
        -------
        :
            The Schedule object.
        """
        schedule_data = json_utils.ScheduleJSONDecoder().decode(data)
        name = schedule_data["name"]
        sched = Schedule.__new__(Schedule)
        sched.__setstate__(schedule_data)

        return sched

    def plot_circuit_diagram_mpl(
        self, figsize: Tuple[int, int] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Union[Axes, List[Axes]]]:
        """
        Creates a circuit diagram visualization of the schedule using matplotlib.

        The circuit diagram visualization visualizes the schedule at the quantum circuit
        layer.
        This visualization provides no timing information, only showing the order of
        operations.
        Because quantify-scheduler uses a hybrid gate-pulse paradigm, operations for
        which no information is specified at the gate level are visualized using an
        icon (e.g., a stylized wavy pulse) depending on the information specified at
        the quantum device layer.

        Alias of :func:`.circuit_diagram.circuit_diagram_matplotlib`.

        Parameters
        ----------
        schedule
            the schedule to render.
        figsize
            matplotlib figsize.
        ax
            Axis handle to use for plotting.

        Returns
        -------
        fig
            matplotlib figure object.
        ax
            matplotlib axis object.
        """
        warnings.warn(
            "`plot_circuit_diagram_mpl` will be removed from this module in "
            "quantify-scheduler >= 0.6.0.\n"
            "Instead, use `plot_circuit_diagram`",
            DeprecationWarning,
        )
        return self.plot_circuit_diagram(figsize, ax, "mpl")

    def plot_circuit_diagram(
        self,
        figsize: Tuple[int, int] = None,
        ax: Optional[Axes] = None,
        plot_backend: Literal["mpl"] = "mpl",
    ) -> Tuple[Figure, Union[Axes, List[Axes]]]:
        """
        Creates a circuit diagram visualization of the schedule using the specified
        plotting backend.

        The circuit diagram visualization visualizes the schedule at the quantum circuit
        layer.
        This visualization provides no timing information, only showing the order of
        operations.
        Because quantify-scheduler uses a hybrid gate-pulse paradigm, operations for
        which no information is specified at the gate level are visualized using an
        icon (e.g., a stylized wavy pulse) depending on the information specified at
        the quantum device layer.

        Alias of :func:`.circuit_diagram.circuit_diagram_matplotlib`.

        Parameters
        ----------
        schedule
            the schedule to render.
        figsize
            matplotlib figsize.
        ax
            Axis handle to use for plotting.
        plot_backend
            Plotting backend to use, currently only 'mpl' is supported

        Returns
        -------
        fig
            matplotlib figure object.
        ax
            matplotlib axis object.
        """
        # NB imported here to avoid circular import
        # pylint: disable=import-outside-toplevel
        if plot_backend == "mpl":
            import quantify_scheduler.visualization.circuit_diagram as cd

            return cd.circuit_diagram_matplotlib(schedule=self, figsize=figsize, ax=ax)

        raise ValueError(
            f"plot_backend must be equal to 'mpl', value given: {repr(plot_backend)}"
        )

    # pylint: disable=too-many-arguments
    def plot_pulse_diagram_mpl(
        self,
        port_list: Optional[List[str]] = None,
        sampling_rate: float = 1e9,
        modulation: Literal["off", "if", "clock"] = "off",
        modulation_if: float = 0.0,
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """
        Creates a visualization of all the pulses in a schedule using matplotlib.

        The pulse diagram visualizes the schedule at the quantum device layer.
        For this visualization to work, all operations need to have the information
        present (e.g., pulse info) to represent these on the quantum-circuit level and
        requires the absolute timing to have been determined.
        This information is typically added when the quantum-device level compilation is
        performed.

        Alias of :func:`.pulse_diagram.pulse_diagram_matplotlib`.

        port_list :
            A list of ports to show. if set to `None` will use the first
            8 ports it encounters in the sequence.
        modulation :
            Determines if modulation is included in the visualization.
        modulation_if :
            Modulation frequency used when modulation is set to "if".
        sampling_rate :
            The time resolution used to sample the schedule in Hz.
        ax:
            Axis onto which to plot.
        """
        warnings.warn(
            "`plot_pulse_diagram_mpl` will be removed from this module in "
            "quantify-scheduler >= 0.6.0.\n"
            "Instead, use `plot_pulse_diagram`",
            DeprecationWarning,
        )
        return self.plot_pulse_diagram(
            port_list, sampling_rate, modulation, modulation_if, "mpl", {"ax": ax}
        )

    # pylint: disable=too-many-arguments
    def plot_pulse_diagram(
        self,
        port_list: Optional[List[str]] = None,
        sampling_rate: float = 1e9,
        modulation: Literal["off", "if", "clock"] = "off",
        modulation_if: float = 0.0,
        plot_backend: Literal["mpl", "plotly"] = "mpl",
        plot_kwargs: Optional[dict] = None,
    ) -> Union[Tuple[Figure, Axes], go.Figure]:
        """
        Creates a visualization of all the pulses in a schedule using the specified
        plotting backend.

        The pulse diagram visualizes the schedule at the quantum device layer.
        For this visualization to work, all operations need to have the information
        present (e.g., pulse info) to represent these on the quantum-circuit level and
        requires the absolute timing to have been determined.
        This information is typically added when the quantum-device level compilation is
        performed.

        Alias of :func:`.pulse_diagram.pulse_diagram_matplotlib` and
        :func:`.pulse_diagram.pulse_diagram_plotly`.

        port_list :
            A list of ports to show. if set to `None` will use the first
            8 ports it encounters in the sequence.
        modulation :
            Determines if modulation is included in the visualization.
        modulation_if :
            Modulation frequency used when modulation is set to "if".
        sampling_rate :
            The time resolution used to sample the schedule in Hz.
        plot_backend:
            Plotting library to use, can either be 'mpl' or 'plotly'.
        plot_kwargs:
            Dictionary of keyword arguments to pass to the plotting backend

        Returns
        -------
        Union[Tuple[Figure, Axes], :class:`!plotly.graph_objects.Figure`]
            the plot
        """
        if plot_kwargs is None:
            plot_kwargs = {}
        if plot_backend == "mpl":
            # NB imported here to avoid circular import
            # pylint: disable=import-outside-toplevel
            from quantify_scheduler.visualization.pulse_diagram import (
                pulse_diagram_matplotlib,
            )

            return pulse_diagram_matplotlib(
                schedule=self,
                sampling_rate=sampling_rate,
                port_list=port_list,
                modulation=modulation,
                modulation_if=modulation_if,
                **plot_kwargs,
            )
        if plot_backend == "plotly":
            # NB imported here to avoid circular import
            # pylint: disable=import-outside-toplevel
            from quantify_scheduler.visualization.pulse_diagram import (
                pulse_diagram_plotly,
            )

            return pulse_diagram_plotly(
                schedule=self,
                sampling_rate=sampling_rate,
                port_list=port_list,
                modulation=modulation,
                modulation_if=modulation_if,
                **plot_kwargs,
            )
        raise ValueError(
            f"plot_backend must be equal to either 'mpl' or 'plotly', "
            f"value given: {repr(plot_backend)}"
        )

    @property
    def timing_table(self) -> pd.io.formats.style.Styler:
        """
        A styled pandas dataframe containing the absolute timing of pulses and
        acquisitions in a schedule.

        This table is constructed based on the abs_time key in the
        :attr:`~quantify_scheduler.schedules.schedule.ScheduleBase.schedulables`.
        This requires the timing to have been determined.

        Parameters
        ----------
        schedule
            a schedule for which the absolute timing has been determined.

        Returns
        -------
        :
            styled_timing_table, a pandas Styler containing a dataframe with
            an overview of the timing of the pulses and acquisitions present in the
            schedule. The data frame can be accessed through the .data attribute of
            the Styler.

        Raises
        ------
        ValueError
            When the absolute timing has not been determined during compilation.
        """

        timing_table = pd.DataFrame(
            columns=[
                "waveform_op_id",  # a readable id based on the operation
                "port",
                "clock",
                "is_acquisition",  # a bool which helps determine if an operation is
                # an acquisition or not. (True is it is an acquisition operation)
                "abs_time",  # start of the operation in absolute time (s)
                "duration",  # duration of the operation in absolute time (s)
                "operation",
                "wf_idx",
            ]
        )

        timing_table_list = [timing_table]
        for schedulable in self.schedulables.values():
            if "abs_time" not in schedulable:
                # when this exception is encountered
                raise ValueError("Absolute time has not been determined yet.")
            operation = self.operations[schedulable["operation_repr"]]

            # iterate over pulse information
            for i, pulse_info in enumerate(operation["pulse_info"]):
                abs_time = pulse_info["t0"] + schedulable["abs_time"]
                df_row = {
                    "waveform_op_id": schedulable["operation_repr"] + f"_p_{i}",
                    "port": pulse_info["port"],
                    "clock": pulse_info["clock"],
                    "abs_time": abs_time,
                    "duration": pulse_info["duration"],
                    "is_acquisition": False,
                    "operation": schedulable["operation_repr"],
                    "wf_idx": i,
                }
                timing_table_list.append(pd.DataFrame(df_row, index=range(1)))

            # iterate over acquisition information
            for i, acq_info in enumerate(operation["acquisition_info"]):
                abs_time = acq_info["t0"] + schedulable["abs_time"]
                df_row = {
                    "waveform_op_id": schedulable["operation_repr"] + f"_acq_{i}",
                    "port": acq_info["port"],
                    "clock": acq_info["clock"],
                    "abs_time": abs_time,
                    "duration": acq_info["duration"],
                    "is_acquisition": True,
                    "operation": schedulable["operation_repr"],
                    "wf_idx": i,
                }
                timing_table_list.append(pd.DataFrame(df_row, index=range(1)))
        timing_table = pd.concat(timing_table_list, ignore_index=True)
        # apply a style so that time is easy to read.
        # this works under the assumption that we are using timings on the order of
        # nanoseconds.
        styled_timing_table = timing_table.style.format(
            {
                "abs_time": lambda val: f"{val*1e9:,.1f} ns",
                "duration": lambda val: f"{val*1e9:,.1f} ns",
            }
        )
        return styled_timing_table


class Schedule(ScheduleBase):  # pylint: disable=too-many-ancestors
    """
    A modifiable schedule.

    Operations :class:`quantify_scheduler.operations.operation.Operation` can be added
    using the :meth:`~.Schedule.add` method, allowing precise
    specification *when* to perform an operation using timing constraints.

    When adding an operation, it is not required to specify how to represent this
    :class:`quantify_scheduler.operations.operation.Operation` on all layers.
    Instead, this information can be added later during
    :ref:`compilation <sec-compilation>`.
    This allows the user to effortlessly mix the gate- and pulse-level descriptions as
    required for many (calibration) experiments.

    """  # pylint: disable=line-too-long

    schema_filename = "schedule.json"

    def __init__(self, name: str, repetitions: int = 1, data: dict = None) -> None:
        """
        Initialize a new instance of Schedule.

        Parameters
        ----------
        name
            The name of the schedule
        repetitions
            The amount of times the schedule will be repeated, by default 1
        data
            A dictionary containing a pre-existing schedule., by default None
        """

        # validate the input data to ensure it is valid schedule data
        super().__init__()

        # ensure keys exist
        self.data["operation_dict"] = {}
        self.data["schedulables"] = {}
        self.data["resource_dict"] = {}
        self.data["name"] = "nameless"
        self.data["repetitions"] = repetitions

        # This is used to define baseband pulses and is expected to always be present
        # in any schedule.
        self.add_resource(
            resources.BasebandClockResource(resources.BasebandClockResource.IDENTITY)
        )

        if name is not None:
            self.data["name"] = name

        if data is not None:
            self.data.update(data)

    def add_resources(self, resources_list: list) -> None:
        """Add wrapper for adding multiple resources"""
        for resource in resources_list:
            self.add_resource(resource)

    def add_resource(self, resource) -> None:
        """
        Add a resource such as a channel or qubit to the schedule.
        """
        assert resources.Resource.is_valid(resource)
        if resource.name in self.data["resource_dict"]:
            raise ValueError(f"Key {resource.name} is already present")

        self.data["resource_dict"][resource.name] = resource

    # pylint: disable=too-many-arguments
    def add(
        self,
        operation: Operation,
        rel_time: float = 0,
        ref_op: Schedulable = None,
        ref_pt: Literal["start", "center", "end"] = "end",
        ref_pt_new: Literal["start", "center", "end"] = "start",
        label: str = None,
    ) -> Schedulable:
        """
        Add an :class:`quantify_scheduler.operations.operation.Operation` to the
        schedule.

        Parameters
        ----------
        operation
            The operation to add to the schedule
        rel_time
            relative time between the reference operation and the added operation.
            the time is the time between the "ref_pt" in the reference operation and
            "ref_pt_new" of the operation that is added.
        ref_op
            reference schedulable. If set to :code:`None`, will default
            to the last added operation.
        ref_pt
            reference point in reference operation must be one of
            ('start', 'center', 'end').
        ref_pt_new
            reference point in added operation must be one of
            ('start', 'center', 'end').
        label
            a unique string that can be used as an identifier when adding operations.
            if set to None, a random hash will be generated instead.
        Returns
        -------
        :
            returns the schedulable created on the schedule
        """
        assert isinstance(operation, Operation)

        if label is None:
            label = str(uuid4())

        operation_id = str(operation)
        self.data["operation_dict"][operation_id] = operation
        element = Schedulable(name=label, operation_repr=operation_id, schedule=self)
        element.add_timing_constraint(rel_time, ref_op, ref_pt, ref_pt_new)
        self.data["schedulables"].update({label: element})

        return element

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state
        for schedulable in self.schedulables.values():
            schedulable.schedule = weakref.proxy(self)


class Schedulable(JSONSchemaValMixin, UserDict):
    """
    This class represents an element on a schedule. All elements on a schedule are
    schedulables. A schedulable contains all information regarding the timing of this
    element as well as the operation being executing by this element.
    This operation is currently represented by an operation ID.

    Schedulables can contain an arbitrary number of timing constraints to determine the
    timing. Multiple different contraints are currently resolved by delaying the element
    until after all timing constraints have been met, to aid compatibility.
    To specify an exact timing between two schedulables, please ensure to only specify
    exactly one timing constraint.
    """

    schema_filename = "schedulable.json"

    def __init__(self, name, operation_repr, schedule, data: dict = None):
        """

        Parameters
        ----------
        name
            The name of this schedulable, by which it can be referenced by other
            schedulables. Separate schedulables cannot share the same name
        operation_repr
            The operation which is to be executed by this schedulable
        schedule
            The schedule to which the schedulable is added. This allows to scheduable to
            find other elements on the schedule
        """
        super().__init__()
        if data is not None:
            self.data = data
            return

        # assert the name is unique
        name_is_unique = (
            len([item for item in schedule["schedulables"].keys() if item == name]) == 0
        )
        if not name_is_unique:
            raise ValueError(f'Name "{name}" must be unique.')

        self.data["name"] = name
        self.data["operation_repr"] = operation_repr
        self.data["timing_constraints"] = []

        # the next lines are to prevent breaking the existing API
        self.data["label"] = name

        self.schedule = weakref.proxy(schedule)
        # self.schedule = schedule

    def add_timing_constraint(
        self,
        rel_time: float = 0,
        ref_schedulable: Schedulable = None,
        ref_pt: Literal["start", "center", "end"] = "end",
        ref_pt_new: Literal["start", "center", "end"] = "start",
    ):
        """
        A timing constraint constrains the operation in time by specifying the time
        (:code:`"rel_time"`) between a reference schedulable and the added schedulable.
        The time can be specified with respect to the "start", "center", or "end" of
        the operations.
        The reference schedulable (:code:`"ref_schedulable"`) is specified using its
        name property.
        See also :attr:`~.ScheduleBase.schedulables`.

        Parameters
        ----------
        rel_time
            relative time between the reference schedulable and the added schedulable.
            the time is the time between the "ref_pt" in the reference operation and
            "ref_pt_new" of the operation that is added.
        ref_schedulable
            name of the reference schedulable. If set to :code:`None`, will default
            to the last added operation.
        ref_pt
            reference point in reference schedulable must be one of
            ('start', 'center', 'end').
        ref_pt_new
            reference point in added schedulable must be one of
            ('start', 'center', 'end').
        """

        # assert that the reference operation exists
        if (
            ref_schedulable is not None
            and str(ref_schedulable) not in self.schedule.data["schedulables"].keys()
        ):
            raise ValueError(
                f'Reference "{ref_schedulable}" does not exist in schedule.'
            )

        timing_constr = {
            "rel_time": rel_time,
            "ref_schedulable": ref_schedulable,
            "ref_pt_new": ref_pt_new,
            "ref_pt": ref_pt,
        }
        self.data["timing_constraints"].append(timing_constr)

    def __str__(self):
        return str(self.data["name"])

    def __repr__(self) -> str:
        """
        Returns the string representation  of this instance.

        This represenation can always be evalued to create a new instance.

        .. code-block::

            eval(repr(operation))

        Returns
        -------
        :
        """
        cls = f"{self.__class__.__name__}"
        return (
            f"{cls}(name='{self.data['name']}', "
            f"operation_repr='', "
            f"schedule='', "
            f"data={self.data})"
        )

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state


# pylint: disable=too-many-ancestors
class CompiledSchedule(ScheduleBase):
    """
    A schedule that contains compiled instructions ready for execution using
    the :class:`~.InstrumentCoordinator`.

    The :class:`CompiledSchedule` differs from a :class:`.Schedule` in
    that it is considered immutable (no new operations or resources can be added), and
    that it contains :attr:`~.compiled_instructions`.

    .. tip::

        A :class:`~.CompiledSchedule` can be obtained by compiling a
        :class:`~.Schedule` using :func:`~quantify_scheduler.compilation.qcompile`.

    """  # pylint: disable=line-too-long

    schema_filename = "schedule.json"

    def __init__(self, schedule: Schedule) -> None:

        # validate the input data to ensure it is valid schedule data
        super().__init__()

        self._hardware_timing_table: pd.DataFrame = pd.DataFrame()

        # N.B. this relies on a bit of a dirty monkey patch way of adding these
        # properties. Not so nice.
        if hasattr(schedule, "_hardware_timing_table"):
            self._hardware_timing_table = schedule._hardware_timing_table

        self._hardware_waveform_dict: Dict[str, np.ndarray] = {}
        if hasattr(schedule, "_hardware_waveform_dict"):
            self._hardware_waveform_dict = schedule._hardware_waveform_dict

        # ensure keys exist
        self.data["compiled_instructions"] = {}

        # deepcopy is used to prevent side effects when the
        # original (mutable) schedule is modified
        self.data.update(deepcopy(schedule.data))

    @property
    def compiled_instructions(self) -> Dict[str, Resource]:
        """
        A dictionary containing compiled instructions.

        The contents of this dictionary depend on the backend it was compiled for.
        However, we assume that the general format consists of a dictionary in which
        the keys are instrument names corresponding to components added to a
        :class:`~.InstrumentCoordinator`, and the
        values are the instructions for that component.

        These values typically contain a combination of sequence files, waveform
        definitions, and parameters to configure on the instrument.
        """  # pylint: disable=line-too-long
        return self.data["compiled_instructions"]

    @classmethod
    def is_valid(cls, object_to_be_validated) -> bool:
        """
        Checks if the contents of the object_to_be_validated are valid
        according to the schema. Additionally checks if the object_to_be_validated is
        an instance of :class:`~.CompiledSchedule`
        """
        valid_schedule = super().is_valid(object_to_be_validated)
        if valid_schedule:
            return isinstance(object_to_be_validated, CompiledSchedule)

        return False

    @property
    def hardware_timing_table(self) -> pd.io.formats.style.Styler:
        """
        Returns a timing table representing all operations at the Control-hardware
        layer.

        Note that this timing table is typically different from the `.timing_table` in
        that it contains more hardware specific information such as channels, clock
        cycles and samples and corrections for things such as gain.

        This hardware timing table is intended to provide a more

        This table is constructed based on the timing_table and modified during
        compilation in one of the hardware back ends and optionally added to the
        schedule. Not all back ends support this feature.
        """
        styled_hardware_timing_table = self._hardware_timing_table.style.format(
            {
                "abs_time": lambda val: f"{val*1e9:,.1f} ns",
                "duration": lambda val: f"{val*1e9:,.1f} ns",
                "clock_cycle_start": lambda val: f"{val:,.1f}",
                "sample_start": lambda val: f"{val:,.1f}",
            }
        )

        return styled_hardware_timing_table

    @property
    def hardware_waveform_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns a waveform dictionary representing all waveforms at the Control-hardware
        layer.

        Where the waveforms are represented as abstract waveforms in the Operations,
        this dictionary contains the numerical arrays that are uploaded to the hardware.

        This dictionary is constructed during compilation in the hardware back ends and
         optionally added to the schedule. Not all back ends support this feature.
        """
        return self._hardware_waveform_dict


@dataclass
class AcquisitionMetadata:
    """
    Class to provide a description of the shape and type of data that a schedule will
    return when executed.

    .. note::

        The acquisition protocol, bin-mode and return types are assumed to be the same
        for all acquisitions in a schedule.
    """

    acq_protocol: str
    """The acquisition protocol that is used for all acquisitions in the schedule."""
    bin_mode: enums.BinMode
    """How the data is stored in the bins indexed by acq_channel and acq_index."""
    acq_return_type: Type
    """The datatype returned by the individual acquisitions."""
    acq_indices: Dict[int, List[int]]
    """A dictionary containing the acquisition channel as key and a list of acquisition
    indices that are used for every channel."""
