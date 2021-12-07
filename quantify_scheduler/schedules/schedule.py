# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the core concepts of the scheduler."""
from __future__ import annotations

import json
from abc import ABC
from collections import UserDict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4

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
    - timing_constraints - a list of all timing constraints added between operations.

    The :class:`~.Schedule` provides an API to create schedules.
    The :class:`~.CompiledSchedule` represents a schedule after
    it has been compiled for execution on a backend.


    The :class:`~.Schedule` contains information on the
    :attr:`~.ScheduleBase.operations` and
    :attr:`~.ScheduleBase.timing_constraints`.
    The :attr:`~.ScheduleBase.operations` is a dictionary of all
    unique operations used in the schedule and contain the information on *what*
    operation to apply *where*.
    The :attr:`~.ScheduleBase.timing_constraints` is a list of
    dictionaries describing timing constraints between operations, i.e. when to apply
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
    def timing_constraints(self) -> List[Dict[str, Any]]:
        """
        A list of dictionaries describing timing constraints between operations.

        A timing constraint constrains the operation in time by specifying the time
        (:code:`"rel_time"`) between a reference operation and the added operation.
        The time can be specified with respect to a reference point (:code:`"ref_pt"')
        on the reference operation (:code:`"ref_op"`) and a reference point on the next
        added operation (:code:`"ref_pt_new"').
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
        return self.data["timing_constraints"]

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
            f'{len(self.data["timing_constraints"])}  (unique) operations.'
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

        return Schedule(name, data=schedule_data)

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
        # NB imported here to avoid circular import
        # pylint: disable=import-outside-toplevel
        import quantify_scheduler.visualization.circuit_diagram as cd

        return cd.circuit_diagram_matplotlib(schedule=self, figsize=figsize, ax=ax)

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
        # NB imported here to avoid circular import
        # pylint: disable=import-outside-toplevel
        from quantify_scheduler.visualization.pulse_diagram import (
            pulse_diagram_matplotlib,
        )

        return pulse_diagram_matplotlib(
            schedule=self,
            sampling_rate=sampling_rate,
            ax=ax,
            port_list=port_list,
            modulation=modulation,
            modulation_if=modulation_if,
        )

    @property
    def timing_table(self) -> pd.io.formats.style.Styler:
        """
        A styled pandas dataframe containing the absolute timing of pulses and
        acquisitions in a schedule.

        This table is constructed based on the abs_time key in the
        :attr:`~quantify_scheduler.schedules.schedule.ScheduleBase.timing_constraints`.
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

        for t_constr in self.timing_constraints:
            if "abs_time" not in t_constr:
                # when this exception is encountered
                raise ValueError("Absolute time has not been determined yet.")
            operation = self.operations[t_constr["operation_repr"]]

            # iterate over pulse information
            for i, pulse_info in enumerate(operation["pulse_info"]):
                abs_time = pulse_info["t0"] + t_constr["abs_time"]
                df_row = {
                    "waveform_op_id": t_constr["operation_repr"] + f"_p_{i}",
                    "port": pulse_info["port"],
                    "clock": pulse_info["clock"],
                    "abs_time": abs_time,
                    "duration": pulse_info["duration"],
                    "is_acquisition": False,
                    "operation": t_constr["operation_repr"],
                    "wf_idx": i,
                }
                timing_table = timing_table.append(df_row, ignore_index=True)

            # iterate over acquisition information
            for i, acq_info in enumerate(operation["acquisition_info"]):
                abs_time = acq_info["t0"] + t_constr["abs_time"]
                df_row = {
                    "waveform_op_id": t_constr["operation_repr"] + f"_acq_{i}",
                    "port": acq_info["port"],
                    "clock": acq_info["clock"],
                    "abs_time": abs_time,
                    "duration": acq_info["duration"],
                    "is_acquisition": True,
                    "operation": t_constr["operation_repr"],
                    "wf_idx": i,
                }
                timing_table = timing_table.append(df_row, ignore_index=True)

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
        self.data["timing_constraints"] = []
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
        ref_op: str = None,
        ref_pt: Literal["start", "center", "end"] = "end",
        ref_pt_new: Literal["start", "center", "end"] = "start",
        label: str = None,
    ) -> str:
        """
        Add an :class:`quantify_scheduler.operations.operation.Operation` to the
        schedule and specify timing constraints.

        A timing constraint constrains the operation in time by specifying the time
        (:code:`"rel_time"`) between a reference operation and the added operation.
        The time can be specified with respect to the "start", "center", or "end" of
        the operations.
        The reference operation (:code:`"ref_op"`) is specified using its label
        property.
        See also :attr:`~.ScheduleBase.timing_constraints`.

        Parameters
        ----------
        operation
            The operation to add to the schedule
        rel_time
            relative time between the reference operation and the added operation.
            the time is the time between the "ref_pt" in the reference operation and
            "ref_pt_new" of the operation that is added.
        ref_op
            label of the reference operation. If set to :code:`None`, will default
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
            returns the (unique) label of the last added operation.
        """
        assert isinstance(operation, Operation)

        if label is None:
            label = str(uuid4())
        else:
            # assert that the label of the operation does not exists in the
            # timing constraints.
            label_is_unique = (
                len(
                    [
                        item
                        for item in self.data["timing_constraints"]
                        if item["label"] == label
                    ]
                )
                == 0
            )
            if not label_is_unique:
                raise ValueError(f'Label "{label}" must be unique.')

        # assert that the reference operation exists
        if ref_op is not None:
            ref_exists = (
                len(
                    [
                        item
                        for item in self.data["timing_constraints"]
                        if item["label"] == ref_op
                    ]
                )
                == 1
            )
            if not ref_exists:
                raise ValueError(f'Reference "{ref_op}" does not exist in schedule.')

        operation_id = str(operation)
        self.data["operation_dict"][operation_id] = operation
        timing_constr = {
            "label": label,
            "rel_time": rel_time,
            "ref_op": ref_op,
            "ref_pt_new": ref_pt_new,
            "ref_pt": ref_pt,
            "operation_repr": operation_id,
        }
        self.data["timing_constraints"].append(timing_constr)

        return label


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
