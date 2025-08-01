# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing the core concepts of the scheduler."""

from __future__ import annotations

import dataclasses
from abc import ABC
from collections import UserDict
from collections.abc import Hashable, MutableMapping
from copy import copy
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from quantify_scheduler import enums, resources
from quantify_scheduler.backends.types.common import ThresholdedTriggerCountMetadata
from quantify_scheduler.helpers.collections import make_hash
from quantify_scheduler.helpers.importers import (
    export_python_object_to_path_string,
)
from quantify_scheduler.json_utils import (
    JSONSchemaValMixin,
    JSONSerializable,
)
from quantify_scheduler.operations.control_flow_library import ConditionalOperation, LoopOperation
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.resources import Resource

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

DictOrdered = dict
"""
An ordered dictionary type hint,
which makes it clear and obvious
that order is significant and used by the logic.
Note: dict is ordered from Python version 3.7.
Note: collections.OrderedDict can be slow in some cases.
"""


class ScheduleBase(JSONSchemaValMixin, JSONSerializable, UserDict, ABC):
    """
    Interface to be used for :class:`~.Schedule`.

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
    - schedulables - an ordered dictionary of all timing constraints added
        between operations; when multiple schedulables have the same
        absolute time, the order defined in the dictionary decides precedence.

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

    .. jsonschema:: https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/quantify_scheduler/schemas/schedule.json

    """

    @property
    def name(self) -> str:
        """Returns the name of the schedule."""
        return self["name"]

    @property
    def repetitions(self) -> int:
        """
        Returns the amount of times this Schedule will be repeated.

        Returns
        -------
        :
            The repetitions count.

        """
        return self["repetitions"]

    @repetitions.setter
    def repetitions(self, value: int) -> None:
        if value <= 0:
            raise ValueError(
                f"Attempting to set repetitions for the schedule. "
                f"Must be a positive number. Got {value}."
            )
        self["repetitions"] = int(value)

    @property
    def operations(self) -> dict[str, Operation | Schedule]:
        """
        A dictionary of all unique operations used in the schedule.

        This specifies information on *what* operation to apply *where*.

        The keys correspond to the :attr:`~.Operation.hash` and values are instances
        of :class:`quantify_scheduler.operations.operation.Operation`.
        """
        return self["operation_dict"]

    @property
    def schedulables(self) -> DictOrdered[str, Schedulable]:
        """
        Ordered dictionary of schedulables describing timing and order of operations.

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

            ['label', 'rel_time', 'ref_op', 'ref_pt_new', 'ref_pt', 'operation_id']

        The label is used as a unique identifier that can be used as a reference for
        other operations, the operation_id refers to the hash of an
        operation in :attr:`~.ScheduleBase.operations`.

        .. note::

            timing constraints are not intended to be modified directly.
            Instead use the :meth:`~.Schedule.add`

        """
        return self["schedulables"]

    @property
    def resources(self) -> dict[str, Resource]:
        """
        A dictionary containing resources.

        Keys are names (str), values are instances of
        :class:`~quantify_scheduler.resources.Resource`.
        """
        return self["resource_dict"]

    def __hash__(self) -> int:
        return make_hash(self.data)

    @property
    def hash(self) -> str:
        """A hash based on the contents of the Schedule."""
        return str(hash(self))

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        return (
            f'{self.__class__.__name__} "{self["name"]}" containing '
            f"({len(self['operation_dict'])}) "
            f"{len(self.schedulables)}  (unique) operations."
        )

    def get_used_port_clocks(self) -> set[tuple[str, str]]:
        """
        Extracts which port-clock combinations are used in this schedule.

        Returns
        -------
        :
            All (port, clock) combinations that operations in this schedule uses

        """
        port_clocks_used = set()
        for op_data in self.operations.values():
            port_clocks_used |= op_data.get_used_port_clocks()
        return port_clocks_used

    def plot_circuit_diagram(
        self,
        figsize: tuple[int, int] | None = None,
        ax: Axes | None = None,
        plot_backend: Literal["mpl"] = "mpl",
    ) -> tuple[Figure | None, Axes | list[Axes]]:
        """
        Create a circuit diagram visualization of the schedule using the specified plotting backend.

        The circuit diagram visualization depicts the schedule at the quantum circuit
        layer. Because quantify-scheduler uses a hybrid gate-pulse paradigm, operations
        for which no information is specified at the gate level are visualized using an
        icon (e.g., a stylized wavy pulse) depending on the information specified at
        the quantum device layer.

        Alias of :func:`quantify_scheduler.schedules._visualization.circuit_diagram.circuit_diagram_matplotlib`.

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



        Each gate, pulse, measurement, and any other operation are plotted in the order
        of execution, but no timing information is provided.

        .. admonition:: Example
            :class: tip

            .. jupyter-execute::

                from quantify_scheduler import Schedule
                from quantify_scheduler.operations.gate_library import Reset, X90, CZ, Rxy, Measure

                sched = Schedule(f"Bell experiment on q0-q1")

                sched.add(Reset("q0", "q1"))
                sched.add(X90("q0"))
                sched.add(X90("q1"), ref_pt="start", rel_time=0)
                sched.add(CZ(qC="q0", qT="q1"))
                sched.add(Rxy(theta=45, phi=0, qubit="q0") )
                sched.add(Measure("q0", acq_index=0))
                sched.add(Measure("q1", acq_index=0), ref_pt="start")

                sched.plot_circuit_diagram();

        .. note::

            Gates that are started simultaneously on the same qubit will overlap.

            .. jupyter-execute::

                from quantify_scheduler import Schedule
                from quantify_scheduler.operations.gate_library import X90, Measure

                sched = Schedule(f"overlapping gates")

                sched.add(X90("q0"))
                sched.add(Measure("q0"), ref_pt="start", rel_time=0)
                sched.plot_circuit_diagram();

        .. note::

            If the pulse's port address was not found then the pulse will be plotted on the
            'other' timeline.

        """  # noqa: E501
        # NB imported here to avoid circular import

        if plot_backend == "mpl":
            import quantify_scheduler.schedules._visualization.circuit_diagram as cd

            return cd.circuit_diagram_matplotlib(schedule=self, figsize=figsize, ax=ax)

        raise ValueError(f"plot_backend must be equal to 'mpl', value given: {plot_backend!r}")

    def plot_pulse_diagram(
        self,
        port_list: list[str] | None = None,
        sampling_rate: float = 1e9,
        modulation: Literal["off", "if", "clock"] = "off",
        modulation_if: float = 0.0,
        plot_backend: Literal["mpl", "plotly"] = "mpl",
        x_range: tuple[float, float] = (-np.inf, np.inf),
        combine_waveforms_on_same_port: bool = False,
        **backend_kwargs,
    ) -> tuple[Figure, Axes] | go.Figure:
        """
        Create a visualization of all the pulses in a schedule using the specified plotting backend.

        The pulse diagram visualizes the schedule at the quantum device layer.
        For this visualization to work, all operations need to have the information
        present (e.g., pulse info) to represent these on the quantum-circuit level and
        requires the absolute timing to have been determined.
        This information is typically added when the quantum-device level compilation is
        performed.

        Alias of
        :func:`quantify_scheduler.schedules._visualization.pulse_diagram.pulse_diagram_matplotlib`
        and
        :func:`quantify_scheduler.schedules._visualization.pulse_diagram.pulse_diagram_plotly`.

        Parameters
        ----------
        port_list :
            A list of ports to show. If ``None`` (default) the first 8 ports encountered in the sequence are used.
        modulation :
            Determines if modulation is included in the visualization.
        modulation_if :
            Modulation frequency used when modulation is set to "if".
        sampling_rate :
            The time resolution used to sample the schedule in Hz.
        plot_backend:
            Plotting library to use, can either be 'mpl' or 'plotly'.
        x_range:
            The range of the x-axis that is plotted, given as a tuple (left limit, right
            limit). This can be used to reduce memory usage when plotting a small section of
            a long pulse sequence. By default (-np.inf, np.inf).
        combine_waveforms_on_same_port:
            By default False. If True, combines all waveforms on the same port into one
            single waveform. The resulting waveform is the sum of all waveforms on that
            port (small inaccuracies may occur due to floating point approximation). If
            False, the waveforms are shown individually.
        backend_kwargs:
            Keyword arguments to be passed on to the plotting backend. The arguments
            that can be used for either backend can be found in the documentation of
            :func:`quantify_scheduler.schedules._visualization.pulse_diagram.pulse_diagram_matplotlib`
            and
            :func:`quantify_scheduler.schedules._visualization.pulse_diagram.pulse_diagram_plotly`.

        Returns
        -------
        Union[tuple[Figure, Axes], :class:`!plotly.graph_objects.Figure`]
            the plot


        .. admonition:: Example
            :class: tip

            A simple plot with matplotlib can be created as follows:

            .. jupyter-execute::

                from quantify_scheduler.backends.graph_compilation import SerialCompiler
                from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
                from quantify_scheduler.operations.pulse_library import (
                    DRAGPulse, SquarePulse, RampPulse, VoltageOffset,
                )
                from quantify_scheduler.resources import ClockResource

                schedule = Schedule("Multiple waveforms")
                schedule.add(DRAGPulse(G_amp=0.2, D_amp=0.2, phase=0, duration=4e-6, port="P", clock="C"))
                schedule.add(RampPulse(amp=0.2, offset=0.0, duration=6e-6, port="P"))
                schedule.add(SquarePulse(amp=0.1, duration=4e-6, port="Q"), ref_pt='start')
                schedule.add_resource(ClockResource(name="C", freq=4e9))

                quantum_device = QuantumDevice("quantum_device")
                device_compiler = SerialCompiler("Device compiler", quantum_device)
                compiled_schedule = device_compiler.compile(schedule)

                _ = compiled_schedule.plot_pulse_diagram(sampling_rate=20e6)

            The backend can be changed to the plotly backend by specifying the
            ``plot_backend=plotly`` argument. With the plotly backend, pulse
            diagrams include a separate plot for each port/clock
            combination:

            .. jupyter-execute::

                _ = compiled_schedule.plot_pulse_diagram(sampling_rate=20e6, plot_backend='plotly')

            The same can be achieved in the default ``plot_backend`` (``matplotlib``)
            by passing the keyword argument ``multiple_subplots=True``:

            .. jupyter-execute::

                _ = compiled_schedule.plot_pulse_diagram(sampling_rate=20e6, multiple_subplots=True)

            By default, waveforms overlapping in time on the same port are shown separately:

            .. jupyter-execute::

                schedule = Schedule("Overlapping waveforms")
                schedule.add(VoltageOffset(offset_path_I=0.25, offset_path_Q=0.0, port="Q"))
                schedule.add(SquarePulse(amp=0.1, duration=4e-6, port="Q"), rel_time=2e-6)
                schedule.add(VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="Q"), ref_pt="start", rel_time=2e-6)

                compiled_schedule = device_compiler.compile(schedule)

                _ = compiled_schedule.plot_pulse_diagram(sampling_rate=20e6)

            This behaviour can be changed with the parameter ``combine_waveforms_on_same_port``:

            .. jupyter-execute::

                _ = compiled_schedule.plot_pulse_diagram(sampling_rate=20e6, combine_waveforms_on_same_port=True)

        """  # noqa: E501
        # NB imported here to avoid circular import

        from quantify_scheduler.schedules._visualization.pulse_diagram import sample_schedule

        sampled_pulses_and_acqs = sample_schedule(
            self,
            sampling_rate=sampling_rate,
            port_list=port_list,
            modulation=modulation,
            modulation_if=modulation_if,
            x_range=x_range,
            combine_waveforms_on_same_port=combine_waveforms_on_same_port,
        )

        if plot_backend == "mpl":
            # NB imported here to avoid circular import

            from quantify_scheduler.schedules._visualization.pulse_diagram import (
                pulse_diagram_matplotlib,
            )

            return pulse_diagram_matplotlib(
                sampled_pulses_and_acqs=sampled_pulses_and_acqs,
                title=self["name"],
                **backend_kwargs,
            )
        if plot_backend == "plotly":
            # NB imported here to avoid circular import

            from quantify_scheduler.schedules._visualization.pulse_diagram import (
                pulse_diagram_plotly,
            )

            return pulse_diagram_plotly(
                sampled_pulses_and_acqs=sampled_pulses_and_acqs,
                title=self["name"],
                **backend_kwargs,
            )
        raise ValueError(
            f"plot_backend must be equal to either 'mpl' or 'plotly', value given: {plot_backend!r}"
        )

    @classmethod
    def _generate_timing_table_list(
        cls,
        operation: Operation | ScheduleBase,
        time_offset: float,
        timing_table_list: list,
        operation_id: str | None,
    ) -> None:
        if isinstance(operation, ScheduleBase):
            for schedulable in operation.schedulables.values():
                if "abs_time" not in schedulable:
                    # when this exception is encountered
                    raise ValueError(
                        "Absolute time has not been determined yet. Please compile your schedule."
                    )
                cls._generate_timing_table_list(
                    operation.operations[schedulable["operation_id"]],
                    time_offset + schedulable["abs_time"],
                    timing_table_list,
                    schedulable["operation_id"],
                )
        elif isinstance(operation, LoopOperation):
            for i in range(operation.data["control_flow_info"]["repetitions"]):
                cls._generate_timing_table_list(
                    operation.body,
                    time_offset + i * operation.body.duration,
                    timing_table_list,
                    operation_id,
                )
        elif isinstance(operation, ConditionalOperation):
            cls._generate_timing_table_list(
                operation.body,
                time_offset,
                timing_table_list,
                operation_id,
            )
        else:
            for i, op_info in chain(
                enumerate(operation["pulse_info"]),
                enumerate(operation["acquisition_info"]),
            ):
                t0 = time_offset + op_info["t0"]
                df_row = {
                    "waveform_op_id": str(operation) + f"_acq_{i}",
                    "port": op_info["port"],
                    "clock": op_info["clock"],
                    "abs_time": t0,
                    "duration": op_info["duration"],
                    "is_acquisition": "acq_channel" in op_info or "bin_mode" in op_info,
                    "operation": str(operation),
                    "wf_idx": i,
                    "operation_hash": operation_id,
                }
                timing_table_list.append(pd.DataFrame(df_row, index=range(1)))

    @property
    def timing_table(self) -> pd.io.formats.style.Styler:
        """
        A styled pandas dataframe containing the absolute timing of pulses and acquisitions in a schedule.

        This table is constructed based on the ``abs_time`` key in the
        :attr:`~quantify_scheduler.schedules.schedule.ScheduleBase.schedulables`.
        This requires the timing to have been determined.

        The table consists of the following columns:

        - `operation`: a ``repr`` of :class:`~quantify_scheduler.operations.operation.Operation` corresponding to the pulse/acquisition.
        - `waveform_op_id`: an id corresponding to each pulse/acquisition inside an :class:`~quantify_scheduler.operations.operation.Operation`.
        - `port`: the port the pulse/acquisition is to be played/acquired on.
        - `clock`: the clock used to (de)modulate the pulse/acquisition.
        - `abs_time`: the absolute time the pulse/acquisition is scheduled to start.
        - `duration`: the duration of the pulse/acquisition that is scheduled.
        - `is_acquisition`: whether the pulse/acquisition is an acquisition or not (type ``numpy.bool_``).
        - `wf_idx`: the waveform index of the pulse/acquisition belonging to the Operation.
        - `operation_hash`: the unique hash corresponding to the :class:`~.Schedulable` that the pulse/acquisition belongs to.

        .. admonition:: Example

            .. jupyter-execute::
                :hide-code:

                from quantify_scheduler.backends import SerialCompiler
                from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
                from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
                from quantify_scheduler.operations.gate_library import (
                    Measure,
                    Reset,
                    X,
                    Y,
                )
                from quantify_scheduler.schedules.schedule import Schedule
                from quantify_scheduler.schemas.examples import utils

                compiler = SerialCompiler("compiler")
                q0 = BasicTransmonElement("q0")
                q4 = BasicTransmonElement("q4")

                for device_element in [q0, q4]:
                    device_element.rxy.amp180(0.115)
                    device_element.rxy.motzoi(0.1)
                    device_element.clock_freqs.f01(7.3e9)
                    device_element.clock_freqs.f12(7.0e9)
                    device_element.clock_freqs.readout(8.0e9)
                    device_element.measure.acq_delay(100e-9)

                quantum_device = QuantumDevice(name="quantum_device0")
                quantum_device.add_element(q0)
                quantum_device.add_element(q4)

                device_config = quantum_device.generate_device_config()
                hardware_config = utils.load_json_example_scheme(
                    "qblox_hardware_config_transmon.json"
                )
                hardware_config["hardware_options"].pop("distortion_corrections")
                quantum_device.hardware_config(hardware_config)

                compiler = SerialCompiler("compiler")
                compiler.quantum_device = quantum_device

            .. jupyter-execute::

                schedule = Schedule("demo timing table")
                schedule.add(Reset("q0", "q4"))
                schedule.add(X("q0"))
                schedule.add(Y("q4"))
                schedule.add(Measure("q0", acq_channel=0, acq_index=0))
                schedule.add(Measure("q4", acq_channel=1, acq_index=0))

                compiled_schedule = compiler.compile(schedule)
                compiled_schedule.timing_table

        Parameters
        ----------
        schedule
            a schedule for which the absolute timing has been determined.

        Returns
        -------
        :
            styled_timing_table, a pandas Styler containing a dataframe with
            an overview of the timing of the pulses and acquisitions present in the
            schedule. The dataframe can be accessed through the .data attribute of
            the Styler.

        Raises
        ------
        ValueError
            When the absolute timing has not been determined during compilation.

        """  # noqa: E501
        timing_table_list = []
        self._generate_timing_table_list(self, 0, timing_table_list, None)
        timing_table = pd.concat(timing_table_list, ignore_index=True)
        timing_table = timing_table.sort_values(by="abs_time")
        # apply a style so that time is easy to read.
        # this works under the assumption that we are using timings on the order of
        # nanoseconds.
        styled_timing_table = timing_table.style.format(
            {
                "abs_time": lambda val: f"{val * 1e9:,.1f} ns",
                "duration": lambda val: f"{val * 1e9:,.1f} ns",
            }
        )
        return styled_timing_table

    def get_schedule_duration(self) -> float:
        """
        Return the duration of the schedule.

        Returns
        -------
        schedule_duration : float
            Duration of current schedule

        """
        schedule_duration = 0

        # find last timestamp
        for schedulable in self.schedulables.values():
            timestamp = schedulable["abs_time"]
            operation_id = schedulable["operation_id"]

            operation = self["operation_dict"][operation_id]
            tmp_time = timestamp + operation.duration

            # keep track of longest found schedule
            schedule_duration = max(tmp_time, schedule_duration)

        schedule_duration *= self.repetitions
        return schedule_duration

    @property
    def duration(self) -> float | None:
        """
        Determine the cached duration of the schedule.

        Will return None if get_schedule_duration() has not been called before.
        """
        return self.get("duration", None)

    def __getstate__(self) -> dict[str, Any]:
        data = copy(self.data)
        # For serialization, we need to keep the order
        # of keys in the serialized data too.
        data["schedulables"] = list(data["schedulables"].items())
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": data,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Logic to allow legacy (old serialized, saved) and current serialization.
        data = state["data"] if ("deserialization_type" in state) and ("data" in state) else state

        if isinstance(data["schedulables"], list):
            # Schedulables can be a list of pair of key values to store
            # the order of schedulables too in the serialized data.
            data["schedulables"] = {k: v for k, v in data["schedulables"]}
        self.data = data

    @classmethod
    def is_valid(cls, schedule: ScheduleBase) -> bool:
        """Check if schedule adheres to JSON schema."""
        super().is_valid(schedule)  # type: ignore
        return True


class Schedule(ScheduleBase):
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

    Parameters
    ----------
    name
        The name of the schedule, by default "schedule"
    repetitions
        The amount of times the schedule will be repeated, by default 1
    data
        A dictionary containing a pre-existing schedule, by default None

    """

    schema_filename = "schedule.json"

    def __init__(
        self, name: str = "schedule", repetitions: int = 1, data: dict | None = None
    ) -> None:
        # validate the input data to ensure it is valid schedule data
        super().__init__()

        # ensure keys exist
        self["operation_dict"] = {}
        self["resource_dict"] = {}
        self["name"] = name
        self["repetitions"] = repetitions

        # Note the order of schedulables is important.
        # If two schedulables have the same absolute time,
        # the order is determined by the order of their keys.
        self["schedulables"] = {}

        # This is used to define baseband pulses and is expected to always be present
        # in any schedule.
        self.add_resource(resources.BasebandClockResource(resources.BasebandClockResource.IDENTITY))
        # This is used to define operations on marker and digital channels.
        self.add_resource(resources.DigitalClockResource(resources.DigitalClockResource.IDENTITY))

        if data is not None:
            self.data.update(data)

    def add_resources(self, resources_list: list) -> None:
        """Add wrapper for adding multiple resources."""
        for resource in resources_list:
            self.add_resource(resource)

    def add_resource(self, resource: Resource) -> None:
        """Add a resource such as a channel or device element to the schedule."""
        if not isinstance(resource, Resource):
            raise ValueError(
                f"Attempting to add resource to schedule. '{resource}' is not a Resource instance."
            )
        if resource.name in self["resource_dict"]:
            raise ValueError(f"Key {resource.name} is already present")

        self["resource_dict"][resource.name] = resource

    def add(
        self,
        operation: Operation | Schedule,
        rel_time: float = 0,
        ref_op: Schedulable | str | None = None,
        ref_pt: Literal["start", "center", "end"] | None = None,
        ref_pt_new: Literal["start", "center", "end"] | None = None,
        label: str | None = None,
    ) -> Schedulable:
        """
        Add an operation or a subschedule to the schedule.

        Parameters
        ----------
        operation
            The operation to add to the schedule, or another schedule to add
            as a subschedule.
        rel_time
            relative time between the reference operation and the added operation.
            the time is the time between the "ref_pt" in the reference operation and
            "ref_pt_new" of the operation that is added.
        ref_op
            reference schedulable. If set to :code:`None`, will default
            based on the chosen :code:`SchedulingStrategy`. If ASAP is chosen, the
            previously added schedulable is the reference schedulable. If ALAP is chose,
            the reference schedulable is the schedulable added immediately after this
            schedulable.
        ref_pt
            reference point in reference operation must be one of
            :code:`"start"`, :code:`"center"`, :code:`"end"`, or :code:`None`; in case
            of :code:`None`,
            :func:`~quantify_scheduler.compilation._determine_absolute_timing` assumes
            :code:`"end"`.
        ref_pt_new
            reference point in added operation must be one of
            :code:`"start"`, :code:`"center"`, :code:`"end"`, or :code:`None`; in case
            of :code:`None`,
            :func:`~quantify_scheduler.compilation._determine_absolute_timing` assumes
            :code:`"start"`.
        label
            a unique string that can be used as an identifier when adding operations.
            if set to `None`, a random hash will be generated instead.

        Returns
        -------
        :
            Returns the schedulable created in the schedule.

        """
        if label is None:
            label = str(uuid4())

        self._validate_add_arguments(operation, label)

        # ensure the schedulable name is unique
        if label in self.schedulables:
            raise ValueError(f"Schedulable name '{label}' must be unique.")

        return self._add(operation, rel_time, ref_op, ref_pt, ref_pt_new, label)

    def _add(
        self,
        operation: Operation | Schedule,
        rel_time: float = 0,
        ref_op: Schedulable | str | None = None,
        ref_pt: Literal["start", "center", "end"] | None = None,
        ref_pt_new: Literal["start", "center", "end"] | None = None,
        label: str | None = None,
    ) -> Schedulable:
        operation_id = operation.hash
        self["operation_dict"][operation_id] = operation
        element = Schedulable(name=label, operation_id=operation_id)
        element.add_timing_constraint(
            rel_time=rel_time,
            ref_schedulable=ref_op,
            ref_pt=ref_pt,
            ref_pt_new=ref_pt_new,
        )
        self.schedulables.update({label: element})

        return element

    def _validate_add_arguments(
        self,
        operation: Operation | Schedule,
        label: str,
    ) -> None:
        if not isinstance(operation, (Operation, Schedule)):
            raise ValueError(
                f"Attempting to add operation to schedule. "
                f"The provided object '{operation=}' is not"
                " an instance of Operation or Schedule"
            )

        # ensure the schedulable name is unique
        if label in self.schedulables:
            raise ValueError(f"Schedulable name '{label}' must be unique.")


class Schedulable(JSONSchemaValMixin, UserDict):
    """
    A representation of an element on a schedule.

    All elements on a schedule are schedulables. A schedulable contains all
    information regarding the timing of this element as well as the operation
    being executed by this element. This operation is currently represented by
    an operation ID.

    Schedulables can contain an arbitrary number of timing constraints to
    determine the timing. Multiple different constraints are currently resolved
    by delaying the element until after all timing constraints have been met, to
    aid compatibility. To specify an exact timing between two schedulables,
    please ensure to only specify exactly one timing constraint.

    Parameters
    ----------
    name
        The name of this schedulable, by which it can be referenced by other
        schedulables. Separate schedulables cannot share the same name.
    operation_id
        Reference to the operation which is to be executed by this schedulable.

    """

    schema_filename = "schedulable.json"

    def __init__(self, name: str, operation_id: str) -> None:
        super().__init__()

        self["name"] = name
        self["operation_id"] = operation_id
        self["timing_constraints"] = []

        # the next lines are to prevent breaking the existing API
        self["label"] = name

    def add_timing_constraint(
        self,
        rel_time: float = 0,
        ref_schedulable: Schedulable | str | None = None,
        ref_pt: Literal["start", "center", "end"] | None = None,
        ref_pt_new: Literal["start", "center", "end"] | None = None,
    ) -> None:
        """
        Add timing constraint.

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
            reference point in reference operation must be one of
            :code:`"start"`, :code:`"center"`, :code:`"end"`, or :code:`None`; in case
            of :code:`None`,
            :meth:`~quantify_scheduler.compilation._determine_absolute_timing` assumes
            :code:`"end"`.
        ref_pt_new
            reference point in added operation must be one of
            :code:`"start"`, :code:`"center"`, :code:`"end"`, or :code:`None`; in case
            of :code:`None`,
            :meth:`~quantify_scheduler.compilation._determine_absolute_timing` assumes
            :code:`"start"`.

        """
        # Save as str to help serialization of schedules.
        if ref_schedulable is not None:
            ref_schedulable = str(ref_schedulable)

        timing_constr = TimingConstraint(
            rel_time=rel_time,
            ref_schedulable=ref_schedulable,
            ref_pt_new=ref_pt_new,
            ref_pt=ref_pt,
        )
        self["timing_constraints"].append(timing_constr)

    def __hash__(self) -> int:
        return make_hash(self.data)

    @property
    def hash(self) -> str:
        """A hash based on the contents of the Operation."""
        return str(hash(self))

    def __str__(self) -> str:
        return str(self["name"])

    def __getstate__(self) -> dict[str, Any]:
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": self.data,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.data = state["data"]


@dataclasses.dataclass
class TimingConstraint:
    """Datastructure to store the information on a Timing Constraint."""

    ref_schedulable: str
    """The schedulable against which `ref_pt` and `rel_time` are defined."""
    ref_pt: Literal["start", "center", "end"]
    """The point on `ref_schedulable` against which `rel_time` is defined."""
    ref_pt_new: Literal["start", "center", "end"]
    """The point on the to be added schedulable against which `rel_time` is defined."""
    rel_time: float
    """The time between `ref_pt` and `ref_pt_new`."""

    @property
    def data(self) -> dict:
        """Representation of this TimingConstraint as a dictionary."""
        return dataclasses.asdict(self)

    def __getitem__(self, key: str) -> str | Literal["start", "center", "end"]:
        return self.data[key]

    def __hash__(self) -> int:
        return make_hash(self.data)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": self.data,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state["data"])


@dataclasses.dataclass
class AcquisitionChannelData:
    """Datastructure to store metadata for the given acquisition channel."""

    acq_index_dim_name: str
    """Acquisition index dimension name."""
    protocol: str
    """Acquisition protocol."""
    bin_mode: enums.BinMode
    """Bin mode."""
    coords: dict | list[dict]
    """
    Coords for each acquisition.

    For binned types this is a list of coords for each acquisition index,
    and for trace and trigger count types, this is only one value.
    """


AcquisitionChannelsData = dict[Hashable, AcquisitionChannelData]
"""
Dictionary mapping each acq_channel to their corresponding
hardware independent acquisition channel data.
"""


class CompiledSchedule(ScheduleBase):
    """
    A schedule that contains compiled instructions ready for execution using the :class:`~.InstrumentCoordinator`.

    The :class:`CompiledSchedule` differs from a :class:`.Schedule` in
    that it is considered immutable (no new operations or resources can be added), and
    that it contains :attr:`~.compiled_instructions`.

    .. tip::

        A :class:`~.CompiledSchedule` can be obtained by compiling a
        :class:`~.Schedule` using :meth:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler.compile`.

    """  # noqa: E501

    schema_filename = "schedule.json"

    def __init__(self, schedule: Schedule) -> None:
        # validate the input data to ensure it is valid schedule data
        super().__init__()

        self._hardware_timing_table: pd.DataFrame = pd.DataFrame()

        # N.B. this relies on a bit of a dirty monkey patch way of adding these
        # properties. Not so nice.
        if hasattr(schedule, "_hardware_timing_table"):
            self._hardware_timing_table = schedule._hardware_timing_table

        self._hardware_waveform_dict: dict[str, np.ndarray] = {}
        if hasattr(schedule, "_hardware_waveform_dict"):
            self._hardware_waveform_dict = schedule._hardware_waveform_dict

        # ensure keys exist
        self["compiled_instructions"] = {}

        self.data.update(schedule.data)

    @property
    def compiled_instructions(self) -> MutableMapping[str, Resource]:
        """
        A dictionary containing compiled instructions.

        The contents of this dictionary depend on the backend it was compiled for.
        However, we assume that the general format consists of a dictionary in which
        the keys are instrument names corresponding to components added to a
        :class:`~.InstrumentCoordinator`, and the
        values are the instructions for that component.

        These values typically contain a combination of sequence files, waveform
        definitions, and parameters to configure on the instrument.
        """
        return self["compiled_instructions"]

    @classmethod
    def is_valid(cls, object_to_be_validated: Any) -> bool:  # noqa: ANN401
        """
        Check if the contents of the object_to_be_validated are valid.

        Additionally checks if the object_to_be_validated is
        an instance of :class:`~.CompiledSchedule`.
        """
        valid_schedule = super().is_valid(object_to_be_validated)
        if valid_schedule:
            return isinstance(object_to_be_validated, CompiledSchedule)

        return False

    @property
    def hardware_timing_table(self) -> pd.io.formats.style.Styler:
        """
        Return a timing table representing all operations at the Control-hardware layer.

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
                "abs_time": lambda val: f"{val * 1e9:,.1f} ns",
                "duration": lambda val: f"{val * 1e9:,.1f} ns",
                "clock_cycle_start": lambda val: f"{val:,.1f}",
                "sample_start": lambda val: f"{val:,.1f}",
            }
        )

        return styled_hardware_timing_table

    @property
    def hardware_waveform_dict(self) -> dict[str, np.ndarray]:
        """
        Return a waveform dictionary representing all waveforms at the Control-hardware layer.

        Where the waveforms are represented as abstract waveforms in the Operations,
        this dictionary contains the numerical arrays that are uploaded to the hardware.

        This dictionary is constructed during compilation in the hardware back ends and
         optionally added to the schedule. Not all back ends support this feature.
        """
        return self._hardware_waveform_dict

    def __getstate__(self) -> dict[str, Any]:
        # Create copy because we might change data,
        # but only a shallow copy to save performance.
        state = super().__getstate__()
        state["data"] = copy(state["data"])
        state["data"]["compiled_instructions"] = copy(state["data"]["compiled_instructions"])
        compiled_instructions = state["data"]["compiled_instructions"]
        for component_key, component in self.compiled_instructions.items():
            if "acq_channels_data" in component:
                # Acquisition channels data should be generated
                # for all backends, but for now it's strictly
                # added to the instrument coordinator components.
                # We need custom serialization/deserialization here,
                # because JSON does not allow the keys of "acq_channels_data" to
                # be anything other than strings. So we store this dict as
                # a list of dicts. If "acq_channels_data" is `{k1: v1, k2: v2}`,
                # then the serialized version is
                # `[{"key": k1, "value": v1}, {"key": k2, "value": v2}]`.
                serialized_acq_channels_data = [
                    dict(key=k, value=v) for k, v in component["acq_channels_data"].items()
                ]
                compiled_instructions[component_key] = copy(compiled_instructions[component_key])
                compiled_instructions[component_key]["acq_channels_data"] = (
                    serialized_acq_channels_data
                )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # We need custom serialization/deserialization here,
        # because JSON does not allow the keys of "acq_channels_data" to
        # be anything other than strings.
        def _acq_channel_data(
            v: AcquisitionChannelData | dict,
        ) -> AcquisitionChannelData:
            return v if isinstance(v, AcquisitionChannelData) else AcquisitionChannelData(**v)

        data = state["data"]
        for component in data["compiled_instructions"].values():
            if "acq_channels_data" in component:
                deserialized_acq_channels = {
                    elem["key"]: _acq_channel_data(elem["value"])
                    for elem in component["acq_channels_data"]
                }
                component["acq_channels_data"] = deserialized_acq_channels
        super().__setstate__(state)


@dataclasses.dataclass
class AcquisitionChannelMetadata:
    """A description of the acquisition channel and it's indices."""

    acq_channel: Hashable
    """The acquisition channel given in the schedule."""
    acq_indices: list[int]
    """The indices reserved for this acquisition channel."""
    thresholded_trigger_count: ThresholdedTriggerCountMetadata | None = None
    """
    Optional metadata for ThresholdedTriggerCount. Must be filled in if the this protocol is used.
    The metadata is allowed to be different per acquisition channel.
    """

    def __getstate__(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": data,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state["data"])


@dataclasses.dataclass
class AcquisitionMetadata:
    """
    A description of the shape and type of data that a schedule will return when executed.

    .. note::

        The acquisition protocol, bin-mode and return types are assumed to be the same
        for all acquisitions in a schedule.
    """

    acq_protocol: str
    """The acquisition protocol that is used for all acquisitions in the schedule."""
    bin_mode: enums.BinMode
    """How the data is stored in the bins indexed by acq_channel and acq_index."""
    acq_return_type: type
    """The datatype returned by the individual acquisitions."""
    acq_channels_metadata: dict[int, AcquisitionChannelMetadata]
    """A dictionary mapping a numeric key, to the corresponding channel metadata."""
    repetitions: int
    """How many times the acquisition was repeated on this specific sequencer."""

    def __getstate__(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": data,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state["data"])
        self.acq_channels_metadata = {}
        for numeric_key, acq_channel_metadata in state["data"]["acq_channels_metadata"].items():
            # TODO this is ugly, but won't be needed after changing these classes to
            # pydantic models.
            thresholded_trigger_count = (
                ThresholdedTriggerCountMetadata(**acq_channel_metadata["thresholded_trigger_count"])
                if acq_channel_metadata["thresholded_trigger_count"] is not None
                else None
            )
            self.acq_channels_metadata[int(numeric_key)] = AcquisitionChannelMetadata(
                acq_channel_metadata["acq_channel"],
                acq_channel_metadata["acq_indices"],
                thresholded_trigger_count=thresholded_trigger_count,
            )

    def acq_channel_metadata_by_acq_channel_name(
        self, acq_channel: Hashable
    ) -> AcquisitionChannelMetadata:
        """Retrieve acq_channel_metadata by acq_channel."""
        for md in self.acq_channels_metadata.values():
            if md.acq_channel == acq_channel:
                return md
        else:  # noqa: PLW0120  # ruff doesn't pick up return statement
            raise KeyError(f"{acq_channel=} is not present.")
