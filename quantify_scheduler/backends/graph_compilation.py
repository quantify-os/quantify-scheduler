# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Graph compilation backend of quantify-scheduler."""
from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import validator

from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule
from quantify_scheduler.structure.model import (
    DataStructure,
    deserialize_class,
    deserialize_function,
)

if TYPE_CHECKING:
    from quantify_scheduler.device_under_test.quantum_device import QuantumDevice


class CompilationError(RuntimeError):
    """Custom exception class for failures in compilation of quantify schedules."""


# pylint: disable=too-few-public-methods
class SimpleNodeConfig(DataStructure):
    """
    Datastructure specifying the structure of a simple compiler pass config.

    See also :class:`~.SimpleNode`.
    """

    name: str
    """The name of the compilation pass."""
    compilation_func: Callable[[Schedule, Any], Schedule]
    """
    The function to perform the compilation pass as an
    importable string (e.g., "package_name.my_module.function_name").
    """

    @validator("compilation_func", pre=True)
    def _import_compilation_func_if_str(
        cls, fun: Callable[[Schedule, Any], Schedule]  # noqa: N805
    ) -> Callable[[Schedule, Any], Schedule]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore


class OperationCompilationConfig(DataStructure):
    """
    Information required to compile an individual operation to the quantum-device layer.

    From a point of view of :ref:`sec-compilation` this information is needed
    to convert an operation defined on a quantum-circuit layer to an operation
    defined on a quantum-device layer.
    """

    factory_func: Callable[..., Operation]
    """
    A callable designating a factory function used to create the representation
    of the operation at the quantum-device level.
    """
    factory_kwargs: Dict[str, Any]
    """
    A dictionary containing the keyword arguments and corresponding values to use
    when creating the operation by evaluating the factory function.
    """
    gate_info_factory_kwargs: Optional[List[str]]
    """
    A list of keyword arguments of the factory function for which the value must
    be retrieved from the `gate_info` of the operation.
    """

    @validator("factory_func", pre=True)
    def _import_factory_func_if_str(
        cls, fun: Union[str, Callable[..., Operation]]  # noqa: N805
    ) -> Callable[..., Operation]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore


# pylint: disable=line-too-long
class DeviceCompilationConfig(DataStructure):
    """
    Information required to compile a schedule to the quantum-device layer.

    From a point of view of :ref:`sec-compilation` this information is needed
    to convert a schedule defined on a quantum-circuit layer to a schedule
    defined on a quantum-device layer.

    .. admonition:: Examples
        :class: dropdown

        The DeviceCompilationConfig is structured such that it should allow the
        specification of the circuit-to-device compilation for many different qubit
        platforms.
        Here we show a basic configuration for a two-transmon quantum device.
        In this example, the DeviceCompilationConfig is created by parsing a dictionary
        containing the relevant information.

        .. important::

            Although it is possible to manually create a configuration using
            dictionaries, this is not recommended. The
            :class:`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice`
            is responsible for managing and generating configuration files.

        .. jupyter-execute::

            import pprint
            from quantify_scheduler.backends.graph_compilation import (
                DeviceCompilationConfig
            )
            from quantify_scheduler.schemas.examples.device_example_cfgs import (
                example_transmon_cfg
            )

            pprint.pprint(example_transmon_cfg)


        The dictionary can be parsed using the :code:`parse_obj` method.

        .. jupyter-execute::

            device_cfg = DeviceCompilationConfig.parse_obj(example_transmon_cfg)
            device_cfg
    """

    backend: Callable[[Schedule, Any], Schedule]
    """
    A . separated string specifying the location of the compilation backend this
    configuration is intended for e.g.,
    :code:`"quantify_scheduler.backends.circuit_to_device.compile_circuit_to_device"`.
    """
    clocks: Dict[str, float]
    """
    A dictionary specifying the clock frequencies available on the device e.g.,
    :code:`{"q0.01": 6.123e9}`.
    """
    elements: Dict[str, Dict[str, OperationCompilationConfig]]
    """
    A dictionary specifying the elements on the device, what operations can be
    applied to them and how to compile these.
    """
    edges: Dict[str, Dict[str, OperationCompilationConfig]]
    """
    A dictionary specifying the edges, links between elements on the device to which
    operations can be applied, and the operations that can be applied to them and how
    to compile these.
    """

    @validator("backend", pre=True)
    def _import_backend_if_str(
        cls, fun: Callable[[Schedule, Any], Schedule]  # noqa: N805
    ) -> Callable[[Schedule, Any], Schedule]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore


class LatencyCorrection(float):
    """
    Latency correction in seconds for a port-clock combination.

    Positive values delay the operations on the corresponding port-clock combination,
    while negative values shift the operation backwards in time with respect to other
    operations in the schedule.

    .. note::

        If the port-clock combination of a signal is not specified in the corrections,
        it is set to zero in compilation. The minimum correction over all port-clock
        combinations is then subtracted to allow for negative latency corrections and to
        ensure minimal wait time (see
        :meth:`~quantify_scheduler.backends.corrections.determine_relative_latency_corrections`).

    .. admonition:: Example
        :class: dropdown

        Let's say we have specified two latency corrections in the CompilationConfig:

        .. code-block:: python

            compilation_config.hardware_options.latency_corrections = {
                "q0:res-q0.ro": LatencyCorrection(-20e-9),
                "q0:mw-q0.01": LatencyCorrection(120e9),
            }

        In this case, all operations on port ``"q0:mw"`` and clock ``"q0.01"`` will
        be delayed by 140 ns with respect to operations on port ``"q0:res"`` and
        clock ``"q0.ro"``.
    """


class DistortionCorrection(DataStructure):
    """Distortion correction information for a port-clock combination."""

    filter_func: str
    """The function applied to the waveforms."""
    input_var_name: str
    """The argument to which the waveforms will be passed in the filter_func."""
    kwargs: Dict[str, Union[List, NDArray]]
    """The keyword arguments that are passed to the filter_func."""
    clipping_values: Optional[List]
    """
    The optional boundaries to which the corrected pulses will be clipped,
    upon exceeding.


    .. admonition:: Example
        :class: dropdown

        .. code-block:: python

            compilation_config.hardware_options.distortion_corrections = {
                "q0:fl-cl0.baseband": DistortionCorrection(
                    filter_func = "scipy.signal.lfilter",
                    input_var_name = "x",
                    kwargs = {
                        "b": [0, 0.25, 0.5],
                        "a": [1]
                    },
                    clipping_values = [-2.5, 2.5]
                )
            }
    """

    class Config:  # noqa: D106
        arbitrary_types_allowed = True
        # This is needed because NDArray does not have a validator.


class ModulationFrequencies(DataStructure):
    """
    Modulation frequencies for a port-clock combination.

    .. admonition:: Example
        :class: dropdown

        .. code-block:: python

            compilation_config.hardware_options.modulation_frequencies = {
                "q0:res-q0.ro": ModulationFrequencies(
                    interm_freq = None,
                    lo_freq = 6e9,
                )
            }
    """

    interm_freq: Optional[float]
    """The intermodulation frequency (IF) used for this port-clock combination."""
    lo_freq: Optional[float]
    """The local oscillator frequency (LO) used for this port-clock combination."""


class MixerCorrections(DataStructure):
    """
    Mixer corrections for a port-clock combination.

    .. admonition:: Example
        :class: dropdown

        .. code-block:: python

            compilation_config.hardware_options.mixer_corrections = {
                "q0:mw-q0.01": MixerCorrections(
                    dc_offset_i = -0.0542,
                    dc_offset_q = -0.0328,
                    amp_ratio = 0.95,
                    phase_error_deg= 0.07,
                )
            }
    """

    dc_offset_i: Optional[float]
    """The DC offset on the I channel used for this port-clock combination."""
    dc_offset_q: Optional[float]
    """The DC offset on the Q channel used for this port-clock combination."""
    amp_ratio: Optional[float]
    """The mixer gain ratio used for this port-clock combination."""
    phase_error: Optional[float]
    """The mixer phase error used for this port-clock combination."""


class HardwareOptions(DataStructure):
    """
    Datastructure containing the hardware options for each port-clock combination.

    .. admonition:: Examples
        :class: dropdown

        Here, the HardwareOptions datastructure is created by parsing a
        dictionary containing the relevant information.

        .. jupyter-execute::

            import pprint
            from quantify_scheduler.backends.graph_compilation import (
                HardwareOptions
            )
            from quantify_scheduler.schemas.examples.utils import (
                load_json_example_scheme
            )

        Example for the Qblox backend:

        .. jupyter-execute::

            qblox_hw_options_dict=load_json_example_scheme("qblox_hardware_options.json")
            pprint.pprint(qblox_hw_options_dict)

        The dictionary can be parsed using the :code:`parse_obj` method.

        .. jupyter-execute::

            qblox_hw_options = HardwareOptions.parse_obj(qblox_hw_options_dict)
            qblox_hw_options

        For the Zurich Instruments backend:

        .. jupyter-execute::

            zi_hw_options_dict=load_json_example_scheme("zhinst_hardware_options.json")
            pprint.pprint(zi_hw_options_dict)
            zi_hw_options = HardwareOptions.parse_obj(zi_hw_options_dict)
            zi_hw_options
    """

    latency_corrections: Optional[Dict[str, LatencyCorrection]]
    """
    Dictionary containing the latency corrections (values) that should be applied
    to operations on a certain port-clock combination (keys).
    """
    distortion_corrections: Optional[Dict[str, DistortionCorrection]]
    """
    Dictionary containing the distortion corrections (values) that should be applied
    to waveforms on a certain port-clock combination (keys).
    """
    modulation_frequencies: Optional[Dict[str, ModulationFrequencies]]
    """
    Dictionary containing the modulation frequencies (values) that should be used
    for signals on a certain port-clock combination (keys).
    """
    mixer_corrections: Optional[Dict[str, MixerCorrections]]
    """
    Dictionary containing the mixer corrections (values) that should be used
    for signals on a certain port-clock combination (keys).
    """


class Connectivity(DataStructure):
    """
    Connectivity between the control hardware and port-clock combinations.

    Describes how the instruments are connected to port-clock combinations on the
    quantum device.
    """


# pylint: disable=too-few-public-methods
class CompilationConfig(DataStructure):
    """
    Base class for a compilation config.

    Subclassing is generally required to create useful compilation configs, here extra
    fields can be defined.
    """

    name: str
    """The name of the compiler."""
    version: str = "v0.4"
    """The version of the `CompilationConfig` to facilitate backwards compatibility."""
    backend: Type[QuantifyCompiler]
    """A reference string to the `QuantifyCompiler` class used in the compilation."""
    device_compilation_config: Optional[Union[DeviceCompilationConfig, Dict]] = None
    """
    The `DeviceCompilationConfig` used in the compilation from the quantum-circuit
    layer to the quantum-device layer.
    """
    hardware_options: Optional[HardwareOptions] = None
    """
    The `HardwareOptions` used in the compilation from the quantum-device layer to
    the control-hardware layer.
    """
    connectivity: Optional[Union[Connectivity, Dict]] = None
    """
    Datastructure representing how the port-clocks on the quantum device are
    connected to the control hardware.
    """
    # Dicts for legacy support for the old hardware config and device config

    @validator("backend", pre=True)
    def _import_backend_if_str(
        cls, class_: Union[Type[QuantifyCompiler], str]  # noqa: N805
    ) -> Type[QuantifyCompiler]:
        if isinstance(class_, str):
            return deserialize_class(class_)
        return class_  # type: ignore

    @validator("connectivity")
    def _latencies_in_hardware_config(cls, connectivity):  # noqa: N805
        # if connectivity contains a hardware config with latency corrections
        if isinstance(connectivity, Dict) and "latency_corrections" in connectivity:
            warnings.warn(
                "Latency corrections should be specified in the "
                "`backends.graph_compilation.HardwareOptions` instead of "
                "the hardware configuration as of quantify-scheduler >= 0.15.0",
                FutureWarning,
            )
        return connectivity

    @validator("connectivity")
    def _distortions_in_hardware_config(cls, connectivity):  # noqa: N805
        # if connectivity contains a hardware config with distortion corrections
        if isinstance(connectivity, Dict) and "distortion_corrections" in connectivity:
            warnings.warn(
                "Distortion corrections should be specified in the "
                "`backends.graph_compilation.HardwareOptions` instead of "
                "the hardware configuration as of quantify-scheduler >= 0.15.0",
                FutureWarning,
            )
        return connectivity


class CompilationNode:
    """A node representing a compiler pass."""

    def __init__(self, name: str):
        """
        Initialize a node representing a compiler pass.

        .. note::

            To compile, the :meth:`~.CompilationNode.compile` method should be used.

        Parameters
        ----------
        name
            The name of the node. Should be unique if it is added to a (larger)
            compilation
            graph.
        """
        self.name = name

    # used as the key in a networkx graph so we like this to be a simple string.
    def __repr__(self):  # noqa: D105
        return self.name

    # used as a label when visualizing using networkx
    def __str__(self):  # noqa: D105
        return self.name

    def _compilation_func(
        self, schedule: Union[Schedule, DataStructure], config: DataStructure
    ) -> Union[Schedule, DataStructure]:
        """
        Private compilation method of this CompilationNode.

        It should be completely stateless whenever inheriting from the CompilationNode,
        this is the object that should be modified.
        """
        # note that for linear/serial compilation graphs, the input and output is always
        # a Schedule class but for more advanced compilers, a graph might want to do
        # several steps in parallel. For this reason the base class supports a more
        # relaxed Union of types as the type hint.
        # How this Datastructure is allowed to look like depends
        # on https://gitlab.com/quantify-os/quantify-scheduler/-/issues/311

        raise NotImplementedError

    def compile(
        self, schedule: Union[Schedule, DataStructure], config: DataStructure
    ) -> Union[Schedule, DataStructure]:
        """
        Execute a compilation pass.

        This method takes a :class:`~.Schedule` and returns a new (updated)
        :class:`~.Schedule` using the information provided in the config.
        """
        # this is the public facing compile method.
        # it wraps around the self._compilation_func, but also contains the common logic
        # to support (planned) features like caching and parallel evaluation.

        # classes inheriting from this node should overwrite the _compilation_func and
        # not the public facing compile.
        return self._compilation_func(schedule=schedule, config=config)


# pylint: disable=too-few-public-methods
class SimpleNode(CompilationNode):
    """A node representing a single compilation pass."""

    def __init__(self, name: str, compilation_func: Callable):
        """
        Initialize a node representing a single compilation pass.

        .. note::

            To compile, the :meth:`~.CompilationNode.compile` method should be used.

        Parameters
        ----------
        name
            The name of the node. Should be unique if it is added to a (larger)
            compilation graph.
        compilation_func
            A Callable that will be wrapped in this object. A compilation function
            should take the intermediate representation (commonly :class:`~.Schedule`)
            and a config as inputs and returns a new (modified) intermediate
            representation.
        """
        super().__init__(name=name)
        self.compilation_func = compilation_func

    def _compilation_func(
        self, schedule: Schedule, config: Union[DataStructure, dict]
    ) -> Schedule:
        # note that in contrast to the CompilationNode parent class, the compilation
        # function has a much stricter type hint as this is for use in a SerialCompiler
        # which constitutes a linear graph.

        # note, the type hint indicates both datastructures and dicts as valid configs.
        # In the future we should only support DataStructures for the compiler options
        # to have stricter typing and error handling. Dict is for legacy support.
        return self.compilation_func(schedule=schedule, config=config)


# pylint: disable=abstract-method
class QuantifyCompiler(CompilationNode):
    """
    A compiler for quantify :class:`~.Schedule` s.

    The compiler defines a directed acyclic graph containing
    :class:`~.CompilationNode` s. In this graph, nodes represent
    modular compilation passes.
    """

    def __init__(self, name, quantum_device: Optional[QuantumDevice] = None) -> None:
        """
        Initialize a QuantifyCompiler for quantify :class:`~.Schedule` s.

        Parameters
        ----------
        name
            name of the compiler instance
        quantum_device
            quantum_device from which a :class:`~.CompilationConfig` will be generated
            if None is provided for the compile step
        """
        super().__init__(name=name)

        # current implementations use networkx directed graph to store the task graph
        # that is (typically) determined at compile time. It's fine for subclasses
        # to specify a different type for this datastructure as long as the public
        # interfaces are the same.
        self._task_graph: nx.DiGraph = None

        self._input_node = None
        self._ouput_node = None

        self.quantum_device = quantum_device

    def compile(
        self, schedule: Schedule, config: Optional[CompilationConfig] = None
    ) -> CompiledSchedule:
        """
        Compile a :class:`~.Schedule` using the information provided in the config.

        Parameters
        ----------
        schedule
            the schedule to compile.
        config
            describing the information required to compile the schedule.
            If not specified, self.quantum_device will be used to generate
            the config.

        Returns
        -------
        CompiledSchedule:
            a compiled schedule containing the compiled instructions suitable
            for execution on a (hardware) backend.

        """
        # this is the public facing compile method.
        # it wraps around the self._compilation_func, but also contains the common logic
        # to support (planned) features like caching and parallel evaluation.

        # classes inheriting from this node should overwrite the _compilation_func and
        # not the public facing compile.
        if config is None:
            if self.quantum_device is None:
                raise RuntimeError("Either quantum_device or config must be specified")
            config = self.quantum_device.generate_compilation_config()
        return self._compilation_func(schedule=schedule, config=config)

    @property
    def input_node(self):
        """
        Node designated as the default input for compilation.

        If not specified will return None.
        """
        return self._input_node

    @property
    def output_node(self):
        """
        Node designated as the default output for compilation.

        If not specified will return None.
        """
        return self._ouput_node

    def construct_graph(self, config: CompilationConfig):
        """Construct the compilation graph based on a provided config."""
        raise NotImplementedError

    def draw(
        self, ax: Axes = None, figsize: Tuple[float, float] = (20, 10), **options
    ) -> Axes:
        """
        Draws the graph defined by this backend using matplotlib.

        Will attempt to position the nodes using the "dot" algorithm for directed
        acyclic graphs from graphviz if available.
        See https://pygraphviz.github.io/documentation/stable/install.html for
        installation instructions of pygraphviz and graphviz.

        If not available will use the Kamada Kawai positioning algorithm.


        Parameters
        ----------
        ax
            Matplotlib axis to plot the figure on
        figsize
            Optional figure size, defaults to something slightly larger that fits the
            size of the nodes.
        options
            optional keyword arguments that are passed to
            :code:`networkx.draw_networkx`.

        """
        if self._task_graph is None:
            raise RuntimeError(
                "Task graph has not been initialized. Consider compiling a Schedule "
                "using .compile or calling .construct_graph"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.set_title(self.name)
        options_dict = {
            "font_size": 10,
            "node_size": 2200,
            "node_color": "white",
            "edgecolors": "C0",
        }
        options_dict.update(options)

        # attempt to use "dot" layout from graphviz.
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self._task_graph, prog="dot")
        except ImportError:
            pos = nx.kamada_kawai_layout(self._task_graph)

        nx.draw_networkx(self._task_graph, pos=pos, ax=ax, **options_dict)
        ax.set_axis_off()

        return ax


class SerialCompiler(QuantifyCompiler):
    """A compiler that executes compilation passes sequentially."""

    def construct_graph(self, config: SerialCompilationConfig):
        """
        Construct the compilation graph based on a provided config.

        For a serial backend, it is just a list of compilation passes.
        """
        if self._task_graph is None:
            self._task_graph = nx.DiGraph(name=self.name)

        # removes any old pre-existing graph removing any statefulness
        # this should be removed in the future once we want to support features
        # like caching and visualization of compilation errors.
        self._task_graph.clear()

        for i, compilation_pass in enumerate(config.compilation_passes):
            node = SimpleNode(
                name=compilation_pass.name,
                compilation_func=compilation_pass.compilation_func,
            )
            # the first node is a bit special as no edge can be added
            if i == 0:
                self._input_node = node
            else:
                self._task_graph.add_edge(last_added_node, node)  # noqa: F821
            last_added_node = node  # noqa: F841

        self._ouput_node = node

    def _compilation_func(
        self, schedule: Schedule, config: SerialCompilationConfig
    ) -> CompiledSchedule:
        """
        Compile a schedule using the backend and the information provided in the config.

        Parameters
        ----------
        schedule
            The schedule to compile.
        config
            A dictionary containing the information needed to compile the schedule.
            Nodes in this compiler specify what key they need information from in this
            dictionary.
        """
        self.construct_graph(config=config)
        # if there is only 1 node there is no shortest_path defined
        if self.input_node == self.output_node:
            path = [self.input_node]
        else:
            try:
                path = nx.shortest_path(
                    self._task_graph, self.input_node, self.output_node
                )
            except nx.exception.NetworkXNoPath as e:
                raise CompilationError(
                    "No path between the input and output nodes"
                ) from e

        # exclude the input and output from the path to use to compile
        for node in path:
            schedule = node.compile(
                schedule=schedule,
                config=config,
            )

        # mark the schedule as "Compiled" before returning at the final step.
        # in the future CompiledSchedule will likely become an attribute of a
        # single Schedule class, see
        # also https://gitlab.com/quantify-os/quantify-scheduler/-/issues/311
        return CompiledSchedule(schedule)


# pylint: disable=too-few-public-methods
class SerialCompilationConfig(CompilationConfig):
    """
    A compilation config for a simple serial compiler.

    Specifies compilation as a list of compilation passes.
    """

    backend: Type[SerialCompiler] = SerialCompiler
    compilation_passes: List[SimpleNodeConfig]

    @validator("backend", pre=True)
    def _import_backend_if_str(
        cls, class_: Union[Type[SerialCompiler], str]  # noqa: N805
    ) -> Type[SerialCompiler]:
        if isinstance(class_, str):
            return deserialize_class(class_)
        return class_  # type: ignore
