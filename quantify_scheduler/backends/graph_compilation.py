# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Graph compilation backend of quantify-scheduler."""
from __future__ import annotations

from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from pydantic import Field, field_serializer, field_validator

from quantify_scheduler.backends.types.common import HardwareCompilationConfig
from quantify_scheduler.helpers.importers import export_python_object_to_path_string
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

    @field_serializer("compilation_func")
    def _serialize_compilation_func(self, v):
        return export_python_object_to_path_string(v)

    @field_validator("compilation_func", mode="before")
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

    factory_func: Callable[..., Union[Operation, Schedule]]
    """
    A callable designating a factory function used to create the representation
    of the operation at the quantum-device level.
    """
    factory_kwargs: Dict[str, Any]
    """
    A dictionary containing the keyword arguments and corresponding values to use
    when creating the operation by evaluating the factory function.
    """
    gate_info_factory_kwargs: Optional[List[str]] = None
    """
    A list of keyword arguments of the factory function for which the value must
    be retrieved from the ``gate_info`` of the operation.
    """

    @field_serializer("factory_func")
    def _serialize_factory_func(self, v):
        return export_python_object_to_path_string(v)

    @field_validator("factory_func", mode="before")
    def _import_factory_func_if_str(
        cls, fun: Union[str, Callable[..., Operation]]  # noqa: N805
    ) -> Callable[..., Operation]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore


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


        The dictionary can be parsed using the :code:`model_validate` method.

        .. jupyter-execute::

            device_cfg = DeviceCompilationConfig.model_validate(example_transmon_cfg)
            device_cfg
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
    scheduling_strategy: Literal["asap", "alap"] = "asap"
    """
    The scheduling strategy used when determining the absolute timing of each
    operation of the schedule.
    """
    compilation_passes: List[SimpleNodeConfig] = Field(
        default=[
            {
                "name": "circuit_to_device",
                "compilation_func": "quantify_scheduler.backends.circuit_to_device."
                + "compile_circuit_to_device_with_config_validation",
            },
            {
                "name": "set_pulse_and_acquisition_clock",
                "compilation_func": "quantify_scheduler.backends.circuit_to_device."
                + "set_pulse_and_acquisition_clock",
            },
            {
                "name": "resolve_control_flow",
                "compilation_func": "quantify_scheduler.compilation.resolve_control_flow",
            },
            {
                "name": "determine_absolute_timing",
                "compilation_func": "quantify_scheduler.compilation._determine_absolute_timing",
            },
            {
                "name": "flatten",
                "compilation_func": "quantify_scheduler.compilation.flatten_schedule",
            },
        ],
        validate_default=True,
    )
    """
    The list of compilation nodes that should be called in succession to compile a
    schedule to the quantum-device layer.
    """


class CompilationConfig(DataStructure):
    """
    Base class for a compilation config.

    Subclassing is generally required to create useful compilation configs, here extra
    fields can be defined.
    """

    name: str
    """The name of the compiler."""
    version: str = "v0.6"
    """The version of the ``CompilationConfig`` to facilitate backwards compatibility."""
    keep_original_schedule: bool = True
    """
    If ``True``, the compiler will not modify the schedule argument.
    If ``False``, the compilation modifies the schedule, thereby
    making the original schedule unusable for further usage; this
    improves compilation time. Warning: if ``False``, the returned schedule
    references objects from the original schedule, please refrain from modifying
    the original schedule after compilation in this case!
    """
    backend: Type[QuantifyCompiler]
    """A reference string to the :class:`~QuantifyCompiler` class used in the compilation."""
    device_compilation_config: Optional[Union[DeviceCompilationConfig, Dict]] = None
    """
    The :class:`~DeviceCompilationConfig` used in the compilation from the quantum-circuit
    layer to the quantum-device layer.
    """
    hardware_compilation_config: Optional[HardwareCompilationConfig] = None
    """
    The ``HardwareCompilationConfig`` used in the compilation from the quantum-device
    layer to the control-hardware layer.
    """
    debug_mode: bool = False
    """
    Debug mode can modify the compilation process,
    so that debugging of the compilation process is easier.
    """

    @field_serializer("backend")
    def _serialize_backend_func(self, v):
        return export_python_object_to_path_string(v)

    @field_validator("backend", mode="before")
    def _import_backend_if_str(
        cls, class_: Union[Type[QuantifyCompiler], str]  # noqa: N805
    ) -> Type[QuantifyCompiler]:
        if isinstance(class_, str):
            return deserialize_class(class_)
        return class_  # type: ignore


class CompilationNode:
    """
    A node representing a compiler pass.

    .. note::

        To compile, the :meth:`~.CompilationNode.compile` method should be used.

    Parameters
    ----------
    name
        The name of the node. Should be unique if it is added to a (larger)
        compilation
        graph.
    """

    def __init__(self, name: str):
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


class SimpleNode(CompilationNode):
    """
    A node representing a single compilation pass.

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

    def __init__(self, name: str, compilation_func: Callable):
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


class QuantifyCompiler(CompilationNode):
    """
    A compiler for quantify :class:`~.Schedule` s.

    The compiler defines a directed acyclic graph containing
    :class:`~.CompilationNode` s. In this graph, nodes represent
    modular compilation passes.

    Parameters
    ----------
    name
        name of the compiler instance
    quantum_device
        quantum_device from which a :class:`~.CompilationConfig` will be generated
        if None is provided for the compile step
    """

    def __init__(self, name, quantum_device: Optional[QuantumDevice] = None) -> None:
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
        self,
        schedule: Schedule,
        config: Optional[CompilationConfig] = None,
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
        if config.keep_original_schedule:
            schedule = deepcopy(schedule)
        # Reset schedule compiled instructions
        schedule["compiled_instructions"] = {}
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

        compilation_passes = []
        if config.device_compilation_config is not None:
            compilation_passes.extend(
                config.device_compilation_config.compilation_passes
            )
        if config.hardware_compilation_config is not None:
            compilation_passes.extend(
                config.hardware_compilation_config.compilation_passes
            )

        for i, compilation_pass in enumerate(compilation_passes):
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
        if isinstance(schedule, CompiledSchedule):
            return schedule
        return CompiledSchedule(schedule)


class SerialCompilationConfig(CompilationConfig):
    """
    A compilation config for a simple serial compiler.

    Specifies compilation as a list of compilation passes.
    """

    backend: Type[SerialCompiler] = SerialCompiler

    @field_serializer("backend")
    def _serialize_backend_func(self, v):
        return export_python_object_to_path_string(v)

    @field_validator("backend", mode="before")
    def _import_backend_if_str(
        cls, class_: Union[Type[SerialCompiler], str]  # noqa: N805
    ) -> Type[SerialCompiler]:
        if isinstance(class_, str):
            return deserialize_class(class_)
        return class_  # type: ignore
