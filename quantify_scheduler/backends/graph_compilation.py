# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
from __future__ import annotations

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
from pydantic import validator
from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule
from quantify_scheduler.structure.model import (
    DataStructure,
    deserialize_class,
    deserialize_function,
)

if TYPE_CHECKING:
    from quantify_scheduler.device_under_test.quantum_device import QuantumDevice


class CompilationError(RuntimeError):
    """
    Custom exception class for failures in compilation of quantify schedules.
    """


# pylint: disable=too-few-public-methods
class SimpleNodeConfig(DataStructure):
    """
    Datastructure specifying the structure of a simple compiler pass config
    (also see :class:`~.SimpleNode`).

    Parameters
    ----------
    name
        the name of the compilation pass
    compilation_func
        the function to perform the compilation pass as an
        importable string (e.g., "package_name.my_module.function_name").
    compilation_options
        the options passed to the compilation function along with the intermediate
        representation.
    """

    name: str
    compilation_func: Callable[[Schedule, Any], Schedule]
    # N.B. custom node configs could inherit and put a stronger type check/schema
    # on options for a particular node.
    compilation_options: Optional[Dict]

    @validator("compilation_func", pre=True)
    @classmethod
    def import_compilation_func_if_str(
        cls, fun: Callable[[Schedule, Any], Schedule]
    ) -> Callable[[Schedule, Any], Schedule]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore


class HardwareOption(DataStructure):
    """
    Base class for hardware options, such as
    :class:`~quantify_scheduler.backends.corrections.LatencyCorrections`.
    """


class Connectivity(DataStructure):
    """
    Describes how the instruments are connected to port-clock combinations on the
    quantum device.
    """


# pylint: disable=too-few-public-methods
class CompilationConfig(DataStructure):
    """
    Base class for a compilation config. Subclassing is generally required to create
    useful compilation configs, here extra fields can be defined.

    Parameters
    ----------
    name
        The name of the compiler.
    version
        The version of the `CompilationConfig` to facilitate backwards compatibility.
    backend
        A reference string to the `QuantifyCompiler` class used in the compilation.
    device_compilation_config
        The `DeviceCompilationConfig` used in the compilation from the quantum-circuit
        layer to the quantum-device layer.
    hardware_options
        A list of `HardwareOption`s used in the compilation from the quantum-device
        layer to the control-hardware layer.
    connectivity
        Datastructure representing how the port-clocks on the quantum device are
        connected to the control hardware.
    """

    name: str
    version: str = "v0.1"
    backend: Type[QuantifyCompiler]
    device_compilation_config: Optional[Union[DeviceCompilationConfig, Dict]] = None
    hardware_options: Optional[List[HardwareOption]] = None
    connectivity: Optional[Union[Connectivity, Dict]] = None
    # Dicts for legacy support for the old hardware config and device config

    @validator("backend", pre=True)
    @classmethod
    def import_backend_if_str(
        cls, class_: Union[Type[QuantifyCompiler], str]
    ) -> Type[QuantifyCompiler]:
        if isinstance(class_, str):
            return deserialize_class(class_)
        return class_  # type: ignore


class CompilationNode:
    """
    A node representing a compiler pass.
    """

    def __init__(self, name: str):
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
        self.name = name

    # used as the key in a networkx graph so we like this to be a simple string.
    def __repr__(self):
        return self.name

    # used as a label when visualizing using networkx
    def __str__(self):
        return self.name

    def _compilation_func(
        self, schedule: Union[Schedule, DataStructure], config: DataStructure
    ) -> Union[Schedule, DataStructure]:
        """
        This is the private compilation method. It should be completely stateless
        whenever inheriting from the CompilationNode, this is the object that should
        be modified.
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
        Execute a compilation pass, taking a :class:`~.Schedule` and using the
        information provided in the config to return a new (updated)
        :class:`~.Schedule`.
        """

        # this is the public facing compile method.
        # it wraps around the self._compilation_func, but also contains the common logic
        # to support (planned) features like caching and parallel evaluation.

        # classes inheriting from this node should overwrite the _compilation_func and
        # not the public facing compile.
        return self._compilation_func(schedule=schedule, config=config)


# pylint: disable=too-few-public-methods
class SimpleNode(CompilationNode):
    """
    A node representing a simple compiler pass consisting of calling a single
    compilation function.
    """

    def __init__(self, name: str, compilation_func: Callable):
        """
        A node representing a simple compiler pass consisting of calling a single
        compilation function.

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
        return self.compilation_func(schedule, config)


# pylint: disable=abstract-method
class QuantifyCompiler(CompilationNode):
    """
    A compiler for quantify :class:`~.Schedule` s.
    The compiler defines a directed acyclic graph containing
    :class:`~.CompilationNode` s. In this graph, nodes represent modular
    compilation passes.
    """

    def __init__(
        self,
        name,
        quantum_device: Optional[
            "quantify_scheduler.device_under_test.quantum_device.QuantumDevice"
        ] = None,
    ) -> None:
        """
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
            describing the information required to compile the schedule. If not specified,
            self.quantum_device will be used to generate the config.

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
        """
        Construct the compilation graph based on a provided config.
        """
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
    """
    A compiler that dynamically generates a graph of serial compilation
    passes upon calling the compile method.
    """

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
                self._task_graph.add_edge(last_added_node, node)
            last_added_node = node

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
        for i, node in enumerate(path):
            schedule = node.compile(
                schedule=schedule,
                config=config.compilation_passes[i].compilation_options,
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
    @classmethod
    def import_backend_if_str(
        cls, class_: Union[Type[SerialCompiler], str]
    ) -> Type[SerialCompiler]:
        if isinstance(class_, str):
            return deserialize_class(class_)
        return class_  # type: ignore
