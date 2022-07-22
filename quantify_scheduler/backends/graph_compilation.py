# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple, Callable
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from quantify_scheduler.structure import DataStructure
from quantify_scheduler import Schedule, CompiledSchedule
from quantify_scheduler.helpers.importers import import_python_object_from_string


class SimpleNodeConfig(DataStructure):
    """
    A config specifying the structure of a simple compilation config.

    Parameters
    ----------
    name: the name of the compilation pass
    compilation_func:
        the function to perform the compilation pass as an
        importable string (e.g., "package_name.my_module.function_name").
    compilation_options:
        the options passed to the compilation function along with the intermediate
        representation.

    """

    name: str
    compilation_func: str
    # N.B. custom node configs could inherit and put a stronger type check/schema
    # on options for a particular node.
    compilation_options: Dict


class CompilationConfig(DataStructure):
    """
    Base class for a CompilationConfig.
    Subclassing is generaly required to create useful configs, here extra fields can
    be defined.
    """

    name: str
    backend: str


class SerialCompilationConfig(CompilationConfig):
    """
    A compilation config for a simple serial compiler.
    Specifies compilation as a list of compilation passes.
    """

    backend: str = "quantify_scheduler.backends.graph_compilation.SerialBackend"
    compilation_passes: List[SimpleNodeConfig]


class CompilationNode:
    def __init__(self, name: str):
        """
        A node representing a compiler pass.

        Parameters
        ----------
        name:
            The name of the node. Should be unique if it is added to a (larger) compilation
            graph.

        .. note:

            Note that to compile, the :meth:`~.CompilationNode.compile` method should be
            used.
        """
        self.name = name

    # used as the key in a networkx graph so we like this to be a simple string.
    def __repr__(self):
        return self.name

    # used as a label when visualizing using networkx
    def __str__(self):
        return self.name

    def _compilation_func(self, schedule: Schedule, config: DataStructure) -> Schedule:
        # this is the private compilation method. It should be completely stateless
        # whenever inheriting from the CompilationNode, this is the object that should
        # be modified.
        raise NotImplementedError

    def compile(self, schedule: Schedule, config: DataStructure) -> Schedule:
        """
        Execute a compilation pass, taking a :class:~.Schedule` and using the
        information provided in the config to return a new (updated) :class:~.Schedule`.
        """

        # this is the public facing compile method.
        # it wraps around the self._compilation_func, but also contains the common logic
        # to support (planned) features like caching and parallel evaluation.

        # classes inheriting from this node should overwrite the _compilation_func and
        # not the public facing compile.
        return self._compilation_func(schedule=schedule, config=config)


class SimpleNode(CompilationNode):
    def __init__(self, name: str, compilation_func: Callable):
        """
        A node representing a compiler pass.

        Parameters
        ----------
        name:
            The name of the node. Should be unique if it is added to a (larger) compilation
            graph.
        compilation_func:
            A Callable that will be wrapped in this object. A compilation function should
            takes the intermediate representation (commonly a :class:`~.Schedule`) and
            a config as an input and returns a new (modified) intermediate representation.

        .. note::

            Note that to compile, the :meth:`~.CompilationNode.compile` method should be
            used.
        """

        self.name = name
        self.compilation_func = compilation_func

    def _compilation_func(
        self, schedule: Schedule, config: Union[DataStructure, dict]
    ) -> Schedule:
        # note, the type hint indicates both datastructures and dicts. In the future
        # we should only support DataStructures for the compiler options to have
        # stricter typing and error handling. Dict is for legacy support.
        return self.compilation_func(schedule, config)


class CompilationPass(CompilationNode):
    """
    A datastructure containing the information required to perform a (modular)
    compilation pass.

    Parameters
    ----------
    name
    compilation_func:
    config_key
    config_validator
    """

    def __init__(self, name, compilation_func, config_key, config_validator):
        self.name = name  #: str
        self.compilation_func = compilation_func  #: Callable
        self.config_key = config_key  #: Union[None, str]
        self.config_validator = config_validator  #: Union[None, Type[DataStructure]]

    # a node in networkx must be a hashable object
    def __hash__(self):
        return hash(
            (self.name, self.compilation_func, self.config_key, self.config_validator)
        )

    def compile(self, schedule: Schedule, config: dict) -> Schedule:
        """
        Performs the compilation pass specified by the compilation function and
        the configuration provided to this node.

        Parameters
        ----------
        Schedule
            The schedule to compile
        config
            A dictionary containing the information needed to compile the schedule.
            The `config_key` attribute of this node specifies the information to use
            from this dictionary.
        """
        if self.config_key is not None:
            node_config = config[self.config_key]
            if self.config_validator is not None:
                node_config = self.config_validator.parse_obj(node_config)
        else:
            node_config = None

        # using positional arguments as not all compilation functions have the right
        # function signature.
        # schedule = self.compilation_func(schedule=schedule, config=node_config)
        schedule = self.compilation_func(schedule, node_config)
        return schedule


class CompilationBackend(nx.DiGraph, CompilationNode):
    """
    A compilation backend defines a directed acyclic graph.
    In this graph, nodes represent modular compilation passes.

    Definition
    ----------
    A **quantify backend** defines a directed acyclic graph of functions
    that when executed fulfill the following input output requirements. The input is a
    :class:`~.Schedule` and a configuration file, and the output consists of platform
    specific hardware instructions.


    .. tip::

        If you have graphviz installed you can visualize the compilation backend
        as a directed graph using the following commands:

        .. code-block::

            import networkx as nx
            gvG = nx.nx_agraph.to_agraph(my_graph)
            gvG.draw('compilation_graph.svg', prog="dot")

    """

    def __init__(self, name, incoming_graph_data=None, **attr):
        super().__init__()
        self.name = name

        self._input_node = None
        self._ouput_node = None

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._ouput_node

    def draw(
        self, ax: Axes = None, figsize: Tuple[float, float] = (20, 10), **options
    ) -> Axes:
        """
        Draws the graph defined by this backend using matplotlib.

        Will attempt to position the nodes using the "dot" algorithm for directed
        acyclic graphs from graphviz
        if available.
        See https://pygraphviz.github.io/documentation/stable/install.html for
        installation instructions of pygraphviz and graphviz.

        If not available will use the Kamada Kawai positioning algorithm.


        Parameters
        ----------
        ax:
            Matplotlib axis to plot the figure on
        figsize:
            Optional figure size, defaults to something slightly larger that fits the
            size of the nodes.
        options:
            optional keyword arguments that are passed to
            :func:`networkx.draw_networkx`.

        """
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
            pos = nx.drawing.nx_agraph.graphviz_layout(self, prog="dot")
        except ImportError:
            pos = nx.kamada_kawai_layout(self)

        nx.draw_networkx(self, pos=pos, ax=ax, **options_dict)
        ax.set_axis_off()

        return ax


class SerialBackend(CompilationBackend):
    """
    A compilation backend that dynamically generates a graph of serial compilation
    passes upon calling the compile method.
    """

    def construct_graph(self, config: SerialCompilationConfig):
        """
        Construct the compilation graph based on a provided config.

        For a serial backend, it is just a list of compilation passes.
        """

        # removes any old pre-existing graph removing any statefulness
        # this should be removed in the future once we want to improve performance a bit
        self.clear()

        # check legacy

        for i, compilation_pass in enumerate(config.compilation_passes):
            compilation_func = import_python_object_from_string(
                compilation_pass.compilation_func
            )
            node = SimpleNode(
                name=compilation_pass.name, compilation_func=compilation_func
            )
            # the first node is a bit special as no edge can be added
            if i == 0:
                self._input_node = node
            else:
                self.add_edge(last_added_node, node)
            last_added_node = node

        self._ouput_node = node

    def _compilation_func(self, schedule: Schedule, config: dict) -> CompiledSchedule:
        """
        Compile a schedule using the backend and the information provided in the config

        Parameters
        ----------
        Schedule
            The schedule to compile
        config
            A dictionary containing the information needed to compile the schedule.
            Nodes in this backend specify what key they need information from in this
            dictionary.
        """
        try:
            path = nx.shortest_path(self, self.input_node, self.output_node)
        except nx.exception.NetworkXNoPath as e:
            raise CompilationError("No path between the input and output nodes")
        # exclude the input and output from the path to use to compile
        for node in path:
            schedule = node.compile(schedule=schedule, config=config)

        # mark the schedule as "Compiled" before returning at the final step.
        return CompiledSchedule(schedule)
