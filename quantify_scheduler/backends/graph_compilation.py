# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import networkx as nx
from typing import Any, Callable, Dict, List, Optional, Union, Type
from quantify_scheduler.structure import DataStructure
from quantify_scheduler import Schedule, CompiledSchedule
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Tuple


class CompilationPass(DataStructure):
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

    name: str
    compilation_func: Callable
    config_key: Union[None, str]
    config_validator: Union[None, Type[DataStructure]]

    # a node in networkx must be a hashable object
    def __hash__(self):
        return hash(
            (self.name, self.compilation_func, self.config_key, self.config_validator)
        )

    # used as the key in a networkx graph so we like this to be a simple string.
    def __repr__(self):
        return self.name

    # used as a label when visualizing using networkx
    def __str__(self):
        return self.name

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


class CompilationBackend(nx.DiGraph):
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

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        # these two nodes are special nodes and serve as the default input and output
        # nodes of the graph when calling the compile method
        self.add_node("input")
        self.add_node("output")

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

    def compile(self, schedule: Schedule, config: dict) -> CompiledSchedule:
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
            path = nx.shortest_path(self, "input", "output")
        except nx.exception.NetworkXNoPath as e:
            raise CompilationError("No path between the input and output nodes")
        # exclude the input and output from the path to use to compile
        for node in path[1:-1]:
            schedule = node.compile(schedule=schedule, config=config)

        # mark the schedule as "Compiled" before returning at the final step.
        return CompiledSchedule(schedule)

    def compose(self, G: nx.DiGraph, H: nx.DiGraph) -> nx.DiGraph:
        """
        Create a new CompilationBackend composed of two compilation backends.
        Connects the two graphs by connected nodes connected to the output of G to
        nodes connected to the input of H.
        """
        output_nodes_G = list(G.predecessors("output"))
        G.remove_node("output")
        input_nodes_H = list(H.successors("input"))
        H.remove_node("input")

        composed_graph = nx.compose(G, H)
        for output_node in output_nodes_G:
            for input_node in input_nodes_H:
                composed_graph.add_edge(output_node, input_node)
        return composed_graph


class CompilationError(RuntimeError):
    pass
