import networkx as nx
from typing import Any, Callable, Dict, List, Optional, Union, Type
from quantify_scheduler.structure import DataStructure
from quantify_scheduler import Schedule, CompiledSchedule


class CompilationNode(DataStructure):
    """
    A datastructure containing the information required to perform a (modular)
    compilation step.

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


class CompilationBackend(nx.DiGraph):
    """
    A compilation backend is defined by a directed acyclic graph.
    In this graph, nodes represent modular compilation steps.

    Definition
    ----------
    A **quantify backend** is defined by a directed acyclic graph of functions
    that when executed fulfil the following input output requirements. The input is a
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

    def compile(self, schedule: Schedule, cfg: dict) -> CompiledSchedule:
        """
        Compile a schedule using the backend and the information provided in the config

        Parameters
        ----------
        Schedule
            The schedule to compile
        cfg
            A dictionary containing the information needed to compile the schedule.
            Nodes in this backend specify what key they need information from in this
            dictionary.
        """

        path = nx.shortest_path(self, "input", "output")
        # exclude the input and output from the path to use to compile
        for node in path[1:-1]:
            if node.config_key is not None:
                node_config = cfg[node.config_key]
                if node.config_validator is not None:
                    node_config = node.config_validator.parse_obj(node_config)
            else:
                node_config = None

            schedule = node.compilation_func(schedule=schedule, config=node_config)

        compiled_schedule = CompiledSchedule(schedule)

        return compiled_schedule


class CompilationError(RuntimeError):
    pass
