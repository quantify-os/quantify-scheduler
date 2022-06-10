from quantify_scheduler.backends.graph_compilation import (
    CompilationNode,
    CompilationBackend,
)


from quantify_scheduler.backends.circuit_to_device import (
    DeviceCompilationConfig,
    compile_circuit_to_device,
)
from quantify_scheduler import compilation

circuit_to_device = CompilationNode.parse_obj(
    {
        "name": "circuit_to_device",
        "compilation_func": compile_circuit_to_device,
        "config_key": "device_cfg",
        "config_validator": DeviceCompilationConfig,
    }
)

determine_absolute_timing = CompilationNode.parse_obj(
    {
        "name": "determine_absolute_timing",
        "compilation_func": compilation.determine_absolute_timing,
        "config_key": None,
        "config_validator": None,
    }
)


class DeviceCompile(CompilationBackend):
    """
    Backend for compiling a schedule from the Quantum-circuit to the
    Quantum-device layer.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

        self.add_node(circuit_to_device)
        self.add_edge("input", circuit_to_device)

        self.add_node(determine_absolute_timing)
        self.add_edge(circuit_to_device, determine_absolute_timing)

        self.add_edge(determine_absolute_timing, "output")
