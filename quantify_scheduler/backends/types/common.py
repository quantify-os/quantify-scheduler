# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Common python dataclasses for multiple backends."""
from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
)

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.helpers.importers import export_python_object_to_path_string
from quantify_scheduler.structure.model import (
    DataStructure,
    deserialize_function,
)
from quantify_scheduler.structure.types import Graph, NDArray

if TYPE_CHECKING:
    from quantify_scheduler.backends.graph_compilation import SimpleNodeConfig
    from quantify_scheduler.schedules.schedule import Schedule

LatencyCorrection = float
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


class SoftwareDistortionCorrection(DataStructure):
    """
    Software distortion correction information for a port-clock combination.

    .. admonition:: Example
        :class: dropdown

        .. jupyter-execute::

            from quantify_scheduler.backends.types.common import (
                SoftwareDistortionCorrection
            )

            distortion_corrections = {
                "q0:fl-cl0.baseband": SoftwareDistortionCorrection(
                    filter_func="scipy.signal.lfilter",
                    input_var_name="x",
                    kwargs={
                        "b": [0, 0.25, 0.5],
                        "a": [1]
                    },
                    clipping_values=[-2.5, 2.5]
                )
            }
    """

    filter_func: str
    """The function applied to the waveforms."""
    input_var_name: str
    """The argument to which the waveforms will be passed in the filter_func."""
    kwargs: dict[str, list | NDArray]
    """The keyword arguments that are passed to the filter_func."""
    clipping_values: list | None = None
    """
    The optional boundaries to which the corrected pulses will be clipped,
    upon exceeding."""
    sampling_rate: float = 1e9
    """The sample rate of the corrected pulse, in Hz."""

    @field_validator("clipping_values")
    def _only_two_clipping_values(cls, clipping_values):
        if clipping_values and len(clipping_values) != 2:
            raise KeyError(
                f"Clipping values should contain only two values, min and max.\n"
                f"clipping_values: {clipping_values}"
            )
        return clipping_values


# @deprecated does not work with pydantic models.
class DistortionCorrection(DataStructure):
    """
    Software distortion correction information for a port-clock combination.

    Note: deprecated and to be removed after 0.23.0. Please use
    ``SoftwareDistortionCorrection`` instead.

    .. admonition:: Example
        :class: dropdown

        .. jupyter-execute::

            from quantify_scheduler.backends.types.common import (
                SoftwareDistortionCorrection
            )

            distortion_corrections = {
                "q0:fl-cl0.baseband": SoftwareDistortionCorrection(
                    filter_func="scipy.signal.lfilter",
                    input_var_name="x",
                    kwargs={
                        "b": [0, 0.25, 0.5],
                        "a": [1]
                    },
                    clipping_values=[-2.5, 2.5]
                )
            }
    """

    def __init__(self, **data) -> None:
        super().__init__(**data)
        warnings.warn(
            "DistortionCorrection is deprecated and will be removed from "
            "quantify-scheduler after 0.23.0. "
            "Please use SoftwareDistortionCorrection instead.",
            FutureWarning,
        )

    filter_func: str
    """The function applied to the waveforms."""
    input_var_name: str
    """The argument to which the waveforms will be passed in the filter_func."""
    kwargs: dict[str, list | NDArray]
    """The keyword arguments that are passed to the filter_func."""
    clipping_values: list | None = None
    """
    The optional boundaries to which the corrected pulses will be clipped,
    upon exceeding."""
    sampling_rate: float = 1e9


class HardwareDistortionCorrection(DataStructure):
    """Parent class for hardware distortion correction."""


class ModulationFrequencies(DataStructure):
    """
    Modulation frequencies for a port-clock combination.

    .. admonition:: Example
        :class: dropdown

        .. jupyter-execute::

            from quantify_scheduler.backends.types.common import (
                ModulationFrequencies
            )

            modulation_frequencies = {
                "q0:res-q0.ro": ModulationFrequencies(
                    interm_freq=None,
                    lo_freq=6e9,
                )
            }
    """

    interm_freq: float | None = None
    """The intermodulation frequency (IF) used for this port-clock combination."""
    lo_freq: float | None = None
    """The local oscillator frequency (LO) used for this port-clock combination."""


class MixerCorrections(DataStructure):
    """
    Mixer corrections for a port-clock combination.

    .. admonition:: Example
        :class: dropdown

        .. jupyter-execute::

            from quantify_scheduler.backends.types.common import (
                MixerCorrections
            )

            mixer_corrections = {
                "q0:mw-q0.01": MixerCorrections(
                    dc_offset_i = -0.0542,
                    dc_offset_q = -0.0328,
                    amp_ratio = 0.95,
                    phase_error= 0.07,
                )
            }
    """

    dc_offset_i: float = 0.0
    """The DC offset on the I channel used for this port-clock combination."""
    dc_offset_q: float = 0.0
    """The DC offset on the Q channel used for this port-clock combination."""
    amp_ratio: float = 1.0
    """The mixer gain ratio used for this port-clock combination."""
    phase_error: float = 0.0
    """The mixer phase error used for this port-clock combination."""


class HardwareOptions(DataStructure):
    """
    Datastructure containing the hardware options for each port-clock combination.

    This datastructure contains the HardwareOptions that are currently shared among
    the existing backends. Subclassing is required to add backend-specific options,
    see e.g.,
    :class:`~quantify_scheduler.backends.types.qblox.QbloxHardwareOptions`,
    :class:`~quantify_scheduler.backends.types.zhinst.ZIHardwareOptions`.
    """

    crosstalk: dict[str, dict[str, float | complex]] | None = None
    """
    Dictionary containing the crosstalk values between ports on the quantum device.
    The crosstalk values are given as a dictionary of dictionaries, where the outer
    dictionary keys are the source ports and the inner dictionary keys are the target
    ports.
    """
    latency_corrections: dict[str, LatencyCorrection] | None = None
    """
    Dictionary containing the latency corrections (values) that should be applied
    to operations on a certain port-clock combination (keys).
    """
    distortion_corrections: dict[str, SoftwareDistortionCorrection] | None = None
    """
    Dictionary containing the distortion corrections (values) that should be applied
    to waveforms on a certain port-clock combination (keys).
    """
    modulation_frequencies: dict[str, ModulationFrequencies] | None = None
    """
    Dictionary containing the modulation frequencies (values) that should be used
    for signals on a certain port-clock combination (keys).
    """
    mixer_corrections: dict[str, MixerCorrections] | None = None
    """
    Dictionary containing the mixer corrections (values) that should be used
    for signals on a certain port-clock combination (keys).
    """
    model_config = ConfigDict(
        extra="forbid",
        # ensures exceptions are raised when passing extra argument that are not
        # part of a model when initializing.
        validate_assignment=True,
        # run validation when assigning attributes
        arbitrary_types_allowed=True,
    )


class LocalOscillatorDescription(DataStructure):
    """Information needed to specify a Local Oscillator in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["LocalOscillator"]
    """The field discriminator for this HardwareDescription datastructure."""
    instrument_name: str | None = None
    """The QCoDeS instrument name corresponding to this Local Oscillator."""
    generic_icc_name: str | None = None
    """The name of the :class:`~.GenericInstrumentCoordinatorComponent`
    corresponding to this Local Oscillator."""
    frequency_param: str = "frequency"
    """The QCoDeS parameter that is used to set the LO frequency."""
    power_param: str = "power"
    """The QCoDeS parameter that is used to set the LO power."""
    power: int | None = None
    """The power setting for this Local Oscillator."""

    @field_validator("generic_icc_name")
    def _only_default_generic_icc_name(cls, generic_icc_name):
        if generic_icc_name is not None and generic_icc_name != constants.GENERIC_IC_COMPONENT_NAME:
            raise NotImplementedError(
                f"Specified name '{generic_icc_name}' as a generic instrument "
                f"coordinator component, but the Qblox backend currently only "
                f"supports using the default name "
                f"'{constants.GENERIC_IC_COMPONENT_NAME}'"
            )
        return generic_icc_name


class IQMixerDescription(DataStructure):
    """Information needed to specify an IQ Mixer in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["IQMixer"]
    """The field discriminator for this HardwareDescription datastructure."""


class OpticalModulatorDescription(DataStructure):
    """Information needed to specify an optical modulator in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["OpticalModulator"]
    """The field discriminator for this HardwareDescription datastructure."""


class HardwareDescription(DataStructure):
    """
    Specifies a piece of hardware and its instrument-specific settings.

    Each supported instrument type should have its own datastructure that inherits from
    this class.
    For examples, see :class:`~quantify_scheduler.backends.types.qblox.ClusterDescription`,
    :class:`~quantify_scheduler.backends.types.zhinst.ZIHDAWG4Description`,
    :class:`~.LocalOscillatorDescription`.

    This datastructure is used to specify the control-hardware ports that
    are included in the :class:`~.Connectivity` graph.
    """

    instrument_type: str
    """The instrument type."""


class Connectivity(DataStructure):
    """
    Connectivity between ports on the quantum device and control hardware inputs/outputs.

    The connectivity graph can be parsed from a list of edges, which are given by a
    set of two strings that each correspond to an input/output on an instrument or a port
    on the quantum device.

    .. note::
        To specify connections between more than one pair of ports at once, one can
        also specify a list of ports within the edge input (see example below, and also
        see :ref:`sec-connectivity-examples`).

    The connectivity graph can be drawn using :meth:`~.draw`, which groups the nodes
    according to the instrument name (specified by the string before the first ``"."``
    in the node name; the name is omitted for the quantum device).

    .. admonition:: Example
        :class: dropdown

        .. jupyter-execute::

            from quantify_scheduler.backends.types.common import (
                Connectivity
            )

            connectivity_dict = {
                "graph": [
                    ("awg0.channel_0", "q0:mw"),
                    ("awg0.channel_1", "q1:mw"),
                    ("rom0.channel_0", ["q0:res", "q1:res"]),
                ]
            }

            connectivity = Connectivity.model_validate(connectivity_dict)
            connectivity.draw()
    """

    graph: Graph
    """
    The connectivity graph consisting of i/o ports (nodes) on the quantum device and on
    the control hardware, and their connections (edges).
    """

    @field_validator("graph", mode="before")
    def _unroll_lists_of_ports_in_edges_input(cls, graph):
        if isinstance(graph, list):
            list_of_edges = []
            for edge_input in graph:
                ports_0 = edge_input[0]
                ports_1 = edge_input[1]
                if not isinstance(ports_0, list):
                    ports_0 = [ports_0]
                if not isinstance(ports_1, list):
                    ports_1 = [ports_1]
                list_of_edges.extend([(p0, p1) for p0 in ports_0 for p1 in ports_1])
            graph = list_of_edges
        return graph

    def draw(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (20, 10),
        **options,
    ) -> Axes:
        """
        Draw the connectivity graph using matplotlib.

        The nodes are positioned using a multipartite layout, where the nodes
        are grouped by instrument (identified by the first part of the node name).


        Parameters
        ----------
        ax
            Matplotlib axis to plot the figure on.
        figsize
            Optional figure size, defaults to something slightly larger that fits the
            size of the nodes.
        options
            optional keyword arguments that are passed to
            :code:`networkx.draw_networkx`.

        Returns
        -------
        :
            Matplotlib axis on which the figure is plotted.

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        options_dict = {
            "font_size": 10,
            "node_size": 1000,
            "node_color": "white",
            "edgecolors": "C0",
        }
        options_dict.update(options)

        # Group nodes by instrument:
        node_labels = {}
        grouped_nodes = {}
        for node in self.graph:
            if "." in node:
                node_instrument, node_port = node.split(sep=".", maxsplit=1)
                self.graph.nodes[node]["subset"] = node_instrument
            else:
                node_instrument = ""
                node_port = node
                self.graph.nodes[node]["subset"] = "quantum_device"
            if node_instrument not in grouped_nodes:
                grouped_nodes[node_instrument] = []
            grouped_nodes[node_instrument].append(node)
            node_labels[node] = node_port

        pos = nx.drawing.multipartite_layout(self.graph)

        # Draw boxes around instrument ports:
        for instrument, nodes in grouped_nodes.items():
            min_node_pos_x = min(pos[node][0] for node in nodes)
            max_node_pos_x = max(pos[node][0] for node in nodes)
            min_node_pos_y = min(pos[node][1] for node in nodes)
            max_node_pos_y = max(pos[node][1] for node in nodes)

            instrument_anchor = (min_node_pos_x - 0.05, min_node_pos_y - 0.05)  # type: ignore
            instrument_width = max_node_pos_x - min_node_pos_x + 0.1  # type: ignore
            instrument_height = max_node_pos_y - min_node_pos_y + 0.1  # type: ignore
            ax.add_patch(  # type: ignore
                Rectangle(
                    xy=instrument_anchor,
                    width=instrument_width,
                    height=instrument_height,
                    fill=False,
                    color="b",
                )
            )
            ax.text(x=min_node_pos_x, y=max_node_pos_y + 0.1, s=instrument, color="b")  # type: ignore

        nx.draw_networkx(self.graph, pos=pos, ax=ax, labels=node_labels, **options_dict)
        ax.set_axis_off()  # type: ignore

        return ax  # type: ignore


class HardwareCompilationConfig(DataStructure):
    """
    Information required to compile a schedule to the control-hardware layer.

    From a point of view of :ref:`sec-compilation` this information is needed
    to convert a schedule defined on a quantum-device layer to compiled instructions
    that can be executed on the control hardware.

    This datastructure defines the overall structure of a ``HardwareCompilationConfig``.
    Specific hardware backends should customize fields within this structure by inheriting
    from this class and specifying their own `"config_type"`, see e.g.,
    :class:`~quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`,
    :class:`~quantify_scheduler.backends.zhinst_backend.ZIHardwareCompilationConfig`.
    """

    config_type: type[HardwareCompilationConfig] = Field(
        default="quantify_scheduler.backends.types.common.HardwareCompilationConfig",
        validate_default=True,
    )
    """
    A reference to the ``HardwareCompilationConfig`` DataStructure for the backend
    that is used.
    """
    hardware_description: dict[str, HardwareDescription]
    """
    Datastructure describing the control hardware instruments in the setup and their
    high-level settings.
    """
    hardware_options: HardwareOptions
    """
    The :class:`~quantify_scheduler.backends.types.common.HardwareOptions` used in the
    compilation from the quantum-device layer to the control-hardware layer.
    """
    connectivity: Connectivity | dict  # Dict for legacy support for the old hardware config
    """
    Datastructure representing how ports on the quantum device are connected to ports
    on the control hardware.
    """
    compilation_passes: list[SimpleNodeConfig] = []
    """
    The list of compilation nodes that should be called in succession to compile a
    schedule to instructions for the control hardware.
    """

    @field_serializer("config_type")
    def _serialize_backend_func(self, v):
        return export_python_object_to_path_string(v)

    @field_validator("config_type", mode="before")
    def _import_config_type_if_str(
        cls, config_type: type[HardwareCompilationConfig]  # noqa: N805
    ) -> Callable[[Schedule, Any], Schedule]:
        if isinstance(config_type, str):
            return deserialize_function(config_type)
        return config_type  # type: ignore

    @model_validator(mode="after")
    def _check_connectivity_graph_nodes_format(self):
        if isinstance(self.connectivity, Connectivity):
            for node in self.connectivity.graph:
                if "." in node:
                    instrument_name = node.split(sep=".")[0]
                    if instrument_name not in self.hardware_description:
                        raise ValueError(
                            f"Invalid node. Instrument '{instrument_name}'"
                            f" not found in hardware description."
                        )
                    self.connectivity.graph.nodes[node]["instrument_name"] = instrument_name
                elif ":" in node:
                    self.connectivity.graph.nodes[node]["instrument_name"] = "QuantumDevice"
                else:
                    raise ValueError(
                        "Invalid node format. " "Must be 'instrument.port' or 'qubit_name:port'."
                    )
        return self

    @model_validator(mode="after")
    def _connectivity_old_style_hw_cfg_empty_hw_options_and_descriptions(self):
        if isinstance(self.connectivity, dict):
            if self.hardware_description != {}:
                raise ValueError(
                    "Hardware description must be empty "
                    "when using old-style hardware config dictionary."
                )
            default_hw_options = HardwareOptions()
            for key, hw_option in self.hardware_options:
                if hw_option != getattr(default_hw_options, key):
                    raise ValueError(
                        "Hardware options must be empty "
                        "when using old-style hardware config dictionary."
                    )
        return self
