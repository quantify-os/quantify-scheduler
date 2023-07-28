# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Common python dataclasses for multiple backends."""
import warnings
from typing import Dict, Literal, Optional, Union, List, Callable, Any

from pydantic import validator

from quantify_scheduler.structure.model import DataStructure, deserialize_function
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.structure.types import NDArray


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


class LocalOscillatorDescription(DataStructure):
    """Information needed to specify a Local Oscillator in the :class:`~.CompilationConfig`."""

    instrument_type: Literal["LocalOscillator"]
    """The field discriminator for this HardwareDescription datastructure."""
    instrument_name: Optional[str]
    """The QCoDeS instrument name corresponding to this Local Oscillator."""
    generic_icc_name: Optional[str]
    """The name of the :class:`~.GenericInstrumentCoordinatorComponent` corresponding to this Local Oscillator."""
    frequency_param: str = "frequency"
    """The QCoDeS parameter that is used to set the LO frequency."""
    power_param: str = "power"
    """The QCoDeS parameter that is used to set the LO power."""
    power: Optional[int]
    """The power setting for this Local Oscillator."""


class HardwareDescription(DataStructure):
    """Specifies a piece of hardware and its instrument-specific settings."""

    instrument_type: str
    """The instrument type."""


class Connectivity(DataStructure):
    """Connectivity between ports on the quantum device and on the control hardware."""


class HardwareCompilationConfig(DataStructure):
    """
    Information required to compile a schedule to the control-hardware layer.

    From a point of view of :ref:`sec-compilation` this information is needed
    to convert a schedule defined on a quantum-device layer to compiled instructions
    that can be executed on the control hardware.

    This datastructure defines the overall structure of a `HardwareCompilationConfig`.
    Specific hardware backends may customize fields within this structure by inheriting
    from this class, see e.g.,
    :class:`~quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig`,
    :class:`~quantify_scheduler.backends.zhinst_backend.ZIHardwareCompilationConfig`.
    """

    backend: Callable[[Schedule, Any], Schedule]
    """
    A . separated string specifying the location of the compilation backend this
    configuration is intended for.
    """
    hardware_description: Dict[str, HardwareDescription]
    """
    Datastructure describing the control hardware instruments in the setup and their
    high-level settings.
    """
    connectivity: Union[
        Connectivity, Dict
    ]  # Dict for legacy support for the old hardware config
    """
    Datastructure representing how ports on the quantum device are connected to ports
    on the control hardware.
    """
    hardware_options: Optional[HardwareOptions]
    """
    The `HardwareOptions` used in the compilation from the quantum-device layer to
    the control-hardware layer.
    """

    @validator("backend", pre=True)
    def _import_backend_if_str(
        cls, fun: Callable[[Schedule, Any], Schedule]  # noqa: N805
    ) -> Callable[[Schedule, Any], Schedule]:
        if isinstance(fun, str):
            return deserialize_function(fun)
        return fun  # type: ignore

    @validator("connectivity")
    def _latencies_in_hardware_config(cls, connectivity):  # noqa: N805
        # if connectivity contains a hardware config with latency corrections
        if isinstance(connectivity, Dict) and "latency_corrections" in connectivity:
            warnings.warn(
                "Latency corrections should be specified in the "
                "`backends.types.common.HardwareOptions` instead of "
                "the hardware configuration as of quantify-scheduler >= 0.19.0",
                FutureWarning,
            )
        return connectivity

    @validator("connectivity")
    def _distortions_in_hardware_config(cls, connectivity):  # noqa: N805
        # if connectivity contains a hardware config with distortion corrections
        if isinstance(connectivity, Dict) and "distortion_corrections" in connectivity:
            warnings.warn(
                "Distortion corrections should be specified in the "
                "`backends.types.common.HardwareOptions` instead of "
                "the hardware configuration as of quantify-scheduler >= 0.19.0",
                FutureWarning,
            )
        return connectivity
