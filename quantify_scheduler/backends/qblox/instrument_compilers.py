# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler classes for Qblox backend."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from quantify_scheduler.backends.qblox import compiler_abc
from quantify_scheduler.backends.qblox.analog import (
    AnalogModuleCompiler,
    BasebandModuleCompiler,
    RFModuleCompiler,
)
from quantify_scheduler.backends.qblox.constants import (
    MAX_NUMBER_OF_INSTRUCTIONS_QCM,
    MAX_NUMBER_OF_INSTRUCTIONS_QRM,
    MAX_NUMBER_OF_INSTRUCTIONS_QTM,
    NUMBER_OF_SEQUENCERS_QCM,
    NUMBER_OF_SEQUENCERS_QRM,
    NUMBER_OF_SEQUENCERS_QTM,
)
from quantify_scheduler.backends.qblox.enums import (
    QbloxFilterConfig,
    QbloxFilterMarkerDelay,
)
from quantify_scheduler.backends.qblox.timetag import TimetagSequencerCompiler
from quantify_scheduler.backends.types.common import (
    HardwareDistortionCorrection,
)
from quantify_scheduler.backends.types.qblox import (
    BoundedParameter,
    OpInfo,
    StaticAnalogModuleProperties,
    StaticTimetagModuleProperties,
    TimetagModuleSettings,
    TimetagSequencerSettings,
)

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox import compiler_container
    from quantify_scheduler.backends.qblox_backend import (
        _ClusterCompilationConfig,
        _ClusterModuleCompilationConfig,
        _LocalOscillatorCompilationConfig,
        _SequencerCompilationConfig,
    )


class LocalOscillatorCompiler(compiler_abc.InstrumentCompiler):
    """
    Implementation of an :class:`~quantify_scheduler.backends.qblox.compiler_abc.InstrumentCompiler` that compiles for a generic LO. The main
    difference between this class and the other compiler classes is that it doesn't take
    pulses and acquisitions.


    Parameters
    ----------
    parent
        Reference to the parent container object.
    name
        QCoDeS name of the device it compiles for.
    total_play_time
        Total time execution of the schedule should go on for. This parameter is
        used to ensure that the different devices, potentially with different clock
        rates, can work in a synchronized way when performing multiple executions of
        the schedule.
    instrument_cfg
        The compiler config referring to this instrument.
    """

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        instrument_cfg: _LocalOscillatorCompilationConfig,
    ):
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self.freq_param_name = "frequency"
        self._frequency = instrument_cfg.frequency
        self.power_param_name = "power"
        self._power = instrument_cfg.hardware_description.power

    @property
    def frequency(self) -> float | None:
        """
        Getter for the frequency.

        Returns
        -------
        :
            The current frequency.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value: float):
        """
        Sets the lo frequency for this device if no frequency is specified, but raises
        an exception otherwise.

        Parameters
        ----------
        value
            The frequency to set it to.

        Raises
        ------
        ValueError
            Occurs when a frequency has been previously set and attempting to set the
            frequency to a different value than what it is currently set to. This would
            indicate an invalid configuration in the hardware mapping.
        """
        if self._frequency is not None:
            if value != self._frequency:
                raise ValueError(
                    f"Attempting to set LO {self.name} to frequency {value}, "
                    f"while it has previously already been set to "
                    f"{self._frequency}!"
                )
        self._frequency = value

    def compile(self, debug_mode, repetitions: int = 1) -> dict[str, Any] | None:
        """
        Compiles the program for the LO InstrumentCoordinator component.

        Parameters
        ----------
        debug_mode
            Debug mode can modify the compilation process,
            so that debugging of the compilation process is easier.
        repetitions
            Number of times execution the schedule is repeated.

        Returns
        -------
        :
            Dictionary containing all the information the InstrumentCoordinator
            component needs to set the parameters appropriately.
        """
        if self._frequency is None:
            return None
        return {
            f"{self.name}.{self.freq_param_name}": self._frequency,
            f"{self.name}.{self.power_param_name}": self._power,
        }


class QCMCompiler(BasebandModuleCompiler):
    """QCM specific implementation of the qblox compiler."""

    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = False  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QCM  # type: ignore
    static_hw_properties = StaticAnalogModuleProperties(  # type: ignore
        instrument_type="QCM",
        max_sequencers=NUMBER_OF_SEQUENCERS_QCM,
        max_awg_output_voltage=2.5,
        mixer_dc_offset_range=BoundedParameter(min_val=-2.5, max_val=2.5, units="V"),
        channel_name_to_connected_io_indices={
            "complex_output_0": (0, 1),
            "complex_output_1": (2, 3),
            "real_output_0": (0,),
            "real_output_1": (1,),
            "real_output_2": (2,),
            "real_output_3": (3,),
            "digital_output_0": (0,),
            "digital_output_1": (1,),
            "digital_output_2": (2,),
            "digital_output_3": (3,),
        },
    )

    def _configure_hardware_distortion_corrections(
        self, distortion_configs: dict[int, Any] | None = {}
    ):  # Unused arg to help type checker
        """Assign distortion corrections to settings of instrument compiler."""
        distortion_configs = self._get_distortion_configs_per_output()
        super()._configure_hardware_distortion_corrections(distortion_configs)

        for output in distortion_configs:
            output_settings = self._settings.distortion_corrections[output]
            marker_debug_mode_enable = distortion_configs[output][
                "marker_debug_mode_enable"
            ]
            if not isinstance(
                distortion_configs[output]["distortion_corrections"], list
            ):
                dc_list = [distortion_configs[output]["distortion_corrections"]]
            else:
                dc_list = distortion_configs[output]["distortion_corrections"]
            for dc in dc_list:
                for key in dc.model_fields_set:
                    value = getattr(dc, key)
                    for i in range(4):
                        if key == f"exp{i}_coeffs" and value is not None:
                            if (
                                len(value) != 2
                                or value[0] < 6
                                or value[1] < -1
                                or value[1] >= 1
                            ):
                                raise ValueError(
                                    "The exponential overshoot correction has two "
                                    "coefficients with ranges of [6,inf) and [-1,1)."
                                )
                            self._configure_filter(
                                getattr(output_settings, f"exp{i}"),
                                value,
                                marker_debug_mode_enable,
                            )
                    if key == "fir_coeffs" and value is not None:
                        if len(value) != 32 or np.any(value) < -2 or np.any(value) >= 2:
                            raise ValueError(
                                "The FIR filter has 32 coefficients "
                                "with a range of [-2,2) each."
                            )
                        self._configure_filter(
                            output_settings.fir, value, marker_debug_mode_enable
                        )

    def _get_distortion_configs_per_output(self):
        module_distortion_configs = {}
        corrections = self.instrument_cfg.hardware_options.distortion_corrections
        if corrections is not None:
            for portclock in corrections:
                if (
                    path := self.instrument_cfg.portclock_to_path.get(portclock, None)
                ) is not None:
                    correction_cfg = corrections[portclock]
                    # `correction_cfg` can also be a `SoftwareDistortionCorrection`
                    if isinstance(correction_cfg, (HardwareDistortionCorrection, list)):
                        _, _, output_name = path.split(".")
                        output_number = int(output_name.split("_")[-1])
                        channel_description = getattr(
                            self.instrument_cfg.hardware_description, output_name
                        )
                        marker_debug_mode_enable = (
                            channel_description.marker_debug_mode_enable
                            if channel_description is not None
                            else False
                        )
                        if module_distortion_configs.get(output_number) is None:
                            if isinstance(correction_cfg, HardwareDistortionCorrection):
                                output_number = int(output_name.split("_")[-1])
                                module_distortion_configs[output_number] = {
                                    "distortion_corrections": correction_cfg,
                                    "marker_debug_mode_enable": marker_debug_mode_enable,
                                }
                            elif isinstance(correction_cfg, list):
                                output_number = 2 * int(output_name.split("_")[-1])
                                module_distortion_configs[output_number] = {
                                    "distortion_corrections": correction_cfg[0],
                                    "marker_debug_mode_enable": marker_debug_mode_enable,
                                }
                                output_number += 1
                                module_distortion_configs[output_number] = {
                                    "distortion_corrections": correction_cfg[1],
                                    "marker_debug_mode_enable": marker_debug_mode_enable,
                                }
                        else:
                            raise ValueError(
                                f"Attempting to set distortion corrections to"
                                f"{output_name} using portclock {portclock}, while it"
                                f"has previously already been set on this output."
                            )
        return module_distortion_configs

    def _configure_filter(self, filt, coefficient, marker_debug_mode_enable):
        filt.coeffs = coefficient
        filt.config = QbloxFilterConfig.ENABLED
        if marker_debug_mode_enable:
            filt.marker_delay = QbloxFilterMarkerDelay.DELAY_COMP


class QRMCompiler(BasebandModuleCompiler):
    """QRM specific implementation of the qblox compiler."""

    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = True  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QRM  # type: ignore
    static_hw_properties = StaticAnalogModuleProperties(  # type: ignore
        instrument_type="QRM",
        max_sequencers=NUMBER_OF_SEQUENCERS_QRM,
        max_awg_output_voltage=0.5,
        mixer_dc_offset_range=BoundedParameter(min_val=-0.5, max_val=0.5, units="V"),
        channel_name_to_connected_io_indices={
            "complex_output_0": (0, 1),
            "complex_input_0": (0, 1),
            "real_output_0": (0,),
            "real_output_1": (1,),
            "real_input_0": (0,),
            "real_input_1": (1,),
            "digital_output_0": (0,),
            "digital_output_1": (1,),
            "digital_output_2": (2,),
            "digital_output_3": (3,),
        },
    )


class QCMRFCompiler(RFModuleCompiler):
    """QCM-RF specific implementation of the qblox compiler."""

    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = False  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QCM  # type: ignore
    static_hw_properties = StaticAnalogModuleProperties(  # type: ignore
        instrument_type="QCM_RF",
        max_sequencers=NUMBER_OF_SEQUENCERS_QCM,
        max_awg_output_voltage=None,
        mixer_dc_offset_range=BoundedParameter(min_val=-50, max_val=50, units="mV"),
        channel_name_to_connected_io_indices={
            "complex_output_0": (0, 1),
            "complex_output_1": (2, 3),
            "digital_output_0": (0,),
            "digital_output_1": (1,),
        },
        channel_name_to_digital_marker={
            "complex_output_0": 0b0001,
            "complex_output_1": 0b0010,
        },
        default_marker=0b0011,
    )


class QRMRFCompiler(RFModuleCompiler):
    """QRM-RF specific implementation of the qblox compiler."""

    # Ignore pyright because a "static property" does not exist (in the standard library).
    supports_acquisition = True  # type: ignore
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QRM  # type: ignore
    static_hw_properties = StaticAnalogModuleProperties(  # type: ignore
        instrument_type="QRM_RF",
        max_sequencers=NUMBER_OF_SEQUENCERS_QRM,
        max_awg_output_voltage=None,
        mixer_dc_offset_range=BoundedParameter(min_val=-50, max_val=50, units="mV"),
        channel_name_to_connected_io_indices={
            "complex_output_0": (0, 1),
            "complex_input_0": (0, 1),
            "digital_output_0": (0,),
            "digital_output_1": (1,),
        },
        default_marker=0b0011,
    )


class QTMCompiler(compiler_abc.ClusterModuleCompiler):
    """
    QTM specific implementation of the qblox compiler.

    Parameters
    ----------
    parent: :class:`quantify_scheduler.backends.qblox.instrument_compilers.ClusterCompiler`
        Reference to the parent object.
    name
        Name of the `QCoDeS` instrument this compiler object corresponds to.
    total_play_time
        Total time execution of the schedule should go on for. This parameter is
        used to ensure that the different devices, potentially with different clock
        rates, can work in a synchronized way when performing multiple executions of
        the schedule.
    instrument_cfg
        The instrument compilation config referring to this device.
    """

    def __init__(
        self,
        parent: ClusterCompiler,
        name: str,
        total_play_time: float,
        instrument_cfg: _ClusterModuleCompilationConfig,
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self.sequencers: dict[str, TimetagSequencerCompiler] = {}

        self._settings: TimetagModuleSettings = (  # type: ignore
            TimetagModuleSettings.extract_settings_from_mapping(instrument_cfg)
        )

    @property
    def max_number_of_instructions(self) -> int:
        """The maximum number of Q1ASM instructions supported by this module type."""
        return MAX_NUMBER_OF_INSTRUCTIONS_QTM

    @property
    def supports_acquisition(self) -> bool:
        """Specifies whether the device can perform acquisitions."""
        return True

    @property
    def static_hw_properties(self) -> StaticTimetagModuleProperties:
        """
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
        """
        return StaticTimetagModuleProperties(
            instrument_type="QTM",
            max_sequencers=NUMBER_OF_SEQUENCERS_QTM,
            channel_name_to_connected_io_indices={
                f"digital_{io}_{idx}": (idx,)
                for io in ("input", "output")
                for idx in range(8)
            },
        )

    def _construct_sequencer_compiler(
        self,
        index: int,  # noqa: ARG002 ignore unused argument
        sequencer_cfg: _SequencerCompilationConfig,
    ) -> TimetagSequencerCompiler:
        def get_index_from_channel_name() -> int:
            """
            Get the sequencer index.

            The QTM has no channel map yet, so the sequencer index = the channel index,
            and there is always only one channel index.
            """
            input_idx = self.static_hw_properties._get_connected_input_indices(
                channel_name
            )
            output_idx = self.static_hw_properties._get_connected_output_indices(
                channel_name
            )
            if len(input_idx) > 0:
                return input_idx[0]

            # If it's not an input channel, it must be an output channel.
            return output_idx[0]

        channel_name = sequencer_cfg.channel_name
        settings = TimetagSequencerSettings.initialize_from_config_dict(
            sequencer_cfg=sequencer_cfg,
            channel_name=channel_name,
            connected_output_indices=self.static_hw_properties._get_connected_output_indices(
                channel_name
            ),
            connected_input_indices=self.static_hw_properties._get_connected_input_indices(
                channel_name
            ),
        )
        return TimetagSequencerCompiler(
            parent=self,
            index=get_index_from_channel_name(),
            static_hw_properties=self.static_hw_properties,
            settings=settings,
            sequencer_cfg=sequencer_cfg,
        )

    def prepare(self) -> None:
        """
        Performs the logic needed before being able to start the compilation. In effect,
        this means assigning the pulses and acquisitions to the sequencers and
        calculating the relevant frequencies in case an external local oscillator is
        used.
        """
        self._construct_all_sequencer_compilers()
        self.distribute_data()
        for seq in self.sequencers.values():
            seq.prepare()


class ClusterCompiler(compiler_abc.InstrumentCompiler):
    """
    Compiler class for a Qblox cluster.

    Parameters
    ----------
    parent
        Reference to the parent object.
    name
        Name of the `QCoDeS` instrument this compiler object corresponds to.
    total_play_time
        Total time execution of the schedule should go on for.
    instrument_cfg
        The instrument compiler config referring to this device.
    """

    compiler_classes: dict[str, type] = {
        "QCM": QCMCompiler,
        "QRM": QRMCompiler,
        "QCM_RF": QCMRFCompiler,
        "QRM_RF": QRMRFCompiler,
        "QTM": QTMCompiler,
    }
    """References to the individual module compiler classes that can be used by the
    cluster."""

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        instrument_cfg: _ClusterCompilationConfig,
    ):
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self.instrument_cfg: _ClusterCompilationConfig  # Help typechecker
        self._op_infos: dict[tuple[str, str], list[OpInfo]] = defaultdict(list)
        self.instrument_compilers = self.construct_module_compilers()
        self.portclock_to_path = instrument_cfg.portclock_to_path

    def add_op_info(self, port: str, clock: str, op_info: OpInfo) -> None:
        """
        Assigns a certain pulse or acquisition to this device.

        Parameters
        ----------
        port
            The port this waveform is sent to (or acquired from).
        clock
            The clock for modulation of the pulse or acquisition. Can be a BasebandClock.
        op_info
            Data structure containing all the information regarding this specific
            pulse or acquisition operation.
        """
        self._op_infos[(port, clock)].append(op_info)

    def construct_module_compilers(self) -> dict[str, AnalogModuleCompiler]:
        """
        Constructs the compilers for the modules inside the cluster.

        Returns
        -------
        :
            A dictionary with the name of the module as key and the value its
            compiler.
        """
        module_compilers = {}
        module_configs = self.instrument_cfg._extract_module_compilation_configs()

        for module_idx, cfg in module_configs.items():
            module_name = f"{self.name}_module{module_idx}"
            compiler_type: type = self.compiler_classes[
                cfg.hardware_description.instrument_type
            ]

            module_compilers[module_name] = compiler_type(
                parent=self,
                name=module_name,
                total_play_time=self.total_play_time,
                instrument_cfg=cfg,
            )
        return module_compilers

    def prepare(self) -> None:
        """Prepares the instrument compiler for compilation by assigning the data."""
        self.distribute_data()
        for compiler in self.instrument_compilers.values():
            compiler.prepare()

    def distribute_data(self) -> None:
        """
        Distributes the pulses and acquisitions assigned to the cluster over the
        individual module compilers.
        """
        for compiler in self.instrument_compilers.values():
            for portclock in compiler.portclocks:
                port, clock = portclock.split("-")
                portclock_tuple = (port, clock)
                if portclock_tuple in self._op_infos:
                    for pulse in self._op_infos[portclock_tuple]:
                        compiler.add_op_info(port, clock, pulse)

    def compile(self, debug_mode: bool, repetitions: int = 1) -> dict[str, Any]:
        """
        Performs the compilation.

        Parameters
        ----------
        debug_mode
            Debug mode can modify the compilation process,
            so that debugging of the compilation process is easier.
        repetitions
            Amount of times to repeat execution of the schedule.

        Returns
        -------
        :
            The part of the compiled instructions relevant for this instrument.
        """
        program = {}
        program["settings"] = {
            "reference_source": self.instrument_cfg.hardware_description.ref
        }

        sequence_to_file = self.instrument_cfg.hardware_description.sequence_to_file

        for compiler in self.instrument_compilers.values():
            instrument_program = compiler.compile(
                repetitions=repetitions,
                sequence_to_file=sequence_to_file,
                debug_mode=debug_mode,
            )
            if instrument_program is not None and len(instrument_program) > 0:
                program[compiler.name] = instrument_program

        return program
