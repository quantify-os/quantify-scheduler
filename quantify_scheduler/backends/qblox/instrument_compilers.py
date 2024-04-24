# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler classes for Qblox backend."""
from __future__ import annotations

from collections import abc, defaultdict
from typing import TYPE_CHECKING, Any

from quantify_scheduler.backends.qblox import compiler_abc
from quantify_scheduler.backends.qblox.constants import (
    MAX_NUMBER_OF_INSTRUCTIONS_QCM,
    MAX_NUMBER_OF_INSTRUCTIONS_QRM,
    NUMBER_OF_SEQUENCERS_QCM,
    NUMBER_OF_SEQUENCERS_QRM,
)
from quantify_scheduler.backends.types.qblox import (
    BoundedParameter,
    LOSettings,
    OpInfo,
    StaticHardwareProperties,
)

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox import compiler_container


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
        The part of the hardware mapping dict referring to this instrument.
    """

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
    ):
        def _extract_parameter(
            parameter_dict: dict[str, float | None]
        ) -> tuple[str, float | None]:
            items: abc.ItemsView = parameter_dict.items()
            return list(items)[0]

        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
        )
        self._settings = LOSettings.from_mapping(instrument_cfg)
        self.freq_param_name, self._frequency = _extract_parameter(
            self._settings.frequency
        )
        self.power_param_name, self._power = _extract_parameter(self._settings.power)

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


class QCMCompiler(compiler_abc.BasebandModuleCompiler):
    """QCM specific implementation of the qblox compiler."""

    supports_acquisition = False
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QCM
    static_hw_properties = StaticHardwareProperties(
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


class QRMCompiler(compiler_abc.BasebandModuleCompiler):
    """QRM specific implementation of the qblox compiler."""

    supports_acquisition = True
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QRM
    static_hw_properties = StaticHardwareProperties(
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


class QCMRFCompiler(compiler_abc.RFModuleCompiler):
    """QCM-RF specific implementation of the qblox compiler."""

    supports_acquisition = False
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QCM
    static_hw_properties = StaticHardwareProperties(
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


class QRMRFCompiler(compiler_abc.RFModuleCompiler):
    """QRM-RF specific implementation of the qblox compiler."""

    supports_acquisition = True
    max_number_of_instructions = MAX_NUMBER_OF_INSTRUCTIONS_QRM
    static_hw_properties = StaticHardwareProperties(
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
        The part of the hardware configuration dictionary referring to this device. This is one
        of the inner dictionaries of the overall hardware config.
    latency_corrections
        Dict containing the delays for each port-clock combination. This is
        specified in the top layer of hardware config.
    """

    compiler_classes: dict[str, type] = {
        "QCM": QCMCompiler,
        "QRM": QRMCompiler,
        "QCM_RF": QCMRFCompiler,
        "QRM_RF": QRMRFCompiler,
    }
    """References to the individual module compiler classes that can be used by the
    cluster."""

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
    ):
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
            latency_corrections=latency_corrections,
        )
        self._op_infos: dict[tuple[str, str], list[OpInfo]] = defaultdict(list)
        self.instrument_compilers = self.construct_instrument_compilers()
        self.latency_corrections = latency_corrections

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

    def construct_instrument_compilers(
        self,
    ) -> dict[str, compiler_abc.ClusterModuleCompiler]:
        """
        Constructs the compilers for the modules inside the cluster.

        Returns
        -------
        :
            A dictionary with the name of the instrument as key and the value its
            compiler.
        """
        instrument_compilers = {}
        for name, cfg in self.instrument_cfg.items():
            if not isinstance(cfg, dict):
                continue  # not an instrument definition
            if "instrument_type" not in cfg:
                raise KeyError(
                    f"Module {name} of cluster {self.name} is specified in "
                    f"the config, but does not specify an 'instrument_type'."
                    f"\n\nValid values: {self.compiler_classes.keys()}"
                )
            instrument_type: str = cfg["instrument_type"]
            if instrument_type not in self.compiler_classes:
                raise KeyError(
                    f"Specified unknown instrument_type {instrument_type} as"
                    f" a module for cluster {self.name}. Please select one "
                    f"of: {self.compiler_classes.keys()}."
                )
            compiler_type: type = self.compiler_classes[instrument_type]
            instance = compiler_type(
                parent=self,
                name=name,
                total_play_time=self.total_play_time,
                instrument_cfg=cfg,
                latency_corrections=self.latency_corrections,
            )

            instrument_compilers[name] = instance
        return instrument_compilers

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
                port, clock = portclock
                if portclock in self._op_infos:
                    for pulse in self._op_infos[portclock]:
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
        program["settings"] = {"reference_source": self.instrument_cfg["ref"]}

        sequence_to_file = self.instrument_cfg.get("sequence_to_file", None)

        for compiler in self.instrument_compilers.values():
            instrument_program = compiler.compile(
                repetitions=repetitions,
                sequence_to_file=sequence_to_file,
                debug_mode=debug_mode,
            )
            if instrument_program is not None and len(instrument_program) > 0:
                program[compiler.name] = instrument_program

        return program
