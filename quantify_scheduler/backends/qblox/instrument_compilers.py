# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler classes for Qblox backend."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from collections import abc

from quantify_scheduler.backends.qblox import compiler_abc, compiler_container
from quantify_scheduler.backends.qblox.constants import (
    NUMBER_OF_SEQUENCERS_QCM,
    NUMBER_OF_SEQUENCERS_QRM,
)
from quantify_scheduler.backends.types.qblox import (
    BoundedParameter,
    LOSettings,
    MarkerConfiguration,
    StaticHardwareProperties,
)


class LocalOscillator(compiler_abc.InstrumentCompiler):
    """
    Implementation of an `InstrumentCompiler` that compiles for a generic LO. The main
    difference between this class and the other compiler classes is that it doesn't take
    pulses and acquisitions.
    """

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
        """
        Constructor for a local oscillator compiler.

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
        hw_mapping
            The hardware mapping dict for this instrument.
        """

        def _extract_parameter(
            parameter_dict: Dict[str, Optional[float]]
        ) -> Tuple[str, Optional[float]]:
            items: abc.ItemsView = parameter_dict.items()
            return list(items)[0]

        super().__init__(parent, name, total_play_time, hw_mapping)
        self._settings = LOSettings.from_mapping(hw_mapping)
        self.freq_param_name, self._frequency = _extract_parameter(
            self._settings.frequency
        )
        self.power_param_name, self._power = _extract_parameter(self._settings.power)

    @property
    def frequency(self) -> Optional[float]:
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
        -------
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

    def compile(self, repetitions: int = 1) -> Optional[Dict[str, Any]]:
        """
        Compiles the program for the LO InstrumentCoordinator component.

        Parameters
        ----------
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


class QcmModule(compiler_abc.QbloxBasebandModule):
    """
    QCM specific implementation of the qblox compiler.
    """

    supports_acquisition: bool = False
    static_hw_properties: StaticHardwareProperties = StaticHardwareProperties(
        instrument_type="QCM",
        max_sequencers=NUMBER_OF_SEQUENCERS_QCM,
        max_awg_output_voltage=2.5,
        marker_configuration=MarkerConfiguration(init=None, start=0b1111, end=0b0000),
        mixer_dc_offset_range=BoundedParameter(min_val=-2.5, max_val=2.5, units="V"),
        valid_ios=[f"complex_output_{i}" for i in [0, 1]]
        + [f"real_output_{i}" for i in range(4)],
    )


# pylint: disable=invalid-name
class QrmModule(compiler_abc.QbloxBasebandModule):
    """
    QRM specific implementation of the qblox compiler.
    """

    supports_acquisition: bool = True
    static_hw_properties: StaticHardwareProperties = StaticHardwareProperties(
        instrument_type="QRM",
        max_sequencers=NUMBER_OF_SEQUENCERS_QRM,
        max_awg_output_voltage=0.5,
        marker_configuration=MarkerConfiguration(init=None, start=0b1111, end=0b0000),
        mixer_dc_offset_range=BoundedParameter(min_val=-0.5, max_val=0.5, units="V"),
        valid_ios=[f"complex_output_{i}" for i in [0]]
        + [f"real_output_{i}" for i in range(2)]
        + [f"complex_input_{i}" for i in [0]]
        + [f"real_input_{i}" for i in range(2)],
    )


class QcmRfModule(compiler_abc.QbloxRFModule):
    """
    QCM-RF specific implementation of the qblox compiler.
    """

    supports_acquisition: bool = False
    static_hw_properties: StaticHardwareProperties = StaticHardwareProperties(
        instrument_type="QCM-RF",
        max_sequencers=NUMBER_OF_SEQUENCERS_QCM,
        max_awg_output_voltage=0.25,
        marker_configuration=MarkerConfiguration(
            init=0b0011,
            start=0b1100,
            end=0b0000,
            output_map={
                "complex_output_0": 0b0001,
                "complex_output_1": 0b0010,
            },
        ),
        mixer_dc_offset_range=BoundedParameter(min_val=-50, max_val=50, units="mV"),
        valid_ios=[f"complex_output_{i}" for i in [0, 1]]
        + [f"real_output_{i}" for i in range(4)],
    )


class QrmRfModule(compiler_abc.QbloxRFModule):
    """
    QRM-RF specific implementation of the qblox compiler.
    """

    supports_acquisition: bool = True
    static_hw_properties: StaticHardwareProperties = StaticHardwareProperties(
        instrument_type="QRM-RF",
        max_sequencers=NUMBER_OF_SEQUENCERS_QRM,
        max_awg_output_voltage=0.25,
        marker_configuration=MarkerConfiguration(init=0b0011, start=0b1111, end=0b0000),
        mixer_dc_offset_range=BoundedParameter(min_val=-50, max_val=50, units="mV"),
        valid_ios=[f"complex_output_{i}" for i in [0]]
        + [f"real_output_{i}" for i in range(2)]
        + [f"complex_input_{i}" for i in [0]]
        + [f"real_input_{i}" for i in range(2)],
    )


class Cluster(compiler_abc.ControlDeviceCompiler):
    """
    Compiler class for a Qblox cluster.
    """

    compiler_classes: Dict[str, type] = {
        "QCM": QcmModule,
        "QRM": QrmModule,
        "QCM_RF": QcmRfModule,
        "QRM_RF": QrmRfModule,
    }
    """References to the individual module compiler classes that can be used by the
    cluster."""
    supports_acquisition: bool = True
    """Specifies that the Cluster supports performing acquisitions."""

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
        latency_corrections: Optional[Dict[str, float]] = None,
    ):
        """
        Constructor for a Cluster compiler object.

        Parameters
        ----------
        parent
            Reference to the parent object.
        name
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time
            Total time execution of the schedule should go on for.
        hw_mapping
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        latency_corrections
            Dict containing the delays for each port-clock combination. This is
            specified in the top layer of hardware config.
        """
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            hw_mapping=hw_mapping,
            latency_corrections=latency_corrections,
        )
        self.instrument_compilers: dict = self.construct_instrument_compilers()
        self.latency_corrections = latency_corrections

    def construct_instrument_compilers(self) -> Dict[str, compiler_abc.QbloxBaseModule]:
        """
        Constructs the compilers for the modules inside the cluster.

        Returns
        -------
        :
            A dictionary with the name of the instrument as key and the value its
            compiler.
        """
        instrument_compilers = {}
        for name, cfg in self.hw_mapping.items():
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
                self,
                name=name,
                total_play_time=self.total_play_time,
                hw_mapping=cfg,
                latency_corrections=self.latency_corrections,
            )
            assert hasattr(instance, "is_pulsar")
            instance.is_pulsar = False

            instrument_compilers[name] = instance
        return instrument_compilers

    def prepare(self) -> None:
        """
        Prepares the instrument compiler for compilation by assigning the data.
        """
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
                if portclock in self._pulses:
                    for pulse in self._pulses[portclock]:
                        compiler.add_pulse(port, clock, pulse)
                if portclock in self._acquisitions:
                    for acq in self._acquisitions[portclock]:
                        compiler.add_acquisition(port, clock, acq)

    def compile(self, repetitions: int = 1) -> Optional[Dict[str, Any]]:
        """
        Performs the compilation.

        Parameters
        ----------
        repetitions
            Amount of times to repeat execution of the schedule.

        Returns
        -------
        :
            The part of the compiled instructions relevant for this instrument.
        """
        program = {}
        program["settings"] = {"reference_source": self.hw_mapping["ref"]}

        sequence_to_file = self.hw_mapping.get("sequence_to_file", None)
        for compiler in self.instrument_compilers.values():
            instrument_program = compiler.compile(
                repetitions=repetitions, sequence_to_file=sequence_to_file
            )
            if instrument_program is not None and len(instrument_program) > 0:
                program[compiler.name] = instrument_program

        if len(program) == 0:
            program = None
        return program


COMPILER_MAPPING: Dict[str, type] = {
    "Pulsar_QCM": QcmModule,
    "Pulsar_QRM": QrmModule,
    "Pulsar_QCM_RF": QcmRfModule,
    "Pulsar_QRM_RF": QrmRfModule,
    "Cluster": Cluster,
    "LocalOscillator": LocalOscillator,
}
"""Maps the names in the hardware config to their appropriate compiler classes."""
