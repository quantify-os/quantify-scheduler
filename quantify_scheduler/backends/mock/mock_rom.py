# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

# pyright: reportIncompatibleVariableOverride=false

"""Compiler backend for a mock readout module."""
from __future__ import annotations

from typing import Dict, Hashable, Literal, Optional, Type

import numpy as np
import xarray as xr
from pydantic import Field
from qcodes.instrument.base import InstrumentBase

from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.types.common import (
    Connectivity,
    HardwareCompilationConfig,
    HardwareDescription,
    HardwareOptions,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.waveforms import (
    exec_waveform_function,
    modulate_waveform,
)
from quantify_scheduler.instrument_coordinator.components.base import (
    InstrumentCoordinatorComponentBase,
)
from quantify_scheduler.operations.acquisition_library import Trace
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule
from quantify_scheduler.structure.model import DataStructure
from quantify_scheduler.structure.types import NDArray  # noqa: TCH001


class MockReadoutModule:
    """Mock readout module that just supports "TRACE" instruction."""

    def __init__(
        self,
        name: str,
        sampling_rate: float = 1e9,
        gain: float = 1.0,
    ) -> None:
        self.name = name
        self.data = []
        self.waveforms = {}
        self.instructions = []
        self.gain = gain
        self.sampling_rate = sampling_rate

    def upload_waveforms(self, waveforms: Dict[str, NDArray]) -> None:
        """Upload a dictionary of waveforms defined on a 1 ns grid."""
        self.waveforms = waveforms

    def upload_instructions(self, instructions: list[str]) -> None:
        """Upload a sequence of instructions."""
        self.instructions = instructions

    def execute(self) -> None:
        """Execute the instruction sequence (only "TRACE" is supported)."""
        if self.instructions == []:
            raise RuntimeError(
                "No instructions available. Did you upload instructions?"
            )
        for instruction in self.instructions:
            if "TRACE" in instruction:
                self.data = []  # Clear data
                for wf in self.waveforms.values():
                    sampling_idx = np.arange(0, len(wf), int(self.sampling_rate / 1e9))
                    self.data.append(wf[sampling_idx] * self.gain)
            else:
                raise NotImplementedError(f"Instruction {instruction} not supported")

    def get_results(self) -> list[np.ndarray]:
        """Return the results of the execution."""
        if self.data == []:
            raise RuntimeError("No data available. Did you execute the sequence?")
        return self.data


class MockROMGettable:
    """Mock readout module gettable."""

    def __init__(
        self,
        mock_rom: MockReadoutModule,
        waveforms: Dict[str, NDArray],
        instructions: list[str],
        sampling_rate: float = 1e9,
        gain: float = 1.0,
    ) -> None:
        """Initialize a mock rom gettable from a set of (compiled) settings."""
        self.mock_rom = mock_rom
        self.waveforms = waveforms
        self.instructions = instructions
        self.sampling_rate = sampling_rate
        self.gain = gain

    def get(self) -> list[np.ndarray]:
        """Execute the sequence and return the results."""
        # Set the sampling rate and gain
        self.mock_rom.sampling_rate = self.sampling_rate
        self.mock_rom.gain = self.gain
        # Upload waveforms and instructions
        self.mock_rom.upload_waveforms(self.waveforms)
        self.mock_rom.upload_instructions(self.instructions)
        # Execute and return results
        self.mock_rom.execute()
        return self.mock_rom.get_results()


class MockROMAcquisitionConfig(DataStructure):
    """
    Acquisition configuration for the mock readout module.

    This information is used in the instrument coordinator component to convert the
    acquired data to an xarray dataset.
    """

    n_acquisitions: int
    acq_protocols: Dict[int, str]
    bin_mode: BinMode


class MockROMSettings(DataStructure):
    """Settings that can be uploaded to the mock readout module."""

    waveforms: Dict[str, NDArray]
    instructions: list[str]
    sampling_rate: float = 1e9
    gain: float = 1.0
    acq_config: MockROMAcquisitionConfig


class MockROMInstrumentCoordinatorComponent(InstrumentCoordinatorComponentBase):
    """Mock readout module instrument coordinator component."""

    def __new__(
        cls, mock_rom: MockReadoutModule
    ) -> InstrumentCoordinatorComponentBase:  # noqa: D102
        # The InstrumentCoordinatorComponentBase.__new__ currently requires a QCoDeS instrument
        # Create a dummy instrument to be compatible with InstrumentCoordinatorComponentBase.__new__
        instrument = InstrumentBase(name=mock_rom.name)
        instance = super().__new__(cls, instrument)
        return instance

    def __init__(self, mock_rom: MockReadoutModule) -> None:
        # The InstrumentCoordinatorComponentBase.__new__ currently requires a QCoDeS instrument
        # Create a dummy instrument to be compatible with InstrumentCoordinatorComponentBase.__init__
        instrument = InstrumentBase(name=mock_rom.name)
        super().__init__(instrument)
        self.rom = mock_rom
        self.acq_config = None

    @property
    def is_running(self) -> bool:  # noqa: D102
        return True

    def prepare(self, options: MockROMSettings) -> None:
        """Upload the settings to the ROM."""
        self.rom.upload_waveforms(options.waveforms)
        self.rom.upload_instructions(options.instructions)
        self.rom.sampling_rate = options.sampling_rate
        self.rom.gain = options.gain

        self.acq_config = options.acq_config

    def start(self) -> None:
        """Execute the sequence."""
        self.rom.execute()

    def stop(self) -> None:
        """Stop the execution."""

    def retrieve_acquisition(self) -> xr.Dataset:
        """Get the acquired data and return it as an xarray dataset."""
        data = self.rom.get_results()

        # TODO: convert to xarray dataset
        acq_config = self.acq_config
        if acq_config is None:
            raise RuntimeError(
                "Attempting to retrieve acquisition from an instrument coordinator"
                " component that was not prepared. Execute"
                " MockROMInstrumentCoordinatorComponent.prepare(mock_rom_settings) first."
            )
        acq_channel_results: list[Dict[Hashable, xr.DataArray]] = []
        for acq_channel, acq_protocol in acq_config.acq_protocols.items():
            if acq_protocol == "Trace":
                complex_data = data[2 * acq_channel] + 1j * data[2 * acq_channel + 1]
                acq_channel_results.append(
                    {
                        acq_channel: xr.DataArray(
                            # Should be one avaraged array
                            complex_data.reshape((1, -1)),
                            dims=(
                                f"acq_index_{acq_channel}",
                                f"trace_index_{acq_channel}",
                            ),
                            attrs={"acq_protocol": acq_protocol},
                        )
                    }
                )
            else:
                raise NotImplementedError(
                    f"Acquisition protocol {acq_protocol} not supported."
                )

        return xr.merge(acq_channel_results, compat="no_conflicts")

    def wait_done(self, timeout_sec: int = 10) -> None:
        """Wait until the execution is done."""

    def get_hardware_log(self, compiled_schedule: CompiledSchedule) -> None:
        """Return the hardware log."""
        return None


def hardware_compile(  # noqa: PLR0912, PLR0915
    schedule: Schedule, config: CompilationConfig
) -> Schedule:
    """Compile the schedule to the mock ROM."""
    # Type checks and initialization
    if not isinstance(
        config.hardware_compilation_config, MockROMHardwareCompilationConfig
    ):
        raise ValueError("Config should be a MockROMHardwareCompilationConfig object.")
    connectivity = config.hardware_compilation_config.connectivity
    if not isinstance(connectivity, Connectivity):
        raise ValueError("Connectivity should be a Connectivity object.")
    hardware_description = config.hardware_compilation_config.hardware_description
    hardware_options = config.hardware_compilation_config.hardware_options
    instructions = []
    waveforms = {}
    acq_protocols = {}
    n_acquisitions = 0

    # Compile the schedule to the mock ROM
    gain_setting = None
    bin_mode = BinMode.AVERAGE
    sampling_rate = hardware_description["mock_rom"].sampling_rate
    for schedulable in schedule.schedulables.values():
        op = schedule.operations[schedulable["operation_id"]]
        if isinstance(op, Schedule):
            raise NotImplementedError(
                "Nested schedules are not supported by the Mock ROM backend."
            )
        if op.valid_pulse:
            pulse_info = op.data["pulse_info"][0]
            port = pulse_info["port"]
            clock = pulse_info["clock"]
            time_grid = np.arange(0, pulse_info["duration"], 1 / sampling_rate)
            wf_func = pulse_info["wf_func"]
            if wf_func is None:
                continue
            if hardware_options.modulation_frequencies is None:
                raise ValueError(
                    "Modulation frequencies must be specified for the Mock ROM backend."
                )
            else:
                pc_mod_freqs = hardware_options.modulation_frequencies.get(
                    f"{port}-{clock}"
                )
            assert pc_mod_freqs is not None
            assert pc_mod_freqs.interm_freq is not None
            envelope = exec_waveform_function(
                wf_func=wf_func, t=time_grid, pulse_info=pulse_info
            )
            modulated_wf = modulate_waveform(
                time_grid,
                envelope=envelope,
                freq=pc_mod_freqs.interm_freq,
            )
            waveforms[f"{op.hash}_I"] = modulated_wf.real
            waveforms[f"{op.hash}_Q"] = modulated_wf.imag
        elif isinstance(op, Trace):
            acq_info = op.data["acquisition_info"][0]
            port = acq_info["port"]
            clock = acq_info["clock"]
            for node in connectivity.graph["q0:res"]:
                hw_port = node.split(".")[1]
                instructions.append(f"TRACE_{hw_port}")
            if hardware_options.gain is not None:
                if (
                    gain_setting is not None
                    and hardware_options.gain[f"{port}-{clock}"] != gain_setting
                ):
                    raise ValueError(
                        "The gain must be the same for all traces in the schedule."
                    )
                gain_setting = hardware_options.gain[f"{port}-{clock}"]
            n_acquisitions += 1
            acq_protocols[acq_info["acq_channel"]] = acq_info["protocol"]
            bin_mode = acq_info["bin_mode"]
        else:
            raise NotImplementedError(
                f"Operation {op} is not supported by the Mock ROM backend."
            )

    if "compiled_instructions" not in schedule:
        schedule["compiled_instructions"] = {}

    # Add compiled instructions for the mock ROM to the schedule
    settings = MockROMSettings(
        waveforms=waveforms,
        instructions=instructions,
        sampling_rate=sampling_rate,
        acq_config=MockROMAcquisitionConfig(
            n_acquisitions=n_acquisitions,
            acq_protocols=acq_protocols,
            bin_mode=bin_mode,
        ),
    )
    if gain_setting is not None:
        settings.gain = gain_setting
    schedule["compiled_instructions"]["mock_rom"] = settings
    return schedule


class MockROMDescription(HardwareDescription):
    instrument_type: Literal["Mock readout module"] = "Mock readout module"
    sampling_rate: float


class MockROMHardwareOptions(HardwareOptions):
    gain: Optional[Dict[str, float]] = None  # noqa: UP007


class MockROMHardwareCompilationConfig(HardwareCompilationConfig):
    config_type: Type[MockROMHardwareCompilationConfig] = Field(  # noqa: UP006
        default="quantify_scheduler.backends.mock.mock_rom.MockROMHardwareCompilationConfig",
        validate_default=True,
    )
    """
    A reference to the
    :class:`~quantify_scheduler.backends.types.common.HardwareCompilationConfig`
    DataStructure for the Mock ROM backend.
    """
    hardware_description: Dict[str, MockROMDescription]
    hardware_options: MockROMHardwareOptions
    compilation_passes: list[SimpleNodeConfig] = [  # noqa: UP006
        SimpleNodeConfig(
            name="mock_rom_hardware_compile", compilation_func=hardware_compile
        )
    ]


hardware_compilation_config = {
    "config_type": "quantify_scheduler.backends.mock.mock_rom.MockROMHardwareCompilationConfig",
    "hardware_description": {
        "mock_rom": {"instrument_type": "Mock readout module", "sampling_rate": 1.5e9}
    },
    "hardware_options": {
        "gain": {"q0:res-q0.ro": 2.0},
        "modulation_frequencies": {"q0:res-q0.ro": {"interm_freq": 100e6}},
    },
    "connectivity": {"graph": [("mock_rom.input0", "q0:res")]},
}
