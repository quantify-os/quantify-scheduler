# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

# pyright: reportIncompatibleVariableOverride=false

"""Compiler backend for a mock readout module."""

from __future__ import annotations

from collections.abc import Hashable
from random import random
from typing import Literal

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
from quantify_scheduler.helpers.generate_acq_channels_data import (
    AcquisitionChannelsData,
    generate_acq_channels_data,
)
from quantify_scheduler.helpers.waveforms import (
    exec_waveform_function,
    modulate_waveform,
)
from quantify_scheduler.instrument_coordinator.components.base import (
    InstrumentCoordinatorComponentBase,
)
from quantify_scheduler.instrument_coordinator.utility import (
    add_acquisition_coords_binned,
    add_acquisition_coords_nonbinned,
)
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex, Trace
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule
from quantify_scheduler.structure.model import DataStructure
from quantify_scheduler.structure.types import NDArray


class MockReadoutModule:
    """Mock readout module that just supports "TRACE" instruction."""

    def __init__(
        self,
        name: str,
        sampling_rate: float = 1e9,
        gain: float = 1.0,
    ) -> None:
        self.name = name
        self.data = {}
        self.waveforms = {}
        self.instructions = []
        self.gain = gain
        self.sampling_rate = sampling_rate

    def upload_waveforms(self, waveforms: dict[str, NDArray]) -> None:
        """Upload a dictionary of waveforms defined on a 1 ns grid."""
        self.waveforms = waveforms

    def upload_instructions(self, instructions: list[str]) -> None:
        """Upload a sequence of instructions."""
        self.instructions = instructions

    def execute(self) -> None:
        """Execute the instruction sequence (only "TRACE" is supported)."""
        if self.instructions == []:
            raise RuntimeError("No instructions available. Did you upload instructions?")
        self.data = {}  # Clear data
        for instruction in self.instructions:
            if "TRACE" in instruction:
                hardware_acq_location = int(instruction.split("_")[2])
                self.data[hardware_acq_location] = []
                for wf in self.waveforms.values():
                    sampling_idx = np.arange(0, len(wf), int(self.sampling_rate / 1e9))
                    self.data[hardware_acq_location].append(wf[sampling_idx] * self.gain)
            elif "ACQ" in instruction:
                hardware_acq_location = int(instruction.split("_")[2])
                self.data[hardware_acq_location] = []
                self.data[hardware_acq_location] = random()
            else:
                raise NotImplementedError(f"Instruction {instruction} not supported")

    def get_results(self) -> dict:
        """Return the results of the execution."""
        return self.data


class MockROMGettable:
    """Mock readout module gettable."""

    def __init__(
        self,
        mock_rom: MockReadoutModule,
        waveforms: dict[str, NDArray],
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

    def get(self) -> dict:
        """Execute the sequence and return the results."""
        # Set the sampling rate and gain
        self.mock_rom.sampling_rate = self.sampling_rate
        self.mock_rom.gain = self.gain
        # Upload waveforms and instructions
        self.mock_rom.upload_waveforms(self.waveforms)
        self.mock_rom.upload_instructions(self.instructions)
        # Execute and return results
        self.mock_rom.execute()
        data = self.mock_rom.get_results()
        return data


MockHardwareAcqMappingTrace = dict[Hashable, int]
"""
Maps each trace acquisition channel to a hardware acquisition location.
"""
MockHardwareAcqMappingBinned = dict[tuple, int]
"""
Maps each binned acquisition channel to a hardware acquisition location.
The key is a tuple of acquisition channel and acquisition index.
"""


class MockROMSettings(DataStructure):
    """Settings that can be uploaded to the mock readout module."""

    waveforms: dict[str, NDArray]
    instructions: list[str]
    sampling_rate: float = 1e9
    gain: float = 1.0
    acq_channels_data: AcquisitionChannelsData
    hardware_acq_mapping_trace: MockHardwareAcqMappingTrace
    hardware_acq_mapping_binned: MockHardwareAcqMappingBinned


class MockROMInstrumentCoordinatorComponent(InstrumentCoordinatorComponentBase):
    """Mock readout module instrument coordinator component."""

    def __new__(cls, mock_rom: MockReadoutModule) -> InstrumentCoordinatorComponentBase:  # noqa: D102
        # The InstrumentCoordinatorComponentBase.__new__ currently requires a QCoDeS instrument
        # Create a dummy instrument to be compatible with InstrumentCoordinatorComponentBase.__new__
        instrument = InstrumentBase(name=mock_rom.name)
        instance = super().__new__(cls, instrument)
        return instance

    def __init__(self, mock_rom: MockReadoutModule) -> None:
        # The InstrumentCoordinatorComponentBase.__new__ currently requires a QCoDeS instrument
        # Create a dummy instrument to be compatible with
        # InstrumentCoordinatorComponentBase.__init__
        instrument = InstrumentBase(name=mock_rom.name)
        super().__init__(instrument)
        self._rom = mock_rom
        self._hardware_acq_mapping_trace = {}
        self._hardware_acq_mapping_binned = {}
        self._acq_channels_data = None

    @property
    def is_running(self) -> bool:  # noqa: D102
        return True

    def prepare(self, program: MockROMSettings) -> None:
        """Upload the settings to the ROM."""
        self._rom.upload_waveforms(program.waveforms)
        self._rom.upload_instructions(program.instructions)
        self._rom.sampling_rate = program.sampling_rate
        self._rom.gain = program.gain

        self._hardware_acq_mapping_trace = program.hardware_acq_mapping_trace
        self._hardware_acq_mapping_binned = program.hardware_acq_mapping_binned
        self._acq_channels_data = program.acq_channels_data

    def start(self) -> None:
        """Execute the sequence."""
        self._rom.execute()

    def stop(self) -> None:
        """Stop the execution."""

    def retrieve_acquisition(self) -> xr.Dataset:
        """Get the acquired data and return it as an xarray dataset."""
        dataset = xr.Dataset()

        data = self._rom.get_results()

        if self._hardware_acq_mapping_trace is None or self._acq_channels_data is None:
            raise RuntimeError(
                "Attempting to retrieve acquisition from an instrument coordinator"
                " component that was not prepared. Execute"
                " MockROMInstrumentCoordinatorComponent.prepare(mock_rom_settings) first."
            )
        for acq_channel, hardware_acq_location in self._hardware_acq_mapping_trace.items():
            acq_protocol = self._acq_channels_data[acq_channel].protocol
            acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
            if acq_protocol == "Trace":
                complex_data = data[hardware_acq_location][0] + 1j * data[hardware_acq_location][1]
                data_len = len(complex_data)
                complex_data_averaged = complex_data.reshape((1, -1))
                time_dim_name = f"time_{acq_channel}"
                data_array = xr.DataArray(
                    complex_data_averaged,
                    dims=(acq_index_dim_name, time_dim_name),
                    coords={
                        acq_index_dim_name: [0],
                        time_dim_name: np.arange(0, data_len * 1e-9, 1e-9),
                    },
                    attrs={"acq_protocol": acq_protocol},
                )
                new_dataset = xr.Dataset({acq_channel: data_array})
                coords = self._acq_channels_data[acq_channel].coords
                assert isinstance(coords, dict)  # Guaranteed by the acquisition protocol.
                add_acquisition_coords_nonbinned(data_array, coords, acq_index_dim_name)
                dataset = dataset.merge(new_dataset)
            else:
                raise NotImplementedError(f"Acquisition protocol {acq_protocol} not supported.")

        for (
            acq_channel,
            acq_index,
        ), hardware_acq_location in self._hardware_acq_mapping_binned.items():
            acq_protocol = self._acq_channels_data[acq_channel].protocol
            acq_index_dim_name = self._acq_channels_data[acq_channel].acq_index_dim_name
            if acq_protocol == "SSBIntegrationComplex":
                data_array = xr.DataArray(
                    [data[hardware_acq_location]],
                    dims=[acq_index_dim_name],
                    coords={acq_index_dim_name: [acq_index]},
                    attrs={"acq_protocol": acq_protocol},
                )
                coords = self._acq_channels_data[acq_channel].coords
                assert isinstance(coords, list)  # Guaranteed by the acquisition protocol.
                add_acquisition_coords_binned(data_array, coords, acq_index_dim_name)
                new_dataset = xr.Dataset({acq_channel: data_array})
                dataset = dataset.merge(new_dataset)
            else:
                raise NotImplementedError(f"Acquisition protocol {acq_protocol} not supported.")

        return dataset

    def wait_done(self, timeout_sec: int = 10) -> None:
        """Wait until the execution is done."""

    def get_hardware_log(self, compiled_schedule: CompiledSchedule) -> None:
        """Return the hardware log."""
        pass


def hardware_compile(  # noqa: PLR0915
    schedule: Schedule, config: CompilationConfig
) -> Schedule:
    """Compile the schedule to the mock ROM."""
    # Type checks and initialization
    if not isinstance(config.hardware_compilation_config, MockROMHardwareCompilationConfig):
        raise ValueError("Config should be a MockROMHardwareCompilationConfig object.")
    connectivity = config.hardware_compilation_config.connectivity
    if not isinstance(connectivity, Connectivity):
        raise ValueError("Connectivity should be a Connectivity object.")
    hardware_description = config.hardware_compilation_config.hardware_description
    hardware_options = config.hardware_compilation_config.hardware_options
    instructions = []
    waveforms = {}
    next_hardware_acq_location = 0
    hardware_acq_mapping_trace: MockHardwareAcqMappingTrace = {}
    hardware_acq_mapping_binned: MockHardwareAcqMappingBinned = {}

    acq_channels_data, schedulable_to_acq_index = generate_acq_channels_data(schedule)

    # Compile the schedule to the mock ROM
    gain_setting = None
    sampling_rate = hardware_description["mock_rom"].sampling_rate
    for schedulable_label, schedulable in schedule.schedulables.items():
        op = schedule.operations[schedulable["operation_id"]]
        if isinstance(op, Schedule):
            raise NotImplementedError("Nested schedules are not supported by the Mock ROM backend.")
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
                pc_mod_freqs = hardware_options.modulation_frequencies.get(f"{port}-{clock}")
            assert pc_mod_freqs is not None
            assert pc_mod_freqs.interm_freq is not None
            envelope = exec_waveform_function(wf_func=wf_func, t=time_grid, pulse_info=pulse_info)
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
                instructions.append(f"TRACE_{hw_port}_{next_hardware_acq_location}")
            if hardware_options.gain is not None:
                if (
                    gain_setting is not None
                    and hardware_options.gain[f"{port}-{clock}"] != gain_setting
                ):
                    raise ValueError("The gain must be the same for all traces in the schedule.")
                gain_setting = hardware_options.gain[f"{port}-{clock}"]
            hardware_acq_mapping_trace[acq_info["acq_channel"]] = next_hardware_acq_location
            next_hardware_acq_location += 1
        elif (
            isinstance(op, SSBIntegrationComplex)
            and op.data["acquisition_info"][0]["bin_mode"] == BinMode.AVERAGE
        ):
            acq_info = op.data["acquisition_info"][0]
            for node in connectivity.graph["q0:res"]:
                hw_port = node.split(".")[1]
                instructions.append(f"ACQ_{hw_port}_{next_hardware_acq_location}")
            acq_index = schedulable_to_acq_index[((schedulable_label,), 0)]
            hardware_acq_mapping_binned[(acq_info["acq_channel"], acq_index)] = (
                next_hardware_acq_location
            )
            next_hardware_acq_location += 1
        else:
            raise NotImplementedError(f"Operation {op} is not supported by the Mock ROM backend.")

    if "compiled_instructions" not in schedule:
        schedule["compiled_instructions"] = {}

    # Add compiled instructions for the mock ROM to the schedule
    settings = MockROMSettings(
        waveforms=waveforms,
        instructions=instructions,
        sampling_rate=sampling_rate,
        acq_channels_data=acq_channels_data,
        hardware_acq_mapping_trace=hardware_acq_mapping_trace,
        hardware_acq_mapping_binned=hardware_acq_mapping_binned,
    )
    if gain_setting is not None:
        settings.gain = gain_setting
    schedule["compiled_instructions"]["mock_rom"] = settings
    return schedule


class MockROMDescription(HardwareDescription):
    instrument_type: Literal["Mock readout module"] = "Mock readout module"
    sampling_rate: float


class MockROMHardwareOptions(HardwareOptions):
    gain: dict[str, float] | None = None


class MockROMHardwareCompilationConfig(HardwareCompilationConfig):
    config_type: type[MockROMHardwareCompilationConfig] = Field(  # type: ignore
        default="quantify_scheduler.backends.mock.mock_rom.MockROMHardwareCompilationConfig",
        validate_default=True,
    )
    """
    A reference to the
    :class:`~quantify_scheduler.backends.types.common.HardwareCompilationConfig`
    DataStructure for the Mock ROM backend.
    """
    hardware_description: dict[str, MockROMDescription]
    hardware_options: MockROMHardwareOptions
    compilation_passes: list[SimpleNodeConfig] = [
        SimpleNodeConfig(
            name="mock_rom_hardware_compile",
            compilation_func=hardware_compile,  # type: ignore
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
