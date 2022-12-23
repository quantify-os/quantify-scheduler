# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Python dataclasses for quantify-scheduler json-schemas."""
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Optional, Union

from dataclasses_json import DataClassJsonMixin

from quantify_scheduler import enums
from quantify_scheduler.backends.types import common


@unique
class DeviceType(str, Enum):
    """Enum of device types."""

    HDAWG = "HDAWG"
    UHFQA = "UHFQA"
    UHFLI = "UHFLI"
    MFLI = "MFLI"
    PQSC = "PQSC"
    NONE = "none"


@dataclass
class Output(DataClassJsonMixin):
    """
    The definition class for zhinst channel properties.

    This class maps to the zhinst backend JSON "channel"
    properties defined in the hardware mapping.

    Parameters
    ----------
    port :
        The port resource.
    clock :
        The Clock resource.
    clock_frequency:
        The frequency for the clock resource
        (AKA RF/signal frequency).
    mode :
        The output mode type.
    modulation :
        The modulation settings.
    local_oscillator :
        The LocalOscillator name.
    gain1 :
        The output1 IQ modulation gain.
        Accepted value between -1 and + 1. (default = 1.0)
    gain2 :
        The output2 IQ modulation gain.
        Accepted value between -1 and + 1. (default = 1.0)
    trigger :
        The ZI Instrument input trigger. (default = None)
        Setting this will declare the device secondary
    markers :
        The ZI Instrument output triggers. (default = [])
    mixer_corrections :
        The output mixer corrections.
    """

    port: str
    clock: str
    mode: enums.SignalModeType
    modulation: common.Modulation
    local_oscillator: str
    clock_frequency: Optional[float] = None
    gain1: int = 0
    gain2: int = 0
    trigger: Optional[int] = None
    markers: List[Union[str, int]] = field(default_factory=lambda: [])
    mixer_corrections: Optional[common.MixerCorrections] = None


@dataclass
class Device(DataClassJsonMixin):
    """
    The device definition class for zhinst devices.

    This class maps to the zhinst backend JSON "devices"
    properties defined in the hardware mapping.

    Parameters
    ----------
    name :
        The QCodes Instrument name.
    type :
        The instrument model type.
        For example: 'UHFQA', 'HDAWG4', 'HDAWG8'
    ref :
        The reference source type.
    channels :
        The list of channels. (auto generated)
    channel_0 :
        The first physical channel properties.
    channel_1 :
        The second physical channel properties.
    channel_2 :
        The third physical channel properties.
    channel_3 :
        The fourth physical channel properties.
    channelgrouping :
        The HDAWG channelgrouping property. (default = 0) corresponding to a single
        sequencer controlling a pair (2) awg outputs.
    clock_select :
        The clock rate divisor which will be used to get
        the instruments clock rate from the lookup dictionary in
        quantify_scheduler.backends.zhinst_backend.DEVICE_CLOCK_RATES.

        For information see zhinst User manuals, section /DEV..../AWGS/n/TIME
        Examples: base sampling rate (1.8 GHz) divided by 2^clock_select. (default = 0)
    mode :
        The Instruments operation mode.
        (default = enums.InstrumentOperationMode.OPERATING)
    device_type :
        The Zurich Instruments hardware type. (default = DeviceType.NONE)
        This field is automatically populated.
    sample_rate :
        The Instruments sampling clock rate.
        This field is automatically populated.
    n_channels :
        The number of physical channels of this ZI Instrument.
        This field is automatically populated.
    """

    name: str
    type: str
    ref: enums.ReferenceSourceType
    channels: List[Output] = field(init=False)
    channel_0: Output
    channel_1: Optional[Output] = None
    channel_2: Optional[Output] = None
    channel_3: Optional[Output] = None
    clock_select: Optional[int] = 0
    channelgrouping: int = 0
    mode: enums.InstrumentOperationMode = enums.InstrumentOperationMode.OPERATING
    device_type: DeviceType = DeviceType.NONE
    sample_rate: Optional[int] = field(init=False)
    n_channels: int = field(init=False)

    def __post_init__(self):
        """Initializes fields after initializing object."""
        self.channels = [self.channel_0]
        if self.channel_1 is not None:
            self.channels.append(self.channel_1)
        if self.channel_2 is not None:
            self.channels.append(self.channel_2)
        if self.channel_3 is not None:
            self.channels.append(self.channel_3)

        if self.type[-1].isdigit():
            digit = int(self.type[-1])
            self.n_channels = digit
            device_type = self.type[: len(self.type) - 1]
            self.device_type = DeviceType(device_type)
        else:
            self.device_type = DeviceType(self.type)
            self.n_channels = 1


@dataclass
class CommandTableHeader(DataClassJsonMixin):
    """
    The CommandTable header definition.
    """

    version: str = "0.2"
    partial: bool = False


@dataclass
class CommandTableEntryValue(DataClassJsonMixin):
    """
    A CommandTable entry definition with a value.
    """

    value: int


@dataclass
class CommandTableWaveform(DataClassJsonMixin):
    """
    The command table waveform properties.
    """

    index: int
    length: int


@dataclass
class CommandTableEntry(DataClassJsonMixin):
    """
    The definition of a single CommandTable entry.
    """

    index: int
    waveform: "CommandTableWaveform"


@dataclass
class CommandTable(DataClassJsonMixin):
    """
    The CommandTable definition for ZI HDAWG.
    """

    header: "CommandTableHeader" = field(init=False)
    table: List["CommandTableEntry"]

    def __post_init__(self):
        """Initializes fields after initializing object."""
        self.header = CommandTableHeader()


@unique
class QasIntegrationMode(Enum):
    """
    Operation mode of all weighted integration units.

    NORMAL: Normal mode. The integration weights are given
    by the user-programmed filter memory.

    SPECTROSCOPY:  Spectroscopy mode. The integration weights
    are generated by a digital oscillator. This mode offers
    enhanced frequency resolution.
    """

    NORMAL = 0
    SPECTROSCOPY = 1


@unique
class QasResultMode(Enum):
    """UHFQA QAS result mode."""

    CYCLIC = 0
    SEQUENTIAL = 1


@unique
class QasResultSource(Enum):
    """UHFQA QAS result source."""

    CROSSTALK = 0
    THRESHOLD = 1
    ROTATION = 3
    CROSSTALK_CORRELATION = 4
    THRESHOLD_CORRELATION = 5
    INTEGRATION = 7


class WaveformDestination(Enum):
    """The waveform destination enum type."""

    CSV = 0
    WAVEFORM_TABLE = 1


@dataclass
class InstrumentInfo:
    """Instrument information record type."""

    sample_rate: int

    num_samples_per_clock: int  # number of samples per clock cycle (sequencer_rate)
    granularity: int  # waveforms need to be a multiple of this many samples.
    mode: enums.InstrumentOperationMode = enums.InstrumentOperationMode.OPERATING
    sequencer_rate: float = field(init=False)

    def __post_init__(self):
        """Initializes fields after initializing object."""
        self.sequencer_rate = self.num_samples_per_clock / self.sample_rate


@dataclass(frozen=True)
class Instruction:
    """Sequence base instruction record type."""

    waveform_id: str
    abs_time: float
    clock_cycle_start: int

    duration: float

    @staticmethod
    def default():
        """
        Returns a default Instruction instance.

        Returns
        -------
        Instruction :
        """
        return Instruction("None", 0, 0, 0)


@dataclass(frozen=True)
class Acquisition(Instruction):
    """
    This instruction indicates that an acquisition is to be triggered in the UHFQA.
    If a waveform_id is specified, this waveform will be used as the integration weight.
    """

    def __repr__(self):
        return (
            f"Acquisition(waveform_id: {self.waveform_id}"
            f"|abs_time: {self.abs_time * 1e9} ns"
            f"|dt: {self.duration * 1e9} ns"
            f"|c0: {self.clock_cycle_start}"
        )


@dataclass(frozen=True)
class Wave(Instruction):
    """
    This instruction indicates that a waveform  should be played.
    """

    def __repr__(self):
        return (
            f"Wave(waveform_id: {self.waveform_id}"
            f"|abs_time: {self.abs_time * 1e9} ns"
            f"|dt: {self.duration * 1e9} ns"
            f"|c0: {self.clock_cycle_start}"
        )
