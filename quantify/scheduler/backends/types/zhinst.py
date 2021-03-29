# -----------------------------------------------------------------------------
# Description:    Python dataclasses for quantify-scheduler json-schemas.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Optional, Union

from dataclasses_json import DataClassJsonMixin

from quantify.scheduler import enums


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
    mode :
        The output mode type.
    modulation :
        The optional modulation mode type is only required when the SignalModeType
        is set to COMPLEX.
    lo_freq :
        The local oscillator (LO) frequency in Hz. default is 0.
    interm_freq :
        The inter-modulation frequency (IF) in Hz. default is 0.
    phase_shift :
        The IQ modulation phase shift in Degrees. default is 0.
    gain1 :
        The output1 IQ modulation gain (value between -1 and + 1). default is 0.
    gain2 :
        The output2 IQ modulation gain (value between -1 and + 1). default is 0.
    line_gain_db :
        The cable line gain in Decibel. default = 0
    line_trigger_delay :
        The ZI Instrument output triggers. default is -1.
    triggers :
        The ZI Instrument input triggers. default is [].
    markers :
        The ZI Instrument output triggers. default is [].
    """

    port: str
    clock: str
    mode: enums.SignalModeType
    modulation: enums.ModulationModeType
    lo_freq: float = 0
    interm_freq: float = 0
    phase_shift: float = 0
    gain1: int = 0
    gain2: int = 0
    line_gain_db: float = 0
    line_trigger_delay: float = -1
    triggers: List[int] = field(default_factory=lambda: [])
    markers: List[Union[str, int]] = field(default_factory=lambda: [])


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
        The HDAWG channelgrouping property. (default = 0)
    type :
        The Zurich Instruments hardware type. (default = DeviceType.NONE)
        This field is automatically set by the backend.
    """

    name: str
    ref: enums.ReferenceSourceType
    channels: List[Output] = field(init=False)
    channel_0: Output
    channel_1: Optional[Output] = None
    channel_2: Optional[Output] = None
    channel_3: Optional[Output] = None
    channelgrouping: int = 0
    type: DeviceType = DeviceType.NONE

    def __post_init__(self):
        """Initializes fields after initializing object."""
        self.channels = [self.channel_0]
        if self.channel_1 is not None:
            self.channels.append(self.channel_1)
        if self.channel_2 is not None:
            self.channels.append(self.channel_2)
        if self.channel_3 is not None:
            self.channels.append(self.channel_3)


@dataclass
class CommandTableHeader(DataClassJsonMixin):
    """
    The CommandTable header definition.
    """

    version: str = "0.2"
    partial: bool = False


@dataclass
class CommandTableEntryIndex(DataClassJsonMixin):
    """
    A CommandTable entry definition with an index.
    """

    index: int


@dataclass
class CommandTableEntryValue(DataClassJsonMixin):
    """
    A CommandTable entry definition with a value.
    """

    value: int


@dataclass
class CommandTableEntry(DataClassJsonMixin):
    """
    The definition of a single CommandTable entry.
    """

    index: int
    waveform: "CommandTableEntryIndex"


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
class QAS_IntegrationMode(Enum):
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
