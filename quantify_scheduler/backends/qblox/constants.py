# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Constants for compilation to Qblox hardware."""

MAX_NUMBER_OF_INSTRUCTIONS_QCM: int = 16384
"""Maximum supported number of instructions in Q1ASM programs for QCM/QCM-RF."""
MAX_NUMBER_OF_INSTRUCTIONS_QRM: int = 12288
"""Maximum supported number of instructions in Q1ASM programs for QRM/QRM-RF."""
MAX_NUMBER_OF_INSTRUCTIONS_QRC: int = 12288
"""Maximum supported number of instructions in Q1ASM programs for QRC."""
MAX_NUMBER_OF_INSTRUCTIONS_QTM: int = 16384
"""Maximum supported number of instructions in Q1ASM programs for QTM."""
IMMEDIATE_SZ_GAIN = pow(2, 16)
"""Size of gain instruction immediates in Q1ASM programs."""
IMMEDIATE_MAX_WAIT_TIME = pow(2, 16) - 4
"""Max size of wait instruction immediates in Q1ASM programs. Max value allowed by
assembler is 2**16-1, but this is the largest that is a multiple of 4 ns."""
IMMEDIATE_SZ_OFFSET = pow(2, 16)
"""Size of offset instruction immediates in Q1ASM programs."""
REGISTER_SIZE = pow(2, 32) - 1
"""Size of registers in Q1ASM programs."""
NUMBER_OF_QBLOX_ACQ_INDICES: int = 32
"""Maximum number of Qblox acquisition index."""
NUMBER_OF_QBLOX_ACQ_BINS: int = 4096
"""Maximum number of Qblox acquisition bins for each index."""
NCO_PHASE_STEPS_PER_DEG = 1e9 / 360
"""The number of steps per degree for NCO phase instructions arguments."""
NCO_FREQ_STEPS_PER_HZ = 4.0
"""The number of steps per Hz for the NCO set_freq instruction."""
NCO_FREQ_LIMIT_STEPS = 2e9

GRID_TIME = 1  # ns
"""
Clock period of the sequencers. All time intervals used must be multiples of this value.
"""
MIN_TIME_BETWEEN_OPERATIONS = 4  # ns
"""
Minimum time between two operations to prevent FIFO errors.
"""
MIN_TIME_BETWEEN_NCO_OPERATIONS = 4  # ns
"""
Minimum time between two frequency updates or two phase updates..
"""
MIN_TIME_BETWEEN_ACQUISITIONS = 300  # ns
"""Minimum time between two acquisitions to prevent FIFO errors."""
SAMPLING_RATE = 1_000_000_000  # 1GS/s
"""Sampling rate of the Qblox control/readout instruments."""
STITCHED_PULSE_PART_DURATION_NS = 2000
"""Default duration of the individual waveforms that are used to build up a longer
stitched waveform. See
:func:`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_ramp_pulse` for an
example."""
PULSE_STITCHING_DURATION = 100e-9
"""
Duration of the individual pulses when pulse stitching is used.
Only applies to square pulses.
"""
PULSE_STITCHING_DURATION_RAMP = 2000e-9
"""Duration of the individual pulses when RampPulse is concerted to long_ramp_pulse."""
DEFAULT_MIXER_PHASE_ERROR_DEG = 0.0
"""Default phase shift in the instruments for mixer corrections.
"""
MIN_MIXER_PHASE_ERROR_DEG = -45
"""Lowest phase shift that can be configured in the instruments for mixer corrections.
"""
MAX_MIXER_PHASE_ERROR_DEG = 45
"""Highest phase shift that can be configured in the instruments for mixer corrections.
"""
DEFAULT_MIXER_AMP_RATIO = 1.0
"""Default value of the amplitude correction. N.B. This correction is defined
as Q/I."""
MIN_MIXER_AMP_RATIO = 0.5
"""Lowest value the amplitude correction can be set to. N.B. This correction is defined
as Q/I."""
MAX_MIXER_AMP_RATIO = 2.0
"""Highest value the amplitude correction can be set to. N.B. This correction is defined
as Q/I."""
NUMBER_OF_REGISTERS: int = 64
"""Number of registers available in the Qblox sequencers."""
MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS: int = 16384
"""Maximal amount of scope trace acquisition datapoints returned."""
MAX_SAMPLE_SIZE_WAVEFORMS: int = 16384
"""Maximal amount of samples in the waveforms to be uploaded to a sequencer."""
MIN_PHASE_ROTATION_ACQ = 0
"""Minimum value of the sequencer integration result phase rotation in degrees."""
MAX_PHASE_ROTATION_ACQ = 360
"""Maximum value of the sequencer integration result phase rotation in degrees."""
MIN_DISCRETIZATION_THRESHOLD_ACQ = -16777212.0
"""Minimum value of the sequencer discretization threshold
 for discretizing the phase rotation result."""
MAX_DISCRETIZATION_THRESHOLD_ACQ = 16777212.0
"""Maximum value of the sequencer discretization threshold
 for discretizing the phase rotation result."""
MAX_NUMBER_OF_BINS: int = 131072
"""Number of bins available in the Qblox sequencers."""
GENERIC_IC_COMPONENT_NAME: str = "generic"
"""Default name for the generic instrument coordinator component."""
TRIGGER_DELAY: float = 364e-9
"""
Total delay time of the feedback trigger before it is registered after the
end of a thresholded acquisition.
"""
MAX_FEEDBACK_TRIGGER_ADDRESS: int = 15
"""Available trigger addresses on each cluster range from 1,...,15."""
MAX_MIN_INSTRUCTION_WAIT: float = 4e-9
"""
Maximum of minimum wait times for real-time-instructions. e.g. play,
set_cond, acquire, require at least 4ns.
"""
GRID_TIME_TOLERANCE_TIME: float = 0.0011
"""
Tolerance for time values in nanoseconds.

.. versionadded:: 0.21.2
"""
QTM_FINE_DELAY_INT_TO_NS_RATIO = 128
"""
Ratio of the integer fine delay argument value to the actual delay in nanoseconds.

The fine delay argument has a resolution of 1/128 ns.
"""
MAX_QTM_FINE_DELAY_NS = ((1 << 11) - 1) / QTM_FINE_DELAY_INT_TO_NS_RATIO
"""
Maximum fine delay value in nanoseconds for QTM instructions that take a fine delay
argument.

The maximum integer value is based on an 11-bit unsigned integer.
"""
MIN_FINE_DELAY_SPACING_NS = 7  # ns
"""QTM instructions with unequal fine delay must be at least this far apart in time."""
