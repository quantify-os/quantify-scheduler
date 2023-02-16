# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Constants for compilation to Qblox hardware."""


MAX_NUMBER_OF_INSTRUCTIONS_QCM: int = 16384
"""Maximum supported number of instructions in Q1ASM programs for QCM/QCM-RF."""
MAX_NUMBER_OF_INSTRUCTIONS_QRM: int = 12288
"""Maximum supported number of instructions in Q1ASM programs for QRM/QRM-RF."""
IMMEDIATE_SZ_GAIN = pow(2, 16) - 1
"""Size of gain instruction immediates in Q1ASM programs."""
IMMEDIATE_MAX_WAIT_TIME = pow(2, 16) - 4
"""Max size of wait instruction immediates in Q1ASM programs. Max value allowed by
assembler is 2**16-1, but this is the largest that is a multiple of 4 ns."""
IMMEDIATE_SZ_OFFSET = pow(2, 16) - 1
"""Size of offset instruction immediates in Q1ASM programs."""
REGISTER_SIZE = pow(2, 32) - 1
"""Size of registers in Q1ASM programs."""
NCO_PHASE_STEPS_PER_DEG = 1e9 / 360
"""The number of steps per degree for NCO phase instructions arguments."""
NCO_FREQ_STEPS_PER_HZ = 4.0
"""The number of steps per Hz for the NCO set_freq instruction."""
NCO_FREQ_LIMIT_STEPS = 2e9
"""The maximum and minimum frequency expressed in steps for the NCO set_freq instruction.
For the minimum we multiply by -1."""
NCO_SET_FREQ_WAIT = 8  # ns
"""Wait that is issued after NCO set_freq instruction, included via upd_param."""

GRID_TIME = 4  # ns
"""
Clock period of the sequencers. All time intervals used must be multiples of this value.
"""
MIN_TIME_BETWEEN_ACQUISITIONS = 1000  # ns
"""Minimum time between two acquisitions to prevent FIFO errors."""
SAMPLING_RATE = 1_000_000_000  # 1GS/s
"""Sampling rate of the Qblox control/readout instruments."""
PULSE_STITCHING_DURATION = 1e-6
"""Duration of the individual pulses when pulse stitching is used."""
MIN_MIXER_PHASE_ERROR_DEG = -45
"""Lowest phase shift that can be configured in the instruments for mixer corrections.
"""
MAX_MIXER_PHASE_ERROR_DEG = 45
"""Lowest phase shift that can be configured in the instruments for mixer corrections.
"""
MIN_MIXER_AMP_RATIO = 0.5
"""Lowest value the amplitude correction can be set to. N.B. This correction is defined
as Q/I."""
MAX_MIXER_AMP_RATIO = 2.0
"""Highest value the amplitude correction can be set to. N.B. This correction is defined
as Q/I."""
NUMBER_OF_SEQUENCERS_QCM = 6
"""Number of sequencers supported by a QCM/QCM-RF in the latest firmware."""
NUMBER_OF_SEQUENCERS_QRM = 6
"""Number of sequencers supported by a QRM/QRM-RF in the latest firmware."""
NUMBER_OF_REGISTERS: int = 64
"""Number of registers available in the Qblox sequencers."""
MAX_SAMPLE_SIZE_ACQUISITIONS: int = 16384
"""Maximal amount of acquisition datapoints returned."""
MAX_NUMBER_OF_BINS: int = 131072
"""Number of bins available in the Qblox sequencers."""
GENERIC_IC_COMPONENT_NAME: str = "generic"
"""Default name for the generic instrument coordinator component."""
