# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Constants for compilation to Qblox hardware."""


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
"""The number of steps per degree for nco phase instructions arguments."""

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
NUMBER_OF_SEQUENCERS_QCM = 6
"""Number of sequencers supported by a QCM in the latest firmware."""
NUMBER_OF_SEQUENCERS_QRM = 6
"""Number of sequencers supported by a QRM in the latest firmware."""
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
NUMBER_OF_REGISTERS: int = 64
"""Number of registers available in the Qblox sequencers."""
GENERIC_IC_COMPONENT_NAME: str = "generic"
"""Default name for the generic instrument coordinator component."""
MAX_SAMPLE_SIZE_ACQUISITIONS: int = 16384
"""Maximal amount of acquisition datapoints returned."""
