# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Constants for compilation to Qblox hardware."""

IMMEDIATE_SZ_GAIN = pow(2, 16) - 1
"""Size of gain instruction immediates in Q1ASM programs."""
IMMEDIATE_SZ_WAIT = pow(2, 16) - 1
"""Size of wait instruction immediates in Q1ASM programs."""
IMMEDIATE_SZ_OFFSET = pow(2, 16) - 1
"""Size of offset instruction immediates in Q1ASM programs."""
REGISTER_SIZE = pow(2, 32) - 1
"""Size of registers in Q1ASM programs."""

GRID_TIME = 4  # ns
"""
Clock period of the sequencers. All time intervals used must be multiples of this value.
"""
MIN_TIME_BETWEEN_ACQUISITIONS = 1000  # ns
"""Minimum time between two acquisitions to prevent FIFO errors."""
SAMPLING_RATE = 1_000_000_000  # 1GS/s
"""Sampling rate of the Qblox Pulsar series instruments."""
PULSE_STITCHING_DURATION = 1e-6
"""Duration of the individual pulses when pulse stitching is used."""
NUMBER_OF_SEQUENCERS_QCM = 6
"""Number of sequencers supported by Pulsar QCM in the latest firmware."""
NUMBER_OF_SEQUENCERS_QRM = 6
"""Number of sequencers supported by Pulsar QRM in the latest firmware."""
NUMBER_OF_REGISTERS: int = 64
"""Number of registers available in the Qblox sequencers."""
