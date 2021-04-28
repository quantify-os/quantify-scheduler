# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Constants for compilation to Qblox hardware."""

IMMEDIATE_SZ = pow(2, 16) - 1
GRID_TIME = 4  # ns
SAMPLING_RATE = 1_000_000_000  # 1GS/s
PULSE_STITCHING_DURATION = 1e-6
