# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Contains registers that are reserved for a specific function by the qblox backend.
"""

# N.B. this implementation will be replaced with a dynamic allocation of registers
# in a future version. For more details see #190.

# Reserved register in the QASM assembly
REGISTER_LOOP: str = "R0"  # used as the main loop counter


# Registers reserved for managing acquisition registers
REGISTER_ACQ_CH0: str = "R20"
REGISTER_ACQ_CH1: str = "R21"
REGISTER_ACQ_CH2: str = "R22"
REGISTER_ACQ_CH3: str = "R23"
REGISTER_ACQ_CH4: str = "R24"
REGISTER_ACQ_CH5: str = "R25"

REGISTER_ACQ_BIN_IDX_CH0: str = "R26"
REGISTER_ACQ_BIN_IDX_CH1: str = "R27"
REGISTER_ACQ_BIN_IDX_CH2: str = "R28"
REGISTER_ACQ_BIN_IDX_CH3: str = "R29"
REGISTER_ACQ_BIN_IDX_CH4: str = "R30"
REGISTER_ACQ_BIN_IDX_CH5: str = "R31"

REGISTER_ACQ_WEIGHT_IDX0_CH0: str = "R32"
REGISTER_ACQ_WEIGHT_IDX1_CH0: str = "R33"
REGISTER_ACQ_WEIGHT_IDX0_CH1: str = "R34"
REGISTER_ACQ_WEIGHT_IDX1_CH1: str = "R35"
REGISTER_ACQ_WEIGHT_IDX0_CH2: str = "R36"
REGISTER_ACQ_WEIGHT_IDX1_CH2: str = "R37"
REGISTER_ACQ_WEIGHT_IDX0_CH3: str = "R38"
REGISTER_ACQ_WEIGHT_IDX1_CH3: str = "R39"
REGISTER_ACQ_WEIGHT_IDX0_CH4: str = "R40"
REGISTER_ACQ_WEIGHT_IDX1_CH4: str = "R41"
REGISTER_ACQ_WEIGHT_IDX0_CH5: str = "R42"
REGISTER_ACQ_WEIGHT_IDX1_CH5: str = "R45"
