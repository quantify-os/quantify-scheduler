# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""
Module that holds all the string literals that are valid instructions that can be
executed by the sequencer in Qblox hardware.
"""
# Control

ILLEGAL = "illegal"
STOP = "stop"
NOP = "nop"
NEW_LINE = ""

# Jumps
JUMP = "jmp"
LOOP = "loop"
JUMP_GREATER_EQUALS = "jge"
JUMP_LESS_EQUALS = "jle"

# Arithmetic
MOVE = "move"
NOT = "not"
ADD = "add"
SUB = "sub"
AND = "and"
OR = "or"
XOR = "xor"
ARITHMETIC_SHIFT_LEFT = "asl"
ARITHMETIC_SHIFT_RIGHT = "asr"

# Real-time pipeline instructions
SET_MARKER = "set_mrk"
PLAY = "play"
ACQUIRE = "acquire"
ACQUIRE_WEIGHED = "acquire_weighed"
WAIT = "wait"
WAIT_SYNC = "wait_sync"
WAIT_TRIGGER = "wait_trigger"
UPDATE_PARAMETERS = "upd_param"
SET_AWG_GAIN = "set_awg_gain"
SET_ACQ_GAIN = "set_acq_gain"
SET_AWG_OFFSET = "set_awg_offs"
SET_ACQ_OFFSET = "set_acq_offs"
RESET_PHASE = "reset_ph"
SET_NCO_PHASE = "set_ph"
SET_NCO_PHASE_OFFSET = "set_ph_delta"
