# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
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
JUMP_LESS_THAN = "jlt"

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

# Parameter operations
SET_MARKER = "set_mrk"
SET_FREQUENCY = "set_freq"
RESET_PHASE = "reset_ph"
SET_NCO_PHASE_OFFSET = "set_ph"
INCR_NCO_PHASE_OFFSET = "set_ph_delta"
SET_AWG_GAIN = "set_awg_gain"
SET_AWG_OFFSET = "set_awg_offs"
SET_DIGITAL = "set_digital"
SET_TIME_REF = "set_time_ref"
SET_SCOPE_EN = "set_scope_en"

# Real-time pipeline instructions
PLAY = "play"
ACQUIRE = "acquire"
ACQUIRE_WEIGHED = "acquire_weighed"
ACQUIRE_TTL = "acquire_ttl"
WAIT = "wait"
WAIT_SYNC = "wait_sync"
WAIT_TRIGGER = "wait_trigger"
UPDATE_PARAMETERS = "upd_param"
FEEDBACK_SET_COND = "set_cond"
FEEDBACK_TRIGGER_EN = "set_latch_en"
FEEDBACK_TRIGGERS_RST = "latch_rst"
PLAY_PULSE = "play_pulse"
ACQUIRE_TIMETAGS = "acquire_timetags"
ACQUIRE_DIGITAL = "acquire_digital"
UPD_THRES = "upd_thres"
