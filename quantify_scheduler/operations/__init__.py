# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing the standard library of commonly used operations as well as the
:class:`.Operation` class.


.. tip::

    Quantify scheduler can trivially be extended by creating custom operations. Take a
    look at e.g., the pulse library for examples on how to implement custom pulses.

"""
from .acquisition_library import (
    Acquisition,
    NumericalSeparatedWeightedIntegration,
    NumericalWeightedIntegration,
    SSBIntegrationComplex,
    ThresholdedAcquisition,
    Timetag,
    TimetagTrace,
    Trace,
    TriggerCount,
    WeightedIntegratedSeparated,
)
from .control_flow_library import (
    ConditionalOperation,
    ControlFlowOperation,
    ControlFlowSpec,
    LoopOperation,
)
from .gate_library import CNOT, CZ, X90, Y90, Z90, H, Measure, Reset, Rxy, Rz, X, Y, Z
from .nv_native_library import ChargeReset, CRCount
from .operation import Operation
from .pulse_factories import (
    composite_square_pulse,
    nv_spec_pulse_mw,
    phase_shift,
    rxy_drag_pulse,
    rxy_gauss_pulse,
    rxy_hermite_pulse,
    spin_init_pulse,
)
from .pulse_library import (
    ChirpPulse,
    DRAGPulse,
    GaussPulse,
    IdlePulse,
    MarkerPulse,
    NumericalPulse,
    RampPulse,
    ReferenceMagnitude,
    ResetClockPhase,
    SetClockFrequency,
    ShiftClockPhase,
    SkewedHermitePulse,
    SoftSquarePulse,
    SquarePulse,
    StaircasePulse,
    SuddenNetZeroPulse,
    Timestamp,
    VoltageOffset,
    WindowOperation,
)

__all__ = [
    "CNOT",
    "CZ",
    "X90",
    "Y90",
    "Z90",
    "H",
    "Measure",
    "Reset",
    "Rxy",
    "Rz",
    "X",
    "Y",
    "Z",
    "ChargeReset",
    "CRCount",
    "composite_square_pulse",
    "nv_spec_pulse_mw",
    "spin_init_pulse",
    "phase_shift",
    "rxy_drag_pulse",
    "rxy_gauss_pulse",
    "rxy_hermite_pulse",
    "ChirpPulse",
    "DRAGPulse",
    "GaussPulse",
    "IdlePulse",
    "MarkerPulse",
    "NumericalPulse",
    "RampPulse",
    "ReferenceMagnitude",
    "ResetClockPhase",
    "SetClockFrequency",
    "ShiftClockPhase",
    "SkewedHermitePulse",
    "SoftSquarePulse",
    "SquarePulse",
    "StaircasePulse",
    "SuddenNetZeroPulse",
    "Timestamp",
    "VoltageOffset",
    "WindowOperation",
    "Operation",
    "ConditionalOperation",
    "ControlFlowSpec",
    "ControlFlowOperation",
    "LoopOperation",
    "Acquisition",
    "Trace",
    "WeightedIntegratedSeparated",
    "SSBIntegrationComplex",
    "ThresholdedAcquisition",
    "NumericalSeparatedWeightedIntegration",
    "NumericalWeightedIntegration",
    "TriggerCount",
    "TimetagTrace",
    "Timetag",
]
