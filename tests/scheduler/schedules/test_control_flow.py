from unittest.mock import Mock

import pytest

from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.backends.qblox.analog import (
    AnalogSequencerCompiler,
    BasebandModuleCompiler,
)
from quantify_scheduler.backends.qblox.conditional import (
    FeedbackTriggerCondition,
    FeedbackTriggerOperator,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    ConditionalStrategy,
)
from quantify_scheduler.backends.qblox_backend import _SequencerCompilationConfig
from quantify_scheduler.backends.types.common import ModulationFrequencies
from quantify_scheduler.backends.types.qblox import (
    AnalogSequencerSettings,
    BoundedParameter,
    ComplexChannelDescription,
    OpInfo,
    SequencerOptions,
    StaticAnalogModuleProperties,
)
from quantify_scheduler.operations.control_flow_library import (
    Conditional,
    ConditionalOperation,
    Loop,
    LoopOperation,
)
from quantify_scheduler.operations.gate_library import Measure, Rxy, X
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.schemas.examples import utils

from .compiles_all_backends import _CompilesAllBackends


class TestSubschedules(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        inner_schedule = Schedule("inner", repetitions=1)
        ref = inner_schedule.add(Rxy(0, 0, "q0"), label="inner0")
        inner_schedule.add(Rxy(0, 1, "q0"), rel_time=40e-9, ref_op=ref, label="inner1")

        outer_schedule = Schedule("outer", repetitions=10)
        ref = outer_schedule.add(Rxy(1, 0, "q0"), label="outer0")
        outer_schedule.add(inner_schedule, rel_time=80e-9, ref_op=ref)
        outer_schedule.add(Measure("q0"), label="measure")
        cls.uncomp_sched = outer_schedule

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == 10


class TestLoops:
    @classmethod
    def setup_class(cls):
        inner_schedule = Schedule("inner", repetitions=1)
        ref = inner_schedule.add(Rxy(0, 0, "q0"), label="inner0")
        inner_schedule.add(Rxy(0, 1, "q0"), rel_time=40e-9, ref_op=ref, label="inner1")

        outer_schedule = Schedule("outer", repetitions=1)
        ref = outer_schedule.add(Rxy(1, 0, "q0"), label="outer0")

        outer_schedule.add(
            LoopOperation(body=inner_schedule, repetitions=10),
            label="loop",
        )

        outer_schedule.add(Measure("q0"), label="measure")
        cls.uncomp_sched = outer_schedule

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == 1


def test_multiple_conditional_without_acquisition_raises(
    mock_setup_basic_transmon_with_standard_params,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    hardware_config = utils.load_json_example_scheme("qblox_hardware_config_transmon.json")
    quantum_device.hardware_config(hardware_config)
    config = quantum_device.generate_compilation_config()

    schedule = Schedule("")
    schedule1 = Schedule("")
    schedule1.add(X("q0"))

    schedule.add(
        ConditionalOperation(body=schedule1, qubit_name="q0"),
    )
    schedule.add(
        ConditionalOperation(body=schedule1, qubit_name="q0"),
    )

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        RuntimeError,
        match=(
            "Conditional control flow, "
            "``ConditionalOperation"
            '\\(body=Schedule "" containing \\(1\\) 1  \\(unique\\) operations.'
            ",qubit_name='q0',t0=0.0,hardware_buffer_time=0.0\\)``,  "
            "found without a preceding Conditional acquisition. "
        ),
    ):
        _ = compiler.compile(
            schedule,
            config=config,
        )


def test_nested_conditional_control_flow_raises_runtime_warning():
    static_hw_properties = StaticAnalogModuleProperties(
        instrument_type="QRM",
        max_sequencers=6,
        max_awg_output_voltage=None,
        mixer_dc_offset_range=BoundedParameter(0, 0, ""),
    )
    mock_parent_module = Mock(BasebandModuleCompiler)
    sequencer_cfg = _SequencerCompilationConfig(
        sequencer_options=SequencerOptions(),
        hardware_description=ComplexChannelDescription(),
        portclock="q1:mw-q1.01",
        channel_name="complex_out_0",
        channel_name_measure=None,
        latency_correction=0,
        distortion_correction=None,
        lo_name=None,
        modulation_frequencies=ModulationFrequencies.model_validate(
            {"lo_freq": None, "interm_freq": 50e6}
        ),
        mixer_corrections=None,
    )
    sequencer = AnalogSequencerCompiler(
        parent=mock_parent_module,
        index=0,
        static_hw_properties=static_hw_properties,
        sequencer_cfg=sequencer_cfg,
    )

    sequencer.op_strategies = [
        ConditionalStrategy(
            operation_info=OpInfo("Conditional", {}, 0),
            trigger_condition=FeedbackTriggerCondition(
                enable=True, operator=FeedbackTriggerOperator.OR, addresses=[1]
            ),
        ),
        ConditionalStrategy(
            operation_info=OpInfo("Conditional", {}, 0),
            trigger_condition=FeedbackTriggerCondition(
                enable=True, operator=FeedbackTriggerOperator.OR, addresses=[1]
            ),
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match="Nested conditional playback inside schedules "
        "is not supported by the Qblox backend.",
    ):
        sequencer.generate_qasm_program(
            ordered_op_strategies=sequencer._get_ordered_operations(),
            total_sequence_time=0,
            align_qasm_fields=False,
            acq_metadata=None,
            repetitions=1,
        )


def test_deprecated_control_flow_loop_warns():
    schedule = Schedule("Test")
    with pytest.warns(
        FutureWarning,
        match="Using the `control_flow` argument in `Schedule.add` is deprecated, "
        "and will be removed from the public interface",
    ):
        schedule.add(X("q0"), control_flow=Loop(3))


def test_deprecated_control_flow_conditional_warns():
    schedule = Schedule("Test")
    with pytest.warns(
        FutureWarning,
        match="Using the `control_flow` argument in `Schedule.add` is deprecated, "
        "and will be removed from the public interface",
    ):
        schedule.add(X("q0"), control_flow=Conditional("q0"))
