import re

import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.qblox.instrument_compilers import QTMCompiler
from quantify_scheduler.backends.qblox.operation_handling.pulses import MarkerPulseStrategy
from quantify_scheduler.backends.types.qblox import (
    OpInfo,
    QCMDescription,
    QCMRFDescription,
    QRMDescription,
    QRMRFDescription,
    QTMDescription,
    RFDescription,
    StaticTimetagModuleProperties,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    SquarePulse,
)
from quantify_scheduler.resources import ClockResource


def test_constructor():
    op_info = OpInfo(name="", data={}, timing=0)
    strategy = MarkerPulseStrategy(
        operation_info=op_info,
        channel_name="digital_output_0",
        module_options=QRMDescription(),
    )
    assert strategy._pulse_info is op_info
    assert strategy.channel_name == "digital_output_0"


def test_operation_info_property():
    # arrange
    operation_info = OpInfo(name="", data={}, timing=0)
    strategy = MarkerPulseStrategy(
        operation_info=operation_info,
        channel_name="digital_output_0",
        module_options=QRMDescription(),
    )

    # act
    from_property = strategy.operation_info

    # assert
    assert operation_info == from_property


def test_generate_data():
    # arrange
    strategy = MarkerPulseStrategy(
        operation_info=OpInfo(name="", data={}, timing=0),
        channel_name="digital_output_0",
        module_options=QRMDescription(),
    )

    # this is what we want to verify
    data = strategy.generate_data({})

    assert data is None


def test_marker_pulse_compilation_qrm(mock_setup_basic_transmon_with_standard_params):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",  # noqa: E501, line too long
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QRM", "digital_output_1": {}}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_input_0", "q0:res"],
                ["cluster0.module1.digital_output_1", "q0:switch"],
            ]
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test MarkerPulse compilation")
    schedule.add(
        MarkerPulse(
            duration=500e-9,
            port="q0:switch",
        ),
    )
    schedule.add(
        Measure("q0", acq_protocol="SSBIntegrationComplex"),
        rel_time=300e-9,
        ref_pt="start",
    )
    schedule.add_resource(ClockResource(name="q0.res", freq=50e6))

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Assert markers were set correctly, and wait time is correct for QRM
    seq0_analog = (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq0"]
        .sequence["program"]
        .splitlines()
    )
    seq1_digital = (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq1"]
        .sequence["program"]
        .splitlines()
    )
    idx = 0
    for i, string in enumerate(seq0_analog):
        if re.search(r"^\s*reset_ph\s+", string):
            idx = i
            break
    assert re.search(r"^\s*wait\s+300\s*($|#)", seq0_analog[idx + 2])
    idx = 0
    for i, string in enumerate(seq1_digital):
        if re.search(r"^\s*set_mrk\s+2\s*($|#)", string):
            idx = i
            break
    assert idx > 0
    assert re.search(r"^\s*upd_param\s+4\s*($|#)", seq1_digital[idx + 1])
    assert re.search(r"^\s*wait\s+496\s*($|#)", seq1_digital[idx + 2])
    assert re.search(r"^\s*set_mrk\s+0\s*($|#)", seq1_digital[idx + 3])


def test_marker_pulse_compilation_qcm_rf(mock_setup_basic_transmon_with_standard_params):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",  # noqa: E501, line too long
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM_RF", "digital_output_1": {}}},
                "ref": "internal",
            }
        },
        "hardware_options": {"modulation_frequencies": {"q0:res-q0.ro": {"interm_freq": 0}}},
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:res"],
                ["cluster0.module1.digital_output_1", "q0:switch"],
            ]
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test MarkerPulse compilation")
    schedule.add(
        MarkerPulse(
            duration=500e-9,
            port="q0:switch",
        ),
    )
    schedule.add(
        SquarePulse(amp=0.2, duration=300e-9, port="q0:res", clock="q0.ro"),
        rel_time=300e-9,
        ref_pt="start",
    )
    schedule.add_resource(ClockResource(name="q0.res", freq=50e6))
    # Generate compiled schedule for QCM
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Assert markers were set correctly, and wait time is correct for QRM
    seq0_analog = (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq0"]
        .sequence["program"]
        .splitlines()
    )
    seq1_digital = (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq1"]
        .sequence["program"]
        .splitlines()
    )
    idx = 0
    for i, string in enumerate(seq0_analog):
        if re.search(r"^\s*reset_ph\s+", string):
            idx = i
            break
    assert re.search(r"^\s*wait\s+300\s*($|#)", seq0_analog[idx + 2])
    idx = 0
    for i, string in enumerate(seq1_digital):
        if re.search(r"^\s*set_mrk\s+7\s*($|#)", string):
            idx = i
            break
    assert re.search(r"^\s*upd_param\s+4\s*($|#)", seq1_digital[idx + 1])
    assert re.search(r"^\s*wait\s+496\s*($|#)", seq1_digital[idx + 2])
    assert re.search(r"^\s*set_mrk\s+3\s*($|#)", seq1_digital[idx + 3])


def test_marker_pulse_added_to_operation():
    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",  # noqa: E501, line too long
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM_RF", "digital_output_1": {}}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q0:mw-q0.01": {"interm_freq": 100000000.0}}
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:mw"],
                ["cluster0.module1.digital_output_1", "q0:switch"],
            ]
        },
    }

    quantum_device = QuantumDevice("marker_test_device")
    quantum_device.hardware_config(hw_config)

    # Define experiment schedule
    schedule = Schedule("test MarkerPulse add to Operation")
    schedule.add_resource(ClockResource(name="q0.01", freq=5.1e9))
    square_pulse_op = SquarePulse(amp=0.5, duration=1e-9, port="q0:mw", clock="q0.01")
    square_pulse_op.add_pulse(MarkerPulse(duration=100e-9, port="q0:switch", t0=40e-9))
    schedule.add(square_pulse_op)
    schedule.add(IdlePulse(4e-9))

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    _ = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())


class TestInsertQasm:
    def test_non_supported_channel_name(self, empty_qasm_program_qcm):
        strategy = MarkerPulseStrategy(
            operation_info=OpInfo(name="test_pulse", data={}, timing=0),
            channel_name="complex_output_0",
            module_options=QCMDescription(),
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unable to set markers on channel 'complex_output_0' "
                "for instrument QCM and operation test_pulse. "
                "Supported channels: "
            ),
        ):
            strategy.insert_qasm(empty_qasm_program_qcm)

    @staticmethod
    def _assert_correct_markers(
        program, channel_name, channel_marker, default_marker, module_options
    ):
        # If the output is enabled, this default is OR'ed together with the channel marker
        strategy = MarkerPulseStrategy(
            operation_info=OpInfo(name="test_pulse", data={"enable": True}, timing=0),
            channel_name=channel_name,
            module_options=module_options,
        )
        strategy.insert_qasm(program)
        assert len(program.instructions) == 1
        assert program.instructions[0][1] == "set_mrk"
        assert program.instructions[0][2] == str(channel_marker | default_marker)
        # if the output is turned off again, the marker is set back to default
        strategy = MarkerPulseStrategy(
            operation_info=OpInfo(name="test_pulse", data={"enable": False}, timing=0),
            channel_name=channel_name,
            module_options=module_options,
        )
        strategy.insert_qasm(program)
        assert len(program.instructions) == 2
        assert program.instructions[1][1] == "set_mrk"
        assert program.instructions[1][2] == str(default_marker)

    @pytest.mark.parametrize("rf_output_on", [True, False])
    @pytest.mark.parametrize(
        "expected_markers",
        [
            ("digital_output_0", 8, 3),
            ("digital_output_1", 4, 3),
            ("complex_output_0", 1, 1),
            ("complex_output_1", 2, 2),
        ],
        ids=["digital0", "digital1", "complex0", "complex1"],
    )
    def test_default_marker_qcm_rf(
        self, expected_markers: tuple[str, int, int], rf_output_on, empty_qasm_program_qcm_rf
    ):
        channel_name, channel_marker, default_marker = expected_markers
        program = empty_qasm_program_qcm_rf
        assert program.static_hw_properties.default_markers[channel_name] == default_marker
        description = QCMRFDescription()
        assert description.rf_output_on is True
        # if the rf output is on, the default marker is 3.
        default_marker = default_marker if rf_output_on else 0

        description.rf_output_on = rf_output_on
        if rf_output_on and "complex" in channel_name:
            strategy = MarkerPulseStrategy(
                operation_info=OpInfo(name="test_pulse", data={"enable": True}, timing=0),
                channel_name=channel_name,
                module_options=description,
            )
            with pytest.raises(RuntimeError):
                strategy.insert_qasm(program)
            return
        self._assert_correct_markers(
            program, channel_name, channel_marker, default_marker, description
        )

    @pytest.mark.parametrize("rf_output_on", [True, False])
    @pytest.mark.parametrize(
        "expected_markers",
        [
            ("digital_output_0", 4, 2),
            ("digital_output_1", 8, 2),
            ("complex_output_0", 2, 2),
        ],
        ids=["digital0", "digital1", "complex0"],
    )
    def test_default_marker_qrm_rf(
        self, expected_markers: tuple[str, int, int], rf_output_on, empty_qasm_program_qrm_rf
    ):
        channel_name, channel_marker, default_marker = expected_markers
        program = empty_qasm_program_qrm_rf
        assert program.static_hw_properties.default_markers[channel_name] == default_marker
        description = QRMRFDescription()
        assert description.rf_output_on is True
        # if the rf output is on, the default marker is 2. (LSB is not used)
        default_marker = 2 if rf_output_on else 0

        description.rf_output_on = rf_output_on
        if rf_output_on and "complex" in channel_name:
            strategy = MarkerPulseStrategy(
                operation_info=OpInfo(name="test_pulse", data={"enable": True}, timing=0),
                channel_name=channel_name,
                module_options=description,
            )
            with pytest.raises(RuntimeError):
                strategy.insert_qasm(program)
            return
        self._assert_correct_markers(
            program, channel_name, channel_marker, default_marker, description
        )

    @pytest.mark.parametrize(
        "expected_markers",
        [
            ("digital_output_0", 1),
            ("digital_output_1", 2),
            ("digital_output_2", 4),
            ("digital_output_3", 8),
        ],
        ids=["digital0", "digital1", "digital2", "digital3"],
    )
    def test_default_marker_qrm(self, expected_markers: tuple[str, int], empty_qasm_program_qrm):
        channel_name, channel_marker = expected_markers
        program = empty_qasm_program_qrm
        assert program.static_hw_properties.default_markers is None
        # QRM will always have a default marker of 0
        default_marker = 0
        self._assert_correct_markers(
            program, channel_name, channel_marker, default_marker, QRMDescription()
        )

    @pytest.mark.parametrize(
        "expected_markers",
        [
            ("digital_output_0", 1),
            ("digital_output_1", 2),
            ("digital_output_2", 4),
            ("digital_output_3", 8),
        ],
        ids=["digital0", "digital1", "digital2", "digital3"],
    )
    def test_default_marker_qcm(self, expected_markers: tuple[str, int], empty_qasm_program_qcm):
        channel_name, channel_marker = expected_markers
        program = empty_qasm_program_qcm
        assert program.static_hw_properties.default_markers is None
        # QCM will always have a default marker of 0
        default_marker = 0
        self._assert_correct_markers(
            program, channel_name, channel_marker, default_marker, QCMRFDescription()
        )

    def test_default_marker_qtm(self, empty_qasm_program_qtm):
        # QTM does not support marker pulses
        program = empty_qasm_program_qtm
        strategy = MarkerPulseStrategy(
            operation_info=OpInfo(name="test_pulse", data={"enable": True}, timing=0),
            channel_name="",
            module_options=QTMDescription(),
        )
        with pytest.raises(
            TypeError,
            match="Marker Operations are only supported for analog modules, "
            "not for instrument QTM.",
        ):
            strategy.insert_qasm(program)
