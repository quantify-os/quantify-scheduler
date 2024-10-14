# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox backend."""
import copy
import itertools
import json
import math
import os
import re
from contextlib import nullcontext
from copy import deepcopy
from typing import Optional

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError
from qblox_instruments import Cluster, ClusterType

import quantify_scheduler
from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler, corrections
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.qblox import (
    compiler_container,
    constants,
    q1asm_instructions,
    register_manager,
)
from quantify_scheduler.backends.qblox.analog import (
    AnalogSequencerCompiler,
)
from quantify_scheduler.backends.qblox.compiler_container import CompilerContainer
from quantify_scheduler.backends.qblox.enums import (
    DistortionCorrectionLatencyEnum,
    LoCalEnum,
    QbloxFilterConfig,
    QbloxFilterMarkerDelay,
    SidebandCalEnum,
)
from quantify_scheduler.backends.qblox.exceptions import NcoOperationTimingError
from quantify_scheduler.backends.qblox.hardware_config_transmon_old_style import (
    hardware_config as hardware_config_transmon_old_style,
)
from quantify_scheduler.backends.qblox.helpers import (
    assign_pulse_and_acq_info_to_devices,
    generate_port_clock_to_device_map,
    generate_uuid_from_wf_data,
    generate_waveform_data,
    is_multiple_of_grid_time,
    to_grid_time,
)
from quantify_scheduler.backends.qblox.instrument_compilers import (
    ClusterCompiler,
    QCMCompiler,
    QCMRFCompiler,
    QRMCompiler,
    QRMRFCompiler,
)
from quantify_scheduler.backends.qblox.operations import (
    long_ramp_pulse,
    long_square_pulse,
    staircase_pulse,
)
from quantify_scheduler.backends.qblox.operations.stitched_pulse import (
    StitchedPulseBuilder,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox_backend import (
    QbloxHardwareCompilationConfig,
    _ClusterCompilationConfig,
    _ClusterModuleCompilationConfig,
    find_qblox_instruments,
)
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.types.common import HardwareDescription
from quantify_scheduler.backends.types.qblox import (
    BasebandModuleSettings,
    QbloxHardwareDistortionCorrection,
)
from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.helpers.collections import (
    find_all_port_clock_combinations,
    find_inner_dicts_containing_key,
)
from quantify_scheduler.helpers.schedule import _extract_port_clocks_used
from quantify_scheduler.operations.acquisition_library import (
    SSBIntegrationComplex,
    Trace,
)
from quantify_scheduler.operations.control_flow_library import (
    ConditionalOperation,
    LoopOperation,
)
from quantify_scheduler.operations.gate_library import CZ, X90, Measure, Reset, X, Y
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import (
    DRAGPulse,
    IdlePulse,
    MarkerPulse,
    NumericalPulse,
    RampPulse,
    ReferenceMagnitude,
    ResetClockPhase,
    SetClockFrequency,
    ShiftClockPhase,
    SoftSquarePulse,
    SquarePulse,
)
from quantify_scheduler.resources import BasebandClockResource, ClockResource
from quantify_scheduler.schedules.timedomain_schedules import (
    allxy_sched,
    readout_calibration_sched,
)

REGENERATE_REF_FILES: bool = False  # Set flag to true to regenerate the reference files


# --------- Test fixtures ---------


@pytest.fixture
def dummy_cluster():
    cluster: Optional[Cluster] = None

    def _dummy_cluster(
        name: str = "cluster0",
        dummy_cfg: dict = None,
    ) -> Cluster:
        nonlocal cluster
        cluster = Cluster(
            name=name,
            dummy_cfg=(
                dummy_cfg
                if dummy_cfg is not None
                else {2: ClusterType.CLUSTER_QCM, 4: ClusterType.CLUSTER_QRM}
            ),
        )
        return cluster

    yield _dummy_cluster


@pytest.fixture
def pulse_only_schedule():
    sched = Schedule("pulse_only_experiment")
    sched.add(IdlePulse(duration=200e-6))
    sched.add(
        DRAGPulse(
            G_amp=0.5,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
    return sched


@pytest.fixture
def cluster_only_schedule():
    sched = Schedule("cluster_only_schedule")
    sched.add(Reset("q4"))
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q4:mw",
            duration=20e-9,
            clock="q4.01",
            t0=4e-9,
        )
    )
    sched.add(
        DRAGPulse(
            G_amp=0.2,
            D_amp=-0.2,
            phase=90,
            port="q5:mw",
            duration=20e-9,
            clock="q5.01",
            t0=4e-9,
        )
    )
    sched.add(RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q4:mw", clock="q4.01"))
    return sched


@pytest.fixture
def pulse_only_schedule_multiplexed():
    sched = Schedule("pulse_only_experiment")
    sched.add(IdlePulse(duration=200e-6))
    operation = sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    for i in range(1, 4):
        sched.add(
            DRAGPulse(
                G_amp=0.7,
                D_amp=-0.2,
                phase=90,
                port=f"q{i}:mw",
                duration=20e-9,
                clock="q0.01",
                t0=8e-9,
            ),
            ref_op=operation,
            ref_pt="start",
        )

    sched.add(RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
    return sched


@pytest.fixture
def pulse_only_schedule_no_lo():
    sched = Schedule("pulse_only_schedule_no_lo")
    sched.add(IdlePulse(duration=200e-6))
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q1:res",
            duration=20e-9,
            clock="q1.ro",
            t0=4e-9,
        )
    )
    return sched


@pytest.fixture
def identical_pulses_schedule():
    sched = Schedule("identical_pulses_schedule")
    sched.add(IdlePulse(duration=200e-6))
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(
        DRAGPulse(
            G_amp=0.8,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=0,
        )
    )
    return sched


@pytest.fixture
def pulse_only_schedule_with_operation_timing():
    sched = Schedule("pulse_only_schedule_with_operation_timing")
    sched.add(IdlePulse(duration=200e-6))
    first_op = sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(
        RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"),
        ref_op=first_op,
        ref_pt="end",
        rel_time=1e-3,
    )
    return sched


@pytest.fixture
def mixed_schedule_with_acquisition():
    sched = Schedule("mixed_schedule_with_acquisition")
    sched.add(Reset("q0"))
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )

    sched.add(Measure("q0"))
    return sched


@pytest.fixture
def gate_only_schedule():
    sched = Schedule("gate_only_schedule")
    sched.add(Reset("q0"))
    x_gate = sched.add(X("q0"))
    sched.add(Measure("q0"), ref_op=x_gate, rel_time=1e-6, ref_pt="end")
    return sched


@pytest.fixture
def duplicate_measure_schedule():
    sched = Schedule("gate_only_schedule")
    sched.add(Reset("q0"))
    x_gate = sched.add(X("q0"))
    sched.add(Measure("q0", acq_index=0), ref_op=x_gate, rel_time=1e-6, ref_pt="end")
    sched.add(Measure("q0", acq_index=1), ref_op=x_gate, rel_time=3e-6, ref_pt="end")
    return sched


@pytest.fixture
def baseband_square_pulse_schedule():
    sched = Schedule("baseband_square_pulse_schedule")
    sched.add(Reset("q0"))
    sched.add(
        SquarePulse(
            amp=0.0,
            duration=2.5e-6,
            port="q0:mw",
            clock=BasebandClockResource.IDENTITY,
            t0=1e-6,
        )
    )
    sched.add(
        SquarePulse(
            amp=2.0 / 5.0,
            duration=2.5e-6,
            port="q0:mw",
            clock=BasebandClockResource.IDENTITY,
            t0=1e-6,
        )
    )
    return sched


@pytest.fixture
def real_square_pulse_schedule():
    sched = Schedule("real_square_pulse_schedule")
    sched.add(IdlePulse(duration=200e-6))
    sched.add(
        SquarePulse(
            amp=1.0,
            duration=5e-7,
            port="q0:fl",
            clock=BasebandClockResource.IDENTITY,
            t0=1e-6,
        )
    )
    sched.add(
        SquarePulse(
            amp=0.5,
            duration=7e-7,
            port="q1:fl",
            clock=BasebandClockResource.IDENTITY,
            t0=0.5e-6,
        )
    )
    sched.add(
        SquarePulse(
            amp=1.2 / 5.0,
            duration=9e-7,
            port="q2:fl",
            clock=BasebandClockResource.IDENTITY,
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=1.2 / 5.0,
            duration=9e-7,
            port="q3:fl",
            clock=BasebandClockResource.IDENTITY,
            t0=0,
        )
    )
    return sched


@pytest.fixture(name="empty_qasm_program_qcm")
def fixture_empty_qasm_program():
    return QASMProgram(
        static_hw_properties=QCMCompiler.static_hw_properties,
        register_manager=register_manager.RegisterManager(),
        align_fields=True,
        acq_metadata=None,
    )


# --------- Test utility functions ---------
def function_for_test_generate_waveform_data(t, x, y):
    return x * t + y


def test_generate_waveform_data():
    x = 10
    y = np.pi
    duration = 1e-8
    sampling_rate = 1e9

    t_verification = np.arange(start=0, stop=0 + duration, step=1 / sampling_rate)
    verification_data = function_for_test_generate_waveform_data(t_verification, x, y)

    data_dict = {
        "wf_func": __name__ + ".function_for_test_generate_waveform_data",
        "x": x,
        "y": y,
        "duration": duration,
    }
    gen_data = generate_waveform_data(data_dict, sampling_rate)

    assert np.allclose(gen_data, verification_data)


@pytest.mark.parametrize(
    "sampling_rate, duration, sample_size",
    [
        (6.1e-08, 1e9, 61),
        (6.1999e-08, 1e9, 62),
        (6.2001e-08, 1e9, 62),
        (6.249e-08, 1e9, 62),
        (6.25e-08, 1e9, 62),
        (6.31e-08, 1e9, 63),
    ],
)
def test_generate_waveform_data_sample_size(duration, sampling_rate, sample_size):
    data_dict = {
        "wf_func": __name__ + ".function_for_test_generate_waveform_data",
        "x": 10,
        "y": np.pi,
        "duration": duration,
    }
    gen_data = generate_waveform_data(data_dict, sampling_rate)

    assert (
        len(gen_data) == sample_size
    ), f"Sample size {sample_size} is integer nearest to {duration * sampling_rate}"


def test_find_inner_dicts_containing_key():
    test_dict = {
        "foo": "bar",
        "list": [{"key": 1, "hello": "world", "other_key": "other_value"}, 4, "12"],
        "nested": {"hello": "world", "other_key": "other_value"},
    }
    dicts_found = find_inner_dicts_containing_key(test_dict, "hello")
    assert len(dicts_found) == 2
    for inner_dict in dicts_found:
        assert inner_dict["hello"] == "world"
        assert inner_dict["other_key"] == "other_value"


def test_find_all_port_clock_combinations(
    hardware_cfg_cluster_legacy,
    hardware_cfg_rf_legacy,
):
    combined_hw_cfg = copy.deepcopy(hardware_cfg_cluster_legacy)
    combined_hw_cfg["cluster1"] = hardware_cfg_rf_legacy["cluster0"]
    combined_hw_cfg["cluster2"] = hardware_config_transmon_old_style["cluster0"]

    assert set(find_all_port_clock_combinations(combined_hw_cfg)) == {
        ("q1:mw", "q1.01"),
        ("q0:mw", "q0.01"),
        ("q0:res", "q0.ro"),
        ("q0:res", "q0.multiplex"),
        ("q1:res", "q1.ro"),
        ("q3:mw", "q3.01"),
        ("q2:mw", "q2.01"),
        ("q2:res", "q2.ro"),
        ("q3:mw", "q3.01"),
        ("q4:mw", "q4.01"),
        ("q5:res", "q5.ro"),
        ("q5:mw", "q5.01"),
        ("q6:mw", "q6.01"),
        ("q7:mw", "q7.01"),
        ("q4:res", "q4.ro"),
        ("q0:fl", "cl0.baseband"),
        ("q1:fl", "cl0.baseband"),
        ("q2:fl", "cl0.baseband"),
        ("q3:fl", "cl0.baseband"),
        ("q4:fl", "cl0.baseband"),
        ("qe0:optical_readout", "qe0.ge0"),
        ("q0:switch", "digital"),
    }


def test_generate_port_clock_to_device_map(
    compile_config_basic_transmon_qblox_hardware, create_schedule_with_pulse_info
):
    hardware_config = (
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config
    )

    sched = Schedule("All portclocks schedule")
    for port, clock in find_all_port_clock_combinations(
        hardware_config_transmon_old_style
    ):
        sched.add(SquarePulse(port=port, clock=clock, amp=0.25, duration=12e-9))
    sched.add(SquarePulse(port="q7:res", clock="q7.ro", amp=0.25, duration=12e-9))
    sched = create_schedule_with_pulse_info(sched)

    instrument_compilers = CompilerContainer.from_hardware_cfg(
        sched, hardware_config
    ).instrument_compilers
    portclock_map = generate_port_clock_to_device_map(instrument_compilers)

    assert set(portclock_map.keys()) == {
        ("q0:mw-q0.01"),
        ("q4:mw-q4.01"),
        ("q5:mw-q5.01"),
        ("q6:mw-q6.01"),
        ("q7:mw-q7.01"),
        ("q4:res-q4.ro"),
        ("q5:res-q5.ro"),
        ("q0:res-q0.ro"),
        ("q0:fl-cl0.baseband"),
        ("q1:fl-cl0.baseband"),
        ("q2:fl-cl0.baseband"),
        ("q3:fl-cl0.baseband"),
        ("q4:fl-cl0.baseband"),
        ("qe0:optical_readout-qe0.ge0"),
        ("q0:switch-digital"),
    }


def test_generate_port_clock_to_device_map_generated_hardware_config(
    make_schedule_with_measurement, compile_config_basic_transmon_qblox_hardware
):
    sched = make_schedule_with_measurement("q0")
    sched.add(SquarePulse(amp=0.5, duration=1e-6, port="q0:fl", clock="cl0.baseband"))
    compiler = SerialCompiler("compiler")
    compiled_sched = compiler.compile(
        sched, compile_config_basic_transmon_qblox_hardware
    )

    hardware_config = (
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config
    )
    instrument_compilers = CompilerContainer.from_hardware_cfg(
        compiled_sched, hardware_config
    ).instrument_compilers
    portclock_map = generate_port_clock_to_device_map(instrument_compilers)

    assert (None, None) not in portclock_map.keys()
    assert set(portclock_map.keys()) == {
        ("q0:fl-cl0.baseband"),
        ("q0:mw-q0.01"),
        ("q0:res-q0.ro"),
    }


# --------- Test classes and member methods ---------


def test_construct_sequencers(
    make_basic_multi_qubit_schedule,
    compile_config_basic_transmon_qblox_hardware_cluster,
    hardware_cfg_cluster,
    create_schedule_with_pulse_info,
):
    sched = create_schedule_with_pulse_info(
        make_basic_multi_qubit_schedule(["q0", "q1"])
    )

    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster)
    instrument_configs = hardware_cfg._extract_instrument_compilation_configs(
        _extract_port_clocks_used(sched)
    )

    test_cluster = ClusterCompiler(
        parent=None,
        name="cluster0",
        total_play_time=1,
        instrument_cfg=instrument_configs["cluster0"],
    )
    test_module = test_cluster.instrument_compilers["cluster0_module1"]
    sched = make_basic_multi_qubit_schedule(["q0", "q1"])

    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=sched,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )
    assign_pulse_and_acq_info_to_devices(
        schedule=sched,
        device_compilers={"cluster0_module1": test_module},
    )

    test_module._construct_all_sequencer_compilers()
    seq_keys = list(test_module.sequencers.keys())

    assert len(seq_keys) == 2
    assert isinstance(test_module.sequencers[seq_keys[0]], AnalogSequencerCompiler)


def test_construct_sequencers_exceeds_seq(
    make_basic_multi_qubit_schedule,
):
    element_names = [f"q{i}" for i in range(7)]

    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {"instrument_type": "QCM_RF"},
                },
                "ref": "internal",
            },
        },
        "hardware_options": {
            "modulation_frequencies": {},
        },
        "connectivity": {
            "graph": [],
        },
    }

    # Define modulation frequencies with a loop
    for i in range(7):
        port = f"q{i}:mw"
        clock = f"q{i}.01"
        key = f"{port}-{clock}"
        hardware_cfg["hardware_options"]["modulation_frequencies"][key] = {
            "interm_freq": 50000000.0
        }

    for i in range(7):
        module_reference = "cluster0.module1.complex_output_0"
        hardware_cfg["connectivity"]["graph"].append([module_reference, f"q{i}:mw"])

    quantum_device = QuantumDevice("quantum_device")
    quantum_device.hardware_config(hardware_cfg)

    for name in element_names:
        quantum_device.add_element(BasicTransmonElement(name))

    sched = make_basic_multi_qubit_schedule(element_names)
    sched.add_resources([ClockResource(f"{qubit}.01", 5e9) for qubit in element_names])

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(ValueError) as test_error:
        sched = compiler.compile(
            schedule=sched,
            config=quantum_device.generate_compilation_config(),
        )

    assert (
        "Number of simultaneously active port-clock combinations exceeds number of sequencers"
        in test_error.exconly()
    )


def test_find_qblox_instruments(qblox_hardware_config_transmon):
    clusters = find_qblox_instruments(
        hardware_config=qblox_hardware_config_transmon["hardware_description"],
        instrument_type="Cluster",
    )
    assert list(clusters.keys()) == ["cluster0"]
    assert clusters["cluster0"]["modules"]["1"]["instrument_type"] == "QCM"


def test_invalid_channel_names_connectivity(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_compilation_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                    },
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                [f"cluster0.module1.wrong_key", "q0:res"],
            ]
        },
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_compilation_config)

    schedule = Schedule("test valid channel_names")
    schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q0:res", clock="q0.ro"))

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(ValueError, match="Invalid channel name"):
        compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )


def test_missing_module_in_description_raises(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_compilation_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "2": {
                        "instrument_type": "QCM",
                    },
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                [f"cluster0.module1.complex_output_0", "q0:res"],
            ]
        },
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_compilation_config)

    with pytest.raises(KeyError, match="not found in the hardware description"):
        quantum_device.generate_compilation_config()


def test_channel_as_both_input_and_output_qtm():
    hardware_compilation_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QTM",
                    },
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_0", "q0:switch"),
                ("cluster0.module1.digital_input_0", "q0:switch"),
            ]
        },
    }

    quantum_device = QuantumDevice("quantum_device")
    quantum_device.hardware_config(hardware_compilation_config)

    schedule = Schedule("test channel names")

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="The configuration for the QTM module contains channel names with port "
        "numbers that are assigned as both input and output. This is not "
        "allowed. Conflicting channel names:\ndigital_input_0\ndigital_output_0",
    ):
        compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )


# Using the old-style / legacy hardware config dict is deprecated
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_warn_mix_lo_false():
    hardware_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module3": {
                "instrument_type": "QRM",
                "real_input_0": {
                    "lo_name": "red_laser",
                    "mix_lo": False,
                    "portclock_configs": [
                        {
                            "port": "qe0:optical_readout",
                            "clock": "qe0.ge0",
                        },
                    ],
                },
            },
        },
        "red_laser": {
            "instrument_type": "LocalOscillator",
            "frequency": None,
            "power": 1,
        },
    }

    with pytest.warns(FutureWarning) as warn:
        _ = QbloxHardwareCompilationConfig.model_validate(hardware_config)

    assert "Please use quantify_scheduler=0.20.1." in str(warn[1].message)


@pytest.mark.parametrize(
    "edge",
    [
        ["q0:mw", "cluster0.module1.complex_output_0"],
        ["iq_mixer_lo0.if", "cluster0.module1.complex_output_0"],
        ["iq_mixer_lo0.lo", "lo0.output"],
        ["q0:mw", "iq_mixer_lo0.rf"],
    ],
)
def test_validate_connectivity_graph_structure(
    edge,
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                    },
                },
            },
            "lo0": {"instrument_type": "LocalOscillator", "power": 1},
            "iq_mixer_lo0": {"instrument_type": "IQMixer"},
        },
        "hardware_options": {},
        "connectivity": {"graph": [edge]},
    }

    with pytest.raises(ValueError) as error:
        _ = QbloxHardwareCompilationConfig.model_validate(hardware_cfg)

    assert "is a source" in error.exconly()


def test_portclocks(
    make_basic_multi_qubit_schedule,
    compile_config_basic_transmon_qblox_hardware,
):
    sched = make_basic_multi_qubit_schedule(["q0", "q4"])

    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=sched, config=compile_config_basic_transmon_qblox_hardware
    )

    hardware_cfg = (
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config
    )
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        sched, hardware_cfg
    )

    assign_pulse_and_acq_info_to_devices(
        schedule=sched,
        device_compilers=container.clusters,
    )

    compilers = container.instrument_compilers["cluster0"].instrument_compilers
    assert compilers["cluster0_module1"].portclocks == [("q4:mw-q4.01")]
    assert compilers["cluster0_module2"].portclocks == [("q0:mw-q0.01")]


@pytest.mark.parametrize(
    "module, channel_name_to_connected_io_indices",
    [
        (
            QCMCompiler,
            {
                "complex_output_0": (0, 1),
                "complex_output_1": (2, 3),
                "real_output_0": (0,),
                "real_output_1": (1,),
                "real_output_2": (2,),
                "real_output_3": (3,),
                "digital_output_0": (0,),
                "digital_output_1": (1,),
                "digital_output_2": (2,),
                "digital_output_3": (3,),
            },
        ),
        (
            QRMCompiler,
            {
                "complex_output_0": (0, 1),
                "complex_input_0": (0, 1),
                "real_output_0": (0,),
                "real_output_1": (1,),
                "real_input_0": (0,),
                "real_input_1": (1,),
                "digital_output_0": (0,),
                "digital_output_1": (1,),
                "digital_output_2": (2,),
                "digital_output_3": (3,),
            },
        ),
        (
            QCMRFCompiler,
            {
                "complex_output_0": (0, 1),
                "complex_output_1": (2, 3),
                "digital_output_0": (0,),
                "digital_output_1": (1,),
            },
        ),
        (
            QRMRFCompiler,
            {
                "complex_output_0": (0, 1),
                "complex_input_0": (0, 1),
                "digital_output_0": (0,),
                "digital_output_1": (1,),
            },
        ),
    ],
)
def test_validate_channel_name_to_connected_io_indices(
    module, channel_name_to_connected_io_indices
):
    assert (
        module.static_hw_properties.channel_name_to_connected_io_indices
        == channel_name_to_connected_io_indices
    )


def test_compile_simple(
    pulse_only_schedule, compile_config_basic_transmon_qblox_hardware
):
    """Tests if compilation with only pulses finishes without exceptions"""

    compiler = SerialCompiler(name="compiler")
    compiler.compile(
        pulse_only_schedule,
        config=compile_config_basic_transmon_qblox_hardware,
    )


def test_compile_empty_device_compilation_config(hardware_cfg_cluster):
    """Tests if compilation without a device config finishes without exceptions"""

    sched = Schedule("pulse_only_experiment")
    sched.add(IdlePulse(duration=200e-6))
    sched.add(
        DRAGPulse(
            G_amp=0.5,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add_resource(ClockResource("q0.01", freq=5e9))

    quantum_device = QuantumDevice("quantum_device")
    quantum_device.hardware_config(
        QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster)
    )

    compiler = SerialCompiler(name="compiler")
    compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )


def test_compile_with_third_party_instrument(
    pulse_only_schedule, compile_config_basic_transmon_qblox_hardware
):
    def _third_party_compilation_node(
        schedule: Schedule, config: CompilationConfig
    ) -> Schedule:
        schedule["compiled_instructions"]["third_party_instrument"] = {
            "setting": "test"
        }
        return schedule

    config = copy.deepcopy(compile_config_basic_transmon_qblox_hardware)
    config.hardware_compilation_config.hardware_description[
        "third_party_instrument"
    ] = HardwareDescription(instrument_type="ThirdPartyInstrument")
    config.hardware_compilation_config.connectivity.graph.add_edge(
        "third_party_instrument.output", "some_qubit:some_port"
    )
    config.hardware_compilation_config.compilation_passes.insert(
        -1,
        SimpleNodeConfig(
            name="third_party_instrument_compilation",
            compilation_func=_third_party_compilation_node,
        ),
    )

    compiler = SerialCompiler(name="compiler")
    comp_sched = compiler.compile(
        pulse_only_schedule,
        config=config,
    )

    assert (
        comp_sched["compiled_instructions"]["third_party_instrument"]["setting"]
        == "test"
    )


@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_compile_cluster_deprecated_hardware_config(
    cluster_only_schedule, mock_setup_basic_transmon_with_standard_params
):
    sched = cluster_only_schedule
    sched.add_resource(ClockResource("q5.01", freq=5e9))

    hardware_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "latency_corrections": {"q4:mw-q4.01": 8e-9, "q5:mw-q5.01": 4e-9},
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "lo_name": "lo0",
                    "portclock_configs": [
                        {
                            "port": "q4:mw",
                            "clock": "q4.01",
                            "interm_freq": 200e6,
                            "mixer_amp_ratio": 0.9999,
                            "mixer_phase_error_deg": -4.2,
                        }
                    ],
                },
            },
            "cluster0_module2": {
                "instrument_type": "QCM_RF",
                "complex_output_0": {
                    "output_att": 4,
                    "portclock_configs": [
                        {"port": "q5:mw", "clock": "q5.01", "interm_freq": 50e6}
                    ],
                },
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_config)

    compiler = SerialCompiler(name="compiler")

    compiled_sched = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )

    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
            "sequencers"
        ]["seq0"]["sequence"]
        is not None
    )
    assert (
        compiled_sched.compiled_instructions["cluster0"]["cluster0_module2"][
            "sequencers"
        ]["seq0"]["sequence"]
        is not None
    )


@pytest.mark.parametrize("delete_lo0", [False, True])
def test_compile_cluster(
    cluster_only_schedule,
    compile_config_basic_transmon_qblox_hardware,
    delete_lo0: bool,
):
    sched = cluster_only_schedule
    sched.add_resource(ClockResource("q5.01", freq=5e9))

    compiler = SerialCompiler(name="compiler")
    context_mngr = nullcontext()
    if delete_lo0:
        del compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_description[
            "lo0"
        ]
        context_mngr = pytest.raises(RuntimeError)
    with context_mngr as error:
        compiler.compile(
            schedule=sched,
            config=compile_config_basic_transmon_qblox_hardware,
        )

    if delete_lo0:
        assert (
            error.value.args[0]
            == "External local oscillator 'lo0' set to be used for port 'q4:mw' "
            "not found! Make sure it is present in the hardware "
            "configuration."
        )


def test_compile_simple_multiplexing(
    pulse_only_schedule_multiplexed,
    hardware_cfg_qcm_multiplexing,
    mock_setup_basic_transmon_with_standard_params,
):
    """Tests if compilation with only pulses finishes without exceptions"""
    sched = pulse_only_schedule_multiplexed

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_qcm_multiplexing)
    compiler = SerialCompiler(name="compiler")
    compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )


def test_compile_identical_pulses(
    identical_pulses_schedule,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    """Tests if compilation with only pulses finishes without exceptions"""

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        identical_pulses_schedule,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )

    prog = compiled_schedule.compiled_instructions["cluster0"]["cluster0_module1"][
        "sequencers"
    ]["seq0"]["sequence"]
    assert len(prog["waveforms"]) == 2


def test_compile_measure(
    duplicate_measure_schedule,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        duplicate_measure_schedule,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )
    qrm0_seq0_json = full_program["compiled_instructions"]["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]["sequence"]

    assert len(qrm0_seq0_json["weights"]) == 0


# Do not convert hardware config from old-to-new style, some coverage still needed
# for users with old-style configs
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
@pytest.mark.parametrize(
    "operation, instruction_to_check, clock_freq_old, lo1_freq",
    [
        (IdlePulse(duration=64e-9), r"^\s*wait\s+64(\s+|$)", None, 4.8e9),
        (Reset("q1"), r"^\s*wait\s+65532", None, 4.8e9),
        (
            ShiftClockPhase(clock="q1.01", phase_shift=180.0),
            r"^\s*set_ph_delta\s+500000000(\s+|$)",
            None,
            4.8e9,
        ),
        (
            SetClockFrequency(clock="q1.01", clock_freq_new=5.001e9),
            rf"^\s*set_freq\s+{round((2e8 + 5.001e9 - 5e9) * 4)}(\s+|$)",
            5e9,
            4.8e9,
        ),
        (
            SetClockFrequency(clock="q1.01", clock_freq_new=None),
            rf"^\s*set_freq\s+{round((2e8) * 4)}(\s+|$)",
            7e9,
            7.05e9,
        ),
    ],
)
def test_compile_clock_operations(
    mock_setup_basic_transmon_with_standard_params,
    operation: Operation,
    instruction_to_check: str,
    clock_freq_old: Optional[float],
    lo1_freq: float,
):
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "mix_lo": True,
                    "lo_name": "lo0",
                    "portclock_configs": [
                        {
                            "port": "q0:mw",
                            "clock": "cl0.baseband",
                            "interm_freq": 50e6,
                        }
                    ],
                },
                "complex_output_1": {
                    "lo_name": "lo1",
                    "portclock_configs": [{"port": "q1:mw", "clock": "q1.01"}],
                },
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
        "lo1": {
            "instrument_type": "LocalOscillator",
            "frequency": lo1_freq,
            "power": 1,
        },
    }

    sched = Schedule("compile_clock_operations")
    sched.add(operation)
    sched.add(SquarePulse(amp=1, port="q1:mw", clock="q1.01", duration=4e-9))

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )

    if operation.__class__ is SetClockFrequency:
        clock_name = operation.data["pulse_info"][0]["clock"]
        qubit_name, clock_short_name = clock_name.split(".")
        qubit = quantum_device.get_element(qubit_name)
        qubit.clock_freqs[f"f{clock_short_name}"](np.nan)

        with pytest.raises(ValueError) as error:
            _ = compiler.compile(
                schedule=sched,
                config=quantum_device.generate_compilation_config(),
            )
        assert (
            error.value.args[0]
            == f"Operation '{operation}' contains clock '{clock_name}' with an "
            f"undefined (initial) frequency; ensure this resource has been "
            f"added to the schedule or to the device config."
        )

        sched.add_resource(ClockResource(clock_name, clock_freq_old))
        _ = compiler.compile(
            schedule=sched,
            config=quantum_device.generate_compilation_config(),
        )

    program_lines = compiled_sched.compiled_instructions["cluster0"][
        "cluster0_module1"
    ]["sequencers"]["seq0"]["sequence"]["program"].splitlines()
    assert any(
        re.search(instruction_to_check, line) for line in program_lines
    ), "\n".join(line for line in program_lines)


def test_compile_cz_gate(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    1: {"instrument_type": "QCM"},
                    2: {"instrument_type": "QCM_RF"},
                },
                "sequence_to_file": False,
                "ref": "internal",
            }
        },
        "hardware_options": {
            "output_att": {},
            "input_att": {},
            "modulation_frequencies": {
                "q2:mw-q2.01": {"interm_freq": 80000000.0},
                "q3:mw-q3.01": {"interm_freq": 80000000.0},
            },
            "mixer_corrections": {},
        },
        "connectivity": {
            "graph": [
                ("cluster0.module1.real_output_2", "q2:fl"),
                ("cluster0.module1.real_output_3", "q3:fl"),
                ("cluster0.module2.complex_output_0", "q2:mw"),
                ("cluster0.module2.complex_output_1", "q3:mw"),
            ]
        },
    }

    mock_setup = mock_setup_basic_transmon_with_standard_params
    edge_q2_q3 = mock_setup["q2_q3"]
    edge_q2_q3.cz.q2_phase_correction(44)
    edge_q2_q3.cz.q3_phase_correction(63)

    sched = Schedule("schedule")
    sched.add(Reset("q2", "q3"))
    sched.add(CZ(qC="q2", qT="q3"))
    sched.add(SquarePulse(amp=1, port="q2:mw", clock="q2.01", duration=4e-9))
    sched.add(SquarePulse(amp=1, port="q3:mw", clock="q3.01", duration=4e-9))

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )

    program_lines = {
        "q2:fl": compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
            "sequencers"
        ]["seq0"]["sequence"]["program"].splitlines(),
        "q2:mw": compiled_sched.compiled_instructions["cluster0"]["cluster0_module2"][
            "sequencers"
        ]["seq0"]["sequence"]["program"].splitlines(),
        "q3:mw": compiled_sched.compiled_instructions["cluster0"]["cluster0_module2"][
            "sequencers"
        ]["seq1"]["sequence"]["program"].splitlines(),
    }

    assert any(
        re.search(rf"^\s*play\s+0,0,4(\s|$)", line) for line in program_lines["q2:fl"]
    ), "\n".join(line for line in program_lines["q2:fl"])

    assert any(
        re.search(rf"^\s*set_ph_delta\s+122222222(\s|$)", line)
        for line in program_lines["q2:mw"]
    ), "\n".join(line for line in program_lines["q2:mw"])

    assert any(
        re.search(rf"^\s*set_ph_delta\s+175000000(\s|$)", line)
        for line in program_lines["q3:mw"]
    ), "\n".join(line for line in program_lines["q3:mw"])


def test_compile_simple_with_acq(
    dummy_cluster,
    mixed_schedule_with_acquisition,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        mixed_schedule_with_acquisition,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )

    qcm0_seq0_json = full_program["compiled_instructions"]["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]["sequence"]

    qcm0 = dummy_cluster().module2
    qcm0.sequencer0.sequence(qcm0_seq0_json)
    qcm0.arm_sequencer(0)

    uploaded_waveforms = qcm0.get_waveforms(0)
    assert uploaded_waveforms is not None


@pytest.mark.parametrize(
    "reset_clock_phase",
    [True, False],
)
def test_compile_acq_measurement_with_clock_phase_reset(
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_cluster,
    reset_clock_phase,
):
    schedule = Schedule("Test schedule")

    hardware_cfg = copy.deepcopy(hardware_cfg_cluster)
    hardware_cfg["hardware_description"]["cluster0"]["modules"]["3"][
        "sequence_to_file"
    ] = True

    q0, q1 = "q0", "q1"
    times = np.arange(0, 60e-6, 3e-6)
    for i, tau in enumerate(times):
        schedule.add(Reset(q0, q1), label=f"Reset {i}")
        schedule.add(X(q0), label=f"pi {i} {q0}")
        schedule.add(X(q1), label=f"pi {i} {q1}", ref_pt="start")

        schedule.add(
            Measure(q0, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q0}{i}",
        )

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["q0"].measure.reset_clock_phase(reset_clock_phase)
    mock_setup["quantum_device"].hardware_config(hardware_cfg)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule, config=mock_setup["quantum_device"].generate_compilation_config()
    )
    qrm0_seq0_json = compiled_schedule.compiled_instructions["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]["seq_fn"]
    with open(qrm0_seq0_json) as file:
        program = json.load(file)["program"]
    reset_counts = program.count(" reset_ph ")
    expected_counts = (1 + len(times)) if reset_clock_phase else 1
    assert reset_counts == expected_counts, (
        f"Expected qasm program to contain `reset_ph`-instruction {expected_counts} "
        f"times, but found {reset_counts} times instead."
    )


def test_acquisitions_max_index_raises(
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    sched = Schedule("acquisitions_max_index_raises")
    sched.add(Measure("q0", acq_index=0))
    sched.add(Measure("q0", acq_index=0))

    with pytest.raises(ValueError) as error:
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=compile_config_basic_transmon_qblox_hardware_cluster,
        )

    assert (
        "Found 0 as the highest index out of "
        "2 for channel 0, indicating "
        "an acquisition index was skipped or an acquisition index was repeated. "
        "Please make sure the used indices increment by 1 starting from 0. "
        "Problem occurred for port q0:res with clock q0.ro, "
        "which corresponds to seq0 of cluster0_module3." == error.value.args[0]
    )


def test_acquisitions_same_index_raises(
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    sched = Schedule("acquisitions_same_index_raises")
    sched.add(Measure("q0", acq_index=0))
    sched.add(Measure("q0", acq_index=2))
    sched.add(Measure("q0", acq_index=2))

    with pytest.raises(ValueError) as error:
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=compile_config_basic_transmon_qblox_hardware_cluster,
        )

    assert (
        "Found 2 unique indices out of "
        "3 for channel 0, indicating "
        "an acquisition index was skipped or an acquisition index was repeated. "
        "Please make sure the used indices increment by 1 starting from 0. "
        "Problem occurred for port q0:res with clock q0.ro, "
        "which corresponds to seq0 of cluster0_module3." == error.value.args[0]
    )


def test_acquisitions_back_to_back(
    compile_config_basic_transmon_qblox_hardware,
):
    sched = Schedule("acquisitions_back_to_back")
    meas_op = sched.add(Measure("q0", acq_index=0))
    # Add another one too quickly
    sched.add(
        Measure("q0", acq_index=1), ref_op=meas_op, ref_pt="start", rel_time=200e-9
    )

    with pytest.raises(ValueError) as error:
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=compile_config_basic_transmon_qblox_hardware,
        )

    assert (
        "Please ensure a minimum interval of 300 ns between acquisitions"
        in error.value.args[0]
    )


@pytest.mark.filterwarnings(
    "ignore:.*The specified weights and sampling rate lead to a weighted integration.*"
)
def test_deprecated_weighted_acquisition_end_to_end(
    pulse_only_schedule_with_operation_timing,
    compile_config_basic_transmon_qblox_hardware,
):
    sched = pulse_only_schedule_with_operation_timing
    sched.add(Measure("q0", acq_protocol="NumericalWeightedIntegrationComplex"))

    compiler = SerialCompiler(name="compiler")
    with pytest.warns(
        FutureWarning,
        match="0.20.0",
    ):
        compiled_sched = compiler.compile(
            sched,
            config=compile_config_basic_transmon_qblox_hardware,
        )
    assert re.search(
        rf"\n\s*acquire_weighed\s+0,0,0,1,4(\s|$)",
        (
            compiled_sched.compiled_instructions["cluster0"]["cluster0_module4"][
                "sequencers"
            ]["seq0"]["sequence"]["program"]
        ),
    )


@pytest.mark.filterwarnings(
    "ignore:.*The specified weights and sampling rate lead to a weighted integration.*"
)
def test_separated_weighted_acquisition_end_to_end(
    pulse_only_schedule_with_operation_timing,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    sched = pulse_only_schedule_with_operation_timing
    sched.add(Measure("q0", acq_protocol="NumericalSeparatedWeightedIntegration"))

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )
    assert re.search(
        rf"\n\s*acquire_weighed\s+0,0,0,1,4(\s|$)",
        (
            compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
                "sequencers"
            ]["seq0"]["sequence"]["program"]
        ),
    )


@pytest.mark.filterwarnings(
    "ignore:.*The specified weights and sampling rate lead to a weighted integration.*"
)
def test_weighted_acquisition_end_to_end(
    pulse_only_schedule_with_operation_timing,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    sched = pulse_only_schedule_with_operation_timing
    sched.add(Measure("q0", acq_protocol="NumericalWeightedIntegration"))

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )
    assert re.search(
        rf"\n\s*acquire_weighed\s+0,0,0,1,4(\s|$)",
        (
            compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
                "sequencers"
            ]["seq0"]["sequence"]["program"]
        ),
    )


@pytest.mark.filterwarnings(
    "ignore:.*The specified weights and sampling rate lead to a weighted integration.*"
)
def test_separated_weighted_acquisition_too_high_sampling_rate_raises(
    pulse_only_schedule_with_operation_timing,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    sched = pulse_only_schedule_with_operation_timing
    sched.add(Measure("q0", acq_protocol="NumericalSeparatedWeightedIntegration"))
    compile_config_basic_transmon_qblox_hardware_cluster.device_compilation_config.elements[
        "q0"
    ][
        "measure"
    ].factory_kwargs[
        "acq_weights_sampling_rate"
    ] = 5e9

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(ValueError) as exc:
        _ = compiler.compile(
            sched,
            config=compile_config_basic_transmon_qblox_hardware_cluster,
        )
    assert exc.value.args[0] == (
        "Qblox hardware supports a sampling rate up to 1.0e+00 GHz, but a sampling "
        "rate of 5.0e+00 GHz was provided to WeightedAcquisitionStrategy. Please check "
        "the device configuration."
    )


@pytest.mark.filterwarnings(
    "ignore:.*The specified weights and sampling rate lead to a weighted integration.*"
)
def test_weighted_acquisition_too_high_sampling_rate_raises(
    pulse_only_schedule_with_operation_timing,
    compile_config_basic_transmon_qblox_hardware,
):
    sched = pulse_only_schedule_with_operation_timing
    sched.add(Measure("q0", acq_protocol="NumericalWeightedIntegration"))
    compile_config_basic_transmon_qblox_hardware.device_compilation_config.elements[
        "q0"
    ]["measure"].factory_kwargs["acq_weights_sampling_rate"] = 5e9

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(ValueError) as exc:
        _ = compiler.compile(
            sched,
            config=compile_config_basic_transmon_qblox_hardware,
        )
    assert exc.value.args[0] == (
        "Qblox hardware supports a sampling rate up to 1.0e+00 GHz, but a sampling "
        "rate of 5.0e+00 GHz was provided to WeightedAcquisitionStrategy. Please check "
        "the device configuration."
    )


def test_compile_with_rel_time(
    dummy_cluster,
    pulse_only_schedule_with_operation_timing,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        pulse_only_schedule_with_operation_timing,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )

    qcm0_seq0_json = full_program["compiled_instructions"]["cluster0"][
        "cluster0_module1"
    ]["sequencers"]["seq0"]["sequence"]

    qcm0 = dummy_cluster().module2
    qcm0.sequencer0.sequence(qcm0_seq0_json)


def test_compile_with_repetitions(
    mixed_schedule_with_acquisition,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    mixed_schedule_with_acquisition.repetitions = 10

    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        mixed_schedule_with_acquisition,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )

    program_from_json = full_program["compiled_instructions"]["cluster0"][
        "cluster0_module1"
    ]["sequencers"]["seq0"]["sequence"]["program"]
    assert re.search(rf"\n\s*move\s+10,R0", program_from_json)
    assert re.search(rf"\n\s*loop\s+R0,@start", program_from_json)


def _func_for_hook_test(qasm: QASMProgram):
    qasm.instructions.insert(
        0, QASMProgram.get_instruction_as_list(q1asm_instructions.NOP)
    )


def test_qasm_hook(
    pulse_only_schedule,
    mock_setup_basic_transmon_with_standard_params,
    qblox_hardware_config_transmon,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    hardware_compilation_config = copy.deepcopy(qblox_hardware_config_transmon)
    hardware_compilation_config["hardware_options"]["sequencer_options"] = {
        "q0:mw-q0.01": {
            "qasm_hook_func": _func_for_hook_test,
        }
    }

    sched = pulse_only_schedule
    sched.repetitions = 11

    quantum_device.hardware_config(hardware_compilation_config)
    compiler = SerialCompiler(name="compiler")
    comp_sched = compiler.compile(
        sched,
        config=mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )
    program = comp_sched["compiled_instructions"]["cluster0"]["cluster0_module2"][
        "sequencers"
    ]["seq0"]["sequence"]["program"]
    program_lines = program.splitlines()

    assert program_lines[0].strip() == q1asm_instructions.NOP


@pytest.mark.deprecated
def test_qasm_hook_hardware_config(
    pulse_only_schedule, mock_setup_basic_transmon_with_standard_params
):
    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QCM"}},
                "ref": "external",
            }
        },
        "hardware_options": {
            "sequencer_options": {
                "q0:mw-q0.01": {
                    "qasm_hook_func": _func_for_hook_test,
                }
            }
        },
        "connectivity": {"graph": [["cluster0.module1.complex_output_0", "q0:mw"]]},
    }

    sched = pulse_only_schedule

    sched.repetitions = 11
    mock_setup_basic_transmon_with_standard_params["quantum_device"].hardware_config(
        hw_config
    )

    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        sched,
        config=mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )
    program = full_program["compiled_instructions"]["cluster0"]["cluster0_module1"][
        "sequencers"
    ]["seq0"]["sequence"]["program"]
    program_lines = program.splitlines()

    assert q1asm_instructions.NOP == program_lines[0].strip()


def test_qcm_acquisition_error(
    hardware_cfg_cluster,
    make_schedule_with_measurement,
    create_schedule_with_pulse_info,
):
    sched = create_schedule_with_pulse_info(make_schedule_with_measurement("q0"))
    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster)
    instrument_configs = hardware_cfg._extract_instrument_compilation_configs(
        _extract_port_clocks_used(sched)
    )
    module_configs = instrument_configs[
        "cluster0"
    ]._extract_module_compilation_configs()

    qcm = QCMCompiler(
        parent=None,
        name="cluster0_module1",
        total_play_time=10,
        instrument_cfg=module_configs[1],
    )
    with pytest.raises(
        RuntimeError,
        match="QCMCompiler cluster0_module1 does not support acquisitions. "
        "Attempting to add acquisition Acquisition",
    ):
        qcm.add_op_info(
            "port",
            "clock",
            types.OpInfo(
                name="test_acq", data={"acq_channel": 0, "duration": 20e-9}, timing=4e-9
            ),
        )


def test_real_mode_pulses(
    real_square_pulse_schedule,
    hardware_cfg_real_mode,
    mock_setup_basic_transmon,
):
    real_square_pulse_schedule.repetitions = 10
    mock_setup_basic_transmon["quantum_device"].hardware_config(hardware_cfg_real_mode)
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        real_square_pulse_schedule,
        config=mock_setup_basic_transmon[
            "quantum_device"
        ].generate_compilation_config(),
    )

    for output in range(4):
        seq_instructions = full_program.compiled_instructions["cluster0"][
            "cluster0_module1"
        ]["sequencers"][f"seq{output}"]["sequence"]

        for value in seq_instructions["waveforms"].values():
            waveform_data, seq_path = value["data"], value["index"]

            # Asserting that indeed we only have square pulse on I and no signal on Q
            if seq_path == 0:
                assert (np.array(waveform_data) == 1).all()
            elif seq_path == 1:
                assert (np.array(waveform_data) == 0).all()

        assert re.search(rf"play\s+0,0", seq_instructions["program"]), (
            f"Output {output + 1} must be connected to "
            f"sequencer{output} path{iq_order[0]} in real mode."
        )


# Keep this test as extra coverage for old-to-new style conversion
# Using the old-style / legacy hardware config dict is deprecated
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_real_mode_pulses_legacy_hardware_cfg(
    real_square_pulse_schedule,
    hardware_cfg_real_mode_legacy,
    mock_setup_basic_transmon,
):
    real_square_pulse_schedule.repetitions = 10
    mock_setup_basic_transmon["quantum_device"].hardware_config(
        hardware_cfg_real_mode_legacy
    )
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        real_square_pulse_schedule,
        config=mock_setup_basic_transmon[
            "quantum_device"
        ].generate_compilation_config(),
    )

    for output in range(4):
        seq_instructions = full_program.compiled_instructions["cluster0"][
            "cluster0_module1"
        ]["sequencers"][f"seq{output}"]["sequence"]

        for value in seq_instructions["waveforms"].values():
            waveform_data, seq_path = value["data"], value["index"]

            # Asserting that indeed we only have square pulse on I and no signal on Q
            if seq_path == 0:
                assert (np.array(waveform_data) == 1).all()
            elif seq_path == 1:
                assert (np.array(waveform_data) == 0).all()

        assert re.search(rf"play\s+0,0", seq_instructions["program"]), (
            f"Output {output + 1} must be connected to "
            f"sequencer{output} path{iq_order[0]} in real mode."
        )


# --------- Test QASMProgram class ---------


def test_emit(empty_qasm_program_qcm):
    qasm = empty_qasm_program_qcm
    qasm.emit(q1asm_instructions.PLAY, 0, 1, 120)
    qasm.emit(q1asm_instructions.STOP, comment="This is a comment that is added")

    assert len(qasm.instructions) == 2


def test_auto_wait(empty_qasm_program_qcm):
    qasm = empty_qasm_program_qcm
    qasm.auto_wait(120)
    assert len(qasm.instructions) == 1
    qasm.auto_wait(70000)
    assert len(qasm.instructions) == 3  # since it should split the waits
    assert qasm.elapsed_time == 70120
    qasm.auto_wait(700000)
    assert qasm.elapsed_time == 770120
    assert len(qasm.instructions) == 8  # now loops are used
    with pytest.raises(ValueError):
        qasm.auto_wait(-120)


@pytest.mark.parametrize(
    "val, expected_expanded_val",
    [
        (-1, -constants.IMMEDIATE_SZ_GAIN // 2),
        (-0.5, -constants.IMMEDIATE_SZ_GAIN // 4),
        (0.0, 0),
        (0.5, constants.IMMEDIATE_SZ_GAIN // 4),
        (1.0, constants.IMMEDIATE_SZ_GAIN // 2 - 1),
    ],
)
def test_expand_awg_gain_from_normalised_range(val, expected_expanded_val):
    minimal_pulse_data = {"duration": 20e-9}
    acq = types.OpInfo(name="test_acq", data=minimal_pulse_data, timing=4e-9)

    expanded_val = QASMProgram.expand_awg_from_normalised_range(
        val=val,
        immediate_size=constants.IMMEDIATE_SZ_GAIN,
        param="test_param",
        operation=acq,
    )
    assert expanded_val == expected_expanded_val


def test_out_of_range_expand_awg_gain_from_normalised_range():
    minimal_pulse_data = {"duration": 20e-9}
    acq = types.OpInfo(name="test_acq", data=minimal_pulse_data, timing=4e-9)
    with pytest.raises(ValueError):
        QASMProgram.expand_awg_from_normalised_range(
            val=10,
            immediate_size=constants.IMMEDIATE_SZ_GAIN,
            param="test_param",
            operation=acq,
        )


@pytest.mark.parametrize(
    "time, expected_time_ns",
    [(4e-9, 4), (8.001e-9, 8), (4.0008e-9, 4), (1, 1e9), (1 + 1e-12, 1e9)],
)
def test_to_grid_time(time, expected_time_ns):
    assert to_grid_time(time) == expected_time_ns


@pytest.mark.parametrize(
    "time",
    [7e-9, 10e-9, 8.07e-9, 4.09e-9, 1 + 0.1e-9],
)
def test_to_grid_time_raises(time):

    with pytest.raises(ValueError) as error:
        to_grid_time(time, grid_time_ns=constants.MIN_TIME_BETWEEN_OPERATIONS)

    assert (
        "Please ensure that the durations of operations"
        " and wait times between operations are multiples of 4 ns" in str(error.value)
    )


@pytest.mark.parametrize(
    "delta_time, raise_expected",
    [(0.1, False), (0.01, True)],
)
def test_GRID_TIME_TOLERANCE_TIME(delta_time, raise_expected):
    constants.GRID_TIME_TOLERANCE_TIME = delta_time
    ref_time = 8e-9
    ref_time_ns = 8
    delta = 0.05e-9
    if raise_expected:
        with pytest.raises(ValueError) as error:
            to_grid_time(
                ref_time + delta, grid_time_ns=constants.MIN_TIME_BETWEEN_OPERATIONS
            )

        assert (
            "Please ensure that the durations of operations"
            " and wait times between operations are multiples of 4 ns"
            in str(error.value)
        )
    else:
        assert (
            to_grid_time(
                ref_time + delta, grid_time_ns=constants.MIN_TIME_BETWEEN_OPERATIONS
            )
            == ref_time_ns
        )


@pytest.mark.parametrize(
    "time, expected",
    [
        (5e-9, False),
        (11e-9, False),
        (8.06e-9, False),
        (4.09e-9, False),
        (12e-9, True),
        (8.001e-9, True),
        (4.0008e-9, True),
    ],
)
def test_is_multiple_of_min_op_time(time, expected):
    assert (
        is_multiple_of_grid_time(
            time, grid_time_ns=constants.MIN_TIME_BETWEEN_OPERATIONS
        )
        is expected
    )


def test_is_within_min_op_time_even_if_floating_point_error():
    time1, time2 = 8e-9, 12e-9
    assert abs(time1 - time2) < constants.MIN_TIME_BETWEEN_OPERATIONS
    assert to_grid_time(time1) != to_grid_time(time2)


def test_loop(empty_qasm_program_qcm):
    num_rep = 10

    qasm = empty_qasm_program_qcm
    qasm.emit(q1asm_instructions.WAIT_SYNC, 4)
    with qasm.loop("this_loop", repetitions=num_rep):
        qasm.emit(q1asm_instructions.WAIT, 20)
    assert len(qasm.instructions) == 5
    assert qasm.instructions[1][1] == q1asm_instructions.MOVE
    num_rep_used, reg_used = qasm.instructions[1][2].split(",")
    assert int(num_rep_used) == num_rep


@pytest.mark.parametrize("amount", [1, 2, 3, 40])
def test_temp_register(amount, empty_qasm_program_qcm):
    qasm = empty_qasm_program_qcm
    with qasm.temp_registers(amount) as registers:
        for reg in registers:
            assert reg not in qasm.register_manager.available_registers
    for reg in registers:
        assert reg in qasm.register_manager.available_registers


# --------- Test compilation functions ---------
@pytest.mark.parametrize("reset_clock_phase", [True, False])
def test_assign_pulse_and_acq_info_to_devices(
    mock_setup_basic_transmon_with_standard_params,
    mixed_schedule_with_acquisition,
    hardware_cfg_cluster,
    reset_clock_phase,
):
    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster)

    sched = mixed_schedule_with_acquisition
    mock_setup_basic_transmon_with_standard_params["q0"].measure.reset_clock_phase(
        reset_clock_phase
    )

    compiler = SerialCompiler(name="compiler")
    sched_with_pulse_info = compiler.compile(
        schedule=sched,
        config=mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        sched_with_pulse_info, hardware_cfg
    )

    assign_pulse_and_acq_info_to_devices(
        schedule=sched_with_pulse_info,
        device_compilers=container.clusters,
    )
    container.prepare()

    qrm = container.instrument_compilers["cluster0"].instrument_compilers[
        "cluster0_module3"
    ]
    expected_num_of_pulses = 1 if reset_clock_phase is False else 2

    actual_portclocks = list(qrm._portclocks_with_data)
    assert len(actual_portclocks) == 1
    actual_portclock = actual_portclocks[0]

    actual_num_of_pulses = len(
        [
            op_info
            for op_info in qrm._op_infos[actual_portclock]
            if not op_info.is_acquisition
        ]
    )
    actual_num_of_acquisitions = len(
        [
            op_info
            for op_info in qrm._op_infos[actual_portclock]
            if op_info.is_acquisition
        ]
    )
    assert actual_num_of_pulses == expected_num_of_pulses, (
        f"Expected {expected_num_of_pulses} number of pulses, but found "
        f"{actual_num_of_pulses} instead."
    )
    assert actual_num_of_acquisitions == 1, (
        f"Expected 1 number of acquisitions, but found {actual_num_of_acquisitions} "
        "instead."
    )


def test_container_prepare(
    pulse_only_schedule,
    hardware_cfg_cluster,
    compile_config_basic_transmon_qblox_hardware,
):
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=pulse_only_schedule,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        sched, QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster)
    )
    assign_pulse_and_acq_info_to_devices(sched, container.clusters)
    container.prepare()

    for instr in container.instrument_compilers.values():
        instr.prepare()

    assert (
        container.instrument_compilers["cluster0"]
        .instrument_compilers["cluster0_module1"]
        .sequencers["seq0"]
        .frequency
        is not None
    )
    assert container.instrument_compilers["lo0"].frequency is not None


def test_multiple_trace_acquisition_error(
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    sched = Schedule("test_multiple_trace_acquisition_error")
    sched.add(Trace(duration=100e-9, port="q0:res", clock="q0.multiplex"))
    sched.add(Trace(duration=100e-9, port="q0:res", clock="q0.ro"))

    sched.add_resource(ClockResource("q0.multiplex", 3.2e9))

    with pytest.raises(ValueError) as exception:
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched,
            config=compile_config_basic_transmon_qblox_hardware_cluster,
        )
    assert str(exception.value) == (
        f"Both sequencer '0' and '1' "
        f"of 'cluster0_module3' attempts to perform scope mode acquisitions. "
        f"Only one sequencer per device can "
        f"trigger raw trace capture.\n\nPlease ensure that "
        f"only one port-clock combination performs "
        f"raw trace acquisition per instrument."
    )


def test_container_prepare_baseband(
    mock_setup_basic_transmon,
    baseband_square_pulse_schedule,
    hardware_cfg_qcm,
):
    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_qcm)

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=baseband_square_pulse_schedule,
        config=quantum_device.generate_compilation_config(),
    )

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        schedule=sched, hardware_cfg=hardware_cfg
    )
    assign_pulse_and_acq_info_to_devices(
        schedule=sched,
        device_compilers=container.clusters,
    )
    container.prepare()

    assert (
        container.instrument_compilers["cluster0"]
        .instrument_compilers["cluster0_module1"]
        .sequencers["seq0"]
        .frequency
        is not None
    )
    assert container.instrument_compilers["lo0"].frequency is not None


def test_container_prepare_no_lo(
    pulse_only_schedule_no_lo,
    hardware_cfg_cluster,
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=pulse_only_schedule_no_lo,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        sched, QbloxHardwareCompilationConfig.model_validate(hardware_cfg_cluster)
    )
    assign_pulse_and_acq_info_to_devices(
        sched,
        container.clusters,
    )
    container.prepare()

    assert (
        container.instrument_compilers["cluster0"]
        .instrument_compilers["cluster0_module4"]
        .sequencers["seq0"]
        .frequency
        == 8.3e9
    )


def test_from_mapping(
    pulse_only_schedule, compile_config_basic_transmon_qblox_hardware
):
    pulse_only_schedule = _determine_absolute_timing(pulse_only_schedule)
    hardware_cfg = (
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config
    )
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        pulse_only_schedule, hardware_cfg
    )
    for instr_name in hardware_cfg.hardware_description.keys():
        if instr_name in [
            "iq_mixer_lo0",
            "iq_mixer_lo1",
            "iq_mixer_lo_real",
            "lo0",
            "lo1",
            "lo_real",
        ]:
            continue
        assert instr_name in container.instrument_compilers


# Transmon-specific config
def test_extract_instrument_compilation_configs_cluster():
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                        "complex_output_0": {"marker_debug_mode_enable": True},
                    },
                },
            },
            "cluster1": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QRM_RF"},
                    "2": {"instrument_type": "QCM_RF"},
                },
            },
            "cluster2": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                },
            },
        },
        "hardware_options": {
            "latency_corrections": {
                "q4:mw-q4.01": 8e-9,
            },
            "modulation_frequencies": {
                "q4:mw-q4.01": {
                    "interm_freq": 200e6,
                    "lo_freq": None,
                },
                "q7:res-q7.ro": {"interm_freq": 52e6},
            },
            "mixer_corrections": {
                "q4:mw-q4.01": {"amp_ratio": 0.9999, "phase_error": -4.2},
            },
            "input_att": {"q7:res-q7.ro": 12},
            "sequencer_options": {
                "qe0:optical_readout-qe0.ge0": {"ttl_acq_threshold": 0.5}
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q4:mw"],
                ["cluster1.module1.complex_input_0", "q7:res"],
            ]
        },
    }

    hardware_config = QbloxHardwareCompilationConfig.model_validate(hardware_config)

    portclocks_used = {
        ("q4:mw", "q4.01"),
        ("q7:res", "q7.ro"),
    }

    instrument_configs = hardware_config._extract_instrument_compilation_configs(
        portclocks_used
    )

    assert list(instrument_configs.keys()) == [
        "cluster0",
        "cluster1",
    ]

    cluster0 = instrument_configs["cluster0"]
    cluster1 = instrument_configs["cluster1"]

    assert cluster0.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "Cluster",
        "ref": "internal",
        "modules": {
            1: {
                "instrument_type": "QCM",
                "complex_output_0": {"marker_debug_mode_enable": True},
            },
        },
    }

    assert cluster0.hardware_options.model_dump(exclude_unset=True) == {
        "latency_corrections": {"q4:mw-q4.01": 8e-09},
        "modulation_frequencies": {
            "q4:mw-q4.01": {"interm_freq": 200000000.0, "lo_freq": None},
        },
        "mixer_corrections": {
            "q4:mw-q4.01": {
                "amp_ratio": 0.9999,
                "phase_error": -4.2,
            },
        },
    }

    assert cluster0.portclock_to_path == {
        ("q4:mw-q4.01"): "cluster0.module1.complex_output_0",
    }

    assert cluster0.lo_to_path == {}

    assert cluster1.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "Cluster",
        "ref": "internal",
        "modules": {1: {"instrument_type": "QRM_RF"}},
    }

    assert cluster1.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {"q7:res-q7.ro": {"interm_freq": 52000000.0}},
        "input_att": {"q7:res-q7.ro": 12},
    }

    assert cluster1.portclock_to_path == {
        ("q7:res-q7.ro"): "cluster1.module1.complex_input_0"
    }
    assert cluster1.lo_to_path == {}


def test_extract_instrument_compilation_configs_lo():

    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "3": {"instrument_type": "QRM"},
                },
            },
            "lo1": {"instrument_type": "LocalOscillator", "power": 1},
            "iq_mixer_lo1": {"instrument_type": "IQMixer"},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7.2e9},
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_output_0", "iq_mixer_lo1.if"],
                ["lo1.output", "iq_mixer_lo1.lo"],
                ["iq_mixer_lo1.rf", "q4:res"],
            ]
        },
    }

    hardware_config = QbloxHardwareCompilationConfig.model_validate(hardware_config)

    portclocks_used = {("q4:res", "q4.ro")}

    instrument_configs = hardware_config._extract_instrument_compilation_configs(
        portclocks_used
    )

    assert list(instrument_configs.keys()) == ["cluster0", "lo1"]

    cluster0 = instrument_configs["cluster0"]
    lo1 = instrument_configs["lo1"]

    assert cluster0.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {
            "q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7200000000.0},
        },
    }

    assert cluster0.lo_to_path == {
        "lo1": "cluster0.module3.complex_output_0",
    }

    assert lo1.model_dump() == {
        "hardware_description": {
            "instrument_type": "LocalOscillator",
            "instrument_name": "lo1",
            "generic_icc_name": None,
            "frequency_param": "frequency",
            "power_param": "power",
            "power": 1,
        },
        "frequency": 7200000000.0,
    }


def test_extract_module_compilation_configs():
    cluster_compilation_config = {
        "hardware_description": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "sequence_to_file": False,
            "modules": {
                2: {
                    "instrument_type": "QCM_RF",
                },
                3: {
                    "instrument_type": "QRM",
                },
            },
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:mw-q0.01": {"interm_freq": 50000000.0, "lo_freq": None},
                "q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7200000000.0},
            },
            "mixer_corrections": {
                "q4:res-q4.ro": {
                    "dc_offset_i": -0.054,
                    "dc_offset_q": -0.034,
                    "amp_ratio": 0.9997,
                    "phase_error": -4.0,
                }
            },
            "input_gain": {"q4:res-q4.ro": {"gain_I": 2, "gain_Q": 3}},
            "output_att": {"q0:mw-q0.01": 4},
        },
        "portclock_to_path": {
            "q0:mw-q0.01": "cluster0.module2.complex_output_0",
            "q4:res-q4.ro": "cluster0.module3.complex_output_0",
        },
        "lo_to_path": {"lo1": "cluster0.module3.complex_output_0"},
    }

    cluster_compilation_config = _ClusterCompilationConfig.model_validate(
        cluster_compilation_config
    )
    module_configs = cluster_compilation_config._extract_module_compilation_configs()

    assert list(module_configs.keys()) == [2, 3]

    module2 = module_configs[2]
    module3 = module_configs[3]

    assert module2.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "QCM_RF"
    }
    assert module2.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {
            "q0:mw-q0.01": {"interm_freq": 50000000.0, "lo_freq": None}
        },
        "output_att": {"q0:mw-q0.01": 4},
    }

    assert module2.portclock_to_path == {
        ("q0:mw-q0.01"): "cluster0.module2.complex_output_0"
    }
    assert module2.lo_to_path == {}

    assert module3.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "QRM"
    }
    assert module3.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {
            "q4:res-q4.ro": {"interm_freq": None, "lo_freq": 7200000000.0}
        },
        "mixer_corrections": {
            "q4:res-q4.ro": {
                "dc_offset_i": -0.054,
                "dc_offset_q": -0.034,
                "amp_ratio": 0.9997,
                "phase_error": -4.0,
            }
        },
        "input_gain": {"q4:res-q4.ro": {"gain_I": 2, "gain_Q": 3}},
    }

    assert module3.portclock_to_path == {
        ("q4:res-q4.ro"): "cluster0.module3.complex_output_0"
    }
    assert module3.lo_to_path == {"lo1": "cluster0.module3.complex_output_0"}


def test_extract_sequencer_compilation_configs():
    module_compilation_config = {
        "hardware_description": {
            "instrument_type": "QCM",
            "complex_output_0": {
                "marker_debug_mode_enable": True,
            },
        },
        "hardware_options": {
            "latency_corrections": {"q0:mw-q0.01": 8e-09},
            "distortion_corrections": {
                "q0:mw-q0.01": {
                    "filter_func": "scipy.signal.lfilter",
                    "input_var_name": "x",
                    "kwargs": {"b": [0, 0.25, 0.5], "a": [1]},
                    "clipping_values": [-2.5, 2.5],
                    "sampling_rate": 1000000000.0,
                }
            },
            "modulation_frequencies": {
                "q0:mw-q0.01": {"interm_freq": None, "lo_freq": 7800000000.0},
                "q1:mw-q1.01": {"interm_freq": 50000000.0, "lo_freq": None},
            },
            "mixer_corrections": {
                "q0:mw-q0.01": {
                    "amp_ratio": 0.9999,
                    "phase_error": -4.2,
                }
            },
            "sequencer_options": {
                "q0:mw-q0.01": {
                    "ttl_acq_threshold": 0.5,
                }
            },
        },
        "portclock_to_path": {
            "q1:mw-q1.01": "cluster0.module1.complex_output_1",
            "q0:mw-q0.01": "cluster0.module1.complex_output_0",
        },
        "lo_to_path": {"lo0": "cluster0.module1.complex_output_0"},
    }

    module_compilation_config = _ClusterModuleCompilationConfig.model_validate(
        module_compilation_config
    )

    sequencer_configs = (
        module_compilation_config._extract_sequencer_compilation_configs()
    )

    assert list(sequencer_configs.keys()) == [0, 1]

    assert sequencer_configs[0].model_dump(exclude_unset=True) == {
        "allow_off_grid_nco_ops": None,
        "sequencer_options": {"ttl_acq_threshold": 0.5},
        "hardware_description": {"marker_debug_mode_enable": True},
        "portclock": "q0:mw-q0.01",
        "channel_name": "complex_output_0",
        "latency_correction": 8e-09,
        "distortion_correction": {
            "filter_func": "scipy.signal.lfilter",
            "input_var_name": "x",
            "kwargs": {"b": [0, 0.25, 0.5], "a": [1]},
            "clipping_values": [-2.5, 2.5],
            "sampling_rate": 1e9,
        },
        "lo_name": "lo0",
        "modulation_frequencies": {"interm_freq": None, "lo_freq": 7800000000.0},
        "mixer_corrections": {"amp_ratio": 0.9999, "phase_error": -4.2},
    }

    assert sequencer_configs[1].model_dump(exclude_unset=True) == {
        "allow_off_grid_nco_ops": None,
        "sequencer_options": {},
        "hardware_description": {},
        "portclock": "q1:mw-q1.01",
        "channel_name": "complex_output_1",
        "latency_correction": 0.0,
        "distortion_correction": None,
        "lo_name": None,
        "modulation_frequencies": {"interm_freq": 50000000.0, "lo_freq": None},
        "mixer_corrections": None,
    }


# NV-center-specific config (this is a sanity check because of common `optical_control`` port)
def test_extract_instrument_compilation_configs_nv_center(
    qblox_hardware_config_nv_center,
):
    hardware_config = QbloxHardwareCompilationConfig.model_validate(
        deepcopy(qblox_hardware_config_nv_center)
    )

    portclocks_used = {
        ("qe0:optical_control", "qe0.ge1"),
        ("qe0:optical_control", "qe0.ionization"),
        ("qe0:optical_control", "qe0.ge0"),
    }

    instrument_configs = hardware_config._extract_instrument_compilation_configs(
        portclocks_used
    )

    assert list(instrument_configs.keys()) == [
        "cluster0",
        "red_laser",
        "spinpump_laser",
        "green_laser",
    ]

    cluster0 = instrument_configs["cluster0"]

    assert cluster0.hardware_description.model_dump(exclude_unset=True) == {
        "instrument_type": "Cluster",
        "ref": "internal",
        "modules": {2: {"instrument_type": "QCM"}},
    }

    assert cluster0.hardware_options.model_dump(exclude_unset=True) == {
        "modulation_frequencies": {
            "qe0:optical_control-qe0.ge1": {
                "interm_freq": 200000000.0,
                "lo_freq": None,
            },
            "qe0:optical_control-qe0.ionization": {
                "interm_freq": 200000000.0,
                "lo_freq": None,
            },
            "qe0:optical_control-qe0.ge0": {
                "interm_freq": 200000000.0,
                "lo_freq": None,
            },
        }
    }

    assert cluster0.portclock_to_path == {
        ("qe0:optical_control-qe0.ge1"): "cluster0.module2.real_output_0",
        ("qe0:optical_control-qe0.ionization"): "cluster0.module2.real_output_1",
        ("qe0:optical_control-qe0.ge0"): "cluster0.module2.real_output_2",
    }

    assert cluster0.lo_to_path == {
        "spinpump_laser": "cluster0.module2.real_output_0",
        "green_laser": "cluster0.module2.real_output_1",
        "red_laser": "cluster0.module2.real_output_2",
    }


def test_generate_uuid_from_wf_data():
    arr0 = np.arange(10000)
    arr1 = np.arange(10000)
    arr2 = np.arange(10000) + 1

    hash0 = generate_uuid_from_wf_data(arr0)
    hash1 = generate_uuid_from_wf_data(arr1)
    hash2 = generate_uuid_from_wf_data(arr2)

    assert hash0 == hash1
    assert hash1 != hash2


def test_real_mode_container(
    real_square_pulse_schedule,
    hardware_cfg_real_mode,
    mock_setup_basic_transmon,
):
    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_real_mode)

    real_square_pulse_schedule = _determine_absolute_timing(real_square_pulse_schedule)
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        real_square_pulse_schedule, hardware_cfg
    )
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=real_square_pulse_schedule,
        config=quantum_device.generate_compilation_config(),
    )
    assign_pulse_and_acq_info_to_devices(sched, container.clusters)
    container.prepare()
    qcm0 = container.instrument_compilers["cluster0"].instrument_compilers[
        "cluster0_module1"
    ]
    for output, seq_name in enumerate(f"seq{i}" for i in range(3)):
        seq_settings = qcm0.sequencers[seq_name].settings
        assert seq_settings.connected_output_indices[0] == output


@pytest.mark.deprecated
def test_assign_frequencies_baseband_hardware_config(
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_cluster,
):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q0"))
    sched.add(X("q1"))

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    hardware_cfg = hardware_cfg_cluster
    quantum_device.hardware_config(hardware_cfg)

    device_cfg = quantum_device.generate_device_config()
    q0_clock_freq = device_cfg.clocks["q0.01"]
    q1_clock_freq = device_cfg.clocks["q1.01"]

    if0 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q0:mw-q0.01"][
        "interm_freq"
    ]
    if1 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q1:mw-q1.01"][
        "interm_freq"
    ]

    io0_lo_name, io1_lo_name = None, None

    for edge in hardware_cfg["connectivity"]["graph"]:
        source, target = edge
        if source == "cluster0.module1.complex_output_0":
            io0_lo_name = target.replace("iq_mixer_", "").replace(".if", "")
        if source == "cluster0.module1.complex_output_1":
            io1_lo_name = target.replace("iq_mixer_", "").replace(".if", "")

    lo0 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q0:mw-q0.01"][
        "lo_freq"
    ]
    lo1 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q1:mw-q1.01"][
        "lo_freq"
    ]

    assert if0 is not None
    assert if1 is None
    assert lo0 is None
    assert lo1 is not None

    lo0 = q0_clock_freq - if0
    if1 = q1_clock_freq - lo1

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, config=quantum_device.generate_compilation_config()
    )
    compiled_instructions = compiled_schedule["compiled_instructions"]

    generic_icc = constants.GENERIC_IC_COMPONENT_NAME
    assert compiled_instructions[generic_icc][f"{io0_lo_name}.frequency"] == lo0
    assert compiled_instructions[generic_icc][f"{io1_lo_name}.frequency"] == lo1
    assert (
        compiled_instructions["cluster0"]["cluster0_module1"]["sequencers"]["seq1"][
            "modulation_freq"
        ]
        == if1
    )


def test_external_lo_not_present_raises(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q4"))
    sched.add(Measure("q4"))

    compile_config = copy.deepcopy(compile_config_basic_transmon_qblox_hardware)

    # Change to non-existent LO:
    compile_config.hardware_compilation_config.connectivity.graph = nx.relabel_nodes(
        compile_config.hardware_compilation_config.connectivity.graph,
        {"lo0.output": "non_existent_lo.output"},
    )

    with pytest.raises(
        RuntimeError,
        match="External local oscillator 'non_existent_lo' set to "
        "be used for port 'q4:mw' not found! Make "
        "sure it is present in the hardware configuration.",
    ):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(sched, config=compile_config)


def test_assign_frequencies_baseband(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q4"))
    sched.add(Measure("q4"))

    device_cfg = compile_config_basic_transmon_qblox_hardware.device_compilation_config
    mw_clock_freq = device_cfg.clocks["q4.01"]
    ro_clock_freq = device_cfg.clocks["q4.ro"]

    hardware_options = (
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_options
    )
    if_mw = hardware_options.modulation_frequencies["q4:mw-q4.01"].interm_freq
    lo_mw = hardware_options.modulation_frequencies["q4:mw-q4.01"].lo_freq
    if_ro = hardware_options.modulation_frequencies["q4:res-q4.ro"].interm_freq
    lo_ro = hardware_options.modulation_frequencies["q4:res-q4.ro"].lo_freq

    assert if_mw is not None
    assert if_ro is None
    assert lo_mw is None
    assert lo_ro is not None

    lo_mw = mw_clock_freq - if_mw
    if_ro = ro_clock_freq - lo_ro

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, config=compile_config_basic_transmon_qblox_hardware
    )
    compiled_instructions = compiled_schedule["compiled_instructions"]

    connectivity = (
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.connectivity
    )
    mw_iq_mixer_name = [n for n in connectivity.graph["q4:mw"]][0].split(".")[0]
    mw_lo_name = [n for n in connectivity.graph[mw_iq_mixer_name + ".lo"]][0].split(
        "."
    )[0]

    ro_iq_mixer_name = [n for n in connectivity.graph["q4:res"]][0].split(".")[0]
    ro_lo_name = [n for n in connectivity.graph[ro_iq_mixer_name + ".lo"]][0].split(
        "."
    )[0]

    generic_icc = constants.GENERIC_IC_COMPONENT_NAME
    assert compiled_instructions[generic_icc][f"{mw_lo_name}.frequency"] == lo_mw
    assert compiled_instructions[generic_icc][f"{ro_lo_name}.frequency"] == lo_ro
    assert (
        compiled_instructions["cluster0"]["cluster0_module3"]["sequencers"]["seq0"][
            "modulation_freq"
        ]
        == if_ro
    )


@pytest.mark.filterwarnings("ignore:Downconverter frequency 0 supplied.*")
@pytest.mark.parametrize(
    "downconverter_freq0, downconverter_freq1",
    list(itertools.product([None, 0, 9e9], repeat=2)) + [(-1, None), (1e6, None)],
)
def test_assign_frequencies_baseband_downconverter(  # noqa: PLR0912, PLR0915
    hardware_cfg_cluster,
    mock_setup_basic_transmon_with_standard_params,
    downconverter_freq0,
    downconverter_freq1,
):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q0"))
    sched.add(X("q1"))

    hardware_cfg = copy.deepcopy(hardware_cfg_cluster)
    hardware_cfg["hardware_description"]["cluster0"]["modules"]["1"][
        "complex_output_0"
    ] = {"downconverter_freq": downconverter_freq0}
    hardware_cfg["hardware_description"]["cluster0"]["modules"]["1"][
        "complex_output_1"
    ] = {"downconverter_freq": downconverter_freq1}

    io0_lo_name = None

    for edge in hardware_cfg["connectivity"]["graph"]:
        source, target = edge
        if source == "cluster0.module1.complex_output_0":
            io0_lo_name = target.replace("iq_mixer_", "").replace(".if", "")

    if0 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q0:mw-q0.01"][
        "interm_freq"
    ]
    lo1 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q1:mw-q1.01"][
        "lo_freq"
    ]

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    q0 = quantum_device.get_element("q0")
    q1 = quantum_device.get_element("q1")
    q0_clock_freq = q0.clock_freqs.f01()
    q1_clock_freq = q1.clock_freqs.f01()

    quantum_device.hardware_config(hardware_cfg)
    compiler = SerialCompiler(name="compiler")

    context_mngr = nullcontext()
    if (
        downconverter_freq0 is not None
        and (downconverter_freq0 < 0 or downconverter_freq0 < q0_clock_freq)
    ) or (
        downconverter_freq1 is not None
        and (downconverter_freq1 < 0 or downconverter_freq1 < q1_clock_freq)
    ):
        context_mngr = pytest.raises(ValueError)
    with context_mngr as error:
        compiled_schedule = compiler.compile(
            sched, config=quantum_device.generate_compilation_config()
        )
    if error is not None:
        if downconverter_freq0 is not None:
            # Find portclock config
            for edge in hardware_cfg["connectivity"]["graph"]:
                source, target = edge
                if source == "cluster0.module1.complex_output_0":
                    iq_name = target.split(".")[0]

            for edge in hardware_cfg["connectivity"]["graph"]:
                source, target = edge
                if source == f"{iq_name}.rf":
                    portclock_config = {"port": target, "clock": "q0.01"}

            if downconverter_freq0 < 0:
                assert (
                    str(error.value) == f"Downconverter frequency must be positive "
                    f"(downconverter_freq={downconverter_freq0:e}) "
                    f"(for 'seq0' of 'cluster0_module1' with "
                    f"port '{portclock_config['port']}' and "
                    f"clock '{portclock_config['clock']}')"
                )
            elif downconverter_freq0 < q0_clock_freq:
                assert (
                    str(error.value)
                    == "Downconverter frequency must be greater than clock frequency "
                    f"(downconverter_freq={downconverter_freq0:e}, "
                    f"clock_freq={q0_clock_freq:e}) "
                    f"(for 'seq0' of 'cluster0_module1' with "
                    f"port '{portclock_config['port']}' and "
                    f"clock '{portclock_config['clock']}')"
                )
        return

    generic_ic_program = compiled_schedule["compiled_instructions"][
        constants.GENERIC_IC_COMPONENT_NAME
    ]
    qcm_program = compiled_schedule["compiled_instructions"]["cluster0"][
        "cluster0_module1"
    ]
    actual_lo0 = generic_ic_program[f"{io0_lo_name}.frequency"]
    actual_if1 = qcm_program["sequencers"]["seq1"]["modulation_freq"]

    if downconverter_freq0 is None:
        expected_lo0 = q0_clock_freq - if0
    else:
        expected_lo0 = downconverter_freq0 - q0_clock_freq - if0

    if downconverter_freq1 is None:
        expected_if1 = q1_clock_freq - lo1
    else:
        expected_if1 = downconverter_freq1 - q1_clock_freq - lo1

    assert actual_lo0 == expected_lo0, (
        f"LO frequency of channel 0 "
        f"{'without' if downconverter_freq0 in (None, 0) else 'after'} "
        f"downconversion must be equal to {expected_lo0} but is equal to {actual_lo0}"
    )
    assert actual_if1 == expected_if1, (
        f"Modulation frequency of channel 1 "
        f"{'without' if downconverter_freq1 in (None, 0) else 'after'} "
        f"downconversion must be equal to {expected_if1} but is equal to {actual_if1}"
    )


@pytest.mark.deprecated
def test_assign_frequencies_rf_hardware_config(
    mock_setup_basic_transmon, hardware_cfg_rf
):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q2"))
    sched.add(X("q3"))

    hardware_cfg = copy.deepcopy(hardware_cfg_rf)
    if0 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q2:mw-q2.01"].get(
        "interm_freq"
    )
    if1 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q3:mw-q3.01"].get(
        "interm_freq"
    )
    lo0 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q2:mw-q2.01"].get(
        "lo_freq"
    )
    lo1 = hardware_cfg["hardware_options"]["modulation_frequencies"]["q3:mw-q3.01"].get(
        "lo_freq"
    )

    assert if0 is not None
    assert if1 is None
    assert lo0 is None
    assert lo1 is not None

    quantum_device = mock_setup_basic_transmon["quantum_device"]

    q2 = quantum_device.get_element("q2")
    q3 = quantum_device.get_element("q3")
    q2.clock_freqs.f01.set(6.02e9)
    q3.clock_freqs.f01.set(5.02e9)

    q2.rxy.amp180(0.213)
    q3.rxy.amp180(0.215)

    device_cfg = quantum_device.generate_device_config()
    q2_clock_freq = device_cfg.clocks["q2.01"]
    q3_clock_freq = device_cfg.clocks["q3.01"]

    lo0 = q2_clock_freq - if0
    if1 = q3_clock_freq - lo1

    quantum_device.hardware_config(hardware_cfg)
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, quantum_device.generate_compilation_config()
    )
    compiled_instructions = compiled_schedule["compiled_instructions"]
    qcm_program = compiled_instructions["cluster0"]["cluster0_module2"]

    assert qcm_program["settings"]["lo0_freq"] == lo0
    assert qcm_program["settings"]["lo1_freq"] == lo1
    assert qcm_program["sequencers"]["seq1"]["modulation_freq"] == if1


def test_assign_frequencies_rf(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q0"))
    sched.add(Measure("q0"))

    device_cfg = compile_config_basic_transmon_qblox_hardware.device_compilation_config
    mw_clock_freq = device_cfg.clocks["q0.01"]
    ro_clock_freq = device_cfg.clocks["q0.ro"]

    hardware_options = (
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_options
    )
    if_mw = hardware_options.modulation_frequencies["q0:mw-q0.01"].interm_freq
    lo_mw = hardware_options.modulation_frequencies["q0:mw-q0.01"].lo_freq
    if_ro = hardware_options.modulation_frequencies["q0:res-q0.ro"].interm_freq
    lo_ro = hardware_options.modulation_frequencies["q0:res-q0.ro"].lo_freq

    assert if_mw is not None
    assert if_ro is None
    assert lo_mw is None
    assert lo_ro is not None

    lo_mw = mw_clock_freq - if_mw
    if_ro = ro_clock_freq - lo_ro

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, config=compile_config_basic_transmon_qblox_hardware
    )
    compiled_instructions = compiled_schedule["compiled_instructions"]

    assert (
        compiled_instructions["cluster0"]["cluster0_module2"]["settings"]["lo0_freq"]
        == lo_mw
    )
    assert (
        compiled_instructions["cluster0"]["cluster0_module4"]["settings"]["lo0_freq"]
        == lo_ro
    )
    assert (
        compiled_instructions["cluster0"]["cluster0_module4"]["sequencers"]["seq0"][
            "modulation_freq"
        ]
        == if_ro
    )


@pytest.mark.filterwarnings("ignore:Downconverter frequency 0 supplied.*")
@pytest.mark.parametrize(
    "downconverter_freq0, downconverter_freq1, element_names",
    [
        list(pair) + [["q5", "q6"]]
        for pair in list(itertools.product([None, 0, 8.2e9], repeat=2))
        + [(-1, None), (1e6, None)]
    ],
)
def test_assign_frequencies_rf_downconverter(
    qblox_hardware_config_transmon,
    mock_setup_basic_transmon_elements,
    downconverter_freq0,
    downconverter_freq1,
    element_names,
):
    sched = Schedule("two_gate_experiment")
    sched.add(X(element_names[0]))
    sched.add(X(element_names[1]))

    hardware_compilation_config = copy.deepcopy(qblox_hardware_config_transmon)
    hardware_compilation_config["hardware_description"]["cluster0"]["modules"]["2"][
        "complex_output_0"
    ] = {"downconverter_freq": downconverter_freq0}
    hardware_compilation_config["hardware_description"]["cluster0"]["modules"]["2"][
        "complex_output_1"
    ] = {"downconverter_freq": downconverter_freq1}

    mock_setup = mock_setup_basic_transmon_elements
    quantum_device = mock_setup["quantum_device"]
    qubit0 = quantum_device.get_element(element_names[0])
    qubit1 = quantum_device.get_element(element_names[1])
    qubit0.clock_freqs.f01.set(6.02e9)
    qubit1.clock_freqs.f01.set(5.02e9)
    qubit0.rxy.amp180(0.213)
    qubit1.rxy.amp180(0.215)

    device_cfg = quantum_device.generate_device_config()
    qubit0_clock_freq = device_cfg.clocks[f"{qubit0.name}.01"]
    qubit1_clock_freq = device_cfg.clocks[f"{qubit1.name}.01"]

    quantum_device.hardware_config(hardware_compilation_config)
    compiler = SerialCompiler(name="compiler")

    context_mngr = nullcontext()
    if (
        downconverter_freq0 is not None
        and (downconverter_freq0 < 0 or downconverter_freq0 < qubit0_clock_freq)
    ) or (
        downconverter_freq1 is not None
        and (downconverter_freq1 < 0 or downconverter_freq1 < qubit1_clock_freq)
    ):
        context_mngr = pytest.raises(ValueError)
    with context_mngr as error:
        compiled_schedule = compiler.compile(
            sched, config=quantum_device.generate_compilation_config()
        )
    if error is not None:
        if downconverter_freq0 is not None:
            if downconverter_freq0 < 0:
                assert (
                    str(error.value) == f"Downconverter frequency must be positive "
                    f"(downconverter_freq={downconverter_freq0:e}) "
                    f"(for 'seq0' of 'cluster0_module2' with "
                    f"port '{qubit0.name}:mw' and "
                    f"clock '{qubit0.name}.01')"
                )
            elif downconverter_freq0 < qubit0_clock_freq:
                assert (
                    str(error.value)
                    == "Downconverter frequency must be greater than clock frequency "
                    f"(downconverter_freq={downconverter_freq0:e}, "
                    f"clock_freq={qubit0_clock_freq:e}) "
                    f"(for 'seq0' of 'cluster0_module2' with "
                    f"port '{qubit0.name}:mw' and "
                    f"clock '{qubit0.name}.01')"
                )
        return

    qcm_program = compiled_schedule["compiled_instructions"]["cluster0"][
        "cluster0_module2"
    ]
    actual_lo0 = qcm_program["settings"]["lo0_freq"]
    actual_lo1 = qcm_program["settings"]["lo1_freq"]
    actual_if1 = qcm_program["sequencers"]["seq1"]["modulation_freq"]

    if0 = hardware_compilation_config["hardware_options"]["modulation_frequencies"][
        f"{qubit0.ports.microwave()}-{qubit0.name}.01"
    ].get("interm_freq")
    assert if0 is not None
    lo1 = hardware_compilation_config["hardware_options"]["modulation_frequencies"][
        f"{qubit1.ports.microwave()}-{qubit1.name}.01"
    ].get("lo_freq")
    expected_lo1 = lo1

    if downconverter_freq0 is None:
        expected_lo0 = qubit0_clock_freq - if0
    else:
        expected_lo0 = downconverter_freq0 - qubit0_clock_freq - if0

    if downconverter_freq1 is None:
        expected_if1 = qubit1_clock_freq - lo1
    else:
        expected_if1 = downconverter_freq1 - qubit1_clock_freq - lo1

    assert actual_lo0 == expected_lo0, (
        f"LO frequency of channel 0 "
        f"{'without' if downconverter_freq0 in (None, 0) else 'after'} "
        f"downconversion must be equal to {expected_lo0}, but is equal to {actual_lo0}"
    )
    assert actual_lo1 == expected_lo1, (
        f"LO frequency of channel 1 "
        f"{'without' if downconverter_freq1 in (None, 0) else 'after'} "
        f"downconversion must be equal to {expected_lo1}, but is equal to {actual_lo1}"
    )
    assert actual_if1 == expected_if1, (
        f"Modulation frequency of channel 1 "
        f"{'without' if downconverter_freq1 in (None, 0) else 'after'} "
        f"downconversion must be equal to {expected_if1}, but is equal to {actual_if1}"
    )


@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
@pytest.mark.parametrize(
    "use_output, element_names",
    ([use_output, ["q0", "q5"]] for use_output in [True, False]),
)
def test_assign_attenuation_old_style_hardware_config(
    mock_setup_basic_transmon_elements,
    use_output,
    element_names,
):
    """
    Test function that checks if attenuation settings on a QRM-RF compile correctly.
    Also checks if floats are correctly converted to ints (if they are close to ints).
    """
    hardware_cfg = copy.deepcopy(hardware_config_transmon_old_style)

    complex_input = hardware_cfg["cluster0"]["cluster0_module4"]["complex_input_0"]
    complex_output = hardware_cfg["cluster0"]["cluster0_module4"]["complex_output_0"]
    if use_output:
        input_att = complex_output["input_att"]
        complex_input.pop("input_att", None)
    else:
        input_att = complex_input["input_att"]
        complex_output.pop("input_att", None)

    output_att = complex_output.get("output_att")

    assert input_att is not None
    assert output_att is not None

    quantum_device = mock_setup_basic_transmon_elements["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    sched = Schedule("Measurement")
    for qubit_name in element_names:
        qubit = quantum_device.get_element(qubit_name)
        qubit.clock_freqs.readout(5e9)
        qubit.measure.pulse_amp(0.2)
        qubit.measure.acq_delay(40e-9)

        sched.add(Measure(qubit_name))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )
    qrm_rf_program = compiled_schedule["compiled_instructions"]["cluster0"][
        "cluster0_module4"
    ]
    compiled_in0_att = qrm_rf_program["settings"]["in0_att"]
    compiled_out0_att = qrm_rf_program["settings"]["out0_att"]

    assert compiled_in0_att == input_att
    assert compiled_out0_att == output_att

    assert isinstance(compiled_in0_att, int)
    assert isinstance(compiled_out0_att, int)


def test_assign_attenuation(compile_config_basic_transmon_qblox_hardware):
    """
    Test function that checks that attenuation settings on a QRM-RF compile correctly.
    Also checks if floats are correctly converted to ints (if they are close to ints).
    """
    sched = Schedule("Measurement")
    sched.add(Measure("q0"))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, config=compile_config_basic_transmon_qblox_hardware
    )
    compiled_instructions = compiled_schedule["compiled_instructions"]

    in0_att = compiled_instructions["cluster0"]["cluster0_module4"]["settings"][
        "in0_att"
    ]
    out0_att = compiled_instructions["cluster0"]["cluster0_module4"]["settings"][
        "out0_att"
    ]

    assert in0_att == 4
    assert out0_att == 12

    assert isinstance(in0_att, int)
    assert isinstance(out0_att, int)


def test_assign_input_att_both_output_input_raises(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"4": {"instrument_type": "QRM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "input_att": {"q0:res-q0.ro": 10, "q1:res-q1.ro": 10},
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": 50000000.0},
                "q1:res-q1.ro": {"interm_freq": 50000000.0},
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module4.complex_output_0", "q0:res"],
                ["cluster0.module4.complex_input_0", "q1:res"],
            ]
        },
    }

    schedule = Schedule("test_assign_input_att_both_output_input_raises")
    schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q0:res", clock="q0.ro"))
    schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q1:res", clock="q1.ro"))
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    with pytest.raises(ValueError) as exc:
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )

    # assert exception was raised with correct message.
    assert (
        exc.value.args[0] == "'input_att' is defined for both 'complex_input_0' and "
        "'complex_output_0' on module 'cluster0_module4', which is prohibited. "
        "Make sure you define it at a single place."
    )


def test_assign_attenuation_invalid_raises(
    mock_setup_basic_transmon_with_standard_params, hardware_cfg_qcm_rf
):
    """
    Test that setting a float value (that is not close to an int) raises an error.
    """
    sched = Schedule("Single Gate Experiment")
    sched.add(X("q1"))

    hardware_cfg = copy.deepcopy(hardware_cfg_qcm_rf)
    hardware_cfg["hardware_options"]["output_att"] = {"q1:mw-q0.01": 10.3}

    mock_setup_basic_transmon_with_standard_params["quantum_device"].hardware_config(
        hardware_cfg
    )
    with pytest.raises(ValueError):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=mock_setup_basic_transmon_with_standard_params[
                "quantum_device"
            ].generate_compilation_config(),
        )


def test_assign_gain(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("Measurement")
    sched.add(Measure("q4"))

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, config=compile_config_basic_transmon_qblox_hardware
    )
    compiled_instructions = compiled_schedule["compiled_instructions"]

    assert (
        compiled_instructions["cluster0"]["cluster0_module3"]["settings"]["in0_gain"]
        == 2
    )
    assert (
        compiled_instructions["cluster0"]["cluster0_module3"]["settings"]["in1_gain"]
        == 3
    )


@pytest.mark.xfail(
    reason="This should be checked in validators of the QBloxHardwareCompilationConfig datastructure."
)
@pytest.mark.parametrize(
    "portclock, not_supported_option, value",
    [
        # "q4:res-q4.ro" is connected to complex_output of QRM
        ("q4:res-q4.ro", "input_att", 10),
        ("q4:res-q4.ro", "output_att", 10),
        ("q4:res-q4.ro", "output_gain", (2, 3)),
        # "q4:mw-q4.01" is connected to complex_output of QCM
        ("q4:mw-q4.01", "input_att", 10),
        ("q4:mw-q4.01", "output_att", 10),
        ("q4:mw-q4.01", "input_gain", (2, 3)),
        ("q4:mw-q4.01", "output_gain", (2, 3)),
        # "q0:res-q0.ro" is connected to complex_output of QRM-RF
        ("q0:res-q0.ro", "input_gain", (2, 3)),
        ("q0:res-q0.ro", "output_gain", (2, 3)),
        # "q0:mw-q0.01" is connected to complex_output of QCM-RF
        ("q0:mw-q0.01", "input_att", 10),
        ("q0:mw-q0.01", "input_gain", (2, 3)),
        ("q0:mw-q0.01", "output_gain", (2, 3)),
    ],
)
def test_set_power_scaling_invalid(
    mock_setup_basic_transmon_with_standard_params,
    qblox_hardware_config_transmon,
    portclock,
    not_supported_option,
    value,
):
    sched = Schedule("single_gate_experiment")
    sched.add(Measure("q0"))

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    hardware_compilation_config = copy.deepcopy(qblox_hardware_config_transmon)
    hardware_compilation_config["hardware_options"]["power_scaling"][portclock] = {
        not_supported_option: value
    }

    quantum_device.hardware_config(hardware_compilation_config)

    with pytest.raises(ValueError, match="not supported"):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=quantum_device.generate_compilation_config(),
        )


def test_markers(mock_setup_basic_transmon, hardware_cfg_cluster, hardware_cfg_rf):
    def _confirm_correct_markers(
        device_program, default_marker, is_rf=False, sequencer=0
    ):
        answer = default_marker
        qasm = device_program["sequencers"][f"seq{sequencer}"]["sequence"]["program"]

        matches = re.findall(r"set\_mrk +\d+", qasm)
        match = [int(m.replace("set_mrk", "").strip()) for m in matches][0]
        assert match == answer

    # Test for baseband
    sched = Schedule("gate_experiment")
    sched.add(X("q0"))
    sched.add(X("q2"))
    sched.add(Measure("q0"))
    sched.add(Measure("q2"))

    quantum_device = mock_setup_basic_transmon["quantum_device"]

    q0 = quantum_device.get_element("q0")
    q3 = quantum_device.get_element("q3")  # q3 uses the same module as q2
    q2 = quantum_device.get_element("q2")

    q0.rxy.amp180(0.213)
    q3.rxy.amp180(0.211)
    q2.rxy.amp180(0.215)

    q0.clock_freqs.f01(7.3e9)
    q0.clock_freqs.f12(7.0e9)
    q0.clock_freqs.readout(8.0e9)
    q0.measure.acq_delay(100e-9)

    q3.clock_freqs.f01(5.2e9)

    q2.clock_freqs.f01(6.33e9)
    q2.clock_freqs.f12(7.0e9)
    q2.clock_freqs.readout(8.0e9)
    q2.measure.acq_delay(100e-9)

    quantum_device.hardware_config(hardware_cfg_cluster)
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, quantum_device.generate_compilation_config()
    )
    program = compiled_schedule["compiled_instructions"]

    _confirm_correct_markers(program["cluster0"]["cluster0_module1"], 0)
    _confirm_correct_markers(program["cluster0"]["cluster0_module3"], 0)

    # # Test for rf
    sched = Schedule("gate_experiment")
    sched.add(X("q2"))
    sched.add(X("q3"))
    sched.add(Measure("q2"))

    quantum_device.hardware_config(hardware_cfg_rf)
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, quantum_device.generate_compilation_config()
    )

    qcm_rf_program = compiled_schedule["compiled_instructions"]["cluster0"][
        "cluster0_module2"
    ]
    qrm_rf_program = compiled_schedule["compiled_instructions"]["cluster0"][
        "cluster0_module4"
    ]

    _confirm_correct_markers(qcm_rf_program, 0b0001, is_rf=True, sequencer=0)
    _confirm_correct_markers(qcm_rf_program, 0b0010, is_rf=True, sequencer=1)
    _confirm_correct_markers(qrm_rf_program, 0b0011, is_rf=True)


def test_extract_settings_from_mapping(
    make_schedule_with_measurement, create_schedule_with_pulse_info
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {"1": {"instrument_type": "QRM_RF"}},
            },
            "lo0": {"instrument_type": "LocalOscillator", "power": 1},
            "iq_mixer_lo0": {"instrument_type": "IQMixer"},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": None, "lo_freq": 7.8e9},
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "iq_mixer_lo0.if"],
                ["lo0.output", "iq_mixer_lo0.lo"],
                ["iq_mixer_lo0.rf", "q0:res"],
            ]
        },
    }

    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg)
    sched = create_schedule_with_pulse_info(make_schedule_with_measurement("q0"))
    instrument_configs = hardware_cfg._extract_instrument_compilation_configs(
        _extract_port_clocks_used(sched)
    )
    module_configs = instrument_configs[
        "cluster0"
    ]._extract_module_compilation_configs()

    types.BasebandModuleSettings.extract_settings_from_mapping(module_configs[1])
    types.RFModuleSettings.extract_settings_from_mapping(module_configs[1])


def test_cluster_settings(
    compile_config_basic_transmon_qblox_hardware, create_schedule_with_pulse_info
):
    sched = Schedule("pulse_only_experiment")
    sched.add(IdlePulse(duration=200e-6))
    sched.add(
        DRAGPulse(
            G_amp=0.5,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(RampPulse(t0=2e-3, amp=0.5, duration=28e-9, port="q4:mw", clock="q0.04"))
    pulse_only_schedule = create_schedule_with_pulse_info(
        _determine_absolute_timing(sched)
    )

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        pulse_only_schedule,
        compile_config_basic_transmon_qblox_hardware.hardware_compilation_config,
    )
    cluster_compiler = container.instrument_compilers["cluster0"]
    cluster_compiler.prepare()
    cl_qcm0 = cluster_compiler.instrument_compilers["cluster0_module1"]
    assert isinstance(cl_qcm0._settings, BasebandModuleSettings)


class TestAssemblyValid:
    @staticmethod
    def _strip_spaces_and_comments(program: str):
        # helper function for comparing programs
        stripped_program = []
        for line in program.split("\n"):
            if "#" in line:
                line = line.split("#")[0]  # noqa: PLW2901
            line = line.rstrip()  # remove trailing whitespace # noqa: PLW2901
            line = re.sub(  # remove line starting spaces # noqa: PLW2901
                r"^ +", "", line
            )
            line = re.sub(  # replace multiple spaces with one # noqa: PLW2901
                r" +", " ", line
            )
            stripped_program.append(line)

        return stripped_program

    @staticmethod
    def _validate_seq_instructions(seq_instructions, filename):
        baseline_assembly = os.path.join(
            quantify_scheduler.__path__[0],
            "..",
            "tests",
            "baseline_qblox_assembly",
            filename,
        )

        # (Utility for updating ref files during development, does not belong to the test)
        if REGENERATE_REF_FILES:
            with open(baseline_assembly, "w", encoding="utf-8") as file:
                json.dump(seq_instructions, file)

        with open(baseline_assembly) as file:
            baseline_seq_instructions = json.load(file)

        program = TestAssemblyValid._strip_spaces_and_comments(
            seq_instructions["program"]
        )
        ref_program = TestAssemblyValid._strip_spaces_and_comments(
            baseline_seq_instructions["program"]
        )

        assert list(program) == list(ref_program)

    @staticmethod
    def _validate_assembly(compiled_schedule, cluster):
        """
        Test helper that takes a compiled schedule and verifies if the assembly is valid
        by passing it to a cluster. Only QCM (module2) and QRM (module4) modules are
        defined.
        """
        all_modules = {module.name: module for module in cluster.modules}

        for slot_idx in ("2", "4"):
            module = all_modules[f"{cluster.name}_module{slot_idx}"]

            module_seq0_json = compiled_schedule["compiled_instructions"][cluster.name][
                module.name
            ]["sequencers"]["seq0"]["sequence"]
            module.sequencer0.sequence(module_seq0_json)
            module.arm_sequencer(0)
            uploaded_waveforms = module.get_waveforms(0)
            assert uploaded_waveforms is not None

    def test_acq_protocol_append_mode_ssro(
        self,
        request,
        dummy_cluster,
        compile_config_basic_transmon_qblox_hardware,
    ):
        cluster = dummy_cluster()

        sched = readout_calibration_sched("q0", [0, 1], repetitions=256)
        compiler = SerialCompiler(name="compiler")
        compiled_ssro_sched = compiler.compile(
            sched, compile_config_basic_transmon_qblox_hardware
        )

        self._validate_assembly(compiled_schedule=compiled_ssro_sched, cluster=cluster)

        qrm_name = f"{cluster.name}_module4"
        test_name = request.node.name[len("test_") :]
        self._validate_seq_instructions(
            seq_instructions=compiled_ssro_sched["compiled_instructions"][cluster.name][
                qrm_name
            ]["sequencers"]["seq0"]["sequence"],
            filename=f"{test_name}_{qrm_name}_seq0_instr.json",
        )

    def test_acq_protocol_average_mode_allxy(
        self,
        request,
        dummy_cluster,
        compile_config_basic_transmon_qblox_hardware,
    ):
        cluster = dummy_cluster()

        sched = allxy_sched("q0", element_select_idx=np.arange(21), repetitions=256)
        compiler = SerialCompiler(name="compiler")
        compiled_allxy_sched = compiler.compile(
            sched, compile_config_basic_transmon_qblox_hardware
        )

        self._validate_assembly(compiled_schedule=compiled_allxy_sched, cluster=cluster)

        qrm_name = f"{cluster.name}_module4"
        test_name = request.node.name[len("test_") :]
        self._validate_seq_instructions(
            seq_instructions=compiled_allxy_sched["compiled_instructions"][
                cluster.name
            ][qrm_name]["sequencers"]["seq0"]["sequence"],
            filename=f"{test_name}_{qrm_name}_seq0_instr.json",
        )

    def test_assigning_waveform_indices(
        self,
        request,
        dummy_cluster,
        compile_config_basic_transmon_qblox_hardware,
    ):
        cluster = dummy_cluster()

        sched = allxy_sched("q0", element_select_idx=np.arange(21), repetitions=256)
        compiler = SerialCompiler(name="compiler")
        compiled_allxy_sched = compiler.compile(
            sched, compile_config_basic_transmon_qblox_hardware
        )

        self._validate_assembly(compiled_schedule=compiled_allxy_sched, cluster=cluster)

        qcm_name = f"{cluster.name}_module2"
        test_name = request.node.name[len("test_") :]
        self._validate_seq_instructions(
            seq_instructions=compiled_allxy_sched["compiled_instructions"][
                cluster.name
            ][qcm_name]["sequencers"]["seq0"]["sequence"],
            filename=f"{test_name}_{qcm_name}_seq0_instr.json",
        )


def test_acq_declaration_dict_append_mode(
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    repetitions = 256

    ssro_sched = readout_calibration_sched("q0", [0, 1], repetitions=1)
    outer_sched = Schedule("outer", repetitions=repetitions)
    outer_sched.add(LoopOperation(body=ssro_sched, repetitions=3))
    compiler = SerialCompiler(name="compiler")
    compiled_ssro_sched = compiler.compile(
        outer_sched, compile_config_basic_transmon_qblox_hardware_cluster
    )

    qrm0_seq_instructions = compiled_ssro_sched["compiled_instructions"]["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]["sequence"]

    acquisitions = qrm0_seq_instructions["acquisitions"]
    # the only key corresponds to channel 0
    assert set(acquisitions.keys()) == {"0"}
    assert acquisitions["0"] == {"num_bins": 3 * 2 * 256, "index": 0}


def test_acq_declaration_dict_bin_avg_mode(
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    allxy = allxy_sched("q0")
    compiler = SerialCompiler(name="compiler")
    compiled_allxy_sched = compiler.compile(
        allxy, config=compile_config_basic_transmon_qblox_hardware_cluster
    )
    qrm0_seq_instructions = compiled_allxy_sched["compiled_instructions"]["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]["sequence"]

    acquisitions = qrm0_seq_instructions["acquisitions"]

    # the only key corresponds to channel 0
    assert set(acquisitions.keys()) == {"0"}
    assert acquisitions["0"] == {"num_bins": 21, "index": 0}


# Setting latency corrections in the hardware config is deprecated
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_apply_latency_corrections_hardware_config_invalid_raises(
    mock_setup_basic_transmon, hardware_cfg_latency_corrections_invalid
):
    """
    This test function checks that:
    Providing an invalid latency correction specification raises an exception
    when compiling.
    """

    sched = Schedule("Single Gate Experiment on Two Qubits")
    sched.add(X("q0"))
    sched.add(
        SquarePulse(port="q1:mw", clock="q1.01", amp=0.25, duration=12e-9),
        ref_pt="start",
    )
    sched.add_resources(
        [ClockResource("q0.01", freq=5e9), ClockResource("q1.01", freq=5e9)]
    )

    hardware_cfg = copy.deepcopy(hardware_cfg_latency_corrections_invalid)
    hardware_cfg["hardware_options"]["latency_corrections"]["q1:mw-q1.01"] = None
    mock_setup_basic_transmon["quantum_device"].hardware_config(hardware_cfg)
    with pytest.raises(ValidationError):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=mock_setup_basic_transmon[
                "quantum_device"
            ].generate_compilation_config(),
        )


# Setting latency corrections in the hardware config is deprecated
@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_apply_latency_corrections_hardware_config_valid(
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_cluster_latency_corrections,
    create_schedule_with_pulse_info,
):
    """
    This test function checks that:
    Latency correction is set for the correct portclock key
    by checking against the value set in QASM instructions.
    """

    mock_setup = mock_setup_basic_transmon_with_standard_params
    hardware_cfg = hardware_cfg_cluster_latency_corrections
    mock_setup["quantum_device"].hardware_config(
        hardware_cfg_cluster_latency_corrections
    )

    sched = Schedule("Single Gate Experiment on Two Qubits")
    sched.add(X("q0"))
    sched.add(
        SquarePulse(port="q1:mw", clock="q1.01", amp=0.25, duration=12e-9),
        ref_pt="start",
    )
    sched = create_schedule_with_pulse_info(sched)

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=mock_setup["quantum_device"].generate_compilation_config(),
    )

    compiled_data = compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"]
    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(
        hardware_cfg_cluster_latency_corrections
    )

    latency_dict = corrections.determine_relative_latency_corrections(
        schedule=sched,
        hardware_cfg=hardware_cfg,
    )
    latency = int(1e9 * latency_dict[f"q0:mw-q0.01"])

    program_lines = compiled_data["sequencers"]["seq0"]["sequence"][
        "program"
    ].splitlines()
    assert any(
        f"latency correction of {constants.MIN_TIME_BETWEEN_OPERATIONS} + {latency} ns"
        in line
        for line in program_lines
    ), f"instrument={('cluster0', 'cluster0_module1')}, latency={latency}"


def test_apply_latency_corrections_hardware_options_valid(
    compile_config_basic_transmon_qblox_hardware,
):
    """
    This test function checks that:
    Latency correction is set for the correct portclock key
    by checking against the value set in QASM instructions.
    """

    sched = Schedule("Latency experiment")
    sched.add(X("q4"))
    sched.add(
        SquarePulse(port="q4:res", clock="q4.ro", amp=0.25, duration=12e-9),
        ref_pt="start",
    )

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    compiled_instructions = compiled_sched.compiled_instructions

    # Check latency correction for q4:mw-q4.01:
    program_lines_mw = compiled_instructions["cluster0"]["cluster0_module1"][
        "sequencers"
    ]["seq0"]["sequence"]["program"].splitlines()
    assert any(
        f"latency correction of {constants.MIN_TIME_BETWEEN_OPERATIONS} + 8 ns" in line
        for line in program_lines_mw
    )

    # Check latency correction for q4:res-q4.ro:
    program_lines_ro = compiled_instructions["cluster0"]["cluster0_module3"][
        "sequencers"
    ]["seq0"]["sequence"]["program"].splitlines()
    assert any(
        f"latency correction of {constants.MIN_TIME_BETWEEN_OPERATIONS} + 0 ns" in line
        for line in program_lines_ro
    )


# Setting latency corrections in the hardware config is deprecated
@pytest.mark.deprecated
def test_determine_relative_latency_corrections(
    hardware_cfg_cluster_latency_corrections,
    create_schedule_with_pulse_info,
) -> None:
    schedule = Schedule(f"q0 and q1")
    schedule.add(Reset("q0"))
    schedule.add(Reset("q1"))
    schedule.add(X90("q0"))
    schedule.add(X90("q1"))
    schedule = create_schedule_with_pulse_info(schedule)

    hardware_cfg = QbloxHardwareCompilationConfig.model_validate(
        hardware_cfg_cluster_latency_corrections
    )

    generated_latency_dict = corrections.determine_relative_latency_corrections(
        schedule=schedule,
        hardware_cfg=hardware_cfg,
    )
    assert generated_latency_dict == {"q0:mw-q0.01": 2.4e-08, "q1:mw-q1.01": 0.0}


def test_apply_mixer_corrections(
    compile_config_basic_transmon_qblox_hardware,
):
    """
    This test function checks that:
    mixer corrections are set for the correct portclock key
    by checking against the value set in the compiled instructions.
    """
    expected_settings = compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_options.mixer_corrections[
        "q4:res-q4.ro"
    ]

    sched = Schedule("Simple experiment")
    sched.add(
        SquarePulse(port="q4:res", clock="q4.ro", amp=0.25, duration=12e-9),
        ref_pt="start",
    )

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    qrm_compiled_instructions = compiled_sched.compiled_instructions["cluster0"][
        "cluster0_module3"
    ]

    assert (
        qrm_compiled_instructions["settings"]["offset_ch0_path_I"]
        == expected_settings.dc_offset_i
    )
    assert (
        qrm_compiled_instructions["settings"]["offset_ch0_path_Q"]
        == expected_settings.dc_offset_q
    )

    assert (
        qrm_compiled_instructions["sequencers"]["seq0"]["mixer_corr_gain_ratio"]
        == expected_settings.amp_ratio
    )
    assert (
        qrm_compiled_instructions["sequencers"]["seq0"][
            "mixer_corr_phase_offset_degree"
        ]
        == expected_settings.phase_error
    )


def test_compile_sequencer_options(
    mock_setup_basic_transmon_with_standard_params,
    qblox_hardware_config_transmon,
):
    sched = Schedule("Simple experiment")
    sched.add(
        SquarePulse(port="q4:res", clock="q4.ro", amp=0.25, duration=12e-9),
        ref_pt="start",
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    hardware_compilation_config = copy.deepcopy(qblox_hardware_config_transmon)

    hardware_compilation_config["hardware_options"]["sequencer_options"] = {
        "q4:res-q4.ro": {
            "ttl_acq_threshold": 0.2,
            "init_offset_awg_path_I": 0.1,
            "init_offset_awg_path_Q": -0.1,
            "init_gain_awg_path_I": 0.55,
            "init_gain_awg_path_Q": 0.66,
        }
    }

    quantum_device.hardware_config(hardware_compilation_config)
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )
    sequencer_instructions = compiled_sched.compiled_instructions["cluster0"][
        "cluster0_module3"
    ]["sequencers"]["seq0"]

    assert sequencer_instructions["ttl_acq_threshold"] == 0.2
    assert sequencer_instructions["init_offset_awg_path_I"] == 0.1
    assert sequencer_instructions["init_offset_awg_path_Q"] == -0.1
    assert sequencer_instructions["init_gain_awg_path_I"] == 0.55
    assert sequencer_instructions["init_gain_awg_path_Q"] == 0.66


def test_digital_channel_any_clock_name(
    mock_setup_basic_transmon_with_standard_params, assert_equal_q1asm
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QRM"},
                },
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("cluster0.module1.digital_output_1", "q0:switch"),
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
            clock="q0.some_clock",
        ),
    )
    schedule.add(IdlePulse(duration=4e-9))
    schedule.add_resource(BasebandClockResource(name="q0.some_clock"))

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # # Assert markers were set correctly, and wait time is correct for QRM
    seq0_digital = compiled_sched.compiled_instructions["cluster0"]["cluster0_module1"][
        "sequencers"
    ]["seq0"]["sequence"]["program"]
    assert_equal_q1asm(
        seq0_digital,
        """
set_mrk 0 # set markers to 0
 wait_sync 4 
 upd_param 4 
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:   
 reset_ph  
 upd_param 4 
 set_mrk 2 # set markers to 2
 upd_param 4 
 wait 496 # auto generated wait (496 ns)
 set_mrk 0 # set markers to 0
 upd_param 4 
 loop R0,@start 
 stop  
""",
    )


def test_stitched_pulse_compilation_smoke_test(mock_setup_basic_nv_qblox_hardware):
    sched = Schedule("Test schedule")
    port = "qe0:optical_readout"
    clock = "qe0.ge0"
    sched.add_resource(ClockResource(name=clock, freq=470.4e12))

    sched.add(SquarePulse(amp=0.1, duration=1e-6, port=port, clock=clock))

    builder = StitchedPulseBuilder(port=port, clock=clock)
    stitched_pulse = (
        builder.add_pulse(SquarePulse(amp=0.16, duration=5e-6, port=port, clock=clock))
        .add_voltage_offset(0.4, 0.0, duration=5e-6)
        .build()
    )
    sched.add(stitched_pulse)
    sched.add(
        SSBIntegrationComplex(port=port, clock=clock, duration=9.996e-6),
        ref_pt="start",
        rel_time=4e-9,
    )

    sched.add(SquarePulse(amp=0.2, duration=1e-6, port=port, clock=clock))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    compiler = SerialCompiler(name="compiler")
    _ = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )


def test_stitched_pulse_compilation_upd_param_at_end(
    mock_setup_basic_nv_qblox_hardware,
):
    sched = Schedule("Test schedule")
    port = "qe0:optical_readout"
    port2 = "qe0:optical_control"
    clock = "qe0.ge0"
    sched.add_resource(ClockResource(name=clock, freq=470.4e12))

    only_offsets_pulse = (
        StitchedPulseBuilder(port=port, clock=clock)
        .add_voltage_offset(path_I=0.5, path_Q=0.0)
        .add_voltage_offset(path_I=0.0, path_Q=0.0, rel_time=1e-5)
        .build()
    )
    sched.add(only_offsets_pulse)
    # Add a pulse on a different port to ensure a wait is inserted for the
    # sequencer that plays the above pulse
    sched.add(SquarePulse(amp=0.5, duration=1e-7, port=port2, clock=clock))

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    program_with_long_square = compiled_sched.compiled_instructions["cluster0"][
        "cluster0_module4"
    ]["sequencers"]["seq0"]["sequence"]["program"].splitlines()
    for i, string in enumerate(program_with_long_square):
        if "set_awg_offs" in string:
            break
    assert re.search(r"^\s*set_awg_offs\s+16384,0\s+", program_with_long_square[i])
    assert re.search(r"^\s*upd_param\s+4\s+", program_with_long_square[i + 1])
    assert re.search(r"^\s*wait\s+9996\s+", program_with_long_square[i + 2])
    assert re.search(r"^\s*set_awg_offs\s+0,0\s+", program_with_long_square[i + 3])
    assert re.search(r"^\s*upd_param\s+4\s+", program_with_long_square[i + 4])
    assert re.search(r"^\s*wait\s+96\s+", program_with_long_square[i + 5])


def test_q1asm_stitched_pulses(mock_setup_basic_nv_qblox_hardware):
    sched = Schedule("stitched_pulses")
    port = "qe0:optical_readout"
    clock = "qe0.ge0"
    sched.add_resource(ClockResource(name=clock, freq=470.4e12))

    sched.add(
        long_square_pulse(
            amp=0.5,
            duration=4e-6,
            port=port,
            clock=clock,
        ),
        rel_time=4e-9,
        ref_pt="start",
    )
    sched.add(
        long_ramp_pulse(
            amp=1.0,
            duration=4e-6,
            port=port,
            offset=-0.5,
            clock=clock,
        ),
        rel_time=5e-7,
    )
    sched.add(
        staircase_pulse(
            start_amp=-0.5,
            final_amp=0.5,
            num_steps=5,
            duration=4e-6,
            port=port,
            clock=clock,
        ),
        rel_time=5e-7,
    )

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )

    assert (
        """ set_awg_offs 16384,0 # setting offset for long_square_pulse
 upd_param 4 
 wait 3992 # auto generated wait (3992 ns)
 set_awg_offs 0,0 # setting offset for long_square_pulse
 set_awg_gain 16384,0 # setting gain for long_square_pulse
 play 0,0,4 # play long_square_pulse (4 ns)
 wait 500 # auto generated wait (500 ns)
 set_awg_offs -16384,0 # setting offset for long_ramp_pulse
 set_awg_gain 16376,0 # setting gain for long_ramp_pulse
 play 1,1,4 # play long_ramp_pulse (2000 ns)
 wait 1996 # auto generated wait (1996 ns)
 set_awg_offs 0,0 # setting offset for long_ramp_pulse
 set_awg_gain 16376,0 # setting gain for long_ramp_pulse
 play 1,1,4 # play long_ramp_pulse (2000 ns)
 wait 2496 # auto generated wait (2496 ns)
 set_awg_offs -16384,0 # setting offset for staircase_pulse
 upd_param 4 
 wait 796 # auto generated wait (796 ns)
 set_awg_offs -8192,0 # setting offset for staircase_pulse
 upd_param 4 
 wait 796 # auto generated wait (796 ns)
 set_awg_offs 0,0 # setting offset for staircase_pulse
 upd_param 4 
 wait 796 # auto generated wait (796 ns)
 set_awg_offs 8192,0 # setting offset for staircase_pulse
 upd_param 4 
 wait 796 # auto generated wait (796 ns)
 set_awg_offs 16384,0 # setting offset for staircase_pulse
 upd_param 4 
 wait 792 # auto generated wait (792 ns)
 set_awg_offs 0,0 # setting offset for staircase_pulse
 set_awg_gain 16384,0 # setting gain for staircase_pulse
 play 0,0,4 # play staircase_pulse (4 ns)"""
        in compiled_sched.compiled_instructions["cluster0"]["cluster0_module4"][
            "sequencers"
        ]["seq0"]["sequence"]["program"]
    )


def test_auto_compile_long_square_pulses(
    mock_setup_basic_nv_qblox_hardware,
):
    sched = Schedule("long_square_pulse_schedule")
    port = "qe0:optical_readout"
    clock = "qe0.ge0"
    sched.add_resource(ClockResource(name=clock, freq=470.4e12))
    square_pulse = SquarePulse(
        amp=0.2,
        duration=2.5e-6,
        port=port,
        clock=clock,
        t0=1e-6,
    )
    # copy to check later if the compilation does not affect the original operation
    saved_pulse = copy.deepcopy(square_pulse)
    sched.add(square_pulse)
    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=quantum_device.generate_compilation_config(),
    )

    assert (
        list(compiled_sched.operations.values())[0]["pulse_info"]
        == long_square_pulse(
            amp=0.2,
            duration=2.5e-6,
            port=port,
            clock=clock,
            t0=1e-6,
        )["pulse_info"]
    )

    assert square_pulse == saved_pulse


def test_long_acquisition(
    mixed_schedule_with_acquisition,
    compile_config_basic_transmon_qblox_hardware,
    get_subschedule_operation,
):
    compile_config_basic_transmon_qblox_hardware.device_compilation_config.elements[
        "q0"
    ]["measure"].factory_kwargs["pulse_duration"] = 3e-6
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        mixed_schedule_with_acquisition,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    readout_pulse_operation = get_subschedule_operation(compiled_sched, [2, 1])
    assert (
        readout_pulse_operation["pulse_info"]
        == long_square_pulse(amp=0.25, duration=3e-6, port="q0:res", clock="q0.ro")[
            "pulse_info"
        ]
    )


def test_too_long_waveform_doesnt_raise(
    compile_config_basic_transmon_qblox_hardware,
):
    sched = Schedule("Too long waveform")
    sched.add(
        SquarePulse(
            amp=0.5,
            duration=constants.MAX_SAMPLE_SIZE_WAVEFORMS // 2 * 1e-9,
            port="q0:res",
            clock="q0.ro",
        )
    )
    sched.add(
        SquarePulse(
            amp=0.5,
            duration=constants.MAX_SAMPLE_SIZE_WAVEFORMS // 2 * 1e-9,
            port="q0:res",
            clock="q0.ro",
        )
    )
    compiler = SerialCompiler(name="compiler")
    _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)


def test_too_long_waveform_raises(
    compile_config_basic_transmon_qblox_hardware,
):
    sched = Schedule("Too long waveform")
    sched.add(
        DRAGPulse(
            G_amp=0.5,
            D_amp=-0.2,
            phase=90,
            port="q0:res",
            duration=(constants.MAX_SAMPLE_SIZE_WAVEFORMS // 2 + 4) * 1e-9,
            clock="q0.ro",
            t0=4e-9,
        )
    )
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(RuntimeError) as error:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    assert (
        "waveform size" in error.value.args[0] or "sample limit" in error.value.args[0]
    )


def test_too_long_waveform_raises2(
    compile_config_basic_transmon_qblox_hardware,
):
    sched = Schedule("Too long waveform")
    sched.add(
        DRAGPulse(
            G_amp=0.5,
            D_amp=-0.2,
            phase=90,
            port="q0:res",
            duration=constants.MAX_SAMPLE_SIZE_WAVEFORMS // 4 * 1e-9,
            clock="q0.ro",
            t0=4e-9,
        )
    )
    sched.add(
        SoftSquarePulse(
            amp=0.5,
            duration=constants.MAX_SAMPLE_SIZE_WAVEFORMS // 4 * 1e-9,
            port="q0:res",
            clock="q0.ro",
        )
    )
    sched.add(
        RampPulse(
            amp=0.5,
            duration=(constants.MAX_SAMPLE_SIZE_WAVEFORMS // 4 + 4) * 1e-9,
            port="q0:res",
            clock="q0.ro",
        )
    )
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(RuntimeError) as error:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    assert (
        "waveform size" in error.value.args[0] or "sample limit" in error.value.args[0]
    )


def test_set_reference_magnitude_raises(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("amp_ref")
    sched.add(
        SquarePulse(
            amp=0.5,
            duration=20e-9,
            reference_magnitude=ReferenceMagnitude(1.0, "V"),
            port="q0:res",
            clock="q0.ro",
        )
    )
    compiler = SerialCompiler(name="compiler")
    with pytest.warns(
        RuntimeWarning,
        match="reference_magnitude parameter not implemented. This parameter will be ignored.",
    ):
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)


def test_zero_pulse_skip_timing(
    request,
    dummy_cluster,
    compile_config_basic_transmon_qblox_hardware,
):
    cluster = dummy_cluster()

    sched = Schedule("ZeroPulseSkip", repetitions=1)
    sched.add(
        DRAGPulse(
            G_amp=0.0,
            D_amp=0.0,
            phase=90,
            duration=20e-9,
            port="q0:mw",
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(
        SoftSquarePulse(
            amp=0.5,
            duration=20e-9,
            port="q0:mw",
            clock="q0.01",
        )
    )
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched, compile_config_basic_transmon_qblox_hardware
    )

    qcm_name = f"{cluster.name}_module2"
    compiled_instructions = compiled_sched["compiled_instructions"][cluster.name][
        qcm_name
    ]["sequencers"]["seq0"]["sequence"]

    assert len(compiled_instructions["waveforms"]) == 1

    idx = 0
    seq_instructions = compiled_instructions["program"].splitlines()
    for i, line in enumerate(seq_instructions):
        if re.search(r"^\s*wait\s+20\s+", line):
            idx = i
            break

    assert re.search(r"^\s*set_awg_gain\s+16384,0\s+", seq_instructions[idx + 1])
    assert re.search(r"^\s*play\s+0,0,4\s+", seq_instructions[idx + 2])


@pytest.mark.parametrize(
    "pulse_offset, measure_offset, expected",
    [
        (40e-9, 1200e-9, None),
        (20e-9, 1200e-9, "interrupting previous Pulse"),
        (40e-9, 400e-9, "interrupting previous Acquisition"),
        (
            20e-9,
            400e-9,
            "interrupting previous Acquisition | interrupting previous Pulse",
        ),
    ],
)
def test_overlapping_operations_warn(
    compile_config_basic_transmon_qblox_hardware, pulse_offset, measure_offset, expected
):
    sched = Schedule("Overlapping operations", repetitions=1)

    pulse = DRAGPulse(
        G_amp=1.0,
        D_amp=0.0,
        phase=90,
        duration=40e-9,
        port="q0:mw",
        clock="q0.01",
    )
    pulse_ref = sched.add(pulse, label="pulse 1")
    measure_ref = sched.add(
        SSBIntegrationComplex(
            duration=800e-9,
            port="q0:res",
            clock="q0.ro",
            acq_channel=0,
            acq_index=0,
        ),
        ref_op=pulse_ref,
        ref_pt="start",
        label="measure 1",
    )

    sched.add(
        pulse, ref_op=pulse_ref, ref_pt="start", rel_time=pulse_offset, label="pulse 2"
    )

    sched.add(
        pulse,
        ref_op=measure_ref,
        ref_pt="start",
        rel_time=measure_offset,
        label="pulse 3",
    )
    sched.add(
        SSBIntegrationComplex(
            duration=800e-9, port="q0:res", clock="q0.ro", acq_channel=0, acq_index=1
        ),
        ref_op=measure_ref,
        ref_pt="start",
        rel_time=measure_offset,
        label="measure 2",
    )

    compiler = SerialCompiler(name="compiler")

    if expected is None:
        context_mngr = nullcontext()
    else:
        context_mngr = pytest.warns(RuntimeWarning, match=expected)
    with context_mngr:
        compiler.compile(sched, compile_config_basic_transmon_qblox_hardware)


@pytest.mark.parametrize("debug_mode", [True, False])
def test_debug_mode_qasm_aligning(
    qblox_hardware_config_transmon,
    mock_setup_basic_transmon_with_standard_params,
    debug_mode,
):
    """
    Test how the `debug_mode` compilation configuration option
    affects the compiled Q1ASM code. It should add proper aligning.
    """
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_transmon)
    compilation_config = quantum_device.generate_compilation_config()
    compilation_config.debug_mode = debug_mode

    compiler = SerialCompiler(name="compiler")

    # Define experiment schedule
    schedule = Schedule("test align qasm fields")
    schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q0:res", clock="q0.ro"))

    compiled_schedule = compiler.compile(schedule=schedule, config=compilation_config)
    program = compiled_schedule.compiled_instructions["cluster0"]["cluster0_module4"][
        "sequencers"
    ]["seq0"]["sequence"]["program"]

    if not debug_mode:
        expected_program = """ set_mrk 3 # set markers to 3
 wait_sync 4 
 upd_param 4 
 wait 4 # latency correction of 4 + 0 ns
 move 1,R0 # iterator for loop with label start
start:   
 reset_ph  
 upd_param 4 
 set_awg_offs 16384,0 # setting offset for SquarePulse
 upd_param 4 
 wait 992 # auto generated wait (992 ns)
 set_awg_offs 0,0 # setting offset for SquarePulse
 set_awg_gain 16384,0 # setting gain for SquarePulse
 play 0,0,4 # play SquarePulse (4 ns)
 loop R0,@start 
 stop  
"""
    else:
        expected_program = """          set_mrk       3          # set markers to 3                    
          wait_sync     4                                                
          upd_param     4                                                
          wait          4          # latency correction of 4 + 0 ns      
          move          1,R0       # iterator for loop with label start  
  start:                                                                 
          reset_ph                                                       
          upd_param     4                                                
          set_awg_offs  16384,0    # setting offset for SquarePulse      
          upd_param     4                                                
          wait          992        # auto generated wait (992 ns)        
          set_awg_offs  0,0        # setting offset for SquarePulse      
          set_awg_gain  16384,0    # setting gain for SquarePulse        
          play          0,0,4      # play SquarePulse (4 ns)             
          loop          R0,@start                                        
          stop                                                           
"""
    assert program == expected_program, program


@pytest.mark.parametrize(
    "op1, op2, ref_pt2, rel_time2",
    [
        (
            long_square_pulse(amp=0.3, duration=100e-9, port="q0:res", clock="q0.ro"),
            SquarePulse(amp=0.3, duration=32e-9, port="q0:res", clock="q0.ro"),
            "start",
            32e-9,
        ),
        (
            long_square_pulse(amp=0.3, duration=100e-9, port="q0:res", clock="q0.ro"),
            SquarePulse(amp=0.3, duration=32e-9, port="q0:res", clock="q0.ro"),
            "start",
            84e-9,
        ),
        (
            long_square_pulse(amp=0.4, duration=100e-9, port="q0:res", clock="q0.ro"),
            long_square_pulse(amp=0.2, duration=100e-9, port="q0:res", clock="q0.ro"),
            "start",
            52e-9,
        ),
    ],
)
def test_overlapping_pulse_and_voltage_offset_raises1(
    qblox_hardware_config_transmon,
    mock_setup_basic_transmon_with_standard_params,
    op1,
    op2,
    ref_pt2,
    rel_time2,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_transmon)
    compilation_config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")

    schedule = Schedule("test align qasm fields")
    schedule.add(op1)
    schedule.add(op2, ref_pt=ref_pt2, rel_time=rel_time2)

    with pytest.raises(
        RuntimeError,
        match=(
            "contain pulses with voltage offsets that overlap in time on the same "
            "port and clock"
        ),
    ):
        _ = compiler.compile(schedule=schedule, config=compilation_config)


def test_add_acquisition_to_control_module_raises(
    compile_config_basic_transmon_qblox_hardware,
):
    sched = Schedule("Overlapping operations", repetitions=1)

    sched.add(
        SSBIntegrationComplex(
            duration=800e-9,
            port="q0:mw",
            clock="q0.01",
            acq_channel=0,
            acq_index=0,
        ),
    )

    compiler = SerialCompiler(name="compiler")

    # Test explicitly that no warning is raised if there is no overlap
    with pytest.raises(RuntimeError, match="does not support acquisitions"):
        compiler.compile(sched, compile_config_basic_transmon_qblox_hardware)


class TestControlFlow:
    @staticmethod
    def _replace_multiple_spaces(s: str) -> str:
        s = re.sub(r" *\n *", "\n", s)
        return re.sub(r" {2,}", " ", s)

    def _compare_sequence(self, compiled, reference, module):
        if module == "qcm":
            mod = "cluster0_module2"
        elif module == "qrm":
            mod = "cluster0_module4"
        program = self._replace_multiple_spaces(
            compiled.compiled_instructions["cluster0"][mod]["sequencers"]["seq0"][
                "sequence"
            ]["program"]
        )
        reference = self._replace_multiple_spaces(reference)
        assert reference == program

    def test_subschedule(self, compile_config_basic_transmon_qblox_hardware):
        """
        outer:
          - X
          - inner:
            - X90
            - inner2:
              - Y
          - inner2:
            - Y
        """
        inner = Schedule("inner")
        inner.add(X90("q0"))
        inner2 = Schedule("inner2")
        inner2.add(Y("q0"))
        inner.add(inner2)

        outer = Schedule("outer")
        outer.add(X("q0"))
        outer.add(inner)
        outer.add(inner2)

        compiler = SerialCompiler(name="compiler")

        compiled_subschedule = compiler.compile(
            outer, config=compile_config_basic_transmon_qblox_hardware
        )

        outer = Schedule("outer")
        outer.add(X("q0"))
        outer.add(X90("q0"))
        outer.add(Y("q0"))
        outer.add(Y("q0"))

        compiled_ref = compiler.compile(
            outer, config=compile_config_basic_transmon_qblox_hardware
        )

        program_subschedule = compiled_subschedule.compiled_instructions["cluster0"][
            "cluster0_module2"
        ]["sequencers"]["seq0"]["sequence"]["program"]
        program_ref = compiled_ref.compiled_instructions["cluster0"][
            "cluster0_module2"
        ]["sequencers"]["seq0"]["sequence"]["program"]

        assert program_ref == program_subschedule

    def test_complex_loop(self, compile_config_basic_transmon_qblox_hardware):
        """
        - Sched
          - X90
          - Loop 3
            - Inner
              - X
              - Loop 2
                - Inner2
                  - Y
              - Measure
          - X90
          - Loop 4
            - Inner2
              -Y
          - X90
          - Loop 2
            -X
        """

        inner = Schedule("inner")
        inner.add(X("q0"))

        inner2 = Schedule("inner2")
        inner2.add(Y("q0"))

        inner.add(
            LoopOperation(body=inner2, repetitions=2),
        )
        inner.add(Measure("q0"))

        sched = Schedule("amp_ref")
        sched.add(X90("q0"))
        sched.add(LoopOperation(body=inner, repetitions=3))
        sched.add(X90("q0"))
        sched.add(LoopOperation(body=inner2, repetitions=4))
        sched.add(X90("q0"))
        sched.add(LoopOperation(X("q0"), repetitions=2))

        compiler = SerialCompiler(name="compiler")

        compiled = compiler.compile(
            sched, config=compile_config_basic_transmon_qblox_hardware
        )

        reference_sequence_qcm = """ set_mrk 1 # set markers to 1
wait_sync 4
upd_param 4
wait 4 # latency correction of 4 + 0 ns
move 1,R0 # iterator for loop with label start
start:
    reset_ph
    upd_param 4
    set_awg_gain 1882,110 # setting gain for X_90 q0
    play 0,1,4 # play X_90 q0 (20 ns)
    wait 16 # auto generated wait (16 ns)
    move 3,R1 # iterator for loop with label loop11
    loop11:
        set_awg_gain 3765,221 # setting gain for X q0
        play 0,1,4 # play X q0 (20 ns)
        wait 16 # auto generated wait (16 ns)
        move 2,R10 # iterator for loop with label loop16
        loop16:
            set_awg_gain -221,3765 # setting gain for Y q0
            play 1,0,4 # play Y q0 (20 ns)
            wait 16 # auto generated wait (16 ns)
        loop R10,@loop16
        wait 1100 # auto generated wait (1100 ns)
    loop R1,@loop11
    set_awg_gain 1882,110 # setting gain for X_90 q0
    play 0,1,4 # play X_90 q0 (20 ns)
    wait 16 # auto generated wait (16 ns)
    move 4,R1 # iterator for loop with label loop27
    loop27:
        set_awg_gain -221,3765 # setting gain for Y q0
        play 1,0,4 # play Y q0 (20 ns)
        wait 16 # auto generated wait (16 ns)
    loop R1,@loop27
    set_awg_gain 1882,110 # setting gain for X_90 q0
    play 0,1,4 # play X_90 q0 (20 ns)
    wait 16 # auto generated wait (16 ns)
    move 2,R1 # iterator for loop with label loop36
    loop36:
        set_awg_gain 3765,221 # setting gain for X q0
        play 0,1,4 # play X q0 (20 ns)
        wait 16 # auto generated wait (16 ns)
    loop R1,@loop36
loop R0,@start
stop
"""

        reference_sequence_qrm = """ set_mrk 3 # set markers to 3
wait_sync 4
upd_param 4
wait 4 # latency correction of 4 + 0 ns
move 1,R0 # iterator for loop with label start
start:
    reset_ph
    upd_param 4
    wait 20 # auto generated wait (20 ns)
    move 3,R1 # iterator for loop with label loop9
    loop9:
        wait 60 # auto generated wait (60 ns)
        reset_ph
        set_awg_gain 8192,0 # setting gain for SquarePulse
        play 0,0,4 # play SquarePulse (300 ns)
        wait 96 # auto generated wait (96 ns)
        acquire 0,0,4
        wait 996 # auto generated wait (996 ns)
    loop R1,@loop9
    wait 160 # auto generated wait (160 ns)
    loop R0,@start
stop
"""

        self._compare_sequence(compiled, reference_sequence_qcm, "qcm")
        self._compare_sequence(compiled, reference_sequence_qrm, "qrm")

    def test_loop_instruction_generated(
        self, compile_config_basic_transmon_qblox_hardware
    ):
        """
        - Sched
          - Square
          - Loop 3
            - Square2
          - Square
        """
        sched = Schedule("amp_ref")
        sched.add(
            SquarePulse(
                amp=0.5,
                port="q0:res",
                duration=2e-6,
                clock="q0.ro",
            )
        )
        sched.add(
            LoopOperation(
                body=SquarePulse(
                    amp=0.3,
                    port="q0:res",
                    duration=2e-6,
                    clock="q0.ro",
                ),
                repetitions=3,
            ),
        )
        sched.add(
            SquarePulse(
                amp=0.7,
                port="q0:res",
                duration=2e-6,
                clock="q0.ro",
            )
        )

        reference = """ set_mrk 3 # set markers to 3
wait_sync 4
upd_param 4
wait 4 # latency correction of 4 + 0 ns
move 1,R0 # iterator for loop with label start
start:
    reset_ph
    upd_param 4
    set_awg_offs 16384,0 # setting offset for SquarePulse
    upd_param 4
    wait 1992 # auto generated wait (1992 ns)
    set_awg_offs 0,0 # setting offset for SquarePulse
    set_awg_gain 16384,0 # setting gain for SquarePulse
    play 0,0,4 # play SquarePulse (4 ns)
    move 3,R1 # iterator for loop with label loop14
    loop14:
        set_awg_offs 9830,0 # setting offset for SquarePulse
        upd_param 4
        wait 1992 # auto generated wait (1992 ns)
        set_awg_offs 0,0 # setting offset for SquarePulse
        set_awg_gain 9830,0 # setting gain for SquarePulse
        play 0,0,4 # play SquarePulse (4 ns)
    loop R1,@loop14
    set_awg_offs 22938,0 # setting offset for SquarePulse
    upd_param 4
    wait 1992 # auto generated wait (1992 ns)
    set_awg_offs 0,0 # setting offset for SquarePulse
    set_awg_gain 22938,0 # setting gain for SquarePulse
    play 0,0,4 # play SquarePulse (4 ns)
loop R0,@start
stop
"""
        compiler = SerialCompiler(name="compiler")
        compiled = compiler.compile(
            sched, config=compile_config_basic_transmon_qblox_hardware
        )
        self._compare_sequence(compiled, reference, "qrm")


@pytest.mark.parametrize("scaling_factor", [0.999, 1])
@pytest.mark.parametrize("low_amp_path", ("I", "Q"))
@pytest.mark.parametrize("gain", [-11, -2, 0, 1, 2, 14])
def test_very_low_amp_paths(
    scaling_factor,
    low_amp_path,
    gain,
    dummy_cluster,
    compile_config_basic_transmon_qblox_hardware,
):
    # ------------------------ build schedule
    sched = Schedule("Low_amp_test", repetitions=1)
    dur = 201
    y = np.linspace(0, 1, dur)
    m = 2 / pow(2, 16)  # absolute tolerance (threshold)
    t = np.arange(dur) * 1e-9

    if low_amp_path == "Q":
        sched.add(
            NumericalPulse(
                samples=y + np.asarray([scaling_factor * gain * m] * dur) * 1j,
                t_samples=t,
                port="q0:mw",
                clock="q0.01",
            )
        )
    elif low_amp_path == "I":
        sched.add(
            NumericalPulse(
                samples=y * 1j + np.asarray([scaling_factor * gain * m] * dur),
                t_samples=t,
                port="q0:mw",
                clock="q0.01",
            )
        )
    # ------------------------ compile schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched, compile_config_basic_transmon_qblox_hardware
    )
    cluster = dummy_cluster()
    qcm_name = f"{cluster.name}_module2"
    compiled_instructions = compiled_sched["compiled_instructions"][cluster.name][
        qcm_name
    ]["sequencers"]["seq0"]["sequence"]

    seq_instructions = compiled_instructions["program"].splitlines()

    # ------------------------ test
    low_amp_path_gain = round(scaling_factor * gain)
    suppressed_waveform = math.floor(scaling_factor * abs(gain)) == 0
    assert len(compiled_instructions["waveforms"]) == (1 if suppressed_waveform else 2)

    if low_amp_path == "Q":
        assert re.search(
            rf"^\s*set_awg_gain\s+32604,{low_amp_path_gain}\s*", seq_instructions[8]
        )

    elif low_amp_path == "I":
        assert re.search(
            rf"^\s*set_awg_gain\s+{low_amp_path_gain},32604\s*", seq_instructions[8]
        )

    assert re.search(
        rf"^\s*play\s+0,{0 if suppressed_waveform else 1},4\s*", seq_instructions[9]
    )


def test_1_ns_time_grid(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("1 ns timegrid")
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=5e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(IdlePulse(duration=2e-9))
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=6e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    compiler = SerialCompiler(name="compiler")
    compiled = compiler.compile(
        sched, config=compile_config_basic_transmon_qblox_hardware
    )
    assert round(compiled.duration, 12) == 21e-9


@pytest.mark.parametrize("allow_off_grid_nco_ops", [True, False])
def test_1_ns_time_grid_repetition_loop_on_nco_grid(
    compile_config_basic_transmon_qblox_hardware, allow_off_grid_nco_ops
):
    # In the second repetition, the NCO operations would be misaligned with the 4 ns
    # grid.
    compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.allow_off_grid_nco_ops = (
        allow_off_grid_nco_ops
    )
    sched = Schedule("1 ns timegrid", repetitions=2)
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=5e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(IdlePulse(duration=2e-9))
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=6e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    compiler = SerialCompiler(name="compiler")
    if allow_off_grid_nco_ops:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    else:
        with pytest.raises(
            NcoOperationTimingError,
            match="The schedule is repeated with a duration of 21 ns per iteration, which "
            "does not align with the grid time of 4 ns for NCO operations. The duration "
            "must adhere to this grid time to ensure proper alignment of NCO operations "
            "for each iteration.",
        ):
            _ = compiler.compile(
                sched, config=compile_config_basic_transmon_qblox_hardware
            )


@pytest.mark.parametrize("allow_off_grid_nco_ops", [True, False])
def test_1_ns_time_grid_loops_on_nco_grid_start_time(
    compile_config_basic_transmon_qblox_hardware, allow_off_grid_nco_ops
):
    compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.allow_off_grid_nco_ops = (
        allow_off_grid_nco_ops
    )
    sched = Schedule("1 ns timegrid")
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=5e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    inner = Schedule("inner")
    inner.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=8e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    inner.add(ShiftClockPhase(phase_shift=20, clock="q0.01"))
    inner.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=12e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(LoopOperation(body=inner, repetitions=2))
    compiler = SerialCompiler(name="compiler")
    if allow_off_grid_nco_ops:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    else:
        with pytest.raises(
            NcoOperationTimingError,
            match="ControlFlow operation LoopOperation, starting at t=9 ns and ending at "
            "t=65 ns, contains NCO related operations that may not be aligned with the 4 "
            "ns time grid. Please make sure all iterations and/or branches start and end "
            "on the 4 ns time grid.",
        ):
            _ = compiler.compile(
                sched, config=compile_config_basic_transmon_qblox_hardware
            )


@pytest.mark.parametrize("allow_off_grid_nco_ops", [True, False])
def test_1_ns_time_grid_loops_on_nco_grid_duration(
    compile_config_basic_transmon_qblox_hardware, allow_off_grid_nco_ops
):
    compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.allow_off_grid_nco_ops = (
        allow_off_grid_nco_ops
    )
    sched = Schedule("1 ns timegrid")
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=8e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    inner = Schedule("inner")
    inner.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=8e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    inner.add(ShiftClockPhase(phase_shift=20, clock="q0.01"))
    inner.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=11e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(LoopOperation(body=inner, repetitions=2))
    compiler = SerialCompiler(name="compiler")
    if allow_off_grid_nco_ops:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    else:
        with pytest.raises(
            NcoOperationTimingError,
            match="ControlFlow operation LoopOperation, starting at t=12 ns and ending at "
            "t=66 ns, contains NCO related operations that may not be aligned with the 4 "
            "ns time grid. Please make sure all iterations and/or branches start and end "
            "on the 4 ns time grid.",
        ):
            _ = compiler.compile(
                sched, config=compile_config_basic_transmon_qblox_hardware
            )


def test_1_ns_time_grid_loops_no_nco(
    compile_config_basic_transmon_qblox_hardware,
):
    sched = Schedule("1 ns timegrid")
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=5e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    inner = Schedule("inner")
    inner.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=8e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    inner.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=11e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(LoopOperation(body=inner, repetitions=2))
    compiler = SerialCompiler(name="compiler")
    _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)


def test_1_ns_time_grid_half_ns(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("1 ns timegrid")
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=5.5e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(IdlePulse(duration=5e-9))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Attempting to use a time value of 9.499999999999998 ns. "
            "Please ensure that the durations of operations and wait times "
            "between operations are multiples of 1 ns"
        ),
    ):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched,
            config=compile_config_basic_transmon_qblox_hardware,
        )


def test_1_ns_time_grid_less_than_min_op(
    compile_config_basic_transmon_qblox_hardware,
):
    sched = Schedule("1 ns timegrid")
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=2e-9,
            clock="q0.01",
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=0.5,
            port="q0:mw",
            duration=2e-9,
            clock="q0.01",
            t0=0,
        )
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid timing. Attempting to wait for -2 ns before Pulse SquarePulse "
            "(t=2e-09 to 4e-09)"
        ),
    ):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched,
            config=compile_config_basic_transmon_qblox_hardware,
        )


@pytest.mark.parametrize("allow_off_grid_nco_ops", [True, False])
def test_1_ns_time_grid_nco_too_close_set_freq(
    compile_config_basic_transmon_qblox_hardware,
    allow_off_grid_nco_ops,
):
    compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.allow_off_grid_nco_ops = (
        allow_off_grid_nco_ops
    )
    sched = Schedule("1 ns timegrid")
    pulse = SquarePulse(
        amp=0.5,
        port="q0:mw",
        duration=4e-9,
        clock="q0.01",
    )
    sched.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.31e9))
    sched.add(pulse)
    sched.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.32e9))
    sched.add(pulse)
    compiler = SerialCompiler(name="compiler")
    if allow_off_grid_nco_ops:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    else:
        with pytest.raises(
            NcoOperationTimingError,
            match=re.escape(
                'Operation Pulse "SetClockFrequency" (t0=4e-09, duration=0) occurred 4 ns '
                "after the previous frequency update. The minimum time between frequency "
                "updates must be 8 ns."
            ),
        ):
            _ = compiler.compile(
                schedule=sched,
                config=compile_config_basic_transmon_qblox_hardware,
            )


@pytest.mark.parametrize("allow_off_grid_nco_ops", [True, False])
def test_1_ns_time_grid_nco_too_close_set_phase(
    compile_config_basic_transmon_qblox_hardware,
    allow_off_grid_nco_ops,
):
    compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.allow_off_grid_nco_ops = (
        allow_off_grid_nco_ops
    )
    sched = Schedule("1 ns timegrid")
    pulse = SquarePulse(
        amp=0.5,
        port="q0:mw",
        duration=4e-9,
        clock="q0.01",
    )
    sched.add(pulse)
    sched.add(ShiftClockPhase(phase_shift=20, clock="q0.01"))
    sched.add(ShiftClockPhase(phase_shift=20, clock="q0.01"))
    sched.add(pulse)
    compiler = SerialCompiler(name="compiler")
    if allow_off_grid_nco_ops:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    else:
        with pytest.raises(
            NcoOperationTimingError,
            match=re.escape(
                'Operation Pulse "ShiftClockPhase" (t0=4e-09, duration=0) occurred 0 ns after '
                "the previous phase update. The minimum time between phase updates must be 4 "
                "ns."
            ),
        ):
            _ = compiler.compile(
                schedule=sched,
                config=compile_config_basic_transmon_qblox_hardware,
            )


@pytest.mark.parametrize("allow_off_grid_nco_ops", [True, False])
def test_1_ns_time_grid_nco(
    compile_config_basic_transmon_qblox_hardware, allow_off_grid_nco_ops
):
    compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.allow_off_grid_nco_ops = (
        allow_off_grid_nco_ops
    )
    sched = Schedule("1 ns timegrid")
    pulse = SquarePulse(
        amp=0.5,
        port="q0:mw",
        duration=5e-9,
        clock="q0.01",
        t0=4e-9,
    )
    sched.add(pulse)
    sched.add(ResetClockPhase(clock="q0.01"))
    sched.add(pulse)
    compiler = SerialCompiler(name="compiler")
    if allow_off_grid_nco_ops:
        _ = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
    else:
        with pytest.raises(
            NcoOperationTimingError,
            match=re.escape(
                "NCO related operation ResetClockPhase(clock='q0.01',t0=0) cannot start at "
                "t=9 ns. This operation must be on the 4 ns time grid."
            ),
        ):
            _ = compiler.compile(
                schedule=sched,
                config=compile_config_basic_transmon_qblox_hardware,
            )


@pytest.mark.parametrize("allow_off_grid_nco_ops", [True, False])
def test_1_ns_time_grid_latency_corrections(
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_cluster,
    allow_off_grid_nco_ops,
):
    """
    NCO operations are checked for whether they are on the 4 ns grid, but this does
    not take into account latency corrections. Thus we restrict latency corrections to
    also be on the 4 ns grid.
    """
    hardware_cfg = copy.deepcopy(hardware_cfg_cluster)
    hardware_cfg["hardware_options"]["latency_corrections"] = {
        "q0:mw-q0.01": 3e-9,
        "q0:res-q0.ro": 0,
    }
    hardware_cfg["allow_off_grid_nco_ops"] = allow_off_grid_nco_ops
    mock_setup_basic_transmon_with_standard_params["quantum_device"].hardware_config(
        hardware_cfg
    )

    sched = Schedule("1 ns timegrid")
    pulse = SquarePulse(
        amp=0.5,
        port="q0:mw",
        duration=8e-9,
        clock="q0.01",
        t0=4e-9,
    )
    sched.add(pulse)
    sched.add(ResetClockPhase(clock="q0.01"))
    sched.add(Measure("q0"))

    compiler = SerialCompiler(name="compiler")
    if allow_off_grid_nco_ops:
        _ = compiler.compile(
            schedule=sched,
            config=mock_setup_basic_transmon_with_standard_params[
                "quantum_device"
            ].generate_compilation_config(),
        )
    else:
        with pytest.raises(
            NcoOperationTimingError,
            match="The latency correction value of 3 ns for "
            "q0:mw-q0.01 does not align with the grid time of "
            "4 ns for NCO operations. The latency "
            "corrections must adhere to this grid time to ensure proper alignment "
            "of all later operations in the schedule.",
        ):
            _ = compiler.compile(
                schedule=sched,
                config=mock_setup_basic_transmon_with_standard_params[
                    "quantum_device"
                ].generate_compilation_config(),
            )


@pytest.mark.filterwarnings(r"ignore:.*quantify-scheduler.*:FutureWarning")
def test_compile_hardware_distortion_corrections():
    quantum_device = QuantumDevice("qblox_distortions_device")
    compiler = SerialCompiler(name="compiler")

    hardware_compilation_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                    "2": {"instrument_type": "QCM_RF"},
                },
            },
            "lo0": {"instrument_type": "LocalOscillator", "power": 20},
            "iq_mixer0": {"instrument_type": "IQMixer"},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q4:mw-q4.01": {"interm_freq": 200e6},
                "q5:mw-q5.01": {"interm_freq": 50e6},
            },
            "mixer_corrections": {
                "q4:mw-q4.01": {"amp_ratio": 0.9999, "phase_error": -4.2}
            },
            "distortion_corrections": {
                "q0:fl-cl0.baseband": QbloxHardwareDistortionCorrection(
                    exp1_coeffs=[2000, -0.1],
                    fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                ),
                "q4:mw-q4.01": [
                    QbloxHardwareDistortionCorrection(
                        exp1_coeffs=[200, -0.1],
                        fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                    ),
                    QbloxHardwareDistortionCorrection(
                        exp1_coeffs=[20, -0.1],
                        fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                    ),
                ],
            },
        },
        "connectivity": {
            "graph": [
                ("cluster0.module1.complex_output_0", "iq_mixer0.if"),
                ("lo0.output", "iq_mixer0.lo"),
                ("iq_mixer0.rf", "q4:mw"),
                ("cluster0.module1.real_output_2", "q0:fl"),
                ("cluster0.module1.real_output_3", "q1:fl"),
                ("cluster0.module2.complex_output_0", "q5:mw"),
            ]
        },
    }

    sched = Schedule("Qblox hardware distortion corrections test", repetitions=1)
    sched.add_resource(ClockResource(name="q4.01", freq=5e6))
    sched.add(
        SquarePulse(
            amp=0.1,
            port="q0:fl",
            duration=200e-9,
            clock="cl0.baseband",
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=0.1,
            port="q4:mw",
            duration=200e-9,
            clock="q4.01",
            t0=0,
        )
    )

    quantum_device.hardware_config(hardware_compilation_cfg)

    sched = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )

    assert sched.compiled_instructions["cluster0"]["cluster0_module1"]["settings"][
        "distortion_corrections"
    ][2]["exp1"]["coeffs"] == [2000, -0.1]
    assert sched.compiled_instructions["cluster0"]["cluster0_module1"]["settings"][
        "distortion_corrections"
    ][0]["exp1"]["coeffs"] == [200, -0.1]
    assert sched.compiled_instructions["cluster0"]["cluster0_module1"]["settings"][
        "distortion_corrections"
    ][1]["exp1"]["coeffs"] == [20, -0.1]


@pytest.mark.parametrize(
    "portclock, distortion_correction, error_msg",
    [
        (
            "q0:mw-q0.01",
            QbloxHardwareDistortionCorrection(
                exp1_coeffs=[2000, -0.1],
                fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
            ),
            "two corrections are required",
        ),
        (
            "q1:mw-q1.01",
            [
                QbloxHardwareDistortionCorrection(
                    exp1_coeffs=[200, -0.1],
                    fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                ),
                QbloxHardwareDistortionCorrection(
                    exp1_coeffs=[20, -0.1],
                    fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                ),
            ],
            "one correction is required",
        ),
    ],
)
def test_validate_hardware_distortion_corrections_mode(
    portclock, distortion_correction, error_msg
):
    quantum_device = QuantumDevice("qblox_distortions_device")
    compiler = SerialCompiler(name="compiler")

    hardware_compilation_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QCM"},
                },
            },
        },
        "hardware_options": {
            "distortion_corrections": {
                portclock: distortion_correction,
            },
        },
        "connectivity": {
            "graph": [
                ("cluster0.module1.complex_output_0", "q0:mw"),
                ("cluster0.module1.real_output_2", "q1:mw"),
            ]
        },
    }

    port, clock = portclock.split("-")

    sched = Schedule("Qblox hardware distortion corrections test", repetitions=1)
    sched.add_resource(ClockResource(name=clock, freq=5e6))
    sched.add(
        SquarePulse(
            amp=0.1,
            port=port,
            duration=200e-9,
            clock=clock,
            t0=0,
        )
    )

    quantum_device.hardware_config(hardware_compilation_cfg)

    with pytest.raises(ValueError) as error:
        sched = compiler.compile(
            schedule=sched,
            config=quantum_device.generate_compilation_config(),
        )

    assert error_msg in error.value.args[0]


@pytest.mark.parametrize(
    "input_gain, error_msg",
    [
        (
            {"q0:mw-q0.01": 2},
            "A real input gain",
        ),
        (
            {"q1:mw-q1.01": {"gain_I": 2, "gain_Q": 3}},
            "A complex input gain",
        ),
        (
            {"q1:mw-q1.01": 2, "q3:mw-q3.01": {"gain_I": 3, "gain_Q": 2}},
            "Found non-unique input gains",
        ),
    ],
)
def test_validate_input_gain(input_gain, error_msg):
    quantum_device = QuantumDevice("qblox_distortions_device")
    compiler = SerialCompiler(name="compiler")

    hardware_compilation_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {"instrument_type": "QRM"},
                    "2": {"instrument_type": "QRM"},
                },
            },
        },
        "hardware_options": {
            "input_gain": input_gain,
        },
        "connectivity": {
            "graph": [
                ("cluster0.module1.complex_input_0", "q0:mw"),
                ("cluster0.module2.real_input_0", "q1:mw"),
                ("cluster0.module2.real_input_1", "q2:mw"),
                ("cluster0.module2.complex_output_0", "q3:mw"),
            ]
        },
    }

    sched = Schedule("Qblox input gain test", repetitions=1)

    for portclock in input_gain:
        port, clock = portclock.split("-")
        sched.add_resource(ClockResource(name=clock, freq=5e6))
        sched.add(
            SquarePulse(
                amp=0.1,
                port=port,
                duration=200e-9,
                clock=clock,
                t0=0,
            )
        )

    quantum_device.hardware_config(hardware_compilation_cfg)

    with pytest.raises(ValueError) as error:
        sched = compiler.compile(
            schedule=sched,
            config=quantum_device.generate_compilation_config(),
        )

    assert error_msg in error.value.args[0]


def test_distortion_correction_latency_compensation():
    hardware_compilation_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                        "complex_output_0": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.EXP1
                            | DistortionCorrectionLatencyEnum.EXP3
                        },
                        "real_output_2": {
                            "marker_debug_mode_enable": True,
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.FIR,
                        },
                        "real_output_3": {
                            "marker_debug_mode_enable": False,
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.EXP2,
                        },
                        "digital_output_0": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.FIR
                        },
                        "digital_output_1": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                        },
                        "digital_output_2": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                        },
                        "digital_output_3": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.EXP1
                        },
                    },
                    "2": {
                        "instrument_type": "QCM_RF",
                        "complex_output_0": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.EXP1
                            | DistortionCorrectionLatencyEnum.EXP3
                        },
                        "complex_output_1": {
                            "marker_debug_mode_enable": True,
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.FIR,
                        },
                        "digital_output_0": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                            | DistortionCorrectionLatencyEnum.EXP1
                        },
                        "digital_output_1": {
                            "distortion_correction_latency_compensation": DistortionCorrectionLatencyEnum.EXP0
                        },
                    },
                },
            },
            "lo0": {"instrument_type": "LocalOscillator", "power": 20},
            "iq_mixer0": {"instrument_type": "IQMixer"},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q4:mw-q4.01": {"interm_freq": 200e6},
                "q6:mw-q6.01": {"interm_freq": 200e6},
                "q5:mw-q5.01": {"interm_freq": 50e6},
            },
            "mixer_corrections": {
                "q4:mw-q4.01": {"amp_ratio": 0.9999, "phase_error": -4.2}
            },
            "distortion_corrections": {
                "q0:fl-cl0.baseband": QbloxHardwareDistortionCorrection(
                    exp1_coeffs=[2000, -0.1],
                    fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                ),
                "q4:mw-q4.01": [
                    QbloxHardwareDistortionCorrection(
                        exp1_coeffs=[200, -0.1],
                        fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                    ),
                    QbloxHardwareDistortionCorrection(
                        exp1_coeffs=[20, -0.1],
                        fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
                    ),
                ],
            },
        },
        "connectivity": {
            "graph": [
                ("cluster0.module1.complex_output_0", "iq_mixer0.if"),
                ("lo0.output", "iq_mixer0.lo"),
                ("iq_mixer0.rf", "q4:mw"),
                ("cluster0.module1.real_output_2", "q0:fl"),
                ("cluster0.module1.real_output_3", "q1:fl"),
                ("cluster0.module2.complex_output_0", "q5:mw"),
                ("cluster0.module2.complex_output_1", "q6:mw"),
                ("cluster0.module1.digital_output_0", "q0:marker"),
                ("cluster0.module1.digital_output_1", "q1:marker"),
                ("cluster0.module1.digital_output_2", "q2:marker"),
                ("cluster0.module1.digital_output_3", "q3:marker"),
                ("cluster0.module2.digital_output_0", "qq0:marker"),
                ("cluster0.module2.digital_output_1", "qq1:marker"),
            ]
        },
    }

    sched = Schedule("Qblox hardware distortion corrections test", repetitions=1)
    sched.add_resource(ClockResource(name="q4.01", freq=5e6))
    sched.add_resource(ClockResource(name="q5.01", freq=5e6))
    sched.add_resource(ClockResource(name="q6.01", freq=5e9))
    sched.add(
        SquarePulse(
            amp=0.1,
            port="q0:fl",
            duration=200e-9,
            clock="cl0.baseband",
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=0.1,
            port="q1:fl",
            duration=200e-9,
            clock="cl0.baseband",
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=0.1,
            port="q4:mw",
            duration=200e-9,
            clock="q4.01",
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=0.1,
            port="q5:mw",
            duration=200e-9,
            clock="q5.01",
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=0.1,
            port="q6:mw",
            duration=200e-9,
            clock="q6.01",
            t0=0,
        )
    )
    for port in [
        "q0:marker",
        "q1:marker",
        # "q2:marker",
        "q3:marker",
        "qq0:marker",
        # "qq1:marker",
    ]:
        sched.add(
            MarkerPulse(
                port=port,
                duration=200e-9,
                t0=0,
            )
        )
    sched.add(IdlePulse(duration=4e-9))

    quantum_device = QuantumDevice("qblox_distortions_device")
    compiler = SerialCompiler(name="compiler")

    quantum_device.hardware_config(hardware_compilation_cfg)

    sched = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )

    corrections = sched.compiled_instructions["cluster0"]["cluster0_module1"][
        "settings"
    ]["distortion_corrections"]

    ideal_corrections = [
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp1": {
                "coeffs": [200.0, -0.1],
                "config": QbloxFilterConfig.ENABLED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": [
                    1.025,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.0,
                ],
                "config": QbloxFilterConfig.ENABLED,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
        },
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp1": {
                "coeffs": [20.0, -0.1],
                "config": QbloxFilterConfig.ENABLED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": [
                    1.025,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.0,
                ],
                "config": QbloxFilterConfig.ENABLED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
        },
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp1": {
                "coeffs": [2000.0, -0.1],
                "config": QbloxFilterConfig.ENABLED,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": [
                    1.025,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.03,
                    0.02,
                    0.0,
                ],
                "config": QbloxFilterConfig.ENABLED,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
        },
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp1": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
        },
    ]

    assert corrections == ideal_corrections

    corrections = sched.compiled_instructions["cluster0"]["cluster0_module2"][
        "settings"
    ]["distortion_corrections"]

    ideal_corrections = [
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp1": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
        },
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp1": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
        },
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp1": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
        },
        {
            "bt": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp0": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
            "exp1": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp2": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "exp3": {
                "coeffs": None,
                "config": QbloxFilterConfig.BYPASSED,
                "marker_delay": QbloxFilterMarkerDelay.BYPASSED,
            },
            "fir": {
                "coeffs": None,
                "config": QbloxFilterConfig.DELAY_COMP,
                "marker_delay": QbloxFilterMarkerDelay.DELAY_COMP,
            },
        },
    ]

    assert corrections == ideal_corrections


def test_invalid_parameter_ordering(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_config_cluster = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {1: {"instrument_type": "QCM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {"graph": [("cluster0.module1.complex_output_0", "q1:mw")]},
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    schedule = Schedule("test valid channel names")
    schedule.add(SquarePulse(port="q1:mw", clock="q1.01", amp=0.25, duration=60e-9))
    schedule.add(SquarePulse(port="q1:mw", clock="q1.01", amp=0.25, duration=12e-9))
    schedule.add(SetClockFrequency(clock="q1.01", clock_freq_new=5e8), rel_time=-12e-9)
    schedule.add(
        SquarePulse(port="q1:mw", clock="q1.01", amp=0.25, duration=12e-9),
        rel_time=20e-9,
    )

    quantum_device.hardware_config(hardware_config_cluster)
    compiler = SerialCompiler(name="compiler")
    with pytest.raises(
        ValueError,
        match="Invalid timing. Pulse SetClockFrequency \\(t=6.*e-08 to 6.*e-08\\).*\n.* cannot be started at this order or time. "
        "Please try to reorder your operations by adding this operation "
        r"before any other operation \(possibly at the same time\) that happens at that time.",
    ):
        compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )


@pytest.mark.parametrize(
    ["pulse_rel_time", "expected_raises"],
    [
        (104e-9, True),
        (100e-9, False),
        (108e-9, False),
    ],
)
def test_conditional_reset_with_overlapping_pulses_error(
    mock_setup_basic_transmon_with_standard_params,
    qblox_hardware_config_transmon,
    pulse_rel_time,
    expected_raises,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(qblox_hardware_config_transmon)
    config = quantum_device.generate_compilation_config()

    schedule = Schedule("test")
    measure_schedulable = schedule.add(
        Measure(
            "q0",
            acq_protocol="ThresholdedAcquisition",
            feedback_trigger_label="q0",
        )
    )
    schedule.add(
        ConditionalOperation(body=X("q0"), qubit_name="q0"),
        rel_time=364e-9,
    )

    schedule.add(
        SquarePulse(amp=0.1, duration=16e-9, port="q0:mw", clock="q0.01"),
        ref_pt="start",
        rel_time=pulse_rel_time,
        ref_op=measure_schedulable,
    )

    compiler = SerialCompiler(name="compiler")
    if expected_raises:
        with pytest.raises(
            ValueError,
        ):
            compiler.compile(
                schedule,
                config=config,
            )
    else:
        compiler.compile(
            schedule,
            config=config,
        )
