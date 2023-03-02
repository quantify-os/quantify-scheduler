# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox backend."""
import copy
import itertools
import json
import logging
import os
import re
from contextlib import nullcontext
from typing import Dict, Generator, Optional

import numpy as np
import pytest
import quantify_scheduler
from pydantic import ValidationError
from qblox_instruments import Pulsar, PulsarType
from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler, corrections
from quantify_scheduler.backends.qblox import (
    compiler_container,
    constants,
    q1asm_instructions,
    register_manager,
)
from quantify_scheduler.backends.qblox.compiler_abc import Sequencer
from quantify_scheduler.backends.qblox.helpers import (
    assign_pulse_and_acq_info_to_devices,
    convert_hw_config_to_portclock_configs_spec,
    find_all_port_clock_combinations,
    find_inner_dicts_containing_key,
    generate_port_clock_to_device_map,
    generate_uuid_from_wf_data,
    generate_waveform_data,
    to_grid_time,
)
from quantify_scheduler.backends.qblox.instrument_compilers import (
    QcmModule,
    QcmRfModule,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox_backend import hardware_compile
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.types.qblox import (
    BasebandModuleSettings,
    MarkerConfiguration,
)
from quantify_scheduler.compilation import (
    determine_absolute_timing,
    device_compile,
    qcompile,
)
from quantify_scheduler.operations.acquisition_library import Trace
from quantify_scheduler.operations.gate_library import Measure, Reset, X
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import (
    DRAGPulse,
    IdlePulse,
    RampPulse,
    ShiftClockPhase,
    SetClockFrequency,
    SquarePulse,
)
from quantify_scheduler.resources import BasebandClockResource, ClockResource
from quantify_scheduler.schedules.timedomain_schedules import (
    allxy_sched,
    readout_calibration_sched,
)
from tests.fixtures.mock_setup import close_instruments

REGENERATE_REF_FILES: bool = False  # Set flag to true to regenerate the reference files


# --------- Test fixtures ---------
@pytest.fixture
def hardware_cfg_baseband(add_lo1):
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "mix_lo": True,
                "lo_name": "lo0",
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": True,
                        "interm_freq": 50e6,
                    }
                ],
            },
            "complex_output_1": {
                "lo_name": "lo1" if add_lo1 else None,
                "portclock_configs": [{"port": "q1:mw", "clock": "q1.01"}],
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": 4.8e9, "power": 1},
    }


@pytest.fixture
def hardware_cfg_real_mode(
    instruction_generated_pulses_enabled,
):  # pylint: disable=line-too-long
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "real_output_0": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_1",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,
                    },
                ],
            },
            "real_output_1": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_2",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,
                    }
                ],
            },
            "real_output_2": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_3",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,
                    }
                ],
            },
            "real_output_3": {
                "portclock_configs": [
                    {
                        "port": "dummy_port_4",
                        "clock": "cl0.baseband",
                        "instruction_generated_pulses_enabled": instruction_generated_pulses_enabled,
                    }
                ],
            },
        },
    }


@pytest.fixture
def hardware_cfg_multiplexing():
    yield {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "name": "qcm0",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "lo_name": "lo0",
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q1:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q2:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q3:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                    {
                        "port": "q4:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6,
                    },
                ],
            },
            "complex_output_1": {
                "portclock_configs": [{"port": "q1:mw", "clock": "q1.01"}],
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 1},
    }


@pytest.fixture
def hardware_cfg_latency_corrections():
    return {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "latency_corrections": {"q0:mw-q0.01": 2e-8, "q1:mw-q1.01": -5e-9},
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "portclock_configs": [{"port": "q0:mw", "clock": "q0.01"}],
            },
        },
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:mw",
                            "clock": "q1.01",
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_latency_corrections_invalid():
    return {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        # None is not a valid key for the latency corrections
        "latency_corrections": {"q0:mw-q0.01": 2e-8, "q1:mw-q1.01": None},
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "portclock_configs": [{"port": "q0:mw", "clock": "q0.01"}],
            },
        },
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:mw",
                            "clock": "q1.01",
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_qcm_rf():
    return {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:mw",
                            "clock": "q1.01",
                            "interm_freq": 50e6,
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def hardware_cfg_two_qubit_gate():
    return {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "portclock_configs": [
                    {"port": f"{qubit}:fl", "clock": clock}
                    for qubit in ["q2", "q3"]
                    for clock in [BasebandClockResource.IDENTITY, f"{qubit}.01"]
                ]
            },
        },
    }


@pytest.fixture
def dummy_pulsars() -> Generator[Dict[str, Pulsar], None, None]:
    qcm_names = ["qcm0", "qcm1"]
    qrm_names = ["qrm0", "qrm1"]

    close_instruments(qcm_names + qrm_names)

    _pulsars = {}
    for qcm_name in qcm_names:
        _pulsars[qcm_name] = Pulsar(name=qcm_name, dummy_type=PulsarType.PULSAR_QCM)
    for qrm_name in qrm_names:
        _pulsars[qrm_name] = Pulsar(name=qrm_name, dummy_type=PulsarType.PULSAR_QRM)

    yield _pulsars

    close_instruments(qcm_names + qrm_names)


@pytest.fixture
def pulse_only_schedule():
    sched = Schedule("pulse_only_experiment")
    sched.add(Reset("q0"))
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
    sched.add(Reset("q0"))
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
    sched.add(Reset("q1"))
    sched.add(
        SquarePulse(
            amp=0.5,
            phase=0,
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
    sched.add(Reset("q0"))
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
            amp=2.0,
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
    sched.add(Reset("q0"))
    sched.add(
        SquarePulse(
            amp=2.0,
            duration=2.5e-6,
            port="dummy_port_1",
            clock=BasebandClockResource.IDENTITY,
            t0=1e-6,
        )
    )
    sched.add(
        SquarePulse(
            amp=1.0,
            duration=2.0e-6,
            port="dummy_port_2",
            clock=BasebandClockResource.IDENTITY,
            t0=0.5e-6,
        )
    )
    sched.add(
        SquarePulse(
            amp=1.2,
            duration=3.5e-6,
            port="dummy_port_3",
            clock=BasebandClockResource.IDENTITY,
            t0=0,
        )
    )
    sched.add(
        SquarePulse(
            amp=1.2,
            duration=3.5e-6,
            port="dummy_port_4",
            clock=BasebandClockResource.IDENTITY,
            t0=0,
        )
    )
    return sched


@pytest.fixture(name="empty_qasm_program_qcm")
def fixture_empty_qasm_program():
    return QASMProgram(
        QcmModule.static_hw_properties, register_manager.RegisterManager()
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


def test_find_all_port_clock_combinations(load_example_qblox_hardware_config):
    portclocks = find_all_port_clock_combinations(load_example_qblox_hardware_config)
    portclocks = set(portclocks)
    portclocks.discard((None, None))
    answer = {
        ("q1:mw", "q1.01"),
        ("q0:mw", "q0.01"),
        ("q0:res", "q0.ro"),
        ("q0:res", "q0.multiplex"),
        ("q1:res", "q1.ro"),
        ("q3:mw", "q3.01"),
        ("q2:mw", "q2.01"),
        ("q2:res", "q2.ro"),
        ("q3:res", "q3.ro"),
        ("q3:mw", "q3.01"),
        ("q4:mw", "q4.01"),
        ("q5:mw", "q5.01"),
        ("q6:mw", "q6.01"),
        ("q4:res", "q4.ro"),
        ("q5:res", "q5.ro"),
        ("q0:fl", "cl0.baseband"),
        ("q1:fl", "cl0.baseband"),
        ("q2:fl", "cl0.baseband"),
        ("q3:fl", "cl0.baseband"),
        ("q4:fl", "cl0.baseband"),
    }
    assert portclocks == answer


def test_generate_port_clock_to_device_map(load_example_qblox_hardware_config):
    portclock_map = generate_port_clock_to_device_map(
        load_example_qblox_hardware_config
    )
    assert (None, None) not in portclock_map.keys()
    assert len(portclock_map.keys()) == 19


# --------- Test classes and member methods ---------


def test_construct_sequencers(
    make_basic_multi_qubit_schedule,
    compile_config_basic_transmon_qblox_hardware,
    load_example_qblox_hardware_config,
):
    test_module = QcmModule(
        parent=None,
        name="tester",
        total_play_time=1,
        hw_mapping=load_example_qblox_hardware_config["qcm0"],
    )
    sched = make_basic_multi_qubit_schedule(["q0", "q1"])

    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=sched, config=compile_config_basic_transmon_qblox_hardware
    )
    assign_pulse_and_acq_info_to_devices(
        schedule=sched,
        hardware_cfg=load_example_qblox_hardware_config,
        device_compilers={"qcm0": test_module},
    )

    test_module._construct_sequencers()
    seq_keys = list(test_module.sequencers.keys())

    assert len(seq_keys) == 2
    assert isinstance(test_module.sequencers[seq_keys[0]], Sequencer)


def test_construct_sequencers_repeated_portclocks_error(
    make_basic_multi_qubit_schedule,
    load_example_qblox_hardware_config,
    compile_config_basic_transmon_qblox_hardware,
):
    hardware_cfg = copy.deepcopy(load_example_qblox_hardware_config)

    hardware_cfg["qcm0"]["complex_output_0"]["portclock_configs"] = [
        {
            "port": "q0:mw",
            "clock": "q0.01",
            "interm_freq": 50e6,
        },
        {
            "port": "q0:mw",
            "clock": "q0.01",
            "interm_freq": 100e6,
        },
    ]

    test_module = QcmModule(
        parent=None,
        name="tester",
        total_play_time=1,
        hw_mapping=hardware_cfg["qcm0"],
    )
    sched = make_basic_multi_qubit_schedule(["q0", "q1"])  # Schedule with two qubits

    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=sched, config=compile_config_basic_transmon_qblox_hardware
    )
    assign_pulse_and_acq_info_to_devices(
        schedule=sched,
        hardware_cfg=hardware_cfg,
        device_compilers={"qcm0": test_module},
    )

    with pytest.raises(ValueError):
        test_module.sequencers = test_module._construct_sequencers()


@pytest.mark.parametrize(
    "element_names, io",
    [
        (
            [f"q{i}" for i in range(7)],
            "complex_output_0",
        ),
        (["q0"], "real_output_0"),
    ],
)
def test_construct_sequencers_exceeds_seq__invalid_io(
    mock_setup_basic_transmon_elements,
    make_basic_multi_qubit_schedule,
    element_names,
    io,
):
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "cluster0_module1": {
                "instrument_type": "QCM_RF",
                f"{io}": {
                    "portclock_configs": [
                        {
                            "port": f"{qubit}:mw",
                            "clock": f"{qubit}.01",
                            "interm_freq": 50e6,
                        }
                        for qubit in element_names
                    ]
                },
            },
        },
    }

    sched = make_basic_multi_qubit_schedule(element_names)
    sched.add_resources([ClockResource(f"{qubit}.01", 5e9) for qubit in element_names])

    quantum_device = mock_setup_basic_transmon_elements["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(ValueError) as error:
        sched = compiler.compile(
            schedule=sched,
            config=quantum_device.generate_compilation_config(),
        )

    name = "cluster0_module1"
    module_type = QcmRfModule
    valid_ios = [f"complex_output_{i}" for i in [0, 1]]

    assert (
        str(error.value.args[0])
        == f"Number of simultaneously active port-clock combinations exceeds number of "
        f"sequencers. Maximum allowed for {name} ({module_type.__name__}) is {6}!"
        or str(error.value.args[0])
        == f"Invalid hardware config: '{io}' of {name} ({module_type.__name__}) is not a "
        f"valid name of an input/output."
        f"\n\nSupported names for {module_type.__name__}:\n{valid_ios}"
    )


def test_portclocks(
    make_basic_multi_qubit_schedule,
    load_example_qblox_hardware_config,
    compile_config_basic_transmon_qblox_hardware,
):
    sched = make_basic_multi_qubit_schedule(["q3", "q4"])

    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=sched, config=compile_config_basic_transmon_qblox_hardware
    )

    hardware_cfg = load_example_qblox_hardware_config
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        sched, hardware_cfg
    )

    assign_pulse_and_acq_info_to_devices(
        schedule=sched,
        hardware_cfg=hardware_cfg,
        device_compilers=container.instrument_compilers,
    )

    compilers = container.instrument_compilers["cluster0"].instrument_compilers
    assert compilers["cluster0_module1"].portclocks == [("q4:mw", "q4.01")]
    assert compilers["cluster0_module2"].portclocks == [
        ("q5:mw", "q5.01"),
        ("q6:mw", "q6.01"),
    ]


def test_compile_simple(
    pulse_only_schedule, compile_config_basic_transmon_qblox_hardware
):
    """Tests if compilation with only pulses finishes without exceptions"""

    compiler = SerialCompiler(name="compiler")
    compiler.compile(
        pulse_only_schedule, config=compile_config_basic_transmon_qblox_hardware
    )


@pytest.mark.parametrize("delete_lo0", [False, True])
def test_compile_cluster(
    cluster_only_schedule,
    mock_setup_basic_transmon_with_standard_params,
    load_example_qblox_hardware_config,
    delete_lo0: bool,
):
    sched = cluster_only_schedule
    sched.add_resource(ClockResource("q5.01", freq=5e9))

    mock_setup_basic_transmon_with_standard_params["quantum_device"].hardware_config(
        load_example_qblox_hardware_config
    )
    compiler = SerialCompiler(name="compiler")
    context_mngr = nullcontext()
    if delete_lo0:
        del load_example_qblox_hardware_config["lo0"]
        context_mngr = pytest.raises(RuntimeError)
    with context_mngr as error:
        compiler.compile(
            schedule=sched,
            config=mock_setup_basic_transmon_with_standard_params[
                "quantum_device"
            ].generate_compilation_config(),
        )

    if delete_lo0:
        assert (
            error.value.args[0]
            == "External local oscillator 'lo0' set to be used by 'seq0' of "
            "'cluster0_module1' not found! Make sure it is present in the hardware "
            "configuration."
        )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_qcompile_no_device_cfg(load_example_qblox_hardware_config):
    sched = Schedule("One pulse schedule")
    sched.add_resource(ClockResource("q0.01", 3.1e9))
    sched.add(SquarePulse(amp=1 / 4, duration=12e-9, port="q0:mw", clock="q0.01"))

    compiled_schedule = qcompile(sched, hardware_cfg=load_example_qblox_hardware_config)

    wf_and_prog = compiled_schedule.compiled_instructions["qcm0"]["sequencers"]["seq0"][
        "sequence"
    ]
    assert "play" in wf_and_prog["program"]


def test_compile_simple_multiplexing(
    pulse_only_schedule_multiplexed,
    hardware_cfg_multiplexing,
    mock_setup_basic_transmon_with_standard_params,
):
    """Tests if compilation with only pulses finishes without exceptions"""
    sched = pulse_only_schedule_multiplexed

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_multiplexing)
    compiler = SerialCompiler(name="compiler")
    compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )


def test_compile_identical_pulses(
    identical_pulses_schedule, compile_config_basic_transmon_qblox_hardware
):
    """Tests if compilation with only pulses finishes without exceptions"""

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        identical_pulses_schedule, config=compile_config_basic_transmon_qblox_hardware
    )

    prog = compiled_schedule.compiled_instructions["qcm0"]["sequencers"]["seq0"][
        "sequence"
    ]
    assert len(prog["waveforms"]) == 2


def test_compile_measure(
    duplicate_measure_schedule, compile_config_basic_transmon_qblox_hardware
):
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        duplicate_measure_schedule, config=compile_config_basic_transmon_qblox_hardware
    )
    qrm0_seq0_json = full_program["compiled_instructions"]["qrm0"]["sequencers"][
        "seq0"
    ]["sequence"]

    assert len(qrm0_seq0_json["weights"]) == 0


@pytest.mark.parametrize(
    "operation, instruction_to_check, clock_freq_old, add_lo1",
    [
        [
            (IdlePulse(duration=64e-9), f"{'wait':<9}  64", None, add_lo1),
            (Reset("q1"), f"{'wait':<9}  65532", None, add_lo1),
            (
                ShiftClockPhase(clock=clock, phase_shift=180.0),
                f"{'set_ph_delta'}  500000000",
                None,
                add_lo1,
            ),
            (
                SetClockFrequency(clock=clock, clock_freq_new=clock_freq_new),
                f"{'set_freq':<9}  {round((2e8 + clock_freq_new - clock_freq_old)*4)}",
                clock_freq_old,
                add_lo1,
            ),
        ]
        for clock in ["q1.01"]
        for clock_freq_old in [5e9]
        for clock_freq_new in [5.001e9]
        for add_lo1 in [True]
    ][0],
)
def test_compile_clock_operations(
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_baseband,
    operation: Operation,
    instruction_to_check: str,
    clock_freq_old: Optional[float],
    add_lo1: bool,  # pylint: disable=unused-argument
):
    sched = Schedule("compile_clock_operations")
    sched.add(operation)

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_baseband)
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

    program_lines = compiled_sched.compiled_instructions["qcm0"]["sequencers"]["seq0"][
        "sequence"
    ]["program"].splitlines()
    assert any(instruction_to_check in line for line in program_lines), "\n".join(
        line for line in program_lines
    )


def test_compile_cz_gate(
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_two_qubit_gate,
    two_qubit_gate_schedule,
):
    mock_setup = mock_setup_basic_transmon_with_standard_params
    edge_q2_q3 = mock_setup["q2_q3"]
    edge_q2_q3.cz.q2_phase_correction(44)
    edge_q2_q3.cz.q3_phase_correction(63)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_two_qubit_gate)
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=two_qubit_gate_schedule,
        config=quantum_device.generate_compilation_config(),
    )

    program_lines = {}
    for seq in ["seq0", "seq1", "seq2"]:
        program_lines[seq] = compiled_sched.compiled_instructions["qcm0"]["sequencers"][
            seq
        ]["sequence"]["program"].splitlines()

    assert any(
        "play          0,1,4" in line for line in program_lines["seq0"]
    ), "\n".join(line for line in program_lines["seq0"])

    assert any(
        "set_ph_delta  122222222" in line for line in program_lines["seq1"]
    ), "\n".join(line for line in program_lines["seq1"])

    assert any(
        "set_ph_delta  175000000" in line for line in program_lines["seq2"]
    ), "\n".join(line for line in program_lines["seq2"])


def test_compile_simple_with_acq(
    dummy_pulsars,
    mixed_schedule_with_acquisition,
    compile_config_basic_transmon_qblox_hardware,
):
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        mixed_schedule_with_acquisition,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    qcm0_seq0_json = full_program["compiled_instructions"]["qcm0"]["sequencers"][
        "seq0"
    ]["sequence"]

    qcm0 = dummy_pulsars["qcm0"]
    qcm0.sequencer0.sequence(qcm0_seq0_json)
    qcm0.arm_sequencer(0)

    uploaded_waveforms = qcm0.get_waveforms(0)
    assert uploaded_waveforms is not None


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_qcompile_simple_with_acq(
    dummy_pulsars,
    mixed_schedule_with_acquisition,
    load_example_transmon_config,
    load_example_qblox_hardware_config,
):
    full_program = qcompile(
        schedule=mixed_schedule_with_acquisition,
        device_cfg=load_example_transmon_config,
        hardware_cfg=load_example_qblox_hardware_config,
    )
    qcm0_seq0_json = full_program["compiled_instructions"]["qcm0"]["sequencers"][
        "seq0"
    ]["sequence"]

    qcm0 = dummy_pulsars["qcm0"]
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
    load_example_qblox_hardware_config,
    reset_clock_phase,
):
    schedule = Schedule("Test schedule")

    q0, q1 = "q0", "q1"
    times = np.arange(0, 60e-6, 3e-6)
    for i, tau in enumerate(times):
        schedule.add(Reset(q0, q1), label=f"Reset {i}")
        schedule.add(X(q0), label=f"pi {i} {q0}")
        schedule.add(X(q1), label=f"pi {i} {q1}", ref_pt="start")

        schedule.add(
            Measure(q0, acq_index=i, acq_channel=0),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q0}{i}",
        )

    mock_setup = mock_setup_basic_transmon_with_standard_params
    mock_setup["q0"].measure.reset_clock_phase(reset_clock_phase)
    mock_setup["quantum_device"].hardware_config(load_example_qblox_hardware_config)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule, config=mock_setup["quantum_device"].generate_compilation_config()
    )
    qrm0_seq0_json = compiled_schedule.compiled_instructions["qrm0"]["sequencers"][
        "seq0"
    ]["seq_fn"]
    with open(qrm0_seq0_json) as file:
        program = json.load(file)["program"]
    reset_counts = program.count(" reset_ph ")
    expected_counts = (1 + len(times)) if reset_clock_phase else 1
    assert reset_counts == expected_counts, (
        f"Expected qasm program to contain `reset_ph`-instruction {expected_counts} "
        f"times, but found {reset_counts} times instead."
    )


def test_acquisitions_back_to_back(
    mixed_schedule_with_acquisition,
    compile_config_basic_transmon_qblox_hardware,
):
    sched = copy.deepcopy(mixed_schedule_with_acquisition)
    meas_op = sched.add(Measure("q0"))
    # Add another one too quickly
    sched.add(Measure("q0"), ref_op=meas_op, ref_pt="start", rel_time=500e-9)

    with pytest.raises(ValueError) as error:
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=compile_config_basic_transmon_qblox_hardware,
        )

    assert (
        "Please ensure a minimum interval of 1000 ns between acquisitions"
        in error.value.args[0]
    )


@pytest.mark.filterwarnings(
    "ignore::FutureWarning"
)  # Tests both device_compile and hardware_compile, keep for coverage
def test_deprecated_acquisitions_back_to_back(
    mixed_schedule_with_acquisition,
    load_example_transmon_config,
    load_example_qblox_hardware_config,
):
    sched = copy.deepcopy(mixed_schedule_with_acquisition)
    meas_op = sched.add(Measure("q0"))
    # Add another one too quickly
    sched.add(Measure("q0"), ref_op=meas_op, ref_pt="start", rel_time=500e-9)

    sched_with_pulse_info = device_compile(sched, load_example_transmon_config)
    with pytest.raises(ValueError) as error:
        hardware_compile(sched_with_pulse_info, load_example_qblox_hardware_config)

    assert (
        "Please ensure a minimum interval of 1000 ns between acquisitions"
        in error.value.args[0]
    )


def test_compile_with_rel_time(
    dummy_pulsars,
    pulse_only_schedule_with_operation_timing,
    compile_config_basic_transmon_qblox_hardware,
):
    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        pulse_only_schedule_with_operation_timing,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    qcm0_seq0_json = full_program["compiled_instructions"]["qcm0"]["sequencers"][
        "seq0"
    ]["sequence"]

    qcm0 = dummy_pulsars["qcm0"]
    qcm0.sequencer0.sequence(qcm0_seq0_json)


def test_compile_with_repetitions(
    mixed_schedule_with_acquisition,
    compile_config_basic_transmon_qblox_hardware,
):
    mixed_schedule_with_acquisition.repetitions = 10

    compiler = SerialCompiler(name="compiler")
    full_program = compiler.compile(
        mixed_schedule_with_acquisition,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    program_from_json = full_program["compiled_instructions"]["qcm0"]["sequencers"][
        "seq0"
    ]["sequence"]["program"]
    move_line = program_from_json.split("\n")[5]
    move_items = move_line.split()  # splits on whitespace
    args = move_items[1]
    iterations = int(args.split(",")[0])
    assert iterations == 10


def _func_for_hook_test(qasm: QASMProgram):
    qasm.instructions.insert(
        0, QASMProgram.get_instruction_as_list(q1asm_instructions.NOP)
    )


def test_qasm_hook(pulse_only_schedule, mock_setup_basic_transmon_with_standard_params):
    hw_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qrm0": {
            "instrument_type": "Pulsar_QRM",
            "ref": "external",
            "complex_output_0": {
                "portclock_configs": [
                    {
                        "qasm_hook_func": _func_for_hook_test,
                        "port": "q0:mw",
                        "clock": "q0.01",
                    }
                ]
            },
        },
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
    program = full_program["compiled_instructions"]["qrm0"]["sequencers"]["seq0"][
        "sequence"
    ]["program"]
    program_lines = program.splitlines()

    assert program_lines[1].strip() == q1asm_instructions.NOP


def test_qcm_acquisition_error(load_example_qblox_hardware_config):
    qcm = QcmModule(
        None,
        "qcm0",
        total_play_time=10,
        hw_mapping=load_example_qblox_hardware_config["qcm0"],
    )
    qcm._acquisitions[0] = 0

    with pytest.raises(RuntimeError):
        qcm.distribute_data()


@pytest.mark.parametrize("instruction_generated_pulses_enabled", [False])
def test_real_mode_pulses(
    real_square_pulse_schedule,
    hardware_cfg_real_mode,
    mock_setup_basic_transmon,
    instruction_generated_pulses_enabled,  # pylint: disable=unused-argument
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
        seq_instructions = full_program.compiled_instructions["qcm0"]["sequencers"][
            f"seq{output}"
        ]["sequence"]

        for value in seq_instructions["waveforms"].values():
            waveform_data, seq_path = value["data"], value["index"]

            # Asserting that indeed we only have square pulse on I and no signal on Q
            if seq_path == 0:
                assert (np.array(waveform_data) == 1).all()
            elif seq_path == 1:
                assert (np.array(waveform_data) == 0).all()

        if output % 2 == 0:
            iq_order = "0,1"  # I,Q
        else:
            iq_order = "1,0"  # Q,I

        assert re.search(rf"play\s*{iq_order}", seq_instructions["program"]), (
            f"Output {output+1} must be connected to "
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


def test_expand_from_normalised_range():
    minimal_pulse_data = {"duration": 20e-9}
    acq = types.OpInfo(name="test_acq", data=minimal_pulse_data, timing=4e-9)
    expanded_val = QASMProgram.expand_from_normalised_range(
        1, constants.IMMEDIATE_MAX_WAIT_TIME, "test_param", acq
    )
    assert expanded_val == constants.IMMEDIATE_MAX_WAIT_TIME // 2
    with pytest.raises(ValueError):
        QASMProgram.expand_from_normalised_range(
            10, constants.IMMEDIATE_MAX_WAIT_TIME, "test_param", acq
        )


def test_to_grid_time():
    time_ns = to_grid_time(8e-9)
    assert time_ns == 8
    with pytest.raises(ValueError):
        to_grid_time(7e-9)


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
    load_example_qblox_hardware_config,
    reset_clock_phase,
):
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
        sched_with_pulse_info, load_example_qblox_hardware_config
    )
    assign_pulse_and_acq_info_to_devices(
        sched_with_pulse_info,
        container.instrument_compilers,
        load_example_qblox_hardware_config,
    )

    qrm = container.instrument_compilers["qrm0"]
    expected_num_of_pulses = 1 if reset_clock_phase is False else 2
    actual_num_of_pulses = len(qrm._pulses[list(qrm._portclocks_with_data)[0]])
    actual_num_of_acquisitions = len(
        qrm._acquisitions[list(qrm._portclocks_with_data)[0]]
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
    load_example_qblox_hardware_config,
    compile_config_basic_transmon_qblox_hardware,
):
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=pulse_only_schedule,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        sched, load_example_qblox_hardware_config
    )
    assign_pulse_and_acq_info_to_devices(
        sched, container.instrument_compilers, load_example_qblox_hardware_config
    )
    container.prepare()

    for instr in container.instrument_compilers.values():
        instr.prepare()

    assert (
        container.instrument_compilers["qcm0"].sequencers["seq0"].frequency is not None
    )
    assert container.instrument_compilers["lo0"].frequency is not None


def test_multiple_trace_acquisition_error(compile_config_basic_transmon_qblox_hardware):
    sched = Schedule("test_multiple_trace_acquisition_error")
    sched.add(Trace(duration=100e-9, port="q0:res", clock="q0.multiplex"))
    sched.add(Trace(duration=100e-9, port="q0:res", clock="q0.ro"))

    sched.add_resource(ClockResource("q0.multiplex", 3.2e9))

    with pytest.raises(ValueError) as exception:
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=compile_config_basic_transmon_qblox_hardware
        )
    assert str(exception.value) == (
        f"Both sequencer '0' and '1' "
        f"of 'qrm0' attempts to perform scope mode acquisitions. "
        f"Only one sequencer per device can "
        f"trigger raw trace capture.\n\nPlease ensure that "
        f"only one port-clock combination performs "
        f"raw trace acquisition per instrument."
    )


@pytest.mark.parametrize("add_lo1", [False])
def test_container_prepare_baseband(
    mock_setup_basic_transmon,
    baseband_square_pulse_schedule,
    hardware_cfg_baseband,
    add_lo1: bool,  # pylint: disable=unused-argument
):
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_baseband)
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=baseband_square_pulse_schedule,
        config=quantum_device.generate_compilation_config(),
    )

    container = compiler_container.CompilerContainer.from_hardware_cfg(
        schedule=sched, hardware_cfg=hardware_cfg_baseband
    )
    assign_pulse_and_acq_info_to_devices(
        schedule=sched,
        device_compilers=container.instrument_compilers,
        hardware_cfg=hardware_cfg_baseband,
    )
    container.prepare()

    assert (
        container.instrument_compilers["qcm0"].sequencers["seq0"].frequency is not None
    )
    assert container.instrument_compilers["lo0"].frequency is not None


def test_container_prepare_no_lo(
    pulse_only_schedule_no_lo,
    load_example_qblox_hardware_config,
    compile_config_basic_transmon_qblox_hardware,
):
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=pulse_only_schedule_no_lo,
        config=compile_config_basic_transmon_qblox_hardware,
    )
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        sched, load_example_qblox_hardware_config
    )
    assign_pulse_and_acq_info_to_devices(
        sched, container.instrument_compilers, load_example_qblox_hardware_config
    )
    container.prepare()

    assert container.instrument_compilers["qrm1"].sequencers["seq0"].frequency == 8.3e9


def test_container_add_from_type(
    pulse_only_schedule, load_example_qblox_hardware_config
):
    determine_absolute_timing(pulse_only_schedule)
    container = compiler_container.CompilerContainer(pulse_only_schedule)
    container.add_instrument_compiler(
        "qcm0", QcmModule, load_example_qblox_hardware_config["qcm0"]
    )
    assert "qcm0" in container.instrument_compilers
    assert isinstance(container.instrument_compilers["qcm0"], QcmModule)


def test_container_add_from_str(
    pulse_only_schedule, load_example_qblox_hardware_config
):
    determine_absolute_timing(pulse_only_schedule)
    container = compiler_container.CompilerContainer(pulse_only_schedule)
    container.add_instrument_compiler(
        "qcm0", "Pulsar_QCM", load_example_qblox_hardware_config["qcm0"]
    )
    assert "qcm0" in container.instrument_compilers
    assert isinstance(container.instrument_compilers["qcm0"], QcmModule)


def test_container_add_from_path(
    pulse_only_schedule, load_example_qblox_hardware_config
):
    determine_absolute_timing(pulse_only_schedule)
    container = compiler_container.CompilerContainer(pulse_only_schedule)
    container.add_instrument_compiler(
        "qcm0",
        "quantify_scheduler.backends.qblox.instrument_compilers.QcmModule",
        load_example_qblox_hardware_config["qcm0"],
    )
    assert "qcm0" in container.instrument_compilers
    assert isinstance(container.instrument_compilers["qcm0"], QcmModule)


def test_from_mapping(pulse_only_schedule, load_example_qblox_hardware_config):
    determine_absolute_timing(pulse_only_schedule)
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        pulse_only_schedule, load_example_qblox_hardware_config
    )
    for instr_name in load_example_qblox_hardware_config.keys():
        if instr_name == "backend" or "corrections" in instr_name:
            continue
        assert instr_name in container.instrument_compilers


def test_generate_uuid_from_wf_data():
    arr0 = np.arange(10000)
    arr1 = np.arange(10000)
    arr2 = np.arange(10000) + 1

    hash0 = generate_uuid_from_wf_data(arr0)
    hash1 = generate_uuid_from_wf_data(arr1)
    hash2 = generate_uuid_from_wf_data(arr2)

    assert hash0 == hash1
    assert hash1 != hash2


@pytest.mark.parametrize("instruction_generated_pulses_enabled", [False])
def test_real_mode_container(
    real_square_pulse_schedule,
    hardware_cfg_real_mode,
    mock_setup_basic_transmon,
    instruction_generated_pulses_enabled,  # pylint: disable=unused-argument
):
    determine_absolute_timing(real_square_pulse_schedule)
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        real_square_pulse_schedule, hardware_cfg_real_mode
    )
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_real_mode)
    compiler = SerialCompiler(name="compiler")
    sched = compiler.compile(
        schedule=real_square_pulse_schedule,
        config=quantum_device.generate_compilation_config(),
    )
    assign_pulse_and_acq_info_to_devices(
        sched, container.instrument_compilers, hardware_cfg_real_mode
    )
    container.prepare()
    qcm0 = container.instrument_compilers["qcm0"]
    for output, seq_name in enumerate(f"seq{i}" for i in range(3)):
        seq_settings = qcm0.sequencers[seq_name].settings
        assert seq_settings.connected_outputs[0] == output


def test_assign_frequencies_baseband(
    mock_setup_basic_transmon_with_standard_params,
    load_example_qblox_hardware_config,
):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q0"))
    sched.add(X("q1"))

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    hardware_cfg = load_example_qblox_hardware_config
    quantum_device.hardware_config(hardware_cfg)

    device_cfg = quantum_device.generate_device_config()
    q0_clock_freq = device_cfg.clocks["q0.01"]
    q1_clock_freq = device_cfg.clocks["q1.01"]

    if0 = hardware_cfg["qcm0"]["complex_output_0"]["portclock_configs"][0].get(
        "interm_freq"
    )
    if1 = hardware_cfg["qcm0"]["complex_output_1"]["portclock_configs"][0].get(
        "interm_freq"
    )
    io0_lo_name = hardware_cfg["qcm0"]["complex_output_0"]["lo_name"]
    io1_lo_name = hardware_cfg["qcm0"]["complex_output_1"]["lo_name"]
    lo0 = hardware_cfg[io0_lo_name].get("frequency")
    lo1 = hardware_cfg[io1_lo_name].get("frequency")

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
    assert compiled_instructions["qcm0"]["sequencers"]["seq1"]["modulation_freq"] == if1


@pytest.mark.parametrize(
    "downconverter_freq0, downconverter_freq1",
    list(itertools.product([None, 0, 9e9], repeat=2)) + [(-1, None), (1e6, None)],
)
def test_assign_frequencies_baseband_downconverter(
    mock_setup_basic_transmon_with_standard_params,
    load_example_qblox_hardware_config,
    downconverter_freq0,
    downconverter_freq1,
):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q0"))
    sched.add(X("q1"))

    hardware_cfg = copy.deepcopy(load_example_qblox_hardware_config)
    hardware_cfg["qcm0"]["complex_output_0"]["downconverter_freq"] = downconverter_freq0
    hardware_cfg["qcm0"]["complex_output_1"]["downconverter_freq"] = downconverter_freq1

    io0_lo_name = hardware_cfg["qcm0"]["complex_output_0"]["lo_name"]
    io1_lo_name = hardware_cfg["qcm0"]["complex_output_1"]["lo_name"]

    if0 = hardware_cfg["qcm0"]["complex_output_0"]["portclock_configs"][0].get(
        "interm_freq"
    )
    lo1 = hardware_cfg[io1_lo_name].get("frequency")

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
            portclock_config = hardware_cfg["qcm0"]["complex_output_0"][
                "portclock_configs"
            ][0]
            if downconverter_freq0 < 0:
                assert (
                    str(error.value) == f"Downconverter frequency must be positive "
                    f"(downconverter_freq={downconverter_freq0:e}) "
                    f"(for 'seq0' of 'qcm0' with "
                    f"port '{portclock_config['port']}' and "
                    f"clock '{portclock_config['clock']}')."
                )
            elif downconverter_freq0 < q0_clock_freq:
                assert (
                    str(error.value)
                    == "Downconverter frequency must be greater than clock frequency "
                    f"(downconverter_freq={downconverter_freq0:e}, "
                    f"clock_freq={q0_clock_freq:e}) "
                    f"(for 'seq0' of 'qcm0' with "
                    f"port '{portclock_config['port']}' and "
                    f"clock '{portclock_config['clock']}')."
                )
        return

    generic_ic_program = compiled_schedule["compiled_instructions"][
        constants.GENERIC_IC_COMPONENT_NAME
    ]
    qcm_program = compiled_schedule["compiled_instructions"]["qcm0"]
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


def test_assign_frequencies_rf(
    mock_setup_basic_transmon, load_example_qblox_hardware_config
):
    sched = Schedule("two_gate_experiment")
    sched.add(X("q2"))
    sched.add(X("q3"))

    hardware_cfg = load_example_qblox_hardware_config
    if0 = hardware_cfg["qcm_rf0"]["complex_output_0"]["portclock_configs"][0].get(
        "interm_freq"
    )
    if1 = hardware_cfg["qcm_rf0"]["complex_output_1"]["portclock_configs"][0].get(
        "interm_freq"
    )
    lo0 = hardware_cfg["qcm_rf0"]["complex_output_0"].get("lo_freq")
    lo1 = hardware_cfg["qcm_rf0"]["complex_output_1"].get("lo_freq")

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
    qcm_program = compiled_instructions["qcm_rf0"]

    assert qcm_program["settings"]["lo0_freq"] == lo0
    assert qcm_program["settings"]["lo1_freq"] == lo1
    assert qcm_program["sequencers"]["seq1"]["modulation_freq"] == if1


@pytest.mark.parametrize(
    "downconverter_freq0, downconverter_freq1, element_names",
    [
        list(pair) + [["q5", "q6"]]
        for pair in list(itertools.product([None, 0, 8.2e9], repeat=2))
        + [(-1, None), (1e6, None)]
    ],
)
def test_assign_frequencies_rf_downconverter(
    mock_setup_basic_transmon_elements,
    load_example_qblox_hardware_config,
    downconverter_freq0,
    downconverter_freq1,
    element_names,
):
    sched = Schedule("two_gate_experiment")
    sched.add(X(element_names[0]))
    sched.add(X(element_names[1]))

    hardware_cfg = copy.deepcopy(load_example_qblox_hardware_config)
    hardware_cfg["cluster0"]["cluster0_module2"]["complex_output_0"][
        "downconverter_freq"
    ] = downconverter_freq0
    hardware_cfg["cluster0"]["cluster0_module2"]["complex_output_1"][
        "downconverter_freq"
    ] = downconverter_freq1

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

    quantum_device.hardware_config(hardware_cfg)
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
            portclock_config = hardware_cfg["cluster0"]["cluster0_module2"][
                "complex_output_0"
            ]["portclock_configs"][0]
            if downconverter_freq0 < 0:
                assert (
                    str(error.value) == f"Downconverter frequency must be positive "
                    f"(downconverter_freq={downconverter_freq0:e}) "
                    f"(for 'seq0' of 'cluster0_module2' with "
                    f"port '{portclock_config['port']}' and "
                    f"clock '{portclock_config['clock']}')."
                )
            elif downconverter_freq0 < qubit0_clock_freq:
                assert (
                    str(error.value)
                    == "Downconverter frequency must be greater than clock frequency "
                    f"(downconverter_freq={downconverter_freq0:e}, "
                    f"clock_freq={qubit0_clock_freq:e}) "
                    f"(for 'seq0' of 'cluster0_module2' with "
                    f"port '{portclock_config['port']}' and "
                    f"clock '{portclock_config['clock']}')."
                )
        return

    qcm_program = compiled_schedule["compiled_instructions"]["cluster0"][
        "cluster0_module2"
    ]
    actual_lo0 = qcm_program["settings"]["lo0_freq"]
    actual_lo1 = qcm_program["settings"]["lo1_freq"]
    actual_if1 = qcm_program["sequencers"]["seq1"]["modulation_freq"]

    if0 = hardware_cfg["cluster0"]["cluster0_module2"]["complex_output_0"][
        "portclock_configs"
    ][0].get("interm_freq")
    lo1 = hardware_cfg["cluster0"]["cluster0_module2"]["complex_output_1"].get(
        "lo_freq"
    )
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


@pytest.mark.parametrize(
    "element_names, input_att_output",
    [
        (["q5"], True),
        (["q5"], False),
    ],
)
def test_assign_attenuation(
    mock_setup_basic_transmon_elements,
    load_example_qblox_hardware_config,
    element_names,
    input_att_output,
):
    """
    Test function that checks if attenuation settings on a QRM-RF compile correctly.
    Also checks if floats are correctly converted to ints (if they are close to ints).
    """
    sched = Schedule("readout_experiment")
    sched.add(Measure(element_names[0]))

    hardware_cfg = copy.deepcopy(load_example_qblox_hardware_config)

    input_att = 10
    complex_input = hardware_cfg["cluster0"]["cluster0_module4"]["complex_input_0"]
    complex_output = hardware_cfg["cluster0"]["cluster0_module4"]["complex_output_0"]
    if input_att_output:
        complex_output["input_att"] = input_att
        complex_input.pop("input_att", None)
    else:
        complex_input["input_att"] = input_att
        complex_output.pop("input_att", None)

    output_att = hardware_cfg["cluster0"]["cluster0_module4"]["complex_output_0"].get(
        "output_att"
    )

    assert input_att is not None
    assert output_att is not None

    quantum_device = mock_setup_basic_transmon_elements["quantum_device"]
    qubit = quantum_device.get_element(element_names[0])

    qubit.clock_freqs.readout(5e9)
    qubit.measure.pulse_amp(0.2)
    qubit.measure.acq_delay(40e-9)

    quantum_device.hardware_config(hardware_cfg)
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )
    compiled_instructions = compiled_schedule["compiled_instructions"]
    qrm_rf_program = compiled_instructions["cluster0"]["cluster0_module4"]

    compiled_in0_att = qrm_rf_program["settings"]["in0_att"]
    compiled_out0_att = qrm_rf_program["settings"]["out0_att"]

    assert np.isclose(compiled_in0_att, input_att)
    assert np.isclose(compiled_out0_att, output_att)

    assert isinstance(compiled_in0_att, int)
    assert isinstance(compiled_out0_att, int)


def test_assign_input_att_both_output_input_raises(
    make_basic_multi_qubit_schedule,
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module4": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "input_att": 10,
                    "portclock_configs": [
                        {"port": "q0:res", "clock": "q0.ro", "interm_freq": 50e6},
                    ],
                },
                "complex_input_0": {
                    "input_att": 10,
                    "portclock_configs": [
                        {"port": "q1:res", "clock": "q1.ro", "interm_freq": 50e6},
                    ],
                },
            },
        },
    }

    schedule = Schedule("test_assign_input_att_both_output_input_raises")
    schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q0:res", clock="q0.ro"))
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
    hardware_cfg["cluster0"]["cluster0_module1"]["complex_output_0"][
        "output_att"
    ] = 10.3

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


def test_markers(mock_setup_basic_transmon, load_example_qblox_hardware_config):
    # Test for baseband
    sched = Schedule("gate_experiment")
    sched.add(X("q0"))
    sched.add(X("q2"))
    sched.add(Measure("q0"))
    sched.add(Measure("q2"))

    quantum_device = mock_setup_basic_transmon["quantum_device"]

    q0 = quantum_device.get_element("q0")
    q2 = quantum_device.get_element("q2")

    q0.rxy.amp180(0.213)
    q2.rxy.amp180(0.215)

    q0.clock_freqs.f01(7.3e9)
    q0.clock_freqs.f12(7.0e9)
    q0.clock_freqs.readout(8.0e9)
    q0.measure.acq_delay(100e-9)

    q2.clock_freqs.f01(7.3e9)
    q2.clock_freqs.f12(7.0e9)
    q2.clock_freqs.readout(8.0e9)
    q2.measure.acq_delay(100e-9)

    quantum_device.hardware_config(load_example_qblox_hardware_config)
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        sched, quantum_device.generate_compilation_config()
    )
    program = compiled_schedule["compiled_instructions"]

    def _confirm_correct_markers(device_program, mrk_config, is_rf=False):
        answers = (
            mrk_config.init,
            mrk_config.start,
            mrk_config.end,
        )
        qasm = device_program["sequencers"]["seq0"]["sequence"]["program"]

        matches = re.findall(r"set\_mrk +\d+", qasm)
        matches = [int(m.replace("set_mrk", "").strip()) for m in matches]
        if not is_rf:
            matches = [None, *matches]

        for match, answer in zip(matches, answers):
            assert match == answer

    _confirm_correct_markers(
        program["qcm0"], MarkerConfiguration(init=None, start=0b1111, end=0)
    )
    _confirm_correct_markers(
        program["qrm0"], MarkerConfiguration(init=None, start=0b1111, end=0)
    )
    _confirm_correct_markers(
        program["qcm_rf0"],
        MarkerConfiguration(init=0b0011, start=0b1101, end=0),
        is_rf=True,
    )
    _confirm_correct_markers(
        program["qrm_rf0"],
        MarkerConfiguration(init=0b0011, start=0b1111, end=0),
        is_rf=True,
    )


def test_pulsar_rf_extract_from_mapping(load_example_qblox_hardware_config):
    hw_map = load_example_qblox_hardware_config["qcm_rf0"]
    types.PulsarRFSettings.extract_settings_from_mapping(hw_map)


def test_cluster_settings(pulse_only_schedule, load_example_qblox_hardware_config):
    determine_absolute_timing(pulse_only_schedule)
    container = compiler_container.CompilerContainer.from_hardware_cfg(
        pulse_only_schedule, load_example_qblox_hardware_config
    )
    cluster_compiler = container.instrument_compilers["cluster0"]
    cluster_compiler.prepare()
    cl_qcm0 = cluster_compiler.instrument_compilers["cluster0_module1"]
    assert isinstance(cl_qcm0._settings, BasebandModuleSettings)


def assembly_valid(compiled_schedule, qcm0, qrm0):
    """
    Test helper that takes a compiled schedule and verifies if the assembly is valid
    by passing it to a dummy qcm and qrm.

    Assumes only qcm0 and qrm0 are used.
    """

    # test the program for the qcm
    qcm0_seq0_json = compiled_schedule["compiled_instructions"]["qcm0"]["sequencers"][
        "seq0"
    ]["sequence"]
    qcm0.sequencer0.sequence(qcm0_seq0_json)
    qcm0.arm_sequencer(0)
    uploaded_waveforms = qcm0.get_waveforms(0)
    assert uploaded_waveforms is not None

    # test the program for the qrm
    qrm0_seq0_json = compiled_schedule["compiled_instructions"]["qrm0"]["sequencers"][
        "seq0"
    ]["sequence"]
    qrm0.sequencer0.sequence(qrm0_seq0_json)
    qrm0.arm_sequencer(0)
    uploaded_waveforms = qrm0.get_waveforms(0)
    assert uploaded_waveforms is not None


def test_acq_protocol_append_mode_valid_assembly_ssro(
    dummy_pulsars,
    compile_config_basic_transmon_qblox_hardware,
):
    repetitions = 256
    ssro_sched = readout_calibration_sched("q0", [0, 1], repetitions=repetitions)
    compiler = SerialCompiler(name="compiler")
    compiled_ssro_sched = compiler.compile(
        ssro_sched, compile_config_basic_transmon_qblox_hardware
    )
    assembly_valid(
        compiled_schedule=compiled_ssro_sched,
        qcm0=dummy_pulsars["qcm0"],
        qrm0=dummy_pulsars["qrm0"],
    )

    qrm0_seq_instructions = compiled_ssro_sched["compiled_instructions"]["qrm0"][
        "sequencers"
    ]["seq0"]["sequence"]

    baseline_assembly = os.path.join(
        quantify_scheduler.__path__[0],
        "..",
        "tests",
        "baseline_qblox_assembly",
        f"{ssro_sched.name}_qrm0_seq0_instr.json",
    )

    if REGENERATE_REF_FILES:
        with open(baseline_assembly, "w", encoding="utf-8") as file:
            json.dump(qrm0_seq_instructions, file)

    with open(baseline_assembly) as file:
        baseline_qrm0_seq_instructions = json.load(file)
    program = _strip_comments(qrm0_seq_instructions["program"])
    exp_program = _strip_comments(baseline_qrm0_seq_instructions["program"])

    assert list(program) == list(exp_program)


def test_acq_protocol_average_mode_valid_assembly_allxy(
    dummy_pulsars,
    compile_config_basic_transmon_qblox_hardware,
):
    repetitions = 256
    sched = allxy_sched("q0", element_select_idx=np.arange(21), repetitions=repetitions)
    compiler = SerialCompiler(name="compiler")
    compiled_allxy_sched = compiler.compile(
        sched, compile_config_basic_transmon_qblox_hardware
    )

    assembly_valid(
        compiled_schedule=compiled_allxy_sched,
        qcm0=dummy_pulsars["qcm0"],
        qrm0=dummy_pulsars["qrm0"],
    )

    qrm0_seq_instructions = compiled_allxy_sched["compiled_instructions"]["qrm0"][
        "sequencers"
    ]["seq0"]["sequence"]

    baseline_assembly = os.path.join(
        quantify_scheduler.__path__[0],
        "..",
        "tests",
        "baseline_qblox_assembly",
        f"{sched.name}_qrm0_seq0_instr.json",
    )

    if REGENERATE_REF_FILES:
        with open(baseline_assembly, "w", encoding="utf-8") as file:
            json.dump(qrm0_seq_instructions, file)

    with open(baseline_assembly) as file:
        baseline_qrm0_seq_instructions = json.load(file)
    program = _strip_comments(qrm0_seq_instructions["program"])
    exp_program = _strip_comments(baseline_qrm0_seq_instructions["program"])

    assert list(program) == list(exp_program)


def test_acq_declaration_dict_append_mode(compile_config_basic_transmon_qblox_hardware):
    repetitions = 256

    ssro_sched = readout_calibration_sched("q0", [0, 1], repetitions=repetitions)
    compiler = SerialCompiler(name="compiler")
    compiled_ssro_sched = compiler.compile(
        ssro_sched, compile_config_basic_transmon_qblox_hardware
    )

    qrm0_seq_instructions = compiled_ssro_sched["compiled_instructions"]["qrm0"][
        "sequencers"
    ]["seq0"]["sequence"]

    acquisitions = qrm0_seq_instructions["acquisitions"]
    # the only key corresponds to channel 0
    assert set(acquisitions.keys()) == {"0"}
    assert acquisitions["0"] == {"num_bins": 2 * 256, "index": 0}


def test_acq_declaration_dict_bin_avg_mode(
    compile_config_basic_transmon_qblox_hardware,
):
    allxy = allxy_sched("q0")
    compiler = SerialCompiler(name="compiler")
    compiled_allxy_sched = compiler.compile(
        allxy, config=compile_config_basic_transmon_qblox_hardware
    )
    qrm0_seq_instructions = compiled_allxy_sched["compiled_instructions"]["qrm0"][
        "sequencers"
    ]["seq0"]["sequence"]

    acquisitions = qrm0_seq_instructions["acquisitions"]

    # the only key corresponds to channel 0
    assert set(acquisitions.keys()) == {"0"}
    assert acquisitions["0"] == {"num_bins": 21, "index": 0}


def test_convert_hw_config_to_portclock_configs_spec(
    make_basic_multi_qubit_schedule,
    mock_setup_basic_transmon_with_standard_params,
):
    old_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "lo_name": "lo0",
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6,
                    "latency_correction": 8e-9,
                },
            },
            "complex_output_1": {
                "lo_name": "lo1",
                "seq1": {"port": "q1:mw", "clock": "q1.01", "interm_freq": 100e6},
                "seq2": {
                    "port": "q2:mw",
                    "clock": "q2.01",
                    "interm_freq": None,
                    "latency_correction": 4e-9,
                },
            },
        },
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module2": {
                "instrument_type": "QRM",
                "complex_output_0": {
                    "seq0": {
                        "port": "q1:res",
                        "clock": "q1.ro",
                        "interm_freq": 50e6,
                    },
                    "seq1": {
                        "port": "q2:res",
                        "clock": "q2.01",
                        "interm_freq": 50e6,
                        "latency_correction": 4e-9,
                    },
                },
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 20},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": None, "power": 20},
    }

    expected_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "latency_corrections": {
            "q0:mw-q0.01": 8e-9,
            "q2:mw-q2.01": 4e-9,
            "q2:res-q2.01": 4e-9,
        },
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
            "complex_output_0": {
                "lo_name": "lo0",
                "portclock_configs": [
                    {"port": "q0:mw", "clock": "q0.01", "interm_freq": 50e6},
                ],
            },
            "complex_output_1": {
                "lo_name": "lo1",
                "portclock_configs": [
                    {"port": "q1:mw", "clock": "q1.01", "interm_freq": 100e6},
                    {"port": "q2:mw", "clock": "q2.01", "interm_freq": None},
                ],
            },
        },
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module2": {
                "instrument_type": "QRM",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:res",
                            "clock": "q1.ro",
                            "interm_freq": 50e6,
                        },
                        {
                            "port": "q2:res",
                            "clock": "q2.01",
                            "interm_freq": 50e6,
                        },
                    ],
                },
            },
        },
        "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 20},
        "lo1": {"instrument_type": "LocalOscillator", "frequency": None, "power": 20},
    }

    # Test that the conversion works adequately
    migrated_config = convert_hw_config_to_portclock_configs_spec(old_config)
    assert migrated_config == expected_config

    # Test that qblox_backend.hardware_compile is converting automatically
    sched = make_basic_multi_qubit_schedule(["q0", "q1"])
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(old_config)

    with pytest.warns(
        FutureWarning,
        match=r"hardware config adheres to a specification that is deprecated",
    ):
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            schedule=sched, config=quantum_device.generate_compilation_config()
        )


def test_apply_latency_corrections_invalid_raises(
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
    hardware_cfg["latency_corrections"]["q1:mw-q1.01"] = None
    mock_setup_basic_transmon["quantum_device"].hardware_config(hardware_cfg)
    with pytest.raises(ValidationError):
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            sched,
            config=mock_setup_basic_transmon[
                "quantum_device"
            ].generate_compilation_config(),
        )


def test_apply_latency_corrections_valid(
    mock_setup_basic_transmon_with_standard_params, hardware_cfg_latency_corrections
):
    """
    This test function checks that:
    Latency correction is set for the correct portclock key
    by checking against the value set in QASM instructions.
    """

    mock_setup = mock_setup_basic_transmon_with_standard_params
    hardware_cfg = hardware_cfg_latency_corrections
    mock_setup["quantum_device"].hardware_config(hardware_cfg_latency_corrections)

    sched = Schedule("Single Gate Experiment on Two Qubits")
    sched.add(X("q0"))
    sched.add(
        SquarePulse(port="q1:mw", clock="q1.01", amp=0.25, duration=12e-9),
        ref_pt="start",
    )

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        sched,
        config=mock_setup["quantum_device"].generate_compilation_config(),
    )

    for instrument in ["qcm0", ("cluster0", "cluster0_module1")]:
        compiled_data = compiled_sched.compiled_instructions
        config_data = hardware_cfg

        if isinstance(instrument, tuple):
            for key in instrument:
                compiled_data = compiled_data.get(key)
                config_data = config_data.get(key)
        else:
            compiled_data = compiled_data.get(instrument)
            config_data = config_data.get(instrument)

        latency_dict = corrections.determine_relative_latencies(hardware_cfg)
        port = config_data["complex_output_0"]["portclock_configs"][0]["port"]
        clock = config_data["complex_output_0"]["portclock_configs"][0]["clock"]
        latency = int(1e9 * latency_dict[f"{port}-{clock}"])

        program_lines = compiled_data["sequencers"]["seq0"]["sequence"][
            "program"
        ].splitlines()
        assert any(
            f"latency correction of {constants.GRID_TIME} + {latency} ns" in line
            for line in program_lines
        ), f"instrument={instrument}, latency={latency}"


def test_determine_relative_latencies(
    hardware_cfg_latency_corrections,
) -> None:
    generated_latency_dict = corrections.determine_relative_latencies(
        hardware_cfg=hardware_cfg_latency_corrections
    )
    assert generated_latency_dict == {"q0:mw-q0.01": 2.5e-08, "q1:mw-q1.01": 0.0}


def test_apply_latency_corrections_warning(
    mock_setup_basic_transmon, hardware_cfg_latency_corrections, caplog
):
    """
    Checks if warning is raised for a latency correction
    that is not a multiple of 4ns
    """
    mock_setup_basic_transmon["quantum_device"].hardware_config(
        hardware_cfg_latency_corrections
    )
    sched = Schedule("Single Gate Experiment")
    sched.add(
        SquarePulse(port="q0:mw", clock="q0.01", amp=0.25, duration=12e-9),
        ref_pt="start",
    )
    sched.add_resource(ClockResource("q0.01", freq=5e9))

    warning = f"not a multiple of {constants.GRID_TIME}"
    with caplog.at_level(
        logging.WARNING, logger="quantify_scheduler.backends.qblox.qblox_backend"
    ):
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            sched,
            config=mock_setup_basic_transmon[
                "quantum_device"
            ].generate_compilation_config(),
        )
    assert any(warning in mssg for mssg in caplog.messages)


def _strip_comments(program: str):
    # helper function for comparing programs
    stripped_program = []
    for line in program.split("\n"):
        if "#" in line:
            line = line.split("#")[0]
        line = line.rstrip()  # remove trailing whitespace
        stripped_program.append(line)
    return stripped_program


def test_overwrite_gain(mock_setup_basic_transmon_with_standard_params):
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module1": {
                "instrument_type": "QCM",
                "real_output_0": {
                    "input_gain": 5,
                    "portclock_configs": [
                        {"port": "q0:res", "clock": "q0.ro", "interm_freq": 50e6},
                    ],
                },
                "real_output_1": {
                    "input_gain": 10,
                    "portclock_configs": [
                        {"port": "q0:res", "clock": "q0.ro", "interm_freq": 50e6},
                    ],
                },
            },
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test overwrite gain")
    schedule.add(SquarePulse(amp=0.5, duration=1e-6, port="q0:res", clock="q0.ro"))

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler", quantum_device=quantum_device)
    with pytest.raises(ValueError) as exc:
        compiler.compile(schedule=schedule)

    # assert exception was raised with correct message.
    assert (
        exc.value.args[0]
        == "Overwriting gain of real_output_1 of module cluster0_module1 to in0_gain: 10."
        "\nIt was previously set to in0_gain: 5."
    )
