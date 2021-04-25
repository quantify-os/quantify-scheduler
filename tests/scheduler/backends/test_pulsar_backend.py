import json
import tempfile
import pathlib
import inspect
import os
import pytest
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.utils.helpers import NumpyJSONEncoder
from quantify.data.handling import set_datadir
from quantify.scheduler.types import Schedule
from quantify.scheduler.gate_library import Reset, Measure, Rxy, X
from quantify.scheduler.pulse_library import SquarePulse, DRAGPulse, RampPulse
from quantify.scheduler.backends.pulsar_backend import (
    build_waveform_dict,
    build_q1asm,
    generate_sequencer_cfg,
    pulsar_assembler_backend,
    _check_driver_version,
    QCM_DRIVER_VER,
    QRM_DRIVER_VER,
    _extract_interm_freq,
    _invert_hardware_mapping,
    _extract_pulsar_type,
    _extract_gain,
    _extract_io,
    _extract_pulsar_config,
    _sanitize_file_name,
)
from quantify.scheduler.resources import ClockResource
from quantify.scheduler.compilation import qcompile, determine_absolute_timing

import quantify.scheduler.schemas.examples as examples

esp = inspect.getfile(examples)

cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)


try:
    from pulsar_qcm.pulsar_qcm import pulsar_qcm_dummy
    from pulsar_qrm.pulsar_qrm import pulsar_qrm_dummy

    PULSAR_ASSEMBLER = True
except ImportError:
    PULSAR_ASSEMBLER = False


def regenerate_ref_file(filename, contents):
    """
    Must only be used to regenerate a reference file after changes.
    Make sure to check the created file is correct.
    Do not push code that calls this function.
    """
    with open(pathlib.Path(__file__).parent.joinpath(filename), "w") as f:
        f.write(contents)


def test_build_waveform_dict():
    real = np.random.random(int(4e3))
    complex_vals = real + (np.random.random(int(4e3)) * 1.0j)

    pulse_data = {
        "gdfshdg45": complex_vals,
        "6h5hh5hyj": real,
    }
    sequence_cfg = build_waveform_dict(pulse_data, acquisitions={"6h5hh5hyj"})
    assert len(sequence_cfg["waveforms"]["awg"]) == 2
    assert len(sequence_cfg["waveforms"]["acq"]) == 2
    wf_1 = sequence_cfg["waveforms"]["awg"]["gdfshdg45_I"]
    wf_2 = sequence_cfg["waveforms"]["awg"]["gdfshdg45_Q"]
    wf_3 = sequence_cfg["waveforms"]["acq"]["6h5hh5hyj_I"]
    wf_4 = sequence_cfg["waveforms"]["acq"]["6h5hh5hyj_Q"]
    np.testing.assert_array_equal(wf_1["data"], complex_vals.real)
    assert wf_1["index"] == 0
    np.testing.assert_array_equal(wf_2["data"], complex_vals.imag)
    assert wf_2["index"] == 1
    np.testing.assert_array_equal(wf_3["data"], real)
    assert wf_3["index"] == 0
    np.testing.assert_array_equal(wf_4["data"], np.zeros(len(wf_4["data"])))
    assert wf_4["index"] == 1


def test_bad_pulse_timings():
    short_pulse_timings = [(0, "drag_ID", None), (4, "square_id", None)]
    short_wait_timings = [(0, "square_id", None), (6, "square_id", None)]
    short_final_wait = [(0, "square_id", None), (4, "square_id", None)]

    dummy_pulse_data = {
        "awg": {
            "square_id_I": {"data": np.ones(4), "index": 0},
            "square_id_Q": {"data": np.zeros(4), "index": 1},
            "drag_ID_I": {"data": np.ones(2), "index": 2},
            "drag_ID_Q": {"data": np.ones(2), "index": 3},
        }
    }
    pulsar_type = "QCM_sequencer"

    with pytest.raises(
        ValueError,
        match="Generated wait for '0':'drag_ID' caused exception 'duration 2ns < "
        "cycle time 4ns'",
    ):
        build_q1asm(
            short_pulse_timings,
            dummy_pulse_data,
            short_pulse_timings[-1][0] + 4,
            set(),
            1,
            pulsar_type,
        )

    with pytest.raises(
        ValueError,
        match="Generated wait for '0':'square_id' caused exception 'duration 2ns < "
        "cycle time 4ns'",
    ):
        build_q1asm(short_wait_timings, dummy_pulse_data, 10, set(), 1, pulsar_type)

    with pytest.raises(
        ValueError,
        match="Generated wait for '4':'square_id' caused exception 'duration 2ns < "
        "cycle time 4ns'",
    ):
        build_q1asm(short_final_wait, dummy_pulse_data, 10, set(), 1, pulsar_type)


def test_acq_delay():
    t0 = 10000
    acq_delay = 50
    wait_timings = [
        (t0 + 0, -3772038313413949572, None),
        (t0 + acq_delay, -1500324, None),
    ]

    dummy_pulse_data = {
        "awg": {
            "-3772038313413949572_I": {"data": np.ones(4), "index": 0},
            "-3772038313413949572_Q": {"data": np.ones(4), "index": 1},
        },
        "acq": {
            "-1500324_I": {"data": np.ones(4), "index": 0},
            "-1500324_Q": {"data": np.ones(4), "index": 1},
        },
    }

    pulsar_type = "QRM_sequencer"

    program_str = build_q1asm(
        wait_timings, dummy_pulse_data, t0 + 54, {-1500324}, 10, pulsar_type
    )

    # regenerate_ref_file('ref_test_acq_delay_q1asm', program_str)
    with open(
        pathlib.Path(__file__).parent.joinpath("ref_test_acq_delay_q1asm"), "r"
    ) as f:
        assert program_str == f.read()


def test_overflowing_instruction_times():
    real = np.random.random(129380)
    pulse_timings = [(0, "square_ID", None)]
    pulse_data = {
        "awg": {
            "square_ID_I": {"data": real, "index": 0},
            "square_ID_Q": {"data": np.zeros(len(real)), "index": 1},
        }
    }
    pulsar_type = "QCM_sequencer"

    program_str = build_q1asm(
        pulse_timings, pulse_data, len(real), set(), 1, pulsar_type
    )
    # regenerate_ref_file('ref_test_large_plays_q1asm', program_str)
    with open(
        pathlib.Path(__file__).parent.joinpath("ref_test_large_plays_q1asm"),
        "r",
    ) as f:
        assert program_str == f.read()

    pulse_timings.append((229380 + pow(2, 16), "square_ID", None))
    program_str = build_q1asm(pulse_timings, pulse_data, 524296, set(), 1, pulsar_type)
    # regenerate_ref_file('ref_test_large_waits_q1asm', program_str)
    with open(
        pathlib.Path(__file__).parent.joinpath("ref_test_large_waits_q1asm"),
        "r",
    ) as f:
        assert program_str == f.read()


def test_build_q1asm():
    real = np.random.random(4)
    complex_vals = real + (np.random.random(4) * 1.0j)

    pulse_timings = [
        (0, "square_id", None),
        (4, "drag_ID", None),
        (16, "square_id", None),
    ]

    pulse_data = {
        "awg": {
            "square_id_I": {"data": real, "index": 0},
            "square_id_Q": {"data": np.zeros(len(real)), "index": 1},
            "drag_ID_I": {"data": complex_vals.real, "index": 2},
            "drag_ID_Q": {"data": complex_vals.imag, "index": 3},
        },
        "acq": {
            "square_id_I": {"data": np.ones(4), "index": 0},
            "square_id_Q": {"data": np.zeros(4), "index": 1},
        },
    }

    pulsar_type = "QCM_sequencer"

    program_str = build_q1asm(pulse_timings, pulse_data, 20, set(), 1, pulsar_type)
    # regenerate_ref_file('ref_test_build_q1asm', program_str)
    with open(pathlib.Path(__file__).parent.joinpath("ref_test_build_q1asm"), "r") as f:
        assert program_str == f.read()

    program_str_sync = build_q1asm(pulse_timings, pulse_data, 30, set(), 1, pulsar_type)
    # regenerate_ref_file('ref_test_build_q1asm_sync', program_str_sync)
    with open(
        pathlib.Path(__file__).parent.joinpath("ref_test_build_q1asm_sync"), "r"
    ) as f:
        assert program_str_sync == f.read()

    program_str_loop = build_q1asm(
        pulse_timings, pulse_data, 20, set(), 20, pulsar_type
    )
    # regenerate_ref_file('ref_test_build_q1asm_loop', program_str_loop)
    with open(
        pathlib.Path(__file__).parent.joinpath("ref_test_build_q1asm_loop"), "r"
    ) as f:
        assert program_str_loop == f.read()

    program_str_meas = build_q1asm(
        pulse_timings, pulse_data, 20, {"square_id"}, 1, pulsar_type
    )
    # regenerate_ref_file('ref_test_build_q1asm_meas', program_str_meas)
    with open(
        pathlib.Path(__file__).parent.joinpath("ref_test_build_q1asm_meas"), "r"
    ) as f:
        assert program_str_meas == f.read()

    err = r"Provided sequence_duration.*4.*less than the total runtime of this sequence.*20"
    with pytest.raises(ValueError, match=err):
        build_q1asm(pulse_timings, pulse_data, 4, set(), 1, pulsar_type)

    # sequence_duration greater than final timing but less than total runtime
    err = r"Provided sequence_duration.*18.*less than the total runtime of this sequence.*20"
    with pytest.raises(ValueError, match=err):
        build_q1asm(pulse_timings, pulse_data, 18, set(), 1, pulsar_type)


def test_fn_sanitization():
    test_fn = "this_is_a test_for\\an_invalid:file?name*okay?.test"
    sanitized = _sanitize_file_name(test_fn)
    assert sanitized == "this_is_a_test_for_an_invalid_file_name_okay_.test"


def test_generate_sequencer_cfg():
    pulse_timings = [
        (0, "square_1", None),
        (4, "drag_1", None),
        (16, "square_2", None),
    ]

    real = np.random.random(4)
    complex_vals = real + (np.random.random(4) * 1.0j)
    pulse_data = {
        "square_1": [0.0, 1.0, 0.0, 0.0],
        "drag_1": complex_vals,
        "square_2": real,
    }
    pulsar_type = "QCM_sequencer"

    def check_waveform(entry, exp_data, exp_idx):
        assert exp_idx == entry["index"]
        np.testing.assert_array_equal(exp_data, entry["data"])

    sequence_cfg = generate_sequencer_cfg(
        pulse_data, pulse_timings, 20, set(), 1, pulsar_type
    )
    check_waveform(
        sequence_cfg["waveforms"]["awg"]["square_1_I"], [0.0, 1.0, 0.0, 0.0], 0
    )
    check_waveform(sequence_cfg["waveforms"]["awg"]["square_1_Q"], np.zeros(4), 1)
    check_waveform(sequence_cfg["waveforms"]["awg"]["drag_1_I"], complex_vals.real, 2)
    check_waveform(sequence_cfg["waveforms"]["awg"]["drag_1_Q"], complex_vals.imag, 3)
    check_waveform(sequence_cfg["waveforms"]["awg"]["square_2_I"], real, 4)
    check_waveform(sequence_cfg["waveforms"]["awg"]["square_2_Q"], np.zeros(4), 5)
    assert len(sequence_cfg["program"])

    if PULSAR_ASSEMBLER:
        with open("tmp.json", "w") as f:
            f.write(json.dumps(sequence_cfg, cls=NumpyJSONEncoder))
        qcm = pulsar_qcm_dummy("test")
        qcm.sequencer0_waveforms_and_program("tmp.json")
        assert "assembler finished successfully" in qcm.get_assembler_log()
        pathlib.Path("tmp.json").unlink()


@pytest.fixture
def dummy_pulsars():
    if PULSAR_ASSEMBLER:
        _pulsars = []
        for qcm in ["qcm0", "qcm1"]:
            _pulsars.append(pulsar_qcm_dummy(qcm))
        for qrm in ["qrm0", "qrm1"]:
            _pulsars.append(pulsar_qrm_dummy(qrm))
    else:
        _pulsars = []

    # ensures a temporary datadir is used which is excluded from git
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    yield _pulsars

    # teardown
    for instr_name in list(Instrument._all_instruments):
        try:
            inst = Instrument.find_instrument(instr_name)
            inst.close()
        except KeyError:
            pass


def test_pulsar_assembler_backend_pulses_only(dummy_pulsars):
    sched = Schedule("pulse_only_experiment")
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
        )
    )
    sched.add(RampPulse(amp=0.5, duration=24e-9, port="q0:mw", clock="q0.01"))
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    sched, config = pulsar_assembler_backend(sched, HARDWARE_MAPPING)


def test_pulsar_assembler_backend(dummy_pulsars):
    """
    This test uses a full example of compilation for a simple Bell experiment.
    This test can be made simpler the more we clean up the code.
    """
    # Create an empty schedule
    sched = Schedule("Bell experiment")

    # define the resources
    q0, q1 = ("q0", "q1")

    # Define the operations, these will be added to the circuit
    init_all = Reset(q0, q1)  # instantiates
    x90_q0 = Rxy(theta=90, phi=0, qubit=q0)

    # we use a regular for loop as we have to unroll the changing theta variable here
    for theta in np.linspace(0, 360, 21):
        sched.add(init_all)
        sched.add(x90_q0)
        # FIXME real-valued outputs are not yet supported in the pulsar backend. issue #44
        # sched.add(operation=CZ(qC=q0, qT="q1"))
        sched.add(Rxy(theta=theta, phi=0, qubit="q0"))
        sched.add(Rxy(theta=90, phi=0, qubit=q1))
        sched.add(Measure(q0, "q1"), label="M {:.2f} deg".format(theta))

    sched, cfgs = qcompile(
        sched,
        device_cfg=DEVICE_CFG,
        hardware_mapping=HARDWARE_MAPPING,
    )
    import logging

    logging.warning(sched.resources.keys())
    assert len(sched.resources["q0:mw_q0.01"].timing_tuples) == int(21 * 2)
    assert len(sched.resources["q1:mw_q1.01"].timing_tuples) == int(21 * 1)
    # flux pulses FIXME real-valued pulses not allowed yet.
    # because resources that are not used, these keys are missing
    # assert len(sched.resources['q0:fl_cl0.baseband'].timing_tuples) ==  int(21*1)
    # assert len(sched.resources['q1:fl_cl0.baseband'].timing_tuples) ==  int(21*1)

    # FIXME realtime modulation currently disabled awaiting realtime demodulation
    # assert sched.resources['q0:mw_q0.01']['nco_freq'] ==
    #   HARDWARE_MAPPING["qcm0"]["complex_output_0"]["seq0"]["nco_freq"]
    # lo_freq = HARDWARE_MAPPING["qcm0"]["complex_output_1"]["lo_freq"]
    # rf_freq = DEVICE_CFG['qubits']["q1"]["params"]["mw_freq"]
    # assert sched.resources['q1:mw_q1.01']['nco_freq'] == rf_freq - lo_freq


def test_gate_and_pulse(dummy_pulsars):
    sched = Schedule("Chevron Experiment")
    sched.add(X("q0"))
    sched.add(SquarePulse(0.8, 20e-9, "q0:mw", clock="q0.01"))
    sched.add(Rxy(90, 90, "q0"))
    sched.add(SquarePulse(0.4, 20e-9, "q0:mw", clock="q0.01"))
    sched.add_resources([ClockResource("q0.01", 6.02e9)])

    sched, cfgs = qcompile(sched, DEVICE_CFG, HARDWARE_MAPPING)
    with open(cfgs["qcm0"]["seq0"]["seq_fn"], "rb") as cfg:
        prog = json.load(cfg)
        assert len(prog["waveforms"]["awg"]) == 4


def test_bad_driver_vers():
    def subtest(device, version):
        _check_driver_version(device, version)
        error = (
            "Backend requires Pulsar Dummy to have driver version l.o.l., "
            + "found {} installed.".format(version)
        )
        with pytest.raises(ValueError, match=error):
            _check_driver_version(device, "l.o.l.")
        device.close()

    subtest(pulsar_qcm_dummy("qcm_bad_vers"), QCM_DRIVER_VER)
    subtest(pulsar_qrm_dummy("qrm_bad_vers"), QRM_DRIVER_VER)


def test_bad_hardware_mapping():
    duplicate_port_clock = {
        "backend": "quantify.scheduler.backends.pulsar_backend.pulsar_assembler_backend",
        "qcm0": {
            "name": "qcm0",
            "type": "Pulsar_QCM",
            "mode": "complex",
            "ref": "int",
            "IP address": "192.168.0.2",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_freq": None,
                "seq0": {"port": "q0:mw", "clock": "q0.01", "interm_freq": -50e6},
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "lo_freq": None,
                "seq0": {"port": "q0:mw", "clock": "q0.01", "interm_freq": -50e6},
            },
        },
    }
    with pytest.raises(
        ValueError, match="Duplicate port and clock combination: 'q0:mw' and 'q0.01'"
    ):
        _invert_hardware_mapping(duplicate_port_clock)

    bad_output = {
        "backend": "quantify.scheduler.backends.pulsar_backend.pulsar_assembler_backend",
        "qcm0": {
            "name": "qcm0",
            "type": "Pulsar_QCM",
            "mode": "blueberry",
            "IP address": "192.168.0.2",
        },
    }
    with pytest.raises(ValueError, match="Unrecognised output mode"):
        _invert_hardware_mapping(bad_output)


def test_missing_clock():
    sched = Schedule("mis")
    sched.add(SquarePulse(amp=0.8, duration=20e-9, port="q0:mw", clock="q1.01"))
    sched.add_resource(ClockResource("q1.01", freq=6.2e9))
    with pytest.raises(ValueError):
        qcompile(sched, DEVICE_CFG, HARDWARE_MAPPING)


def test_extract():
    portclock_reference = _invert_hardware_mapping(HARDWARE_MAPPING)
    assert portclock_reference == {
        "q0:mw_q0.01": ("qcm0", "complex_output_0", "seq0"),
        "q0:mw_q0.12": ("qcm0", "complex_output_0", "seq1"),
        "q1:mw_q1.01": ("qcm0", "complex_output_1", "seq0"),
        "q1:mw_q1.12": ("qcm0", "complex_output_1", "seq1"),
        "q0:res_q0.ro": ("qrm0", "complex_output_0", "seq0"),
        "q1:res_q1.ro": ("qrm0", "complex_output_0", "seq1"),
        "q2:res_q2.ro": ("qrm0", "complex_output_0", "seq2"),
        "q3:res_q3.ro": ("qrm0", "complex_output_0", "seq3"),
        "q0:fl_cl0.baseband": ("qcm1", "real_output_0", "seq0"),
        "q1:fl_cl0.baseband": ("qcm1", "real_output_1", "seq0"),
        "q2:fl_cl0.baseband": ("qcm1", "real_output_2", "seq0"),
        "c0:fl_cl0.baseband": ("qcm1", "real_output_3", "seq0"),
    }

    for func in [
        _extract_gain,
        _extract_io,
        _extract_pulsar_type,
        _extract_pulsar_config,
    ]:
        with pytest.raises(ValueError):
            func(HARDWARE_MAPPING, portclock_reference, "q0:mw", "q1.12")

    for portclock, (device_name, output, seq) in portclock_reference.items():
        port, clock = portclock.split("_")
        pulsar_cfg = _extract_pulsar_config(
            HARDWARE_MAPPING, portclock_reference, port, clock
        )
        pulsar_type = _extract_pulsar_type(
            HARDWARE_MAPPING, portclock_reference, port, clock
        )
        gain = _extract_gain(HARDWARE_MAPPING, portclock_reference, port, clock)
        io = _extract_io(HARDWARE_MAPPING, portclock_reference, port, clock)
        assert HARDWARE_MAPPING[device_name] == pulsar_cfg
        assert HARDWARE_MAPPING[device_name]["type"] == pulsar_type
        assert HARDWARE_MAPPING[device_name][output]["line_gain_db"] == gain
        assert output == io


def test_extract_interm_freq():
    inverted = _invert_hardware_mapping(HARDWARE_MAPPING)
    lo_freq, interm_freq = _extract_interm_freq(
        HARDWARE_MAPPING, inverted, port="q0:mw", clock="q0.01", clock_freq=5.32e9
    )
    assert interm_freq == -50e6  # Hardcoded in config

    lo_freq, interm_freq = _extract_interm_freq(
        HARDWARE_MAPPING, inverted, port="q0:mw", clock="q0.01", clock_freq=1.32e9
    )
    assert interm_freq == -50e6  # Hardcoded in config

    RF = 4.52e9
    LO = 4.8e9  # lo_freq set in config for output connected to q1:mw
    lo_freq, interm_freq = _extract_interm_freq(
        HARDWARE_MAPPING, inverted, port="q1:mw", clock="q1.01", clock_freq=RF
    )

    # RF = LO + IF
    assert interm_freq == RF - LO

    RF: float = 8.52e9
    LO: float = 7.2e9  # lo_freq set in config for output connected to the feedline
    lo_freq, interm_freq = _extract_interm_freq(
        HARDWARE_MAPPING, inverted, port="q1:res", clock="q1.ro", clock_freq=RF
    )
    assert interm_freq == RF - LO
    assert lo_freq == RF - interm_freq

    invalid_mapping = {
        "backend": "quantify.scheduler.backends.pulsar_backend.pulsar_assembler_backend",
        "qcm0": {
            "name": "qcm0",
            "type": "Pulsar_QCM",
            "mode": "complex",
            "ref": "int",
            "IP address": "192.168.0.2",
            "complex_output_0": {
                "line_gain_db": 0,
                "lo_freq": 6.4e9,
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": -50e6,
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "lo_freq": None,
                "seq0": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "interm_freq": None,
                },
                "seq1": {
                    "port": "q1:mw",
                    "clock": "q1.12",
                    "interm_freq": None,
                },
            },
        },
    }
    invalid_inverted = _invert_hardware_mapping(invalid_mapping)
    with pytest.raises(ValueError):
        # overconstrained example
        _extract_interm_freq(
            invalid_mapping,
            invalid_inverted,
            port="q0:mw",
            clock="q0.01",
            clock_freq=RF,
        )
    with pytest.raises(ValueError):
        # underconstrained example
        _extract_interm_freq(
            invalid_mapping,
            invalid_inverted,
            port="q1:mw",
            clock="q1.01",
            clock_freq=RF,
        )
