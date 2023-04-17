import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.operations.gate_library import Measure, Reset
from quantify_scheduler.operations.nv_native_library import ChargeReset, CRCount
from quantify_scheduler.operations.shared_native_library import SpectroscopyOperation
from quantify_scheduler.schedules.schedule import CompiledSchedule


def test_compilation_spectroscopy_operation_qblox_hardware(
    mock_setup_basic_nv_qblox_hardware,
):
    """SpectroscopyOperation can be compiled to the device layer and to qblox
    instructions.

    Verify that the device representation and the hardware instructions contain
    plausible content.
    """
    schedule = Schedule(name="Two Spectroscopy Pulses", repetitions=1)

    label1 = "Spectroscopy pulse 1"
    label2 = "Spectroscopy pulse 2"
    _ = schedule.add(SpectroscopyOperation("qe0"), label=label1)
    _ = schedule.add(SpectroscopyOperation("qe0"), label=label2)

    # SpectroscopyOperation is added to the operations.
    # It has "gate_info", but no "pulse_info" yet.
    spec_pulse_str = str(SpectroscopyOperation("qe0"))
    assert spec_pulse_str in schedule.operations
    assert "gate_info" in schedule.operations[spec_pulse_str]
    assert schedule.operations[spec_pulse_str]["pulse_info"] == []

    # Operation is added twice to schedulables and has no timing information yet.
    assert label1 in schedule.schedulables
    assert label2 in schedule.schedulables
    assert (
        "abs_time" not in schedule.schedulables[label1].data.keys()
        or schedule.schedulables[label1].data["abs_time"] is None
    )
    assert (
        "abs_time" not in schedule.schedulables[label2].data.keys()
        or schedule.schedulables[label2].data["abs_time"] is None
    )

    # We can plot the circuit diagram
    schedule.plot_circuit_diagram()

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    pulse_duration = quantum_device.get_element(
        "qe0"
    ).spectroscopy_operation.duration.get()

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # The gate_info remains unchanged, but the pulse info has been added
    assert spec_pulse_str in compiled_sched.operations
    assert "gate_info" in compiled_sched.operations[spec_pulse_str]
    assert (
        compiled_sched.operations[spec_pulse_str]["gate_info"]
        == schedule.operations[spec_pulse_str]["gate_info"]
    )
    assert not compiled_sched.operations[spec_pulse_str]["pulse_info"] == []

    # Timing info has been added
    assert "abs_time" in compiled_sched.schedulables[label1].data.keys()
    assert "abs_time" in compiled_sched.schedulables[label2].data.keys()
    assert compiled_sched.schedulables[label1].data["abs_time"] == 0
    duration_pulse_1 = compiled_sched.operations[spec_pulse_str].data["pulse_info"][0][
        "duration"
    ]
    assert compiled_sched.schedulables[label2].data["abs_time"] == pytest.approx(
        0 + duration_pulse_1
    )

    assert isinstance(compiled_sched, CompiledSchedule)
    assert "compiled_instructions" in compiled_sched.data

    assert compiled_sched.timing_table.data.loc[0, "duration"] == pulse_duration
    assert compiled_sched.timing_table.data.loc[1, "duration"] == pulse_duration
    assert compiled_sched.timing_table.data.loc[1, "abs_time"] == pulse_duration

    assert compiled_sched.timing_table.data.loc[0, "is_acquisition"] is False
    assert compiled_sched.timing_table.data.loc[1, "is_acquisition"] is False


def test_compilation_reset_qblox_hardware(mock_setup_basic_nv_qblox_hardware):
    """_Reset can be compiled to the device layer and to qblox
    instructions.

    Verify that the device representation and the hardware instructions contain
    plausible content.
    """
    schedule = Schedule(name="Reset", repetitions=1)
    label = "reset pulse"

    _ = schedule.add(Reset("qe0"), label=label)
    reset_str = str(Reset("qe0"))

    # We can plot the circuit diagram
    schedule.plot_circuit_diagram()

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    pulse_duration = quantum_device.get_element("qe0").reset.duration.get()

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # The gate_info remains unchanged, but the pulse info has been added
    assert reset_str in compiled_sched.operations
    assert "gate_info" in compiled_sched.operations[reset_str]
    assert (
        compiled_sched.operations[reset_str]["gate_info"]
        == schedule.operations[reset_str]["gate_info"]
    )
    assert not compiled_sched.operations[reset_str]["pulse_info"] == []

    # Timing info has been added
    assert "abs_time" in compiled_sched.schedulables[label].data.keys()
    assert compiled_sched.schedulables[label].data["abs_time"] == 0

    assert isinstance(compiled_sched, CompiledSchedule)
    assert "compiled_instructions" in compiled_sched.data

    assert compiled_sched.timing_table.data.loc[0, "duration"] == pulse_duration
    assert compiled_sched.timing_table.data.loc[0, "is_acquisition"] is False


def test_compilation_measure_qblox_hardware(mock_setup_basic_nv_qblox_hardware):
    """Measure can be compiled to the device layer and to qblox
    instructions.

    Verify that the device representation and the hardware instructions contain
    plausible content.
    """
    schedule = Schedule(name="Measure", repetitions=1)
    label = "measure pulse"

    _ = schedule.add(Measure("qe0"), label=label)
    measure_str = str(Measure("qe0"))

    # We can plot the circuit diagram
    schedule.plot_circuit_diagram()

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    quantum_device.get_element("qe0").measure.acq_delay(1e-7)

    pulse_duration = quantum_device.get_element("qe0").measure.pulse_duration()
    acq_duration = quantum_device.get_element("qe0").measure.acq_duration()

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # The gate_info and acquisition_info remains unchanged, but the pulse info has been
    # added
    assert measure_str in compiled_sched.operations
    assert "gate_info" in compiled_sched.operations[measure_str]
    assert (
        compiled_sched.operations[measure_str]["gate_info"]
        == schedule.operations[measure_str]["gate_info"]
    )
    assert "acquisition_info" in compiled_sched.operations[measure_str]
    acquisition_info = compiled_sched.operations[measure_str]["acquisition_info"][0]
    assert acquisition_info["t0"] == 1e-7
    assert acquisition_info["protocol"] == "TriggerCount"

    assert len(compiled_sched.operations[measure_str]["pulse_info"]) > 0

    # Timing info has been added
    assert "abs_time" in compiled_sched.schedulables[label].data.keys()
    assert compiled_sched.schedulables[label].data["abs_time"] == 0

    assert isinstance(compiled_sched, CompiledSchedule)
    assert "compiled_instructions" in compiled_sched.data

    assert compiled_sched.timing_table.data.loc[0, "duration"] == pulse_duration
    assert compiled_sched.timing_table.data.loc[0, "is_acquisition"] is False
    assert compiled_sched.timing_table.data.loc[2, "duration"] == acq_duration
    assert compiled_sched.timing_table.data.loc[2, "is_acquisition"] is True


def test_compilation_charge_reset_qblox_hardware(mock_setup_basic_nv_qblox_hardware):
    """ChargeReset can be compiled to the device layer and to qblox
    instructions.

    Verify that the device representation and the hardware instructions contain
    plausible content.
    """
    schedule = Schedule(name="ChargeReset", repetitions=1)
    label = "charge reset pulse"

    _ = schedule.add(ChargeReset("qe0"), label=label)
    charge_reset_str = str(ChargeReset("qe0"))

    # We can plot the circuit diagram
    schedule.plot_circuit_diagram()

    quantum_device = mock_setup_basic_nv_qblox_hardware["quantum_device"]
    pulse_duration = quantum_device.get_element("qe0").charge_reset.duration.get()

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # The gate_info remains unchanged, but the pulse info has been added
    assert charge_reset_str in compiled_sched.operations
    assert "gate_info" in compiled_sched.operations[charge_reset_str]
    assert (
        compiled_sched.operations[charge_reset_str]["gate_info"]
        == schedule.operations[charge_reset_str]["gate_info"]
    )
    assert not compiled_sched.operations[charge_reset_str]["pulse_info"] == []

    # Timing info has been added
    assert "abs_time" in compiled_sched.schedulables[label].data.keys()
    assert compiled_sched.schedulables[label].data["abs_time"] == 0

    assert isinstance(compiled_sched, CompiledSchedule)
    assert "compiled_instructions" in compiled_sched.data

    assert compiled_sched.timing_table.data.loc[0, "duration"] == pulse_duration
    assert compiled_sched.timing_table.data.loc[0, "is_acquisition"] is False


def test_compilation_cr_count_qblox_hardware(mock_setup_basic_nv):
    """cr_count can be compiled to the device layer and to qblox
    instructions.

    Verify that the device representation and the hardware instructions contain
    plausible content.
    """
    schedule = Schedule(name="cr_count", repetitions=1)
    label = "cr_count pulse"

    _ = schedule.add(CRCount("qe0"), label=label)
    cr_count_str = str(CRCount("qe0"))

    # We can plot the circuit diagram
    schedule.plot_circuit_diagram()

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.get_element("qe0").cr_count.acq_delay(1e-8)

    pulse_duration = quantum_device.get_element("qe0").measure.pulse_duration()
    acq_duration = quantum_device.get_element("qe0").measure.acq_duration()

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # The gate_info and acquisition_info remains unchanged, but the pulse info has been
    # added
    assert cr_count_str in compiled_sched.operations
    assert "gate_info" in compiled_sched.operations[cr_count_str]
    assert (
        compiled_sched.operations[cr_count_str]["gate_info"]
        == schedule.operations[cr_count_str]["gate_info"]
    )
    assert "acquisition_info" in compiled_sched.operations[cr_count_str]
    acquisition_info = compiled_sched.operations[cr_count_str]["acquisition_info"][0]
    assert acquisition_info["t0"] == 1e-8
    assert acquisition_info["protocol"] == "TriggerCount"

    assert len(compiled_sched.operations[cr_count_str]["pulse_info"]) > 0

    # Timing info has been added
    assert "abs_time" in compiled_sched.schedulables[label].data.keys()
    assert compiled_sched.schedulables[label].data["abs_time"] == 0

    assert isinstance(compiled_sched, CompiledSchedule)
    assert "compiled_instructions" in compiled_sched.data
    assert compiled_sched.timing_table.data.loc[0, "duration"] == pulse_duration
    assert compiled_sched.timing_table.data.loc[0, "is_acquisition"] is False
    assert compiled_sched.timing_table.data.loc[2, "duration"] == acq_duration
    assert compiled_sched.timing_table.data.loc[2, "is_acquisition"] is True
