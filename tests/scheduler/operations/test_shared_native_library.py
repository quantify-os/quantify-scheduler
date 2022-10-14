import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.operations.shared_native_library import SpectroscopyOperation
from quantify_scheduler.compilation import device_compile, hardware_compile
from quantify_scheduler.schedules.schedule import CompiledSchedule


def test_compilation_spectroscopy_operation(mock_setup_basic_nv_qblox_hardware):
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

    mock_nv_setup = mock_setup_basic_nv_qblox_hardware
    quantum_device = mock_nv_setup["quantum_device"]
    pulse_duration = quantum_device.get_element(
        "qe0"
    ).spectroscopy_operation.duration.get()

    dev_cfg = quantum_device.generate_device_config()
    schedule_device = device_compile(schedule, dev_cfg)

    # The gate_info remains unchanged, but the pulse info has been added
    assert spec_pulse_str in schedule_device.operations
    assert "gate_info" in schedule_device.operations[spec_pulse_str]
    assert (
        schedule_device.operations[spec_pulse_str]["gate_info"]
        == schedule.operations[spec_pulse_str]["gate_info"]
    )
    assert not schedule_device.operations[spec_pulse_str]["pulse_info"] == []

    # Timing info has been added
    assert "abs_time" in schedule_device.schedulables[label1].data.keys()
    assert "abs_time" in schedule_device.schedulables[label2].data.keys()
    assert schedule_device.schedulables[label1].data["abs_time"] == 0
    duration_pulse_1 = schedule_device.operations[spec_pulse_str].data["pulse_info"][0][
        "duration"
    ]
    assert schedule_device.schedulables[label2].data["abs_time"] == pytest.approx(
        0 + duration_pulse_1
    )

    hardware_cfg = quantum_device.generate_hardware_config()
    assert not "compiled_instructions" in schedule_device.data
    schedule_hardware = hardware_compile(schedule_device, hardware_cfg)

    assert isinstance(schedule_hardware, CompiledSchedule)
    assert "compiled_instructions" in schedule_hardware.data

    assert schedule_hardware.timing_table.data.loc[0, "duration"] == pulse_duration
    assert schedule_hardware.timing_table.data.loc[1, "duration"] == pulse_duration
    assert schedule_hardware.timing_table.data.loc[1, "abs_time"] == pulse_duration
    assert schedule_hardware.timing_table.data.loc[0, "is_acquisition"] is False
    assert schedule_hardware.timing_table.data.loc[1, "is_acquisition"] is False
