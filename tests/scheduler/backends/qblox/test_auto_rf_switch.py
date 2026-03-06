# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox backend auto RF switch dressing."""

import numpy as np
import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends.graph_compilation import SerialCompilationConfig
from quantify_scheduler.backends.qblox.auto_rf_switch import (
    MicrowavePulseInfo,
    VoltageOffsetEvent,
    _collect_all_mw_events_recursive,
    _collect_all_mw_pulses_recursive,
    _collect_microwave_pulses,
    _merge_pulses_into_windows,
    _voltage_offset_events_to_pulses,
    auto_rf_switch_dressing,
)
from quantify_scheduler.backends.qblox.operations.rf_switch_toggle import RFSwitchToggle
from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig
from quantify_scheduler.backends.types.qblox import QbloxCompilerOptions
from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.operations.pulse_library import SquarePulse, VoltageOffset


@pytest.fixture
def hardware_cfg_rf_simple():
    """Simple hardware config for RF testing."""
    return {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {"instrument_type": "QCM_RF", "rf_output_on": False},
                },
                "ref": "internal",
            }
        },
        "hardware_options": {"modulation_frequencies": {"q0:mw-q0.01": {"interm_freq": 50e6}}},
        "connectivity": {"graph": [["cluster0.module1.complex_output_0", "q0:mw"]]},
    }


class TestMergePulsesIntoWindows:
    """Tests for the _merge_pulses_into_windows function."""

    def test_single_pulse(self):
        """Test with a single pulse."""
        pulses = [
            MicrowavePulseInfo(
                schedulable_key="a",
                abs_time=0.0,
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            )
        ]
        windows = _merge_pulses_into_windows(pulses, ramp_buffer=20e-9)
        assert len(windows) == 1
        assert windows[0] == (0.0, 20e-9)

    def test_two_pulses_merged(self):
        """Test that two pulses within 2*ramp_buffer are merged."""
        ramp_buffer = 20e-9
        pulses = [
            MicrowavePulseInfo(
                schedulable_key="a",
                abs_time=0.0,
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            ),
            MicrowavePulseInfo(
                schedulable_key="b",
                abs_time=50e-9,  # gap of 30ns, within 2*20ns=40ns
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            ),
        ]
        windows = _merge_pulses_into_windows(pulses, ramp_buffer=ramp_buffer)
        assert len(windows) == 1
        assert windows[0][0] == 0.0
        assert abs(windows[0][1] - 70e-9) < 1e-15

    def test_two_pulses_separate(self):
        """Test that two pulses far apart create separate windows."""
        ramp_buffer = 20e-9
        pulses = [
            MicrowavePulseInfo(
                schedulable_key="a",
                abs_time=0.0,
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            ),
            MicrowavePulseInfo(
                schedulable_key="b",
                abs_time=100e-9,  # gap of 80ns, larger than 2*20ns=40ns
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            ),
        ]
        windows = _merge_pulses_into_windows(pulses, ramp_buffer=ramp_buffer)
        assert len(windows) == 2
        assert windows[0] == (0.0, 20e-9)
        assert windows[1] == (100e-9, 120e-9)

    def test_three_pulses_two_merged_one_separate(self):
        """Test that first two pulses merge but third is separate."""
        ramp_buffer = 20e-9
        pulses = [
            MicrowavePulseInfo(
                schedulable_key="a",
                abs_time=0.0,
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            ),
            MicrowavePulseInfo(
                schedulable_key="b",
                abs_time=30e-9,  # gap of 10ns, within 40ns
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            ),
            MicrowavePulseInfo(
                schedulable_key="c",
                abs_time=200e-9,  # large gap
                duration=20e-9,
                port="q0:mw",
                clock="q0.01",
            ),
        ]
        windows = _merge_pulses_into_windows(pulses, ramp_buffer=ramp_buffer)
        assert len(windows) == 2
        # first two merged
        assert windows[0][0] == 0.0
        assert abs(windows[0][1] - 50e-9) < 1e-15
        # third separate
        assert abs(windows[1][0] - 200e-9) < 1e-15
        assert abs(windows[1][1] - 220e-9) < 1e-15

    def test_empty_list(self):
        """Test with empty pulse list."""
        windows = _merge_pulses_into_windows([], ramp_buffer=20e-9)
        assert windows == []


class TestCollectMicrowavePulses:
    """Tests for the _collect_microwave_pulses function."""

    def test_collects_mw_pulses(self):
        """Test that microwave pulses are collected correctly."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        pulses = _collect_microwave_pulses(schedule)
        assert len(pulses) == 1
        assert pulses[0].port == "q0:mw"
        assert pulses[0].clock == "q0.01"
        assert pulses[0].duration == 20e-9

    def test_ignores_non_mw_pulses(self):
        """Test that non-microwave pulses are ignored."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:fl", clock="cl0.baseband"))
        schedule = _determine_absolute_timing(schedule)

        pulses = _collect_microwave_pulses(schedule)
        assert len(pulses) == 0

    def test_ignores_rf_switch_toggle(self):
        """Test that RFSwitchToggle operations are not collected."""
        schedule = Schedule("test")
        schedule.add(RFSwitchToggle(duration=100e-9, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        pulses = _collect_microwave_pulses(schedule)
        assert len(pulses) == 0


class TestAutoRfSwitchDressing:
    """Tests for the auto_rf_switch_dressing function."""

    def test_skipped_when_disabled(self, hardware_cfg_rf_simple):
        """Test that pass is skipped when auto_rf_switch is False (default)."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        # Default is auto_rf_switch=False
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        original_schedulable_count = len(schedule.schedulables)
        result = auto_rf_switch_dressing(schedule, config)

        # No RF switch operations should be added
        assert len(result.schedulables) == original_schedulable_count

    def test_skipped_when_compiler_options_none(self, hardware_cfg_rf_simple):
        """Test that pass is skipped when compiler_options is None."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        assert hardware_cfg.compiler_options is None

        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        original_schedulable_count = len(schedule.schedulables)
        result = auto_rf_switch_dressing(schedule, config)

        # No RF switch operations should be added
        assert len(result.schedulables) == original_schedulable_count

    def test_rf_switch_inserted_for_mw_pulse(self, hardware_cfg_rf_simple):
        """Test that RFSwitchToggle is inserted for microwave pulses."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": 20e-9,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        # Should have original pulse + RF switch toggle
        assert len(result.schedulables) == 2

        # Find the RF switch operation
        rf_switch_ops = [op for op in result.operations.values() if isinstance(op, RFSwitchToggle)]
        assert len(rf_switch_ops) == 1

        # Check RF switch duration includes ramp buffers
        # Since pulse starts at t=0, RF switch starts at max(0, 0-20ns) = 0
        # Duration should be: pulse_duration + ramp_buffer = 40ns
        # (can't include pre-ramp buffer since we can't start before t=0)
        rf_switch = rf_switch_ops[0]
        expected_duration = 20e-9 + 20e-9  # pulse + post-buffer
        assert abs(rf_switch.duration - expected_duration) < 1e-12

    def test_adjacent_pulses_merged(self, hardware_cfg_rf_simple):
        """Test that adjacent pulses are merged into single RF switch window."""
        schedule = Schedule("test")
        # Two pulses within 2*ramp_buffer of each other
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        schedule.add(
            SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"),
            rel_time=30e-9,  # gap of 10ns after first pulse ends
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": 20e-9,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        # Find the RF switch operations
        rf_switch_ops = [op for op in result.operations.values() if isinstance(op, RFSwitchToggle)]
        # Should have only 1 RF switch for both merged pulses
        assert len(rf_switch_ops) == 1

    def test_separate_windows_for_far_apart_pulses(self, hardware_cfg_rf_simple):
        """Test separate RF switch windows for pulses far apart."""
        schedule = Schedule("test")
        # First pulse at t=0
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        # Second pulse far away (gap > 2*ramp_buffer)
        schedule.add(
            SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"),
            rel_time=100e-9,  # large gap
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": 20e-9,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        # Find the RF switch operations
        rf_switch_ops = [op for op in result.operations.values() if isinstance(op, RFSwitchToggle)]
        # Should have 2 separate RF switch operations
        assert len(rf_switch_ops) == 2

    def test_no_rf_switch_for_non_mw_pulses(self, hardware_cfg_rf_simple):
        """Test that non-microwave pulses don't get RF switch."""
        schedule = Schedule("test")
        # Add a flux pulse (not microwave)
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:fl", clock="cl0.baseband"))
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": 20e-9,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        # Should have only the original flux pulse, no RF switch
        rf_switch_ops = [op for op in result.operations.values() if isinstance(op, RFSwitchToggle)]
        assert len(rf_switch_ops) == 0


class TestQbloxCompilerOptions:
    """Tests for the QbloxCompilerOptions dataclass."""

    def test_default_values(self):
        """Test default values of QbloxCompilerOptions."""
        options = QbloxCompilerOptions()
        assert options.auto_rf_switch is False
        assert options.auto_rf_switch_ramp_buffer == 20e-9

    def test_custom_values(self):
        """Test setting custom values."""
        options = QbloxCompilerOptions(
            auto_rf_switch=True,
            auto_rf_switch_ramp_buffer=30e-9,
        )
        assert options.auto_rf_switch is True
        assert options.auto_rf_switch_ramp_buffer == 30e-9

    def test_from_dict(self):
        """Test creating from dictionary."""
        options = QbloxCompilerOptions.model_validate(
            {
                "auto_rf_switch": True,
                "auto_rf_switch_ramp_buffer": 25e-9,
            }
        )
        assert options.auto_rf_switch is True
        assert options.auto_rf_switch_ramp_buffer == 25e-9


class TestAutoRfSwitchTiming:
    """Tests for verifying correct timing of RF switch operations.

    These tests are inspired by qblox-scheduler's test_switch_outputs.py.
    """

    def _get_rf_switch_schedulables(self, schedule):
        """Helper to get RF switch schedulables with their operations."""
        rf_switches = []
        for _schedulable_key, schedulable in schedule.schedulables.items():
            operation = schedule.operations.get(schedulable["operation_id"])
            if isinstance(operation, RFSwitchToggle):
                rf_switches.append((schedulable, operation))
        return rf_switches

    def _get_pulse_schedulables(self, schedule):
        """Helper to get pulse schedulables (non-RF switch)."""
        pulses = []
        for _schedulable_key, schedulable in schedule.schedulables.items():
            operation = schedule.operations.get(schedulable["operation_id"])
            if isinstance(operation, SquarePulse):
                pulses.append((schedulable, operation))
        return pulses

    def test_timing_single_pulse_with_headroom(self, hardware_cfg_rf_simple):
        """Test RF switch timing for a single pulse.

        First schedulable gets abs_time=0 after _determine_absolute_timing,
        so pulse is at 0; RF switch starts at 0 (no room before) and spans
        pulse + post ramp_buffer.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9

        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1
        rf_schedulable, rf_operation = rf_switches[0]

        expected_rf_start = 0.0
        np.testing.assert_allclose(
            rf_schedulable.data["abs_time"],
            expected_rf_start,
            err_msg="RF switch should start at 0 when pulse is first",
        )

        expected_rf_duration = pulse_duration + ramp_buffer
        np.testing.assert_allclose(
            rf_operation.duration,
            expected_rf_duration,
            err_msg="RF switch duration should be pulse + post buffer",
        )

    def test_timing_single_pulse_at_start(self, hardware_cfg_rf_simple):
        """Test RF switch timing when pulse starts at t=0.

        When there is no room before the pulse, RF switch starts at t=0
        and only includes post-buffer.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9

        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1
        rf_schedulable, rf_operation = rf_switches[0]

        np.testing.assert_allclose(
            rf_schedulable.data["abs_time"],
            0.0,
            err_msg="RF switch should start at t=0 when pulse is at start",
        )

        # RF switch duration: pulse + post-buffer (no pre-buffer room)
        expected_rf_duration = pulse_duration + ramp_buffer
        np.testing.assert_allclose(
            rf_operation.duration,
            expected_rf_duration,
            err_msg="RF switch duration should be pulse + post-buffer only",
        )

    def test_timing_single_pulse_with_t0_offset(self, hardware_cfg_rf_simple):
        """Test RF switch timing when pulse has t0 offset.

        The t0 offset in pulse_info is accounted for; schedulable at 0 with t0=4e-9
        gives actual pulse start 4e-9, so RF starts at max(0, 4e-9 - ramp_buffer)=0.
        """
        ramp_buffer = 20e-9
        pulse_t0 = 4e-9
        pulse_duration = 20e-9

        schedule = Schedule("test")
        schedule.add(
            SquarePulse(
                amp=0.5,
                duration=pulse_duration,
                port="q0:mw",
                clock="q0.01",
                t0=pulse_t0,
            )
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1
        rf_schedulable, rf_operation = rf_switches[0]

        expected_rf_start = 0.0
        np.testing.assert_allclose(
            rf_schedulable.data["abs_time"],
            expected_rf_start,
            err_msg="RF switch should start at 0 (t0 within ramp)",
        )
        expected_rf_duration = pulse_t0 + pulse_duration + ramp_buffer
        np.testing.assert_allclose(
            rf_operation.duration,
            expected_rf_duration,
            err_msg="RF switch duration should cover t0 + pulse + post buffer",
        )

    def test_timing_merged_pulses(self, hardware_cfg_rf_simple):
        """Test RF switch timing when pulses are merged.

        First pulse at 0, second at 0 + 20e-9 + 30e-9 = 50e-9 (gap after first).
        Single window 0--70e-9; RF switch 0 to 90e-9.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        gap = 30e-9

        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))
        schedule.add(
            SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"),
            rel_time=gap,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1, "Merged pulses should have single RF switch"
        rf_schedulable, rf_operation = rf_switches[0]

        expected_rf_start = 0.0
        np.testing.assert_allclose(
            rf_schedulable.data["abs_time"],
            expected_rf_start,
            err_msg="RF switch should start before first pulse",
        )

        second_pulse_end = pulse_duration + gap + pulse_duration
        expected_rf_end = second_pulse_end + ramp_buffer
        actual_rf_end = rf_schedulable.data["abs_time"] + rf_operation.duration

        np.testing.assert_allclose(
            actual_rf_end,
            expected_rf_end,
            err_msg="RF switch should end after second pulse + buffer",
        )

    def test_timing_separate_pulses(self, hardware_cfg_rf_simple):
        """Test RF switch timing when pulses are far apart.

        First pulse at 0, second at 20e-9 + 100e-9 = 120e-9. Two RF switches.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        gap = 100e-9

        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))
        schedule.add(
            SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"),
            rel_time=gap,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 2, "Separate pulses should have separate RF switches"

        rf_switches_sorted = sorted(rf_switches, key=lambda x: x[0].data["abs_time"])

        rf1_schedulable, rf1_operation = rf_switches_sorted[0]
        expected_rf1_start = 0.0
        np.testing.assert_allclose(
            rf1_schedulable.data["abs_time"],
            expected_rf1_start,
            err_msg="First RF switch should start before first pulse",
        )
        expected_rf1_duration = pulse_duration + ramp_buffer
        np.testing.assert_allclose(
            rf1_operation.duration,
            expected_rf1_duration,
            err_msg="First RF switch (pulse at 0) has pulse + post buffer only",
        )

        rf2_schedulable, rf2_operation = rf_switches_sorted[1]
        second_pulse_start = pulse_duration + gap
        expected_rf2_start = second_pulse_start - ramp_buffer
        np.testing.assert_allclose(
            rf2_schedulable.data["abs_time"],
            expected_rf2_start,
            err_msg="Second RF switch should start before second pulse",
        )
        expected_rf2_duration = ramp_buffer + pulse_duration + ramp_buffer
        np.testing.assert_allclose(
            rf2_operation.duration,
            expected_rf2_duration,
            err_msg="Second RF switch should have full duration",
        )

    def test_timing_boundary_merge(self, hardware_cfg_rf_simple):
        """Test RF switch merging at exactly 2*ramp_buffer gap.

        First pulse at 0 (end 20e-9), second at 60e-9 (gap 40e-9). Merge.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        gap = 2 * ramp_buffer

        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))
        schedule.add(
            SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"),
            rel_time=gap,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1, "Pulses at exactly 2*ramp_buffer gap should merge"

    def test_timing_just_over_boundary_no_merge(self, hardware_cfg_rf_simple):
        """Test RF switch not merging when gap is just over 2*ramp_buffer.

        First pulse at 0, second at 20e-9 + (40e-9+1e-9) = 61e-9. No merge.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        gap = 2 * ramp_buffer + 1e-9

        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))
        schedule.add(
            SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"),
            rel_time=gap,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 2, "Pulses over 2*ramp_buffer gap should not merge"

    def test_timing_three_pulses_partial_merge(self, hardware_cfg_rf_simple):
        """Test RF switch with three pulses where first two merge but third is separate.

        First at 0, second at 50e-9, third at 170e-9. Windows (0,70e-9) and (170e-9,190e-9).
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        small_gap = 30e-9
        large_gap = 100e-9

        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))
        schedule.add(
            SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"),
            rel_time=small_gap,
        )
        schedule.add(
            SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"),
            rel_time=large_gap,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg_rf_simple["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        hardware_cfg = QbloxHardwareCompilationConfig.model_validate(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 2, "Should have 2 RF switches (1 merged, 1 separate)"

        rf_switches_sorted = sorted(rf_switches, key=lambda x: x[0].data["abs_time"])

        rf1_schedulable, rf1_operation = rf_switches_sorted[0]
        second_pulse_end = pulse_duration + small_gap + pulse_duration
        expected_rf1_end = second_pulse_end + ramp_buffer
        actual_rf1_end = rf1_schedulable.data["abs_time"] + rf1_operation.duration

        np.testing.assert_allclose(
            actual_rf1_end,
            expected_rf1_end,
            err_msg="First RF switch should end after second pulse",
        )

        rf2_schedulable, rf2_operation = rf_switches_sorted[1]
        third_pulse_start = second_pulse_end + large_gap
        expected_rf2_start = third_pulse_start - ramp_buffer
        np.testing.assert_allclose(
            rf2_schedulable.data["abs_time"],
            expected_rf2_start,
            err_msg="Second RF switch should start before third pulse",
        )


class TestCollectAllMwPulsesRecursive:
    """Tests for _collect_all_mw_pulses_recursive."""

    def test_collects_regular_mw_pulse(self):
        """Regular MW pulses are collected with correct abs_time."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        pulses = _collect_all_mw_pulses_recursive(schedule, time_offset=0.0)
        assert len(pulses) == 1
        assert pulses[0].port == "q0:mw"
        assert pulses[0].duration == 20e-9

    def test_collects_from_sub_schedule_with_global_time(self):
        """MW pulses inside a sub-schedule get global abs_time."""
        sub_sched = Schedule("sub")
        sub_sched.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))

        root = Schedule("root")
        # Flux pulse occupies [0, 100e-9]; sub-schedule placed right after (rel_time=0
        # means start-of-sub aligned with end of flux pulse, i.e. at 100e-9).
        root.add(SquarePulse(amp=0.0, duration=100e-9, port="q0:fl", clock="cl0.baseband"))
        root.add(sub_sched, rel_time=0)
        root = _determine_absolute_timing(root)

        pulses = _collect_all_mw_pulses_recursive(root, time_offset=0.0)
        assert len(pulses) == 1
        # The sub-schedule starts at 100e-9, so the pulse should have abs_time=100e-9
        assert abs(pulses[0].abs_time - 100e-9) < 1e-12

    def test_ignores_non_mw_ports(self):
        """Flux pulses are not collected."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:fl", clock="cl0.baseband"))
        schedule = _determine_absolute_timing(schedule)

        pulses = _collect_all_mw_pulses_recursive(schedule, time_offset=0.0)
        assert len(pulses) == 0

    def test_voltage_offset_ignored(self):
        """VoltageOffset operations (wf_func=None) are not collected as pulses."""
        from quantify_scheduler.operations.pulse_library import VoltageOffset

        schedule = Schedule("test")
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01")
        )
        schedule = _determine_absolute_timing(schedule)

        pulses = _collect_all_mw_pulses_recursive(schedule, time_offset=0.0)
        assert len(pulses) == 0


class TestAutoRfSwitchSubSchedules:
    """Tests verifying that RF switch ops are placed on the root schedule."""

    def _enable_rf_switch(self, hardware_cfg, ramp_buffer=20e-9):
        hardware_cfg["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        return QbloxHardwareCompilationConfig.model_validate(hardware_cfg)

    def _get_rf_switch_schedulables(self, schedule):
        rf_switches = []
        for _schedulable_key, schedulable in schedule.schedulables.items():
            operation = schedule.operations.get(schedulable["operation_id"])
            if isinstance(operation, RFSwitchToggle):
                rf_switches.append((schedulable, operation))
        return rf_switches

    def test_rf_switch_on_root_not_sub_schedule(self, hardware_cfg_rf_simple):
        """RF switch ops must appear on the root schedule, not inside the sub-schedule.

        Previously the recursive approach inserted RF switch ops inside the sub-schedule,
        which could extend its duration and cause timing problems.
        """
        sub_sched = Schedule("inner")
        sub_sched.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))

        root = Schedule("root")
        root.add(sub_sched)
        root = _determine_absolute_timing(root)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(root, config)

        # RF switch must be on the root schedule
        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1

        # Verify it is NOT inside the sub-schedule
        sub_sched_ops = result.operations
        for op in sub_sched_ops.values():
            from quantify_scheduler.schedules.schedule import ScheduleBase

            if isinstance(op, ScheduleBase):
                for inner_schedulable in op.schedulables.values():
                    inner_op = op.operations.get(inner_schedulable["operation_id"])
                    assert not isinstance(inner_op, RFSwitchToggle), (
                        "RF switch should not be inside sub-schedule"
                    )

    def test_sub_schedule_duration_not_extended(self, hardware_cfg_rf_simple):
        """Inserting RF switch on root must not change the sub-schedule duration."""
        pulse_duration = 20e-9

        sub_sched = Schedule("inner")
        sub_sched.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))

        root = Schedule("root")
        root.add(sub_sched)
        root = _determine_absolute_timing(root)
        # Capture duration after timing is determined (before RF switch insertion)
        original_sub_duration = sub_sched.duration

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(root, config)

        # Find sub-schedule in result and verify its duration is unchanged
        for op in result.operations.values():
            from quantify_scheduler.schedules.schedule import ScheduleBase

            if isinstance(op, ScheduleBase) and op.name == "inner":
                assert op.duration == original_sub_duration, (
                    "Sub-schedule duration must not be changed by RF switch insertion"
                )

    def test_consecutive_sub_schedules_windows_merged(self, hardware_cfg_rf_simple):
        """MW pulses from two consecutive sub-schedules are merged when close enough.

        Previously, sub-schedules were processed independently, so windows at the
        boundary between sub-schedules could not be merged.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        # Gap between end of first sub-schedule and start of second is small
        gap_between_sub_schedules = 30e-9  # < 2 * ramp_buffer → should merge

        sub1 = Schedule("sub1")
        sub1.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))

        sub2 = Schedule("sub2")
        sub2.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))

        root = Schedule("root")
        root.add(sub1)
        root.add(sub2, rel_time=gap_between_sub_schedules)
        root = _determine_absolute_timing(root)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple, ramp_buffer)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(root, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1, (
            "Pulses from consecutive sub-schedules within 2*ramp_buffer should be merged"
        )

    def test_consecutive_sub_schedules_separate_windows(self, hardware_cfg_rf_simple):
        """MW pulses from two sub-schedules far apart create separate RF switch windows."""
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        gap_between_sub_schedules = 200e-9  # > 2 * ramp_buffer → separate

        sub1 = Schedule("sub1")
        sub1.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))

        sub2 = Schedule("sub2")
        sub2.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))

        root = Schedule("root")
        root.add(sub1)
        root.add(sub2, rel_time=gap_between_sub_schedules)
        root = _determine_absolute_timing(root)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple, ramp_buffer)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(root, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 2, (
            "Pulses from sub-schedules far apart should produce separate RF switch windows"
        )

    def test_rf_switch_abs_time_correct_for_sub_schedule_pulse(self, hardware_cfg_rf_simple):
        """RF switch timing is correct for a pulse inside a sub-schedule.

        The sub-schedule starts at 100e-9 (after a flux pulse), so the RF switch
        should start relative to the global timeline, not the sub-schedule's timeline.
        """
        ramp_buffer = 20e-9
        pulse_duration = 20e-9
        sub_start = 100e-9

        sub_sched = Schedule("inner")
        sub_sched.add(SquarePulse(amp=0.5, duration=pulse_duration, port="q0:mw", clock="q0.01"))

        root = Schedule("root")
        root.add(SquarePulse(amp=0.0, duration=sub_start, port="q0:fl", clock="cl0.baseband"))
        root.add(sub_sched, rel_time=0)
        root = _determine_absolute_timing(root)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple, ramp_buffer)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(root, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1
        rf_schedulable, rf_operation = rf_switches[0]

        expected_rf_start = sub_start - ramp_buffer  # 80e-9
        np.testing.assert_allclose(
            rf_schedulable.data["abs_time"],
            expected_rf_start,
            err_msg="RF switch should start ramp_buffer before the pulse's global time",
        )
        expected_rf_end = sub_start + pulse_duration + ramp_buffer  # 140e-9
        actual_rf_end = rf_schedulable.data["abs_time"] + rf_operation.duration
        np.testing.assert_allclose(
            actual_rf_end,
            expected_rf_end,
            err_msg="RF switch should end ramp_buffer after the pulse ends globally",
        )


class TestVoltageOffsetEventsToPulses:
    """Tests for _voltage_offset_events_to_pulses."""

    def test_single_on_off_pair(self):
        """Non-zero then zero event creates one window."""
        events = [
            VoltageOffsetEvent(abs_time=0.0, port="q0:mw", clock="q0.01", is_nonzero=True),
            VoltageOffsetEvent(abs_time=500e-9, port="q0:mw", clock="q0.01", is_nonzero=False),
        ]
        pulses = _voltage_offset_events_to_pulses(events)
        assert len(pulses) == 1
        assert pulses[0].abs_time == 0.0
        assert abs(pulses[0].duration - 500e-9) < 1e-15
        assert pulses[0].port == "q0:mw"
        assert pulses[0].clock == "q0.01"

    def test_multiple_nonzero_before_close(self):
        """Multiple non-zero events before close: window starts at first non-zero."""
        events = [
            VoltageOffsetEvent(abs_time=0.0, port="q0:mw", clock="q0.01", is_nonzero=True),
            VoltageOffsetEvent(abs_time=100e-9, port="q0:mw", clock="q0.01", is_nonzero=True),
            VoltageOffsetEvent(abs_time=500e-9, port="q0:mw", clock="q0.01", is_nonzero=False),
        ]
        pulses = _voltage_offset_events_to_pulses(events)
        assert len(pulses) == 1
        assert pulses[0].abs_time == 0.0
        assert abs(pulses[0].duration - 500e-9) < 1e-15

    def test_two_on_off_pairs(self):
        """Two separate on/off pairs create two windows."""
        events = [
            VoltageOffsetEvent(abs_time=0.0, port="q0:mw", clock="q0.01", is_nonzero=True),
            VoltageOffsetEvent(abs_time=200e-9, port="q0:mw", clock="q0.01", is_nonzero=False),
            VoltageOffsetEvent(abs_time=1000e-9, port="q0:mw", clock="q0.01", is_nonzero=True),
            VoltageOffsetEvent(abs_time=1500e-9, port="q0:mw", clock="q0.01", is_nonzero=False),
        ]
        pulses = _voltage_offset_events_to_pulses(events)
        assert len(pulses) == 2
        assert pulses[0].abs_time == 0.0
        assert abs(pulses[0].duration - 200e-9) < 1e-15
        assert abs(pulses[1].abs_time - 1000e-9) < 1e-15
        assert abs(pulses[1].duration - 500e-9) < 1e-15

    def test_unclosed_window_is_ignored(self):
        """A non-zero event without a subsequent zero event creates no window."""
        events = [
            VoltageOffsetEvent(abs_time=0.0, port="q0:mw", clock="q0.01", is_nonzero=True),
        ]
        pulses = _voltage_offset_events_to_pulses(events)
        assert len(pulses) == 0

    def test_empty_events(self):
        """Empty list returns empty list."""
        assert _voltage_offset_events_to_pulses([]) == []

    def test_two_ports_independent(self):
        """Events on different ports create independent windows."""
        events = [
            VoltageOffsetEvent(abs_time=0.0, port="q0:mw", clock="q0.01", is_nonzero=True),
            VoltageOffsetEvent(abs_time=0.0, port="q1:mw", clock="q1.01", is_nonzero=True),
            VoltageOffsetEvent(abs_time=300e-9, port="q0:mw", clock="q0.01", is_nonzero=False),
            VoltageOffsetEvent(abs_time=600e-9, port="q1:mw", clock="q1.01", is_nonzero=False),
        ]
        pulses = _voltage_offset_events_to_pulses(events)
        assert len(pulses) == 2
        by_port = {p.port: p for p in pulses}
        assert abs(by_port["q0:mw"].duration - 300e-9) < 1e-15
        assert abs(by_port["q1:mw"].duration - 600e-9) < 1e-15


class TestCollectAllMwEventsRecursive:
    """Tests for _collect_all_mw_events_recursive."""

    def test_collects_regular_mw_pulse(self):
        """Regular MW pulses are returned as MicrowavePulseInfo."""
        schedule = Schedule("test")
        schedule.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        schedule = _determine_absolute_timing(schedule)

        pulses, events = _collect_all_mw_events_recursive(schedule, time_offset=0.0)
        assert len(pulses) == 1
        assert pulses[0].port == "q0:mw"
        assert pulses[0].duration == 20e-9
        assert len(events) == 0

    def test_collects_voltage_offset_events(self):
        """VoltageOffset operations are collected as VoltageOffsetEvents."""
        schedule = Schedule("test")
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01")
        )
        schedule.add(
            VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="q0:mw", clock="q0.01"),
            rel_time=500e-9,
        )
        schedule = _determine_absolute_timing(schedule)

        pulses, events = _collect_all_mw_events_recursive(schedule, time_offset=0.0)
        assert len(pulses) == 0
        assert len(events) == 2
        events_sorted = sorted(events, key=lambda e: e.abs_time)
        assert events_sorted[0].is_nonzero is True
        assert events_sorted[1].is_nonzero is False

    def test_voltage_offset_abs_time(self):
        """VoltageOffset events have correct abs_time from the schedulable."""
        schedule = Schedule("test")
        # Add a dummy pulse first to shift timing
        schedule.add(SquarePulse(amp=0.0, duration=100e-9, port="q0:fl", clock="cl0.baseband"))
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01"),
            rel_time=0,
        )
        schedule = _determine_absolute_timing(schedule)

        _, events = _collect_all_mw_events_recursive(schedule, time_offset=0.0)
        assert len(events) == 1
        assert abs(events[0].abs_time - 100e-9) < 1e-12

    def test_ignores_non_mw_voltage_offset(self):
        """VoltageOffset on a non-mw port is not collected."""
        schedule = Schedule("test")
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:fl", clock="cl0.baseband")
        )
        schedule = _determine_absolute_timing(schedule)

        pulses, events = _collect_all_mw_events_recursive(schedule, time_offset=0.0)
        assert len(pulses) == 0
        assert len(events) == 0


class TestAutoRfSwitchVoltageOffset:
    """Tests for RF switch scheduling with VoltageOffset (stitched pulses)."""

    def _enable_rf_switch(self, hardware_cfg, ramp_buffer=20e-9):
        hardware_cfg["compiler_options"] = {
            "auto_rf_switch": True,
            "auto_rf_switch_ramp_buffer": ramp_buffer,
        }
        return QbloxHardwareCompilationConfig.model_validate(hardware_cfg)

    def _get_rf_switch_schedulables(self, schedule):
        rf_switches = []
        for _schedulable_key, schedulable in schedule.schedulables.items():
            operation = schedule.operations.get(schedulable["operation_id"])
            if isinstance(operation, RFSwitchToggle):
                rf_switches.append((schedulable, operation))
        return rf_switches

    def test_rf_switch_inserted_for_stitched_pulse(self, hardware_cfg_rf_simple):
        """RF switch is inserted for a VoltageOffset on/off pair."""
        schedule = Schedule("test")
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01")
        )
        schedule.add(
            VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="q0:mw", clock="q0.01"),
            rel_time=500e-9,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1
        rf_schedulable, rf_operation = rf_switches[0]

        # RF switch should cover the stitched pulse window + ramp buffers.
        # Window is [0, 500e-9]; start = max(0, 0 - 20e-9) = 0.
        np.testing.assert_allclose(rf_schedulable.data["abs_time"], 0.0)
        expected_rf_end = 500e-9 + 20e-9
        actual_rf_end = rf_schedulable.data["abs_time"] + rf_operation.duration
        np.testing.assert_allclose(actual_rf_end, expected_rf_end)

    def test_no_rf_switch_for_zero_only_voltage_offset(self, hardware_cfg_rf_simple):
        """A zero VoltageOffset alone (no non-zero before) creates no RF switch."""
        schedule = Schedule("test")
        schedule.add(
            VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="q0:mw", clock="q0.01")
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)
        assert len(self._get_rf_switch_schedulables(result)) == 0

    def test_unclosed_stitched_pulse_no_rf_switch(self, hardware_cfg_rf_simple):
        """A non-zero VoltageOffset without a closing zero event creates no RF switch."""
        schedule = Schedule("test")
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01")
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)
        assert len(self._get_rf_switch_schedulables(result)) == 0

    def test_stitched_pulse_merged_with_adjacent_mw_pulse(self, hardware_cfg_rf_simple):
        """A stitched pulse window adjacent to a regular MW pulse is merged."""
        ramp_buffer = 20e-9

        schedule = Schedule("test")
        # Stitched pulse: t=0 to t=200e-9
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01")
        )
        schedule.add(
            VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="q0:mw", clock="q0.01"),
            rel_time=200e-9,
        )
        # Regular pulse starting at 230e-9 (gap of 30ns < 2*20ns=40ns → should merge)
        schedule.add(
            SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"),
            rel_time=30e-9,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple, ramp_buffer)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)

        rf_switches = self._get_rf_switch_schedulables(result)
        assert len(rf_switches) == 1, "All within 2*ramp_buffer → single RF switch"

    def test_stitched_pulse_separate_from_far_mw_pulse(self, hardware_cfg_rf_simple):
        """A stitched pulse window far from a regular MW pulse creates two RF switches."""
        ramp_buffer = 20e-9

        schedule = Schedule("test")
        # Stitched pulse: t=0 to t=100e-9
        schedule.add(
            VoltageOffset(offset_path_I=0.5, offset_path_Q=0.0, port="q0:mw", clock="q0.01")
        )
        schedule.add(
            VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port="q0:mw", clock="q0.01"),
            rel_time=100e-9,
        )
        # Regular pulse at t=300e-9 (gap of 200ns >> 2*20ns → separate)
        schedule.add(
            SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"),
            rel_time=200e-9,
        )
        schedule = _determine_absolute_timing(schedule)

        hardware_cfg = self._enable_rf_switch(hardware_cfg_rf_simple, ramp_buffer)
        config = SerialCompilationConfig(name="test", hardware_compilation_config=hardware_cfg)

        result = auto_rf_switch_dressing(schedule, config)
        assert len(self._get_rf_switch_schedulables(result)) == 2
