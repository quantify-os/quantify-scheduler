# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Unit tests for schedules used in two qubit experiments."""

from typing import Type

import numpy as np

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.graph_compilation import SerialCompilationConfig
from quantify_scheduler.schedules import two_qubit_transmon_schedules as ts


class TestChevronCZSched:
    """
        Unit test for the
    :func:`~quantify_scheduler.schedules.two_qubit_transmon_schedules.chevron_cz_sched`
    schedule generating function.
    """

    @classmethod
    def setup_class(cls: Type) -> None:
        """Configure an example sweep for a single flux pulse duration."""
        cls.sched_kwargs = {  # type: ignore
            "lf_qubit": "q0",
            "hf_qubit": "q4",
            "amplitudes": np.linspace(0, 80e-6, 21),
            "duration": 100e-9,
            "repetitions": 10,
        }
        cls.uncomp_sched = ts.chevron_cz_sched(**cls.sched_kwargs)  # type: ignore

    def test_repetitions(self) -> None:
        """Test that the number of repetitions is correct."""
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_sweep(self) -> None:
        """Test that sweep is in correct order and flux pulse amplitude is swept."""
        _labels = [
            [
                f"Reset {i}",
                f"X(q0) {i}",
                f"X(q4) {i}",
                f"SquarePulse(q4:fl) {i}",
                f"Measure(q0,q4) {i}",
            ]
            for i in range(len(self.sched_kwargs["amplitudes"]))
        ]
        labels = [item for sublist in _labels for item in sublist]

        sq_pulse_idx = 0
        for i, schedulable in enumerate(self.uncomp_sched.schedulables.values()):
            # Test that the order of the sweep is the same by checking labels
            assert schedulable["label"] == labels[i]

            # Test that the amplitude of the flux pulse is unchanged
            if schedulable["label"].startswith("SquarePulse"):
                op_hash = schedulable["operation_id"]
                pulse = self.uncomp_sched.operations[op_hash]["pulse_info"][0]
                assert pulse["amp"] == self.sched_kwargs["amplitudes"][sq_pulse_idx]
                sq_pulse_idx += 1

    def test_sched_float_amp_compile(
        self, compile_config_basic_transmon_qblox_hardware: SerialCompilationConfig
    ) -> None:
        """Test that a single datapoint schedule compiles with SerialCompiler."""
        sched_kwargs = {
            "lf_qubit": "q0",
            "hf_qubit": "q4",
            "amplitudes": 0.25,
            "duration": 100e-9,
            "repetitions": 10,
        }  # type: ignore
        sched = ts.chevron_cz_sched(**sched_kwargs)
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=compile_config_basic_transmon_qblox_hardware
        )

    def test_custom_flux_port(
        self, compile_config_basic_transmon_qblox_hardware: SerialCompilationConfig
    ) -> None:
        """Test that custom flux port is respected."""
        sched_kwargs = {
            "lf_qubit": "q0",
            "hf_qubit": "q4",
            "amplitudes": 0.25,
            "duration": 100e-9,
            "repetitions": 10,
            "flux_port": "q1:fl",
        }
        sched = ts.chevron_cz_sched(**sched_kwargs)
        compiler = SerialCompiler(name="compiler")
        compiled_sched = compiler.compile(
            schedule=sched, config=compile_config_basic_transmon_qblox_hardware
        )

        for schedulable in compiled_sched.schedulables.values():
            if schedulable["label"].startswith("SquarePulse"):
                operation = compiled_sched.operations[schedulable["operation_id"]]
                assert operation["pulse_info"][0]["port"] == "q1:fl"

    def test_sched_compile(
        self, compile_config_basic_transmon_qblox_hardware: SerialCompilationConfig
    ) -> None:
        """Test that a sweep schedule compiles with SerialCompiler."""
        sched = ts.chevron_cz_sched(**self.sched_kwargs)
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=compile_config_basic_transmon_qblox_hardware
        )

    def test_operations(self) -> None:
        """Test that the number of operations in the schedule is correct."""
        # The operations property has all unique operations in schedule.
        # 1x Reset operation
        # 2x X gate operations (one on each qubit individually)
        # Num amp Flux operations (amplitude is different for every setpoint)
        # Num amp Measure gates (acq_index different for every setpoint)
        assert len(self.uncomp_sched.operations) == (
            1 + 2 + len(self.sched_kwargs["amplitudes"]) * 2
        )
