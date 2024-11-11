# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Automatic mixer calibration tests."""

import pytest

from quantify_scheduler.backends.qblox.constants import (
    DEFAULT_MIXER_AMP_RATIO,
    DEFAULT_MIXER_PHASE_ERROR_DEG,
)
from quantify_scheduler.backends.qblox.enums import LoCalEnum, SidebandCalEnum
from quantify_scheduler.backends.types.qblox import ValidationWarning
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice


def test_string_literal_works():
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QRM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": 40e6},
            },
            "mixer_corrections": {
                "q0:res-q0.ro": {
                    "auto_lo_cal": "on_lo_interm_freq_change",
                    "auto_sideband_cal": "on_interm_freq_change",
                }
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:res"],
            ]
        },
    }
    quantum_device = QuantumDevice("qdev")
    quantum_device.hardware_config(hardware_config)
    hw_config_class = quantum_device.generate_compilation_config().hardware_compilation_config
    assert (
        hw_config_class.hardware_options.mixer_corrections["q0:res-q0.ro"].auto_lo_cal  # type: ignore
        == LoCalEnum.ON_LO_INTERM_FREQ_CHANGE
    )
    assert (
        hw_config_class.hardware_options.mixer_corrections[  # type: ignore
            "q0:res-q0.ro"
        ].auto_sideband_cal  # type: ignore
        == SidebandCalEnum.ON_INTERM_FREQ_CHANGE
    )


def test_conflicting_settings_warns_nco():
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QRM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": 40e6},
            },
            "mixer_corrections": {
                "q0:res-q0.ro": {
                    "amp_ratio": 0.9999,
                    "phase_error": -4.2,
                    "auto_lo_cal": "on_lo_interm_freq_change",
                    "auto_sideband_cal": "on_interm_freq_change",
                }
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:res"],
            ]
        },
    }
    quantum_device = QuantumDevice("qdev")
    quantum_device.hardware_config(hardware_config)
    with pytest.warns(
        ValidationWarning,
        match="Setting `auto_sideband_cal=on_interm_freq_change` will "
        "overwrite settings `amp_ratio=0.9999` and "
        "`phase_error=-4.2`. To suppress this warning, do not "
        "set either `amp_ratio` or `phase_error` for this port-clock.",
    ):
        hw_config_class = quantum_device.generate_compilation_config().hardware_compilation_config
    assert (
        hw_config_class.hardware_options.mixer_corrections["q0:res-q0.ro"].amp_ratio  # type: ignore
        is DEFAULT_MIXER_AMP_RATIO
    )
    assert (
        hw_config_class.hardware_options.mixer_corrections["q0:res-q0.ro"].phase_error  # type: ignore
        is DEFAULT_MIXER_PHASE_ERROR_DEG
    )
    assert (
        hw_config_class.hardware_options.mixer_corrections["q0:res-q0.ro"].auto_lo_cal  # type: ignore
        == LoCalEnum.ON_LO_INTERM_FREQ_CHANGE
    )
    assert (
        hw_config_class.hardware_options.mixer_corrections[  # type: ignore
            "q0:res-q0.ro"
        ].auto_sideband_cal  # type: ignore
        == SidebandCalEnum.ON_INTERM_FREQ_CHANGE
    )


def test_conflicting_settings_warns_lo():
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"1": {"instrument_type": "QRM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": 40e6},
            },
            "mixer_corrections": {
                "q0:res-q0.ro": {
                    "dc_offset_i": -0.054,
                    "dc_offset_q": -0.034,
                    "auto_lo_cal": "on_lo_interm_freq_change",
                    "auto_sideband_cal": "on_interm_freq_change",
                }
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "q0:res"],
            ]
        },
    }
    quantum_device = QuantumDevice("qdev")
    quantum_device.hardware_config(hardware_config)
    with pytest.warns(
        ValidationWarning,
        match="Setting `auto_lo_cal=on_lo_interm_freq_change` will overwrite settings "
        "`dc_offset_i=-0.054` and "
        "`dc_offset_q=-0.034`. To suppress this warning, do not "
        "set either `dc_offset_i` or `dc_offset_q` for this port-clock.",
    ):
        hw_config_class = quantum_device.generate_compilation_config().hardware_compilation_config
    assert (
        hw_config_class.hardware_options.mixer_corrections["q0:res-q0.ro"].dc_offset_i  # type: ignore
        is None
    )
    assert (
        hw_config_class.hardware_options.mixer_corrections["q0:res-q0.ro"].dc_offset_q  # type: ignore
        is None
    )
    assert (
        hw_config_class.hardware_options.mixer_corrections["q0:res-q0.ro"].auto_lo_cal  # type: ignore
        == LoCalEnum.ON_LO_INTERM_FREQ_CHANGE
    )
    assert (
        hw_config_class.hardware_options.mixer_corrections[  # type: ignore
            "q0:res-q0.ro"
        ].auto_sideband_cal
        == SidebandCalEnum.ON_INTERM_FREQ_CHANGE
    )
