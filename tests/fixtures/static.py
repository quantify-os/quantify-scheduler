from importlib import resources

import pytest


@pytest.fixture(scope="session")
def qdevice_with_basic_nv_element_yaml() -> str:
    return resources.read_text("tests.data", "qdevice_with_basic_nv_element.yaml")


@pytest.fixture(scope="session")
def hwconfig_with_qcm_rf_yaml() -> str:
    return resources.read_text("tests.data", "hwconfig_with_qcm_rf.yaml")
