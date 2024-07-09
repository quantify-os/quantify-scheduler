import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def example_ip() -> str:
    return "192.168.1.100"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
