import pytest


@pytest.fixture(scope="function", autouse=False)
def mock_qblox_instruments_config_manager():
    class MockQbloxConfigurationManager:
        def download_log(self, source, fmt, file):  # noqa: ARG002
            with open(file, "w") as mock_file:
                mock_file.write(f"Mock hardware log for {source}")

    return MockQbloxConfigurationManager()
