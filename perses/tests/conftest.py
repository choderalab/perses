import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu_ci: mark test as useful to run on GPU")
    config.addinivalue_line("markers", "gpu_needed: mark test as GPU required")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")

    if not config.getoption("--runslow"):
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
