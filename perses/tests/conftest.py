import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--gpu-ci", action="store_true", default=False, help="run GPU tests on CI"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu_ci: mark test as useful to run on GPU CI")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu-ci"):
        only_GPU = pytest.mark.skip(reason="--gpu-ci used to only run GPU CI tests")
        for item in items:
            if not "gpu_ci" in item.keywords:
                item.add_marker(only_GPU)
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
