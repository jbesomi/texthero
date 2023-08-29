import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-skip-broken",
        action="store_true",
        default=False,
        help="run tests marked as broken",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "skip_broken: mark test broken")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-skip-broken"):
        return

    skip_broken = pytest.mark.skip(reason="test marked as broken")
    for item in items:
        if "skip_broken" in item.keywords:
            item.add_marker(skip_broken)


def broken_case(*params):
    return pytest.param(*params, marks=(pytest.mark.skip_broken))
