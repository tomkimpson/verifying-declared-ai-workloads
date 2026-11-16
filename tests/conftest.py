import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: windows real captures (minutes)")
