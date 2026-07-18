import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config():
    from sentinel.utils.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def tau_sequence():
    import json
    path = REPO_ROOT / "data" / "interim" / "tau_sequence.json"
    if not path.exists():
        pytest.skip("data/interim/tau_sequence.json not present — run `make data` first")
    return json.load(open(path))["sequence"]
