import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config(ROOT / "config.yaml")
    c["owner_confirmed"] = True
    c["sessions_utc"] = {}
    c["rules"]["competition_end_utc"] = None
    return c
