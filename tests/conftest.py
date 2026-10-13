"""Test isolation: every test runs against a tmp FORGE_ROOT and a fake token."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAUDE_LOCAL_TOKEN", "test-token-NOT-REAL-123456")
    monkeypatch.setenv("CLAUDE_BASE_URL", "http://claude.test/v1")
    monkeypatch.setenv("CLAUDE_MODEL", "sonnet")
    monkeypatch.setenv("GEMINI_FALLBACK", "false")
    for d in ("data", "state", "logs", "backtests"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    yield tmp_path
