"""Tests for brain/gemini_client.py (prompt building, subprocess mock, error scrubbing)."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from brain.gemini_client import GeminiError, GeminiOutput, call_gemini, prompt_from_messages
from forge.config import GeminiSettings


def test_prompt_from_messages_concatenates():
    msgs = [
        {"role": "system", "content": "System mandate and rules."},
        {"role": "user", "content": "Portfolio view and universe table."},
    ]
    prompt = prompt_from_messages(msgs)
    assert prompt == "System mandate and rules.\n\nPortfolio view and universe table."


def test_prompt_from_messages_skips_empty():
    msgs = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Previous output"},
    ]
    prompt = prompt_from_messages(msgs)
    assert prompt == "System instructions\n\nPrevious output"


def test_call_gemini_success():
    cfg = GeminiSettings(model="gemini-3.8-flash-high", bin_path="agy", timeout_s=30.0)
    fake_json = '{"holdings": {"S001": {"weight": 0.2, "rationale": "trend"}}, "cash": 0.8, "notes": "ok", "risk_view": "low"}'

    with patch("shutil.which", return_value="/fake/path/agy"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["/fake/path/agy", "-p", "test"],
                returncode=0,
                stdout=fake_json + "\n",
                stderr="",
            )
            out = call_gemini("Test prompt here", cfg)

            assert isinstance(out, GeminiOutput)
            assert out.text == fake_json
            assert out.usage["prompt_tokens"] > 0
            assert out.usage["completion_tokens"] > 0
            assert out.cost_usd > 0
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert args[0][:4] == ["/fake/path/agy", "--model", "gemini-3.8-flash-high", "--dangerously-skip-permissions"]
            assert args[0][4:] == ["-p", "Test prompt here"]
            assert kwargs["timeout"] == 30.0


def test_call_gemini_missing_bin():
    cfg = GeminiSettings(bin_path="nonexistent_agy_binary")
    with patch("shutil.which", return_value=None):
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(GeminiError, match="binary not found"):
                call_gemini("prompt", cfg)


def test_call_gemini_timeout():
    cfg = GeminiSettings(bin_path="agy", timeout_s=5.0)
    with patch("shutil.which", return_value="/bin/agy"):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["agy"], timeout=5.0)):
            with pytest.raises(GeminiError, match="timed out after 5s"):
                call_gemini("prompt", cfg)


def test_call_gemini_nonzero_exit_scrubs_secrets(monkeypatch):
    secret = "secret-token-gemini-xyz-987"
    monkeypatch.setenv("CLAUDE_LOCAL_TOKEN", secret)
    cfg = GeminiSettings(bin_path="agy")

    with patch("shutil.which", return_value="/bin/agy"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["agy"],
                returncode=1,
                stdout="",
                stderr=f"Error occurred while processing {secret}",
            )
            with pytest.raises(GeminiError) as exc_info:
                call_gemini("prompt", cfg)
            msg = str(exc_info.value)
            assert secret not in msg
            assert "***REDACTED***" in msg
            assert "exited with code 1" in msg


def test_call_gemini_empty_stdout():
    cfg = GeminiSettings(bin_path="agy")
    with patch("shutil.which", return_value="/bin/agy"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["agy"],
                returncode=0,
                stdout="   \n  ",
                stderr="",
            )
            with pytest.raises(GeminiError, match="empty response"):
                call_gemini("prompt", cfg)
