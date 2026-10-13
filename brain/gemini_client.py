"""One-shot Gemini CLI wrapper via the local agy executable.

Used as a fallback when Claude is rate-limited, returns an error, or repeatedly fails validation.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

from forge.config import GeminiSettings, estimate_cost_usd, gemini_settings, redact

log = logging.getLogger("forge.brain.gemini")


class GeminiError(RuntimeError):
    """Raised when the Gemini CLI call fails, times out, or produces invalid output."""


@dataclass(frozen=True)
class GeminiOutput:
    text: str
    usage: dict[str, int]
    cost_usd: float


def prompt_from_messages(messages: Sequence[dict]) -> str:
    """Concatenate system and user messages into a single prompt for one-shot CLI execution."""
    parts = []
    for m in messages:
        content = m.get("content", "")
        if content:
            parts.append(str(content))
    return "\n\n".join(parts)


def call_gemini(prompt: str, settings: GeminiSettings | None = None) -> GeminiOutput:
    """Invoke the agy CLI one-shot with the given prompt.

    Args:
        prompt: Full text prompt (system prompt + user table/instructions).
        settings: Optional settings override; defaults to gemini_settings().

    Returns:
        GeminiOutput containing stdout text, estimated token usage, and estimated cost.

    Raises:
        GeminiError: On missing binary, timeout, non-zero exit code, or empty response.
    """
    cfg = settings or gemini_settings()
    which_bin = shutil.which(cfg.bin_path)
    if which_bin:
        bin_path = which_bin
    else:
        local_bin = Path.home() / ".local" / "bin" / cfg.bin_path
        if local_bin.exists():
            bin_path = str(local_bin)
        else:
            raise GeminiError(f"gemini CLI binary not found: {cfg.bin_path}")

    cmd = [
        bin_path,
        "--model", cfg.model,
        "--dangerously-skip-permissions",
        "-p", prompt,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise GeminiError(f"gemini call timed out after {cfg.timeout_s:.0f}s") from e
    except OSError as e:
        raise GeminiError(redact(f"failed to execute {bin_path}: {e}")) from e

    if proc.returncode != 0:
        err = redact(proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}")
        raise GeminiError(f"gemini CLI exited with code {proc.returncode}: {err}")

    text = proc.stdout.strip()
    if not text:
        raise GeminiError("gemini CLI returned empty response")

    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(text) // 4)
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    cost = estimate_cost_usd(cfg.model, prompt_tokens, completion_tokens)
    return GeminiOutput(text=text, usage=usage, cost_usd=cost)
