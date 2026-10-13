"""Structural guarantee: this codebase can only paper trade."""
import re
from pathlib import Path

import forge.config as cfg

ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = ["forge", "engine", "brain", "kronos", "optimize", "run", "api", "scripts"]
FORBIDDEN = [r"nautilus_trader\.live", r"\bTradingNode\b", r"LiveExecClient", r"LiveDataClient", r"\bib_insync\b",
             r"\balpaca", r"\bccxt\b", r"\bbinance\b", r"\binteractive_brokers\b", r"\bfreqtrade\b", r"\bhummingbot\b",
             r"AccountType\.MARGIN"]


def source_files():
    for d in SRC_DIRS:
        yield from (ROOT / d).rglob("*.py")


def test_no_live_trading_code_paths():
    hits = []
    for f in source_files():
        text = f.read_text()
        for pat in FORBIDDEN:
            if re.search(pat, text):
                hits.append((str(f.relative_to(ROOT)), pat))
    assert hits == []


def test_nautilus_live_package_never_imported():
    import ast
    for f in source_files():
        for node in ast.walk(ast.parse(f.read_text())):
            mods = [node.module] if isinstance(node, ast.ImportFrom) and node.module else \
                   [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not any(m.startswith("nautilus_trader.live") or m.startswith("nautilus_trader.adapters") for m in mods), f


def test_mode_is_paper_and_venue_is_simulated_cash_account():
    assert cfg.MODE == "paper"
    sim = (ROOT / "engine" / "sim.py").read_text()
    assert "AccountType.CASH" in sim and 'Venue("SIM")' in sim
    assert "nautilus_trader.backtest" in sim


def test_dashboard_is_loopback_only_on_8904():
    assert cfg.DASHBOARD_HOST == "127.0.0.1" and cfg.DASHBOARD_PORT == 8904


def test_no_broker_keys_in_environment_template():
    env_names = set(re.findall(r"^([A-Z_]+)=", (ROOT / ".env").read_text(), flags=re.M)) if (ROOT / ".env").exists() else set()
    assert not any(k for k in env_names if any(w in k for w in ("BROKER", "ALPACA", "IBKR", "BINANCE", "API_SECRET")))
