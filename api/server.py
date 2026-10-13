"""Run the dashboard: ``python -m api.server`` (binds 192.168.1.201:8905, LAN-visible).

Paper-trading data only — no secrets are served and all endpoints are GET-only,
so a LAN-visible bind is acceptable.
"""

from __future__ import annotations

import uvicorn

from forge.config import DASHBOARD_HOST, DASHBOARD_PORT


def main(host: str | None = None, port: int | None = None) -> None:
    """Serve ``api.app:app`` with uvicorn. Defaults come from forge.config."""
    bind_host = DASHBOARD_HOST if host is None else host
    bind_port = DASHBOARD_PORT if port is None else int(port)
    if not 1 <= bind_port <= 65535:
        raise ValueError(f"invalid port {bind_port}")
    uvicorn.run("api.app:app", host=bind_host, port=bind_port, log_level="info")


if __name__ == "__main__":
    main()
