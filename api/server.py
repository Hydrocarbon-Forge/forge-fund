"""Run the localhost dashboard: ``python -m api.server`` (SPEC: 127.0.0.1:8905 only).

The dashboard exposes portfolio and decision data, so the host must be a loopback
address. ``main`` refuses anything else before uvicorn is imported or started.
"""

from __future__ import annotations

import ipaddress

import uvicorn

from forge.config import DASHBOARD_HOST, DASHBOARD_PORT


class NonLoopbackHostError(ValueError):
    """Raised when asked to bind the dashboard to a non-loopback address."""


def _require_loopback(host: str) -> str:
    if host == "localhost":
        return host
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        raise NonLoopbackHostError(
            f"refusing to bind dashboard to {host!r}: only loopback hosts are allowed"
        ) from None
    if not addr.is_loopback:
        raise NonLoopbackHostError(
            f"refusing to bind dashboard to {host!r}: only loopback hosts are allowed"
        )
    return host


def main(host: str | None = None, port: int | None = None) -> None:
    """Serve ``api.app:app`` with uvicorn. Defaults come from forge.config."""
    bind_host = _require_loopback(DASHBOARD_HOST if host is None else host)
    bind_port = DASHBOARD_PORT if port is None else int(port)
    if not 1 <= bind_port <= 65535:
        raise ValueError(f"invalid port {bind_port}")
    uvicorn.run("api.app:app", host=bind_host, port=bind_port, log_level="info")


if __name__ == "__main__":
    main()
