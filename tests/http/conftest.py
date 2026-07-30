"""Shared fixtures for HTTP/WS loopback tests: run an ASGI app under uvicorn."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest_asyncio
import uvicorn


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RunningServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}/acp"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/acp"


@pytest_asyncio.fixture
async def serve_asgi() -> AsyncIterator[Callable[[Any], Any]]:
    """Yield a factory that boots an ASGI app under uvicorn and returns a RunningServer."""
    servers: list[uvicorn.Server] = []
    tasks: list[asyncio.Task[Any]] = []

    async def _start(app: Any) -> RunningServer:
        host, port = "127.0.0.1", _free_port()
        config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        servers.append(server)
        task = asyncio.ensure_future(server.serve())
        tasks.append(task)
        # Wait until the server is up.
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.02)
        return RunningServer(host, port)

    try:
        yield _start
    finally:
        for server in servers:
            server.should_exit = True
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=5)
