from __future__ import annotations

from ..interfaces import Client
from ..router import MessageRouter

__all__ = ["build_client_router"]


def build_client_router(client: Client, use_unstable_protocol: bool = False) -> MessageRouter:
    return MessageRouter.from_protocol(Client, client, use_unstable_protocol=use_unstable_protocol)
