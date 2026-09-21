from __future__ import annotations

from ..interfaces import Agent
from ..router import MessageRouter

__all__ = ["build_agent_router"]


def build_agent_router(agent: Agent, use_unstable_protocol: bool = False) -> MessageRouter:
    return MessageRouter.from_protocol(Agent, agent, use_unstable_protocol=use_unstable_protocol)
