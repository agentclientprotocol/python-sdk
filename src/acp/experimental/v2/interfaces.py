from typing import Protocol

from . import schema

__all__ = ["Agent"]


class Agent(Protocol):
    async def initialize(self, request: schema.InitializeRequest) -> schema.InitializeResponse: ...
