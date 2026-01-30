from __future__ import annotations
from ..py38_compatibility import *

from enum import Enum
from typing import Any

__all__ = ["RpcTask", "RpcTaskKind"]


class RpcTaskKind(Enum):
    REQUEST = "request"
    NOTIFICATION = "notification"


@dataclass_with_slots(slots=True)
class RpcTask:
    kind: RpcTaskKind
    message: dict[str, Any]


from .dispatcher import (  # noqa: E402
    DefaultMessageDispatcher,
    MessageDispatcher,
    NotificationRunner,
    RequestRunner,
)
from .queue import InMemoryMessageQueue, MessageQueue  # noqa: E402
from .sender import MessageSender, SenderFactory  # noqa: E402
from .state import InMemoryMessageStateStore, MessageStateStore  # noqa: E402
from .supervisor import TaskSupervisor  # noqa: E402

__all__ += [
    "DefaultMessageDispatcher",
    "InMemoryMessageQueue",
    "InMemoryMessageStateStore",
    "MessageDispatcher",
    "MessageQueue",
    "MessageSender",
    "MessageStateStore",
    "NotificationRunner",
    "RequestRunner",
    "SenderFactory",
    "TaskSupervisor",
]
