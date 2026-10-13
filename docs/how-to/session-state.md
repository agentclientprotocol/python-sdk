# Accumulate session state

Use the experimental `SessionAccumulator` to merge tool call updates and keep message chunks for a UI. Create one accumulator per session so interleaved events from different sessions stay separate.

```python
from typing import Any

from acp import Client
from acp.contrib.session_state import SessionAccumulator
from acp.schema import SessionNotification


class StatefulClient(Client):
    def __init__(self) -> None:
        self.sessions: dict[str, SessionAccumulator] = {}

    async def session_update(
        self, session_id: str, update: Any, **kwargs: Any
    ) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionAccumulator(
                auto_reset_on_session_change=False
            )
        snapshot = self.sessions[session_id].apply(
            SessionNotification(session_id=session_id, update=update)
        )
        print(snapshot.session_id, len(snapshot.agent_messages))
```

The accumulator tracks tool call starts and progress, full `AgentPlanUpdate` entries, the current mode, available commands, and user/agent/thought message chunks. A progress event without a start creates a provisional tool call entry. A later start can replace its fields; the helper does not reconstruct event chronology.

It does not retain every update type or field. Handle session metadata, usage, config options, notices, compaction, incremental plan events, and tool names separately if your UI uses them. Message collections contain chunks; group by `message_id` when present and concatenate text chunks when rendering a message.

`apply()` returns a snapshot; `snapshot()` reads the current one after at least one notification. The helper copies model data and freezes the outer snapshot model, but its dictionaries and nested objects remain mutable. Treat snapshots as read-only in your application.

`subscribe(callback)` invokes a synchronous callback after each update and returns a function that unsubscribes it. Remove the session's accumulator when your application no longer needs it. See the [contrib reference](../reference/experimental/contrib.md) for reset behavior and related tool call helpers.
