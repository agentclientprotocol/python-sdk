# Handle permissions

An agent requests permission for a tool call through `request_permission`. The client displays the supplied options and returns the option the user selected, or a cancellation outcome.

## Return the user's choice

Implement this method on your `Client`. `choose_option` represents your application's UI and returns an option ID or `None`:

```python
from typing import Any

from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallUpdate,
)


async def request_permission(
    self,
    session_id: str,
    tool_call: ToolCallUpdate,
    options: list[PermissionOption],
    **kwargs: Any,
) -> RequestPermissionResponse:
    option_id = await choose_option(tool_call, options)
    if option_id is None:
        return RequestPermissionResponse(outcome=DeniedOutcome())
    if option_id not in {option.option_id for option in options}:
        raise ValueError("Choose an option from the permission request")
    return RequestPermissionResponse(outcome=AllowedOutcome(option_id=option_id))
```

Despite its Python name, `AllowedOutcome` means the user selected an option. That option can be a rejection. Read the option's `kind` to distinguish `allow_once` and `allow_always` from `reject_once` and `reject_always`. `DeniedOutcome` represents cancellation.

When the client cancels a prompt turn with `session/cancel`, resolve every pending permission request for that turn with `DeniedOutcome()`. A dismissed dialog also returns that outcome. The agent must not execute the operation after a cancellation or a rejection. See the protocol's [permission outcomes](https://agentclientprotocol.com/protocol/v1/tool-calls#requesting-permission).

## Request approval from the agent

The experimental contrib helpers associate a permission request with a tracked tool call. Pass your session's tracker to this function. It publishes the pending call and returns whether the user approved it; the caller performs the operation only if it returns `True`:

```python
from acp import Client
from acp.contrib.permissions import PermissionBroker, default_permission_options
from acp.contrib.tool_calls import ToolCallTracker
from acp.schema import AllowedOutcome, RequestPermissionRequest, RequestPermissionResponse


async def request_tool_permission(
    client: Client, session_id: str, tracker: ToolCallTracker
) -> bool:
    start = tracker.start("write-file", title="Write a file", kind="edit", status="pending")
    await client.session_update(session_id=session_id, update=start)

    async def request_permission(
        request: RequestPermissionRequest,
    ) -> RequestPermissionResponse:
        return await client.request_permission(
            session_id=request.session_id,
            tool_call=request.tool_call,
            options=request.options,
        )

    options = default_permission_options()
    broker = PermissionBroker(session_id, request_permission, tracker=tracker)
    response = await broker.request_for("write-file", options=options)
    selected = response.outcome
    option = next(
        (item for item in options if isinstance(selected, AllowedOutcome)
         and item.option_id == selected.option_id),
        None,
    )
    approved = option is not None and option.kind in {"allow_once", "allow_always"}
    progress = tracker.progress(
        "write-file", status="in_progress" if approved else "failed"
    )
    await client.session_update(session_id=session_id, update=progress)
    return approved
```

After an approved operation finishes, send a tool call update with its final status and result. Keep the tracker with your session if you need to update the same tool call later. Your application defines what a persistent approval covers; `PermissionBroker` does not enforce an approval policy or execute tools.

The broker supplies its defaults when you omit `options` or pass `None`. An empty list raises `MissingPermissionOptionsError` before the request is sent. See [contrib reference](../reference/experimental/contrib.md) for the helper contracts and [the Gemini client](https://github.com/agentclientprotocol/python-sdk/blob/main/examples/gemini.py) for a terminal UI that displays the supplied options.
