# Maintain protocol routes

The `Agent` and `Client` protocols in `src/acp/interfaces.py` are the source of
truth for incoming routes. Declare routing metadata alongside each method's
parameter model:

```python
from typing import Any

from acp.meta import AGENT_METHODS
from acp.schema import DeleteSessionRequest, DeleteSessionResponse
from acp.utils import normalize_result, param_model


class SessionHandlers:
    @param_model(
        DeleteSessionRequest,
        method=AGENT_METHODS["session_delete"],
        adapt_result=normalize_result,
    )
    async def delete_session(
        self, session_id: str, **kwargs: Any
    ) -> DeleteSessionResponse: ...
```

`build_agent_router` and `build_client_router` read these declarations through
`MessageRouter.from_protocol`. New protocol declarations automatically become
routes; implementation-only methods are not exposed. Use `kind="notification"`
for notifications and `unstable=True` for methods requiring explicit opt-in.
Optional handlers can declare `optional=True` and `default_result`.

`param_model` takes exactly one type expression: a model, a union such as
`ModelA | ModelB`, or `Annotated[ModelA | ModelB, Field(discriminator="type")]`.
The router validates the original type with Pydantic's `TypeAdapter`, preserving
`Annotated` validation metadata. Union handlers receive the fields common to all
branches by default.

`validate_params` and `adapt_params` provide custom validation and conversion
when the wire representation differs from the Python signature, as with config
options and elicitation. Legacy handlers still receive the validated request
model. Signature generation expands single-model fields and preserves handwritten
union signatures. Connection methods retain model-only decorators for signature generation
and legacy calls; they do not repeat the routing options.
