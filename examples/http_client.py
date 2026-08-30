# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#     "agent-client-protocol[http]",
# ]
# ///
"""Connect to a remote ACP agent over Streamable HTTP (experimental).

Start the server first (``uv run examples/http_server.py``), then run this.
"""

import asyncio
from typing import Any

from acp import connect_to_agent
from acp.http import create_http_stream
from acp.interfaces import Client
from acp.schema import TextContentBlock


class ExampleClient(Client):
    async def request_permission(self, session_id: str, tool_call: Any, options: Any, **kwargs: Any) -> Any:
        # Auto-allow the first option.
        return {"outcome": {"outcome": "selected", "optionId": options[0]["optionId"]}}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        content = getattr(update, "content", None)
        text = getattr(content, "text", None) if content is not None else None
        if text:
            print(f"<< {text}")

    async def write_text_file(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def read_text_file(self, *args: Any, **kwargs: Any) -> Any:
        return {"content": ""}


async def main() -> None:
    transport = create_http_stream("http://localhost:8000/acp")
    conn = connect_to_agent(ExampleClient(), transport)
    try:
        init = await conn.initialize(protocol_version=1)
        print(f"initialized (protocol v{init.protocol_version})")
        session = await conn.new_session(cwd=".", mcp_servers=[])
        print(f"session: {session.session_id}")
        result = await conn.prompt(session_id=session.session_id, prompt=[TextContentBlock(text="hello over http")])
        print(f"stop reason: {result.stop_reason}")
    finally:
        await conn.close()
        await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
