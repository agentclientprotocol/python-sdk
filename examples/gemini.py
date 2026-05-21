from __future__ import annotations

import argparse
import asyncio
import asyncio.subprocess
import contextlib
import json
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    connect_to_agent,
    text_block,
)
from acp.core import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    ClientCapabilities,
    ConfigOptionUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    EnvVariable,
    FileEditToolCallContent,
    FileSystemCapabilities,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionInfoUpdate,
    TerminalOutputResponse,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)


class GeminiClient(Client):
    """Minimal client implementation that can drive the Gemini CLI over ACP."""

    def __init__(self, auto_approve: bool) -> None:
        self._auto_approve = auto_approve

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCallUpdate, **kwargs: Any
    ) -> RequestPermissionResponse:
        if self._auto_approve:
            option = _pick_preferred_option(options)
            if option is None:
                return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
            return RequestPermissionResponse(outcome=AllowedOutcome(option_id=option.option_id, outcome="selected"))

        title = tool_call.title or "<permission>"
        if not options:
            print(f"\n🔐 Permission requested: {title} (no options, cancelling)")
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        print(f"\n🔐 Permission requested: {title}")
        for idx, opt in enumerate(options, start=1):
            print(f"  {idx}. {opt.name} ({opt.kind})")

        loop = asyncio.get_running_loop()
        while True:
            choice = await loop.run_in_executor(None, lambda: input("Select option: ").strip())
            if not choice:
                continue
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    opt = options[idx]
                    return RequestPermissionResponse(
                        outcome=AllowedOutcome(option_id=opt.option_id, outcome="selected")
                    )
            print("Invalid selection, try again.")

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        pathlib_path = Path(path)
        if not pathlib_path.is_absolute():
            raise RequestError.invalid_params({"path": pathlib_path, "reason": "path must be absolute"})
        pathlib_path.parent.mkdir(parents=True, exist_ok=True)
        pathlib_path.write_text(content)
        print(f"[Client] Wrote {pathlib_path} ({len(content)} bytes)")
        return WriteTextFileResponse()

    async def read_text_file(
        self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        pathlib_path = Path(path)
        if not pathlib_path.is_absolute():
            raise RequestError.invalid_params({"path": pathlib_path, "reason": "path must be absolute"})
        text = pathlib_path.read_text()
        print(f"[Client] Read {pathlib_path} ({len(text)} bytes)")
        if line is not None or limit is not None:
            text = _slice_text(text, line, limit)
        return ReadTextFileResponse(content=text)

    async def session_update(  # noqa: C901
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate
        | ConfigOptionUpdate
        | SessionInfoUpdate
        | UsageUpdate,
        **kwargs: Any,
    ) -> None:
        if isinstance(update, AgentMessageChunk):
            _print_text_content(update.content)
        elif isinstance(update, AgentThoughtChunk):
            print("\n[agent_thought]")
            _print_text_content(update.content)
        elif isinstance(update, UserMessageChunk):
            print("\n[user_message]")
            _print_text_content(update.content)
        elif isinstance(update, AgentPlanUpdate):
            print("\n[plan]")
            for entry in update.entries:
                print(f" - {entry.status.upper():<10} {entry.content}")
        elif isinstance(update, ToolCallStart):
            print(f"\n🔧 {update.title} ({update.status or 'pending'})")
        elif isinstance(update, ToolCallProgress):
            status = update.status or "in_progress"
            print(f"\n🔧 Tool call `{update.tool_call_id}` → {status}")
            if update.content:
                for item in update.content:
                    if isinstance(item, FileEditToolCallContent):
                        print(f"  diff: {item.path}")
                    elif isinstance(item, TerminalToolCallContent):
                        print(f"  terminal: {item.terminal_id}")
                    elif isinstance(item, dict):
                        print(f"  content: {json.dumps(item, indent=2)}")
        else:
            print(f"\n[session update] {update}")

    # Optional / terminal-related methods ---------------------------------
    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        print(f"[Client] createTerminal: {command} {args or []} (cwd={cwd})")
        return CreateTerminalResponse(terminal_id="term-1")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        print(f"[Client] terminalOutput: {session_id} {terminal_id}")
        return TerminalOutputResponse(output="", truncated=False)

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        print(f"[Client] releaseTerminal: {session_id} {terminal_id}")
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        print(f"[Client] waitForTerminalExit: {session_id} {terminal_id}")
        return WaitForTerminalExitResponse()

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> KillTerminalResponse | None:
        print(f"[Client] killTerminal: {session_id} {terminal_id}")
        return KillTerminalResponse()


def _pick_preferred_option(options: Iterable[PermissionOption]) -> PermissionOption | None:
    best: PermissionOption | None = None
    for option in options:
        if option.kind in {"allow_once", "allow_always"}:
            return option
        best = best or option
    return best


def _slice_text(content: str, line: int | None, limit: int | None) -> str:
    lines = content.splitlines()
    start = 0
    if line:
        start = max(line - 1, 0)
    end = len(lines)
    if limit:
        end = min(start + limit, end)
    return "\n".join(lines[start:end])


def _print_text_content(content: object) -> None:
    if isinstance(content, TextContentBlock):
        print(content.text)
    elif isinstance(content, ResourceContentBlock):
        print(f"{content.name or content.uri}")
    elif isinstance(content, EmbeddedResourceContentBlock):
        resource = content.resource
        text = getattr(resource, "text", None)
        if text:
            print(text)
        else:
            blob = getattr(resource, "blob", None)
            print(blob if blob else "<embedded resource>")
    elif isinstance(content, dict):
        text = content.get("text")  # type: ignore[union-attr]
        if text:
            print(text)


async def _send_prompt(conn: ClientSideConnection, session_id: str, prompt: str, timeout: float | None) -> None:
    request = conn.prompt(
        session_id=session_id,
        prompt=[text_block(prompt)],
    )
    if timeout is None:
        await request
        return
    await asyncio.wait_for(request, timeout=timeout)


async def interactive_loop(conn: ClientSideConnection, session_id: str, prompt_timeout: float | None) -> None:
    print("Type a message and press Enter to send.")
    print("Commands: :cancel, :exit")

    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("\n> ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line in {":exit", ":quit"}:
            break
        if line == ":cancel":
            await conn.cancel(session_id=session_id)
            continue

        try:
            await _send_prompt(conn, session_id, line, prompt_timeout)
        except asyncio.TimeoutError:
            print("prompt timed out waiting for final ACP response", file=sys.stderr)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(conn.cancel(session_id=session_id), timeout=2)
        except RequestError as err:
            _print_request_error("prompt", err)
        except Exception as exc:
            print(f"Prompt failed: {exc}", file=sys.stderr)


def _resolve_gemini_cli(binary: str | None) -> str:
    if binary:
        return binary
    env_value = os.environ.get("ACP_GEMINI_BIN")
    if env_value:
        return env_value
    resolved = shutil.which("gemini")
    if resolved:
        return resolved
    raise FileNotFoundError("Unable to locate `gemini` CLI, provide --gemini path")


async def run(argv: list[str]) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Interact with the Gemini CLI over ACP.")
    parser.add_argument("--gemini", help="Path to the Gemini CLI binary")
    parser.add_argument("--model", help="Model identifier to pass to Gemini")
    parser.add_argument("--prompt", help="Send one prompt and exit")
    parser.add_argument(
        "--prompt-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for session/prompt to finish; use 0 to disable",
    )
    parser.add_argument("--sandbox", action="store_true", help="Enable Gemini sandbox mode")
    parser.add_argument("--debug", action="store_true", help="Pass --debug to Gemini")
    parser.add_argument("--experimental-acp", action="store_true", help="Use Gemini's deprecated ACP flag")
    parser.add_argument("--skip-trust", action="store_true", help="Trust the current workspace for this session")
    parser.add_argument("--yolo", action="store_true", help="Auto-approve permission prompts")
    args = parser.parse_args(argv[1:])
    prompt_timeout = None if args.prompt_timeout == 0 else args.prompt_timeout

    try:
        gemini_path = _resolve_gemini_cli(args.gemini)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    cmd = [gemini_path, "--experimental-acp" if args.experimental_acp else "--acp"]
    if args.model:
        cmd += ["--model", args.model]
    if args.sandbox:
        cmd.append("--sandbox")
    if args.debug:
        cmd.append("--debug")
    if args.skip_trust:
        cmd.append("--skip-trust")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Suppress Gemini CLI diagnostic noise unless debugging
            stderr=None if args.debug else asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        print(f"Failed to start Gemini CLI: {exc}", file=sys.stderr)
        return 1

    if proc.stdin is None or proc.stdout is None:
        print("Gemini process did not expose stdio pipes.", file=sys.stderr)
        proc.terminate()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        return 1

    client_impl = GeminiClient(auto_approve=args.yolo)
    conn = connect_to_agent(client_impl, proc.stdin, proc.stdout)

    try:
        init_resp = await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
                terminal=True,
            ),
        )
    except RequestError as err:
        _print_request_error("initialize", err)
        await _shutdown(proc, conn)
        return 1
    except Exception as exc:
        print(f"initialize error: {exc}", file=sys.stderr)
        await _shutdown(proc, conn)
        return 1

    print(f"✅ Connected to Gemini (protocol v{init_resp.protocol_version})")

    try:
        session = await conn.new_session(
            cwd=os.getcwd(),
            mcp_servers=[],
        )
    except RequestError as err:
        _print_request_error("new_session", err)
        await _shutdown(proc, conn)
        return 1
    except Exception as exc:
        print(f"new_session error: {exc}", file=sys.stderr)
        await _shutdown(proc, conn)
        return 1

    print(f"📝 Created session: {session.session_id}")

    try:
        if args.prompt is None:
            await interactive_loop(conn, session.session_id, prompt_timeout)
        else:
            await _send_prompt(conn, session.session_id, args.prompt, prompt_timeout)
    except asyncio.TimeoutError:
        print("prompt timed out waiting for final ACP response", file=sys.stderr)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(conn.cancel(session_id=session.session_id), timeout=2)
        return 1
    finally:
        await _shutdown(proc, conn)

    return 0


def _print_request_error(stage: str, err: RequestError) -> None:
    payload = err.to_error_obj()
    message = payload.get("message", "")
    code = payload.get("code")
    print(f"{stage} error ({code}): {message}", file=sys.stderr)
    data = payload.get("data")
    if data is not None:
        try:
            formatted = json.dumps(data, indent=2)
        except TypeError:
            formatted = str(data)
        print(formatted, file=sys.stderr)


async def _shutdown(proc: asyncio.subprocess.Process, conn: ClientSideConnection) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(conn.close(), timeout=2)
    if proc.stdin is not None:
        with contextlib.suppress(Exception):
            proc.stdin.close()
            await asyncio.wait_for(proc.stdin.wait_closed(), timeout=2)
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    try:
        return asyncio.run(run(list(args)))
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
