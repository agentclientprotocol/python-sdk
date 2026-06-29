import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from acp.transports import spawn_stdio_transport

LARGE_LINE_SIZE = 70 * 1024


def _large_line_script(size: int = LARGE_LINE_SIZE) -> str:
    return textwrap.dedent(
        f"""
        import sys
        sys.stdout.write("X" * {size})
        sys.stdout.write("\\n")
        sys.stdout.flush()
        """
    ).strip()


@pytest.mark.asyncio
async def test_spawn_stdio_transport_hits_default_limit() -> None:
    script = _large_line_script()
    async with spawn_stdio_transport(sys.executable, "-c", script) as (reader, _writer, _process):
        # readline() re-raises LimitOverrunError as ValueError on CPython 3.12+.
        with pytest.raises(ValueError):
            await reader.readline()


@pytest.mark.asyncio
async def test_spawn_stdio_transport_custom_limit_handles_large_line() -> None:
    script = _large_line_script()
    async with spawn_stdio_transport(
        sys.executable,
        "-c",
        script,
        limit=LARGE_LINE_SIZE * 2,
    ) as (reader, _writer, _process):
        line = await reader.readline()
        assert len(line) == LARGE_LINE_SIZE + 1


@pytest.mark.asyncio
async def test_run_agent_stdio_buffer_limit() -> None:
    """Test that run_agent with different buffer limits can handle appropriately sized messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Small buffer (1KB) reads a large message (70KB) in chunks
        small_agent = os.path.join(tmpdir, "small_agent.py")
        with open(small_agent, "w") as f:
            f.write(
                textwrap.dedent(
                    """
                    import asyncio
                    from acp.core import run_agent
                    from acp.interfaces import Agent
                    from acp.schema import InitializeResponse

                    class TestAgent(Agent):
                        async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kwargs):
                            return InitializeResponse(protocol_version=protocol_version)

                    asyncio.run(run_agent(TestAgent(), stdio_buffer_limit_bytes=1024))
                    """
                ).strip()
            )

        # Send a 70KB message - should be read in chunks despite the 1KB buffer
        large_msg = (
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"_meta":{"data":"'
            + "X" * LARGE_LINE_SIZE
            + '"}}}\n'
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, small_agent], input=large_msg, capture_output=True, text=True, timeout=2
        )

        assert result.returncode == 0
        assert "LimitOverrunError" not in result.stderr
        assert "Separator is found, but chunk is longer than limit" not in result.stderr
        assert "oversized JSON-RPC frame" not in result.stderr
        assert '"id":1' in result.stdout
        assert '"protocolVersion":1' in result.stdout

        # Test 2: Large buffer (200KB) succeeds with large message (70KB)
        large_agent = os.path.join(tmpdir, "large_agent.py")
        with open(large_agent, "w") as f:
            f.write(
                textwrap.dedent(
                    f"""
                    import asyncio
                    from acp.core import run_agent
                    from acp.interfaces import Agent
                    from acp.schema import InitializeResponse

                    class TestAgent(Agent):
                        async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kwargs):
                            return InitializeResponse(protocol_version=protocol_version)

                    asyncio.run(run_agent(TestAgent(), stdio_buffer_limit_bytes={LARGE_LINE_SIZE * 3}))
                    """
                ).strip()
            )

        # Same message, but with a buffer 3x the size - should handle it
        result = subprocess.run(  # noqa: S603
            [sys.executable, large_agent], input=large_msg, capture_output=True, text=True, timeout=2
        )

        # With a large enough buffer, the agent should at least start successfully
        # (it may have other errors from invalid JSON-RPC, but not buffer overrun)
        if "LimitOverrunError" in result.stderr or "buffer" in result.stderr.lower():
            pytest.fail(f"Large buffer still hit limit error: {result.stderr}")
        assert '"id":1' in result.stdout
        assert '"protocolVersion":1' in result.stdout
