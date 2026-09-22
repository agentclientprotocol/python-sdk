import asyncio
import contextlib
import signal
import sys

import pytest

from acp.transports import spawn_stdio_transport


@pytest.mark.asyncio
@pytest.mark.parametrize("ignore_terminate", [False, True])
async def test_shutdown_with_unread_stdin(ignore_terminate: bool) -> None:
    if ignore_terminate and sys.platform == "win32":
        pytest.skip("Windows terminate() cannot be ignored")
    script = "import signal, time; "
    if ignore_terminate:
        script += "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    script += "print('ready', flush=True); time.sleep(60)"
    transport = spawn_stdio_transport(sys.executable, "-c", script, shutdown_timeout=0.1)
    reader, writer, process = await transport.__aenter__()
    closing = None
    try:
        assert (await asyncio.wait_for(reader.readline(), timeout=5)).strip() == b"ready"
        writer.write(b"x" * (1024 * 1024))
        assert writer.transport.get_write_buffer_size() > 0
        closing = asyncio.create_task(transport.__aexit__(None, None, None))
        await asyncio.wait_for(asyncio.shield(closing), timeout=5)
        assert process.returncode is not None
        if sys.platform != "win32":
            assert process.returncode == -(signal.SIGKILL if ignore_terminate else signal.SIGTERM)
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
        if closing is not None:
            await closing
        else:
            await transport.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_shutdown_flushes_stdin_before_eof() -> None:
    script = "import sys; data = sys.stdin.buffer.read(); print(len(data), flush=True)"
    async with spawn_stdio_transport(sys.executable, "-c", script) as (reader, writer, process):
        writer.write(b"x" * (1024 * 1024))
    assert process.returncode == 0
    assert (await reader.readline()).strip() == b"1048576"


@pytest.mark.asyncio
async def test_shutdown_preserves_body_exception() -> None:
    with pytest.raises(ValueError, match="body failure"):
        async with spawn_stdio_transport(sys.executable, "-c", "import sys; sys.stdin.buffer.read()") as (
            _reader,
            _writer,
            process,
        ):
            raise ValueError("body failure")
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_shutdown_after_child_exit() -> None:
    async with spawn_stdio_transport(sys.executable, "-c", "pass") as (_reader, writer, process):
        await asyncio.wait_for(process.wait(), timeout=5)
        with contextlib.suppress(ConnectionError):
            await writer.drain()
    assert process.returncode == 0
