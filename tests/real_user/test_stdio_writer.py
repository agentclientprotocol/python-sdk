import asyncio
import io
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from acp.stdio import stdio_streams


@pytest.mark.skipif(sys.platform == "win32", reason="native POSIX pipe transport")
def test_stdio_writer_flushes_and_waits_for_close() -> None:
    script = textwrap.dedent("""
        import asyncio
        from acp.stdio import stdio_streams

        async def main():
            _, writer = await stdio_streams()
            writer.write(b"x" * (1024 * 1024))
            await writer.drain()
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=2)

        asyncio.run(main())
    """)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], input=b"", capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"x" * (1024 * 1024)


@pytest.mark.skipif(sys.platform == "win32", reason="native POSIX pipe transport")
def test_stdio_writer_drain_fails_when_reader_closes_under_backpressure() -> None:
    script = textwrap.dedent("""
        import asyncio
        import contextlib
        import sys
        from acp.stdio import stdio_streams

        async def main():
            _, writer = await stdio_streams()
            writer.write(b"x" * (1024 * 1024))
            print("ready", file=sys.stderr, flush=True)
            try:
                try:
                    await asyncio.wait_for(writer.drain(), timeout=2)
                except ConnectionError:
                    pass
                else:
                    raise AssertionError("drain succeeded after the reader closed")
            finally:
                writer.close()
                with contextlib.suppress(ConnectionError):
                    await asyncio.wait_for(writer.wait_closed(), timeout=2)

        asyncio.run(main())
    """)
    with subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as child:
        assert child.stdout is not None
        assert child.stderr is not None
        try:
            assert child.stderr.readline() == b"ready\n"
            child.stdout.close()
            assert child.wait(timeout=10) == 0, child.stderr.read().decode()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("write_error", [False, True])
async def test_windows_stdio_writer_reports_close(monkeypatch: pytest.MonkeyPatch, write_error: bool) -> None:
    class Output(io.BytesIO):
        def write(self, data: bytes) -> int:
            if write_error:
                raise BrokenPipeError("reader closed")
            return super().write(data)

    output = Output()
    # Exercise the public Windows adapter on controlled standard handles.
    monkeypatch.setattr("acp.stdio.platform.system", lambda: "Windows")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=output, flush=output.flush))
    _, writer = await stdio_streams()
    writer.write(b"hello")
    if write_error:
        with pytest.raises(ConnectionError):
            await writer.drain()
        with pytest.raises(BrokenPipeError):
            await asyncio.wait_for(writer.wait_closed(), timeout=2)
    else:
        await writer.drain()
        writer.close()
        writer.close()
        await asyncio.wait_for(writer.wait_closed(), timeout=2)
        assert output.getvalue() == b"hello"
