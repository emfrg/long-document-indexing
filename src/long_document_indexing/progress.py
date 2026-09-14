from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress


async def await_with_progress[T](
    awaitable: Awaitable[T],
    *,
    emit: Callable[[str], None] | None,
    message: str,
    heartbeat_seconds: float = 60.0,
) -> T:
    """Await remote work while periodically reporting that it is still active."""

    if emit is None:
        return await awaitable
    if heartbeat_seconds <= 0:
        raise ValueError("progress heartbeat interval must be positive")

    emit(message)
    started = time.monotonic()
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=heartbeat_seconds,
                )
            except TimeoutError:
                elapsed = round(time.monotonic() - started)
                emit(f"{message} ({elapsed}s elapsed)")
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        raise
