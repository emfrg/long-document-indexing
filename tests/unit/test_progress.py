from __future__ import annotations

import asyncio

from long_document_indexing.progress import await_with_progress


async def test_await_with_progress_emits_initial_message_and_heartbeats() -> None:
    messages: list[str] = []

    async def complete_later() -> str:
        await asyncio.sleep(0.03)
        return "done"

    result = await await_with_progress(
        complete_later(),
        emit=messages.append,
        message="waiting for remote response",
        heartbeat_seconds=0.01,
    )

    assert result == "done"
    assert messages[0] == "waiting for remote response"
    assert any(message.endswith("s elapsed)") for message in messages[1:])


async def test_await_with_progress_is_silent_without_callback() -> None:
    result = await await_with_progress(
        asyncio.sleep(0, result="done"),
        emit=None,
        message="waiting for remote response",
    )

    assert result == "done"
