"""WebSocket progress endpoint — fan-out from the ingest worker to all clients.

Architecture: a single ``relay_progress`` background task drains the worker's
``multiprocessing.Queue`` and broadcasts events to every connected client.
One drainer prevents events from being stolen by per-client pollers.
"""

from __future__ import annotations

import asyncio
import queue

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api import state

router = APIRouter()


@router.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and register it for progress fan-out.

    Events are pushed by :func:`relay_progress`.  This coroutine keeps the
    connection alive with periodic pings and removes the client on disconnect.
    """
    await websocket.accept()
    async with state.ws_clients_lock:
        state.ws_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        async with state.ws_clients_lock:
            state.ws_clients.discard(websocket)


async def relay_progress() -> None:
    """Single background task: drain the worker mp.Queue and broadcast to all WS clients.

    MUST run as exactly ONE asyncio.Task — not per-client — because
    ``multiprocessing.Queue`` has exactly one consumer in this process.
    Runs until cancelled (typically on app shutdown).
    """
    while True:
        h = state.worker_handle
        if h is None:
            await asyncio.sleep(0.1)
            continue

        try:
            # Drain up to 50 events per tick (30 ms budget from Track D1)
            for _ in range(50):
                event: dict = h.progress_queue.get_nowait()
                _update_ingest_state(event)
                await _broadcast(event)
        except queue.Empty:
            pass

        await asyncio.sleep(0.03)  # 30 ms — within the 20–50 ms WebSocket budget


def _update_ingest_state(event: dict) -> None:
    """Update module-level ingest counters from an incoming worker event."""
    etype = event.get("type", "")
    if etype == "chunk_start":
        state.current_chunk = event.get("chunk_index")
        if state.total_chunks is None:
            state.total_chunks = event.get("total_chunks")
    elif etype == "chunk_done":
        if event.get("status") == "accepted":
            state.chunks_accepted += 1
        else:
            state.chunks_rejected += 1
    elif etype == "ingest_complete":
        state.chunks_accepted = event.get("chunks_accepted", state.chunks_accepted)
        state.chunks_rejected = event.get("chunks_rejected", state.chunks_rejected)


async def _broadcast(event: dict) -> None:
    """Send *event* to every connected WebSocket client; prune dead connections."""
    async with state.ws_clients_lock:
        dead: list[WebSocket] = []
        for ws in state.ws_clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.ws_clients.discard(ws)
