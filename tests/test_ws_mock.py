"""
End-to-end WebSocket tests that run without Ollama by forcing the mock engine.

They spin up a real uvicorn server in a background thread and connect with a
real WebSocket client — the same path a browser uses — then verify the JSON
protocol: connect -> session, user_message -> start/token/done, reset ->
reset_ack, and that a bad message yields an error instead of crashing.
"""
import asyncio
import contextlib
import json
import os
import socket
import threading
import time

import pytest
import uvicorn
import websockets

# Force the mock engine BEFORE importing the app so get_engine() picks it up.
os.environ["ENGINE"] = "mock"
from app.main import app  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    # Wait for startup.
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


def test_health(server):
    import httpx

    r = httpx.get(f"http://{server}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=15))


def test_ws_full_turn(server):
    async def scenario():
        async with websockets.connect(f"ws://{server}/ws/chat") as ws:
            hello = json.loads(await ws.recv())
            assert hello["type"] == "session" and hello["session_id"]

            await ws.send(json.dumps({"type": "user_message", "content": "What plans?"}))
            assert json.loads(await ws.recv())["type"] == "start"

            got_token = False
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "token":
                    got_token = True
                elif msg["type"] == "done":
                    assert got_token, "expected at least one streamed token"
                    assert msg["content"]
                    return

    _run(scenario())


def test_ws_reset(server):
    async def scenario():
        async with websockets.connect(f"ws://{server}/ws/chat") as ws:
            await ws.recv()  # session
            await ws.send(json.dumps({"type": "reset"}))
            assert json.loads(await ws.recv())["type"] == "reset_ack"

    _run(scenario())


def test_ws_bad_type_is_reported_not_fatal(server):
    async def scenario():
        async with websockets.connect(f"ws://{server}/ws/chat") as ws:
            await ws.recv()  # session
            await ws.send(json.dumps({"type": "nonsense"}))
            assert json.loads(await ws.recv())["type"] == "error"
            # Connection still alive: a valid message still works afterwards.
            await ws.send(json.dumps({"type": "user_message", "content": "hi"}))
            assert json.loads(await ws.recv())["type"] == "start"

    _run(scenario())


def test_ws_empty_message_is_reported(server):
    async def scenario():
        async with websockets.connect(f"ws://{server}/ws/chat") as ws:
            await ws.recv()  # session
            await ws.send(json.dumps({"type": "user_message", "content": "   "}))
            assert json.loads(await ws.recv())["type"] == "error"

    _run(scenario())
