"""
FastAPI application: REST + WebSocket API and static file serving.

Endpoints
  GET  /                -> serves the chat UI (static/index.html)
  GET  /health          -> liveness check + which model/domain is active
  POST /api/session     -> create a fresh session id
  GET  /api/history/{id}-> full history for a session (REST view of state)
  WS   /ws/chat         -> the real-time streaming chat channel

WebSocket JSON protocol
  client -> server:
    {"type": "user_message", "content": "...", "session_id": "..."}  (session_id optional)
    {"type": "reset", "session_id": "..."}
  server -> client:
    {"type": "session", "session_id": "..."}      # sent on connect
    {"type": "start"}                              # a reply is beginning
    {"type": "token", "content": "..."}            # streamed chunk
    {"type": "done", "content": "<full reply>"}    # reply finished
    {"type": "reset_ack", "session_id": "..."}     # history cleared
    {"type": "error", "message": "..."}            # something went wrong
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .conversation import ConversationManager
from .domain import ACTIVE_DOMAIN
from .llm_engine import LLMError, get_engine

app = FastAPI(title="CPU Local Chatbot", version="1.0.0")

# One manager and one engine for the whole app. The manager holds all sessions;
# the engine is stateless, so sharing a single instance is safe and cheap.
manager = ConversationManager()
engine = get_engine()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# --------------------------------------------------------------------------- #
# REST endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.MODEL_NAME,
        "domain": ACTIVE_DOMAIN.name,
    }


@app.post("/api/session")
async def create_session():
    session = manager.create_session()
    return {"session_id": session.session_id}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    session = manager.get_session(session_id)
    return {"session_id": session_id, "history": manager.history_as_dicts(session)}


# --------------------------------------------------------------------------- #
# WebSocket endpoint
# --------------------------------------------------------------------------- #
@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    # Give the connection its own session up front.
    session = manager.create_session()
    await websocket.send_json({"type": "session", "session_id": session.session_id})

    try:
        while True:
            # ---- receive & validate ----------------------------------------
            try:
                data = await websocket.receive_json()
            except Exception:
                await websocket.send_json(
                    {"type": "error", "message": "Message must be valid JSON."}
                )
                continue

            msg_type = data.get("type")

            # Allow the client to steer which session this socket uses.
            sid = data.get("session_id") or session.session_id
            session = manager.get_session(sid)

            if msg_type == "reset":
                session = manager.reset_session(session.session_id)
                await websocket.send_json(
                    {"type": "reset_ack", "session_id": session.session_id}
                )
                continue

            if msg_type != "user_message":
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type!r}"}
                )
                continue

            user_text = (data.get("content") or "").strip()
            if not user_text:
                await websocket.send_json(
                    {"type": "error", "message": "Empty message ignored."}
                )
                continue

            # ---- run the turn ---------------------------------------------
            session.add("user", user_text)
            prompt = manager.build_prompt(session)

            await websocket.send_json({"type": "start"})
            collected: list[str] = []
            try:
                async for chunk in engine.stream_chat(prompt):
                    collected.append(chunk)
                    await websocket.send_json({"type": "token", "content": chunk})
            except LLMError as exc:
                # Model/backend failed: report it, drop the half-turn from
                # history so the session stays consistent, keep the socket open.
                if session.history and session.history[-1].role == "user":
                    session.history.pop()
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            full_reply = "".join(collected).strip()
            session.add("assistant", full_reply)
            await websocket.send_json({"type": "done", "content": full_reply})

    except WebSocketDisconnect:
        # Client closed the tab / dropped mid-stream. Nothing to clean up beyond
        # letting the coroutine exit; the session stays in memory for reconnect.
        return
    except Exception as exc:  # last-resort guard so the server never crashes
        try:
            await websocket.send_json(
                {"type": "error", "message": f"Unexpected server error: {exc}"}
            )
        except Exception:
            pass


# Serve the rest of the static assets (style.css, app.js) under /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
