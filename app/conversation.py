"""
Conversation manager: sessions, history, and context-window management.

Responsibilities:
  * Hold per-session dialogue history (one Session object per chat).
  * Build the message list we actually send to the model each turn, keeping the
    system prompt pinned and trimming old turns so we stay inside a token budget.
  * Enforce turn-taking (append user, then assistant) cleanly.

No tools, no RAG — the manager only shapes prompts and remembers history.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from .config import settings
from .domain import ACTIVE_DOMAIN


def estimate_tokens(text: str) -> int:
    """Rough token estimate without pulling in a tokenizer.

    A common rule of thumb for English is ~4 characters per token. We only need
    this to be approximately right to keep the prompt under the model's window,
    so a cheap heuristic is fine and keeps the code dependency-free.
    """
    return max(1, len(text) // 4)


@dataclass
class Message:
    role: str          # "user" or "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    history: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))


class ConversationManager:
    """Owns every active session and builds prompts for the LLM."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self.system_prompt = ACTIVE_DOMAIN.system_prompt
        self._system_tokens = estimate_tokens(self.system_prompt)

    # ---- session lifecycle -------------------------------------------------
    def create_session(self) -> Session:
        sid = uuid.uuid4().hex
        session = Session(session_id=sid)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Session:
        """Return the session, creating it if we've never seen this id.

        Being forgiving here means a client can reconnect with an old id and
        keep chatting instead of erroring out.
        """
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return session

    def reset_session(self, session_id: str) -> Session:
        """Wipe history for a session (the UI 'New chat' button calls this)."""
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    # ---- prompt construction ----------------------------------------------
    def build_prompt(self, session: Session) -> List[dict]:
        """Return the messages list to send to the model.

        Context-window strategy (a "pinned system prompt + sliding window"):
          1. The system prompt is ALWAYS included first. It carries the persona,
             the domain knowledge, and the guardrails, so it must never be
             dropped — losing it is how a bot forgets it's a gym assistant.
          2. We then walk the history newest-first, adding messages until either
             (a) we hit MAX_HISTORY_MESSAGES, or (b) adding the next message
             would push us past CONTEXT_TOKEN_BUDGET.
          3. Whatever survived is reversed back into chronological order.

        Result: recent turns are always present (so the bot stays coherent about
        what was just said), and old turns fall out of the window gracefully as
        the chat grows, keeping latency and memory bounded on CPU.
        """
        budget = settings.CONTEXT_TOKEN_BUDGET - self._system_tokens
        kept: List[Message] = []
        used = 0

        for msg in reversed(session.history):
            if len(kept) >= settings.MAX_HISTORY_MESSAGES:
                break
            cost = estimate_tokens(msg.content)
            if used + cost > budget and kept:
                # Stop once we'd overflow — but always keep at least the most
                # recent message so the model has something to respond to.
                break
            kept.append(msg)
            used += cost

        kept.reverse()

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": m.role, "content": m.content} for m in kept)
        return messages

    def history_as_dicts(self, session: Session) -> List[dict]:
        """Full history for the UI (not trimmed) — used by the REST endpoint."""
        return [{"role": m.role, "content": m.content} for m in session.history]
