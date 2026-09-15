"""
Tests for the conversation manager. These need no model — they check that our
context-window logic behaves: the system prompt is always present, recent turns
survive, and old turns fall out once the window is full.
"""
from app.conversation import ConversationManager, estimate_tokens
from app.config import settings


def test_system_prompt_always_first():
    cm = ConversationManager()
    s = cm.create_session()
    s.add("user", "hello")
    prompt = cm.build_prompt(s)
    assert prompt[0]["role"] == "system"
    assert "PulsePoint" in prompt[0]["content"]


def test_recent_message_kept_even_if_huge():
    cm = ConversationManager()
    s = cm.create_session()
    # A single message larger than the whole budget must still be included,
    # otherwise the model would have nothing to answer.
    big = "word " * (settings.CONTEXT_TOKEN_BUDGET * 5)
    s.add("user", big)
    prompt = cm.build_prompt(s)
    assert prompt[-1]["content"] == big


def test_old_turns_are_trimmed():
    cm = ConversationManager()
    s = cm.create_session()
    for i in range(settings.MAX_HISTORY_MESSAGES + 10):
        s.add("user", f"message number {i}")
        s.add("assistant", f"reply number {i}")
    prompt = cm.build_prompt(s)
    history_msgs = prompt[1:]  # drop system
    assert len(history_msgs) <= settings.MAX_HISTORY_MESSAGES
    # The most recent reply must be in the window.
    assert any("reply number" in m["content"] for m in history_msgs)
    # The very first message should have fallen out.
    assert all("message number 0" != m["content"] for m in history_msgs)


def test_reset_clears_history():
    cm = ConversationManager()
    s = cm.create_session()
    s.add("user", "remember this")
    s2 = cm.reset_session(s.session_id)
    assert s2.history == []
    assert cm.build_prompt(s2) == [
        {"role": "system", "content": cm.system_prompt}
    ]


def test_estimate_tokens_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)
