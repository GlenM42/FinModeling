from bot.secretary import sessions


def test_get_session_creates_and_reuses():
    sessions.reset_session(42)
    s1 = sessions.get_session(42, "sys")
    s2 = sessions.get_session(42, "sys - ignored on reuse since session already exists")

    assert s1 is s2
    assert s1.messages[0] == {"role": "system", "content": "sys"}
    sessions.reset_session(42)


def test_append_user_and_raw():
    sessions.reset_session(7)
    session = sessions.get_session(7, "sys")
    sessions.append_user(session, "hello")
    sessions.append_raw(session, {"role": "assistant", "content": "hi"})

    assert session.messages[-2] == {"role": "user", "content": "hello"}
    assert session.messages[-1] == {"role": "assistant", "content": "hi"}
    sessions.reset_session(7)


def test_get_context_for_api_never_orphans_a_tool_message(monkeypatch):
    monkeypatch.setattr(sessions, "MAX_SESSION_TURNS", 3)
    sessions.reset_session(1)
    session = sessions.get_session(1, "system prompt")

    for i in range(6):
        sessions.append_user(session, f"question {i}")
        sessions.append_raw(session, {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": str(i), "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        })
        sessions.append_raw(session, {"role": "tool", "tool_call_id": str(i), "content": "result"})
        sessions.append_raw(session, {"role": "assistant", "content": f"reply {i}"})

    context = sessions.get_context_for_api(session)

    assert context[0] == {"role": "system", "content": "system prompt"}
    # First non-system message must be a user message — never an orphaned tool/assistant message
    assert context[1]["role"] == "user"

    user_count = sum(1 for m in context if m.get("role") == "user")
    assert user_count == 3  # trimmed to MAX_SESSION_TURNS whole turns

    sessions.reset_session(1)


def test_get_context_for_api_keeps_everything_under_the_limit():
    sessions.reset_session(2)
    session = sessions.get_session(2, "sys")
    sessions.append_user(session, "hi")
    sessions.append_raw(session, {"role": "assistant", "content": "hello"})

    context = sessions.get_context_for_api(session)

    assert context == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    sessions.reset_session(2)
