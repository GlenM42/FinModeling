"""
Per-user, in-memory AI-secretary chat history. Resets on process restart — no DB
schema changes, acceptable since each request is a fresh, self-contained ask.
"""

import time
from dataclasses import dataclass, field

from bot.secretary.config import MAX_SESSION_TURNS


@dataclass
class Session:
    user_id: int
    messages: list[dict]
    last_active: float = field(default_factory=time.time)


_sessions: dict[int, Session] = {}


def get_session(user_id: int, system_prompt: str) -> Session:
    session = _sessions.get(user_id)
    if session is None:
        session = Session(user_id=user_id, messages=[{"role": "system", "content": system_prompt}])
        _sessions[user_id] = session
    return session


def append_user(session: Session, text: str) -> None:
    session.messages.append({"role": "user", "content": text})
    session.last_active = time.time()


def append_raw(session: Session, message: dict) -> None:
    """Append an assistant-with-tool_calls message or a tool-result message."""
    session.messages.append(message)
    session.last_active = time.time()


def reset_session(user_id: int) -> None:
    _sessions.pop(user_id, None)


def get_context_for_api(session: Session) -> list[dict]:
    """Returns [system_prompt] + the last MAX_SESSION_TURNS whole turns. A 'turn'
    is one user message plus everything that follows it (assistant tool_calls,
    tool results, final assistant reply) up to the next user message. Trimming
    always drops whole turns from the front — never mid-turn — since a naive
    fixed-size trim could orphan a `tool` message from its paired `assistant`
    tool-call message and produce an invalid message sequence for the API."""
    system = session.messages[0]
    rest = session.messages[1:]

    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    if len(user_indices) > MAX_SESSION_TURNS:
        cutoff = user_indices[-MAX_SESSION_TURNS]
        rest = rest[cutoff:]

    return [system] + rest
