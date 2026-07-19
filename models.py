"""
Data structures for a single user's chat session.

ElenchusAI never persists conversation data to disk or to a database. A
session lives only in this process's memory for as long as the Flask process
is running, keyed by a random id stored in the user's session cookie. When
the application terminates, the process memory is released and every
transcript disappears with it -- satisfying "the application will never save
any data about the user beyond the user session".
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """
    A single turn in a conversation.

    role is either "user" or "assistant". "assistant" messages store only
    the text that was actually shown to the user -- never the raw model
    output before signal-extraction/truncation -- so replaying history back
    to the model can never leak an earlier hidden "thought process".
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    """
    Everything ElenchusAI knows about one open conversation.

    mode: the active persona key, one of config.VALID_MODES. This can change
    mid-conversation only through the Buddy-initiated mode-switch flow.
    history: the full transcript, oldest first. Never truncated in storage;
    chat_engine.py decides how much of it to replay to the model each turn.
    awaiting_action_prompt: True once the Rock/Therapist Friend persona has
    asked "what are you going to do about it", so the front end knows to
    keep showing the "I don't know, you tell me" helper button.
    pending_mode_suggestion: set by the Buddy persona when it proposes
    switching modes, so the front end can render the small switch-mode menu;
    cleared as soon as the user accepts or declines.
    """

    mode: str
    history: List[Message] = field(default_factory=list)
    awaiting_action_prompt: bool = False
    pending_mode_suggestion: Optional[str] = None

    def add_message(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))

    def recent_history(self, max_turns: int) -> List[Message]:
        """Return the last max_turns messages, oldest first."""
        if max_turns <= 0:
            return []
        return self.history[-max_turns:]


class SessionStore:
    """
    Thread-safe in-memory map of session_id -> ChatSession.

    Flask's development server can service requests on multiple threads, so a
    lock guards every read/modify/write against the underlying dict. This
    store is intentionally never backed by a file or database: losing the
    dict (process exit, crash, restart) is the intended way session data is
    "deleted upon application termination".
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def create(self, mode: str) -> str:
        """Create a brand-new session in the given mode and return its id."""
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = ChatSession(mode=mode)
        return session_id

    def get(self, session_id: Optional[str]) -> Optional[ChatSession]:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: Optional[str]) -> None:
        """Explicitly drop a session, e.g. when the user ends the chat."""
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Wipe every in-memory session, e.g. on application shutdown."""
        with self._lock:
            self._sessions.clear()


# A single process-wide store. Imported by app.py. Deliberately a plain
# module-level instance rather than a Flask extension: this app has no
# database and no multi-process deployment target, so the simplest possible
# shared state is the appropriate choice.
session_store = SessionStore()
