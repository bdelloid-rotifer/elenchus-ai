"""
Turns one user message into one displayed persona reply.

This is the only module that talks to the Gemma 4 model. It is responsible
for:
  1. Assembling the message list sent to Ollama: the active persona's system
     prompt (re-included every turn, per the spec) plus a bounded window of
     recent conversation history plus the new user message.
  2. Requesting strict JSON output from the model so the "response" text and
     any control signals are cleanly separable.
  3. Defensively sanitizing that output before it is shown to anyone: the
     model is instructed not to leak its reasoning or exceed two paragraphs,
     but this module does not simply trust that instruction was followed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from config import (
    MAX_HISTORY_TURNS_SENT_TO_MODEL,
    MAX_RESPONSE_PARAGRAPHS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
)
from models import ChatSession
from personas import PERSONAS


@dataclass
class TurnResult:
    """What the rest of the app needs after one model turn."""

    display_text: str
    signals: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# The model tag actually sent to POST /api/chat. Defaults to config.py's
# bare OLLAMA_MODEL, but app.py overrides this at startup via
# configure_active_model() once ollama_manager.ensure_ready() has resolved
# the exact tag Ollama has pulled locally (see ollama_manager.py's
# find_matching_local_model() docstring for why the bare configured name
# and the exact local tag can differ, e.g. "gemma4" vs. "gemma4:26b").
# Module-level and mutable (rather than re-imported per call) so this one
# override applies to every subsequent generate_reply() call for the rest
# of the process's lifetime, without threading a parameter through app.py's
# request handlers.
_active_model = OLLAMA_MODEL


def configure_active_model(model_tag: str) -> None:
    """Override the exact model tag used for every future /api/chat call."""
    global _active_model
    _active_model = model_tag


def _enforce_paragraph_limit(text: str) -> str:
    """
    Defensive backstop for "responses should never exceed two paragraphs".

    Splits on blank lines (the natural paragraph boundary) and keeps only
    the first MAX_RESPONSE_PARAGRAPHS. This only ever trims; it never pads
    or alters wording, so it cannot introduce new content into a response
    that was never shown to the model's own judgment.
    """
    paragraphs = [p for p in text.strip().split("\n\n") if p.strip()]
    if len(paragraphs) <= MAX_RESPONSE_PARAGRAPHS:
        return text.strip()
    return "\n\n".join(paragraphs[:MAX_RESPONSE_PARAGRAPHS])


def _parse_model_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse the model's JSON output, tolerating the common way small local
    models deviate from "output only JSON": wrapping it in a markdown code
    fence. Raises ValueError if no JSON object can be recovered at all.
    """
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        # Strip a leading ```json / ``` fence and a trailing ``` fence.
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[len("json"):]
        candidate = candidate.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc


def _build_messages(session: ChatSession, user_text: str) -> list[dict]:
    """Assemble the full message list to send to Ollama for this turn."""
    persona = PERSONAS[session.mode]

    messages = [{"role": "system", "content": persona.system_prompt}]

    for message in session.recent_history(MAX_HISTORY_TURNS_SENT_TO_MODEL):
        messages.append({"role": message.role, "content": message.content})

    messages.append({"role": "user", "content": user_text})
    return messages


def generate_reply(session: ChatSession, user_text: str) -> TurnResult:
    """
    Call Gemma 4 (via Ollama) for the next persona reply in this session.

    Does not mutate session.history -- the caller (app.py) is responsible
    for recording both the user's message and the resulting display_text
    once it knows the turn succeeded, keeping this function a pure
    request/response step that is easy to test in isolation.
    """
    messages = _build_messages(session, user_text)

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": _active_model,
                "messages": messages,
                "format": "json",
                "stream": False,
            },
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return TurnResult(
            display_text=(
                "I'm having trouble reaching the local model right now. "
                "Make sure Ollama is running and try again in a moment."
            ),
            error=str(exc),
        )

    raw_content = response.json().get("message", {}).get("content", "")

    try:
        parsed = _parse_model_json(raw_content)
    except ValueError as exc:
        # A malformed response is shown to the user as-is (best effort) but
        # flagged as an error so app.py can log it; the paragraph limit
        # still applies so a malfunctioning model can't flood the chat.
        return TurnResult(
            display_text=_enforce_paragraph_limit(raw_content),
            error=str(exc),
        )

    display_text = _enforce_paragraph_limit(str(parsed.get("response", "")).strip())
    signals = {key: value for key, value in parsed.items() if key != "response"}

    return TurnResult(display_text=display_text, signals=signals)
