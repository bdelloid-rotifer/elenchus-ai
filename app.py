"""
ElenchusAI Flask application.

Route map:
  GET  /                 landing page: "What do you need today?"
  POST /start             create a session in the chosen mode, then redirect
  GET  /chat               chat UI for the active session
  POST /api/message        send one user message, get one persona reply
  POST /api/switch_mode     Buddy-initiated mode switch (or decline)
  POST /api/end             explicitly end and discard the current session

Run directly (`python app.py`): this first verifies Ollama and the Gemma 4
model are available on the device (installing/downloading them if not, per
ollama_manager.py) before starting the web server, since the app cannot
function without them.
"""

from __future__ import annotations

import atexit
import sys

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for

import chat_engine
import ollama_manager
import safety
from config import FLASK_HOST, FLASK_PORT, FLASK_SECRET_KEY, VALID_MODES
from models import ChatSession, session_store
from personas import LANDING_OPTIONS, PERSONAS

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Modes a Buddy conversation is allowed to switch into. Buddy itself is
# excluded since the whole point of the flow is leaving Buddy for one of the
# more focused personas.
_SWITCHABLE_MODES = tuple(mode for mode in VALID_MODES if mode != "buddy")


def _current_session() -> ChatSession | None:
    """Look up the ChatSession for this browser's session cookie, if any."""
    return session_store.get(session.get("sid"))


@app.route("/", methods=["GET"])
def landing_page():
    """The "What do you need today?" mode-selection screen."""
    options = [
        {"mode": opt["mode"], "label": opt["label"], "title": PERSONAS[opt["mode"]].title}
        for opt in LANDING_OPTIONS
    ]
    return render_template("index.html", options=options)


@app.route("/start", methods=["POST"])
def start_session():
    """Create a fresh in-memory session in the chosen mode."""
    mode = request.form.get("mode", "")
    if mode not in VALID_MODES:
        abort(400, description="Unknown mode selected.")

    # Starting a brand-new conversation discards any previous one this
    # browser had, consistent with "never save data beyond the user
    # session": there is only ever one live session per browser cookie.
    session_store.delete(session.get("sid"))
    session["sid"] = session_store.create(mode)
    return redirect(url_for("chat_page"))


@app.route("/chat", methods=["GET"])
def chat_page():
    """Render the chat UI, seeded with any history already in the session."""
    chat_session = _current_session()
    if chat_session is None:
        return redirect(url_for("landing_page"))

    persona = PERSONAS[chat_session.mode]
    return render_template(
        "chat.html",
        mode=persona.key,
        mode_title=persona.title,
        history=[{"role": m.role, "content": m.content} for m in chat_session.history],
        awaiting_action_prompt=chat_session.awaiting_action_prompt,
        pending_mode_suggestion=chat_session.pending_mode_suggestion,
        switchable_modes=[
            {"mode": key, "title": PERSONAS[key].title} for key in _SWITCHABLE_MODES
        ],
    )


@app.route("/api/message", methods=["POST"])
def post_message():
    """Handle one user chat turn and return the persona's reply."""
    chat_session = _current_session()
    if chat_session is None:
        return jsonify(error="No active session."), 401

    payload = request.get_json(silent=True) or {}
    user_text = str(payload.get("text", "")).strip()
    if not user_text:
        return jsonify(error="Empty message."), 400

    assessment = safety.assess_input(user_text)
    if assessment.is_crisis:
        # Deterministic path: never touches the model. The persona's
        # personality is intentionally dropped for this turn.
        reply_text = safety.build_crisis_response(assessment.category)
        chat_session.add_message("user", user_text)
        chat_session.add_message("assistant", reply_text)
        return jsonify(
            response=reply_text,
            mode=chat_session.mode,
            mode_title=PERSONAS[chat_session.mode].title,
            ready_for_action_question=chat_session.awaiting_action_prompt,
            suggest_switch=False,
            suggested_mode=None,
            crisis=True,
        )

    result = chat_engine.generate_reply(chat_session, user_text)
    if result.error:
        # Logged server-side only; the user still sees a graceful message.
        print(f"[chat_engine warning] {result.error}", file=sys.stderr)

    chat_session.add_message("user", user_text)
    chat_session.add_message("assistant", result.display_text)

    if chat_session.mode == "buddy":
        suggest_switch = bool(result.signals.get("suggest_switch", False))
        suggested_mode = result.signals.get("suggested_mode")
        chat_session.pending_mode_suggestion = (
            suggested_mode if suggest_switch and suggested_mode in _SWITCHABLE_MODES else None
        )
    else:
        suggest_switch, suggested_mode = False, None
        if "ready_for_action_question" in result.signals:
            chat_session.awaiting_action_prompt = bool(
                result.signals["ready_for_action_question"]
            )

    return jsonify(
        response=result.display_text,
        mode=chat_session.mode,
        mode_title=PERSONAS[chat_session.mode].title,
        ready_for_action_question=chat_session.awaiting_action_prompt,
        suggest_switch=suggest_switch,
        suggested_mode=suggested_mode,
        crisis=False,
    )


@app.route("/api/switch_mode", methods=["POST"])
def switch_mode():
    """
    Act on the Buddy-initiated mode-switch prompt: either move the session
    into a new persona or dismiss the suggestion and keep chatting as Buddy.
    """
    chat_session = _current_session()
    if chat_session is None:
        return jsonify(error="No active session."), 401

    if chat_session.pending_mode_suggestion is None:
        return jsonify(error="No mode switch is pending."), 400

    payload = request.get_json(silent=True) or {}
    choice = payload.get("new_mode")

    if choice == "decline":
        chat_session.pending_mode_suggestion = None
        return jsonify(declined=True)

    if choice not in _SWITCHABLE_MODES:
        return jsonify(error="Invalid mode choice."), 400

    # History is deliberately left untouched: the new persona will see the
    # full prior conversation, including the Buddy exchange, as context.
    chat_session.mode = choice
    chat_session.pending_mode_suggestion = None
    chat_session.awaiting_action_prompt = False

    persona = PERSONAS[choice]
    return jsonify(
        mode=persona.key,
        mode_title=persona.title,
        notice=f"Mode changed to {persona.title}.",
    )


@app.route("/api/end", methods=["POST"])
def end_session():
    """Explicitly discard the current session's in-memory transcript."""
    session_store.delete(session.get("sid"))
    session.pop("sid", None)
    return jsonify(ended=True)


if __name__ == "__main__":
    print("Checking for Ollama and the Gemma 4 model...")
    readiness = ollama_manager.ensure_ready()
    print(readiness.detail)

    if not readiness.model_available:
        print(
            "ElenchusAI cannot start without a working Ollama installation "
            "and the Gemma 4 model. Resolve the issue above and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    # readiness.resolved_model is the exact local tag Ollama reported (e.g.
    # "gemma4:26b"), which can differ from config.py's bare OLLAMA_MODEL.
    # Every /api/chat call must use that exact tag or requests can 404 even
    # though the model is unquestionably installed -- see
    # ollama_manager.find_matching_local_model()'s docstring.
    chat_engine.configure_active_model(readiness.resolved_model)

    # Belt-and-suspenders: even though process exit already releases all
    # session memory, this makes the "delete on termination" behavior an
    # explicit, visible step rather than an implicit side effect of the OS
    # reclaiming memory.
    atexit.register(session_store.clear_all)

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)
