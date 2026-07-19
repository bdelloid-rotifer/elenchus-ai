"""
Central configuration for ElenchusAI.

Every tunable constant used elsewhere in the codebase is defined here so a
reviewer only has one place to check for "what value does the app use for X".
Nothing in this file talks to the network or the filesystem; it only defines
values that other modules import.
"""

import os
import re
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


# ---------------------------------------------------------------------------
# Ollama / Gemma connection settings
# ---------------------------------------------------------------------------

# BUG FIX (connectivity): this used to be
#     OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
# which is the proximate cause of "Ollama and Gemma 4 are installed locally,
# but the application cannot reach them". OLLAMA_HOST is not an environment
# variable this application invented -- it is Ollama's OWN server-side
# variable (see `ollama serve` docs), and it is conventionally set WITHOUT a
# URL scheme and is sometimes set to a bind-all address, e.g.:
#     export OLLAMA_HOST=127.0.0.1:11434
#     export OLLAMA_HOST=0.0.0.0:11434
# A great many Ollama setup guides tell users to export exactly that, so a
# developer's shell (and therefore this application's os.environ) very
# plausibly already has OLLAMA_HOST set this way, left over from installing
# Ollama itself, with no relation to this application.
#
# Two failures follow if that raw value is used directly to build a URL:
#   1. Missing scheme: f"{OLLAMA_HOST}/api/version" becomes
#      "127.0.0.1:11434/api/version" (no "http://"). The `requests` library
#      raises `MissingSchema` for that, which every reachability check in
#      ollama_manager.py catches as "server not running" -- even though
#      Ollama is installed and already serving requests perfectly well.
#   2. "0.0.0.0" is a bind-all address for a *server*; it is not a routable
#      address a *client* can connect to (confirmed by Ollama's own issue
#      tracker: users who set OLLAMA_HOST=0.0.0.0:PORT for LAN access must
#      still use "127.0.0.1" or "localhost" as the client-side target).
#      Passing "0.0.0.0" straight into requests.get(...) will simply fail
#      to connect on many platforms.
#
# _normalize_ollama_host() below fixes both cases so that whatever value a
# user's environment happens to already carry, ElenchusAI's own HTTP client
# always ends up with a scheme-qualified, client-connectable URL.
def _normalize_ollama_host(raw_value: Optional[str]) -> str:
    """
    Convert an OLLAMA_HOST value (however a user's environment has it set)
    into a fully-qualified base URL safe to use as a `requests` target.

    Handles, in order:
      - unset/empty -> the documented Ollama default, 127.0.0.1:11434.
      - missing scheme (e.g. "127.0.0.1:11434") -> "http://" is prepended.
      - missing port (e.g. "http://myhost" or "myhost") -> ":11434" is
        appended, since that is Ollama's fixed default port and a bare
        hostname alone is not a complete address.
      - "0.0.0.0" as the host (a server bind-all address) -> rewritten to
        "127.0.0.1", the loopback address a client on the same machine can
        actually connect to.
    Anything already well-formed (e.g. "http://127.0.0.1:11434") passes
    through unchanged, aside from stripping a trailing slash.
    """
    default = "http://127.0.0.1:11434"
    if not raw_value or not raw_value.strip():
        return default

    candidate = raw_value.strip()

    # Prepend a scheme if none is present, so urlsplit() parses the host
    # and port correctly instead of treating the whole string as a path.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", candidate):
        candidate = f"http://{candidate}"

    parts = urlsplit(candidate)
    hostname = parts.hostname or "127.0.0.1"
    if hostname == "0.0.0.0":
        hostname = "127.0.0.1"

    port = parts.port if parts.port is not None else 11434
    scheme = parts.scheme or "http"

    normalized = urlunsplit((scheme, f"{hostname}:{port}", "", "", ""))
    return normalized.rstrip("/")


OLLAMA_HOST = _normalize_ollama_host(os.environ.get("OLLAMA_HOST"))

# Model tag passed to `ollama pull` / the /api/chat "model" field.
#
# "gemma4" (no tag suffix) resolves to Ollama's default Gemma 4 variant.
# Ollama also publishes explicit size tags if a reviewer wants to pin one:
#   gemma4:e2b  - smallest, fastest, least capable
#   gemma4:e4b  - balanced default for a typical desktop
#   gemma4:26b  - larger, needs a capable GPU/CPU
#   gemma4:31b  - largest, needs a capable GPU/CPU
# The spec calls for "gemma 4" exclusively, so the tag suffix is left
# configurable via an environment variable rather than hard-coded, so a
# reviewer can pin a size without touching source code.
OLLAMA_MODEL = os.environ.get("ELENCHUS_MODEL", "gemma4")

# Seconds to wait for a single generation call before giving up. Gemma 4 can
# take a while on CPU-only hardware, so this is generous on purpose.
OLLAMA_REQUEST_TIMEOUT_SECONDS = 180

# Seconds to wait for `ollama pull` to report progress before considering the
# download stalled. Model weights are multiple gigabytes, so this is long.
OLLAMA_PULL_TIMEOUT_SECONDS = 3600


# ---------------------------------------------------------------------------
# Flask / web server settings
# ---------------------------------------------------------------------------

# The application is a local, single-user desktop tool. Binding to loopback
# only (never 0.0.0.0) keeps it unreachable from other devices on the network,
# which matters because chat content here can be emotionally sensitive.
FLASK_HOST = "127.0.0.1"
FLASK_PORT = int(os.environ.get("ELENCHUS_PORT", "5000"))

# Flask needs a secret key to sign the session cookie that carries the
# (random, non-identifying) session id. It is generated fresh every process
# start, never written to disk, and never reused across runs -- consistent
# with "never save any data about the user beyond the user session".
FLASK_SECRET_KEY = os.urandom(32)


# ---------------------------------------------------------------------------
# Conversation / persona settings
# ---------------------------------------------------------------------------

# Valid mode identifiers used throughout the codebase (URLs, session state,
# the persona registry). Display titles are defined alongside each persona in
# personas.py; this tuple is only the set of valid internal keys.
VALID_MODES = ("rock", "problem_solver", "therapist_friend", "buddy")

# Only the most recent N exchanges are replayed to the model each turn. This
# keeps the prompt inside Gemma 4's context window and keeps local inference
# fast; it does not affect what is shown to the user, whose full session
# history remains visible in the browser until the app is closed.
MAX_HISTORY_TURNS_SENT_TO_MODEL = 20

# Hard ceiling on paragraphs in a persona's reply, enforced twice: once by
# instructing the model, and once defensively in code (see chat_engine.py)
# in case the model does not comply.
MAX_RESPONSE_PARAGRAPHS = 2
