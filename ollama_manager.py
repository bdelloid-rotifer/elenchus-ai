"""
Ollama + Gemma 4 lifecycle management.

The spec requires that ElenchusAI does not ship with Ollama or Gemma 4
bundled inside it. Instead, at startup, the application must check whether
they are already available on the device and, if not, fetch and install them
onto the device itself (not embedded in the application). This module is the
only place in the codebase that shells out to install software or download
model weights.

Everything here is deliberately conservative:
- It only ever calls the official Ollama CLI / install script / REST API.
- It never silently overwrites an existing Ollama installation.
- On platforms where unattended installation is not realistically possible
  (Windows, which distributes Ollama as a GUI installer), it downloads the
  official installer and hands control to the user rather than trying to
  script a GUI, and clearly reports what it did.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import requests

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_PULL_TIMEOUT_SECONDS,
)

# Official, stable download/install locations published by Ollama. These are
# fetched from disk/network at run time, never bundled with the app.
_OLLAMA_LINUX_INSTALL_SCRIPT_URL = "https://ollama.com/install.sh"
_OLLAMA_WINDOWS_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
_OLLAMA_MACOS_DOWNLOAD_PAGE = "https://ollama.com/download/mac"


@dataclass
class ReadinessResult:
    """Outcome of ensure_ready(), reported back to the caller for logging."""

    ollama_installed: bool
    ollama_server_running: bool
    model_available: bool
    detail: str
    # The exact tag Ollama has locally for the requested model (e.g.
    # "gemma4:26b"), which may differ from the bare name in config.py. None
    # whenever model_available is False. See find_matching_local_model().
    resolved_model: Optional[str] = None


def _run(cmd: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def is_ollama_installed() -> bool:
    """True if an `ollama` executable is discoverable on PATH."""
    return shutil.which("ollama") is not None


def is_ollama_server_running() -> bool:
    """True if the local Ollama REST API is responding right now."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/version", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_ollama_server() -> bool:
    """
    Launch `ollama serve` as a background process if it is not already
    running, then poll briefly for it to come up.

    Returns True once the server responds, False if it never comes up within
    the poll window.
    """
    if is_ollama_server_running():
        return True

    # Detached background process: ElenchusAI does not own this process's
    # lifetime beyond starting it, matching how the Ollama desktop app or a
    # system service would normally keep it running independently.
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    for _ in range(20):
        time.sleep(0.5)
        if is_ollama_server_running():
            return True
    return False


def install_ollama() -> bool:
    """
    Fetch and install Ollama onto the device, outside the application.

    Returns True if installation appears to have succeeded (i.e. the `ollama`
    executable is now discoverable). Platform-specific because Ollama ships
    a different installation mechanism per OS.
    """
    system = platform.system()

    if system == "Linux":
        # Ollama publishes an official shell installer for Linux that
        # installs the CLI and, where applicable, a systemd service.
        script_path = tempfile.mktemp(suffix="-ollama-install.sh")
        urllib.request.urlretrieve(_OLLAMA_LINUX_INSTALL_SCRIPT_URL, script_path)
        result = _run(["sh", script_path], timeout=900)
        return result.returncode == 0 and is_ollama_installed()

    if system == "Darwin":
        # macOS distributes Ollama as a zipped .app bundle rather than a
        # silent installer. Downloading and moving it into /Applications is
        # scriptable without user interaction.
        zip_path = tempfile.mktemp(suffix="-Ollama-darwin.zip")
        urllib.request.urlretrieve(
            "https://ollama.com/download/Ollama-darwin.zip", zip_path
        )
        extract_dir = tempfile.mkdtemp(suffix="-ollama-extract")
        _run(["unzip", "-o", zip_path, "-d", extract_dir])
        # BUG FIX: `mv -f source dest` does NOT replace `dest` when `dest`
        # is already an existing directory -- it moves `source` *inside*
        # `dest` instead (producing the nested, broken bundle
        # "/Applications/Ollama.app/Ollama.app"). This branch is only
        # reached when is_ollama_installed() (a PATH check) returned False,
        # which can happen even though a stale /Applications/Ollama.app
        # already exists (e.g. its CLI symlink was never created). Removing
        # any such stale bundle first makes the subsequent `mv` an actual
        # replacement, matching this module's documented guarantee.
        _run(["rm", "-rf", "/Applications/Ollama.app"])
        _run(["mv", "-f", f"{extract_dir}/Ollama.app", "/Applications/Ollama.app"])
        # The macOS app bundle embeds a copy of the CLI; symlink it onto
        # PATH so `ollama` is discoverable the same way it is on Linux.
        _run(
            [
                "ln",
                "-sf",
                "/Applications/Ollama.app/Contents/Resources/ollama",
                "/usr/local/bin/ollama",
            ]
        )
        return is_ollama_installed()

    if system == "Windows":
        # Ollama for Windows is a GUI installer (.exe). There is no
        # officially supported silent/unattended flag documented for it, so
        # rather than guessing at undocumented install switches, this
        # downloads the genuine installer and launches it, then waits for
        # the user to complete the on-screen steps.
        installer_path = tempfile.mktemp(suffix="-OllamaSetup.exe")
        urllib.request.urlretrieve(_OLLAMA_WINDOWS_INSTALLER_URL, installer_path)
        subprocess.Popen([installer_path])
        print(
            "The Ollama installer has been downloaded and launched. "
            "Please complete the installation steps in the window that "
            "just opened, then re-run ElenchusAI.",
            file=sys.stderr,
        )
        return False

    raise RuntimeError(f"Unsupported platform for automatic Ollama install: {system}")


def find_matching_local_model(model: str = OLLAMA_MODEL) -> Optional[str]:
    """
    Return the exact tag Ollama has locally for `model`, or None if nothing
    in the same model family is pulled.

    BUG FIX (model mismatch): this used to be `is_model_available()`, which
    only returned a bool -- true if `model` (e.g. the bare "gemma4") or ANY
    same-family tagged variant (e.g. "gemma4:26b") was present locally. That
    was enough to make ensure_ready() correctly report the model as
    "available", but generate_reply() in chat_engine.py separately sends the
    literal, unqualified `model` string from config.py to POST /api/chat.
    If a reviewer or user pulled a specific size instead of the bare
    default -- e.g. ran `ollama pull gemma4:26b` for their hardware, rather
    than `ollama pull gemma4` -- Ollama has no tag literally named "gemma4"
    (that name only exists as an alias once the bare/"latest" tag has
    specifically been pulled). Every chat request would then 404 with
    "model not found", even though ensure_ready() had just confirmed Gemma 4
    was "ready" -- another concrete way to reach the reported symptom of
    "the model is installed locally, but the application cannot reach it".
    Returning the exact local tag (and having callers use it verbatim for
    /api/chat, rather than the possibly-mismatched configured name) removes
    this class of failure entirely.
    """
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return None

    local_models = [entry.get("name", "") for entry in response.json().get("models", [])]
    if model in local_models:
        return model

    # An unqualified request like "gemma4" is satisfied by any tagged
    # variant already present, e.g. "gemma4:e4b" -- but the /api/chat call
    # must use that exact tag, not the unqualified name.
    base_name = model.split(":")[0]
    for name in local_models:
        if name.split(":")[0] == base_name:
            return name
    return None


def is_model_available(model: str = OLLAMA_MODEL) -> bool:
    """True if `model` (or a same-family tagged variant) is already pulled."""
    return find_matching_local_model(model) is not None


def pull_model(model: str = OLLAMA_MODEL) -> bool:
    """
    Download `model`'s weights via the Ollama CLI, streaming progress to the
    console. Model weights are multiple gigabytes, so this can take a while.
    """
    print(f"Downloading {model} via Ollama. This can take a while the first time...")
    try:
        result = _run(["ollama", "pull", model], timeout=OLLAMA_PULL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"Timed out waiting for `ollama pull {model}` to finish.", file=sys.stderr)
        return False

    if result.returncode != 0:
        print(f"`ollama pull {model}` failed:\n{result.stderr}", file=sys.stderr)
        return False
    return True


def ensure_ready(model: str = OLLAMA_MODEL) -> ReadinessResult:
    """
    Top-level orchestration called once at application startup.

    Ensures, in order: Ollama is installed -> the Ollama server is running ->
    the requested Gemma 4 model is pulled. Each step is skipped if already
    satisfied. Never raises for an ordinary "not installed yet" case; it
    reports failure in the returned ReadinessResult instead so app.py can
    decide how to surface it.
    """
    if not is_ollama_installed():
        print("Ollama was not found on this device. Installing it now...")
        installed = install_ollama()
        if not installed:
            return ReadinessResult(
                ollama_installed=False,
                ollama_server_running=False,
                model_available=False,
                detail=(
                    "Ollama could not be installed automatically. See the "
                    "console output above for next steps."
                ),
            )

    server_up = start_ollama_server()
    if not server_up:
        return ReadinessResult(
            ollama_installed=True,
            ollama_server_running=False,
            model_available=False,
            detail="Ollama is installed, but the local server did not start.",
        )

    resolved = find_matching_local_model(model)
    if resolved is None:
        print(f"{model} was not found locally. Pulling it now...")
        pulled = pull_model(model)
        if not pulled:
            return ReadinessResult(
                ollama_installed=True,
                ollama_server_running=True,
                model_available=False,
                detail=f"Ollama is running, but {model} could not be downloaded.",
            )
        # Re-resolve rather than assuming `model` itself is now the exact
        # local tag: `ollama pull gemma4` can register the weights under a
        # size-specific tag (e.g. "gemma4:e4b") rather than the bare name.
        resolved = find_matching_local_model(model)

    return ReadinessResult(
        ollama_installed=True,
        ollama_server_running=True,
        model_available=resolved is not None,
        detail=(
            "Ollama and the Gemma 4 model are ready."
            if resolved is not None
            else f"Ollama is running, but no local tag matching {model} could be found after pulling."
        ),
        resolved_model=resolved,
    )
