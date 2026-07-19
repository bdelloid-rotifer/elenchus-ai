"""
Crisis safeguard layer.

This module is the one part of ElenchusAI that is deliberately *not* left to
the language model's judgment. If a user's message indicates they may be in
danger of harming themselves or someone else, the application must respond
with a fixed, predictable message and real human resources -- every time,
regardless of what persona is active and regardless of how Gemma 4 might
otherwise have chosen to respond. Detection therefore runs as plain Python
pattern matching on the raw user input, entirely before any call to Ollama,
so a jailbroken or unusually-phrased model response can never suppress it.

Two categories are handled here because they are safety-critical and
time-sensitive:
  - self-harm / suicidal ideation
  - intent to harm another person

Everything else the spec calls "not for this application" (homework,
schoolwork, workplace tasks, coding help, lewd content) is deliberately
*not* handled here. Those are scope violations, not emergencies, so they are
instead handled by instructing the persona itself (see personas.py) to
decline and redirect in-character. That keeps this module small, auditable,
and focused only on the cases where a hard-coded, non-model response is the
right design choice.

This is a heuristic keyword/phrase matcher, not a clinical or legal
determination. It is intentionally biased toward false positives (better to
show resources unnecessarily than to miss a real crisis). A production
deployment should pair this with a maintained, professionally-reviewed
classifier; the patterns below are a conservative starting baseline.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class CrisisCategory(Enum):
    SELF_HARM = "self_harm"
    HARM_TO_OTHERS = "harm_to_others"


@dataclass
class SafetyAssessment:
    """Result of screening one piece of user input."""

    is_crisis: bool
    category: Optional[CrisisCategory] = None


# Each pattern is matched case-insensitively against the raw user message.
# Patterns are phrase-level (not single words) to keep the false-positive
# rate manageable -- e.g. matching "kill myself" but not the standalone word
# "kill", which appears in countless harmless sentences.
_SELF_HARM_PATTERNS: List[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bkill(ing)? myself\b",
        r"\bend(ing)? my life\b",
        r"\bwant(ing)? to die\b",
        r"\bdon'?t want to (be alive|live anymore|exist anymore)\b",
        r"\b(commit(ting)?|attempt(ing)?) suicide\b",
        r"\bsuicidal\b",
        r"\bself[\s-]?harm(ing)?\b",
        r"\bhurt(ing)? myself\b",
        r"\bcut(ting)? myself\b",
        r"\bno reason to (go on|keep living|live)\b",
        r"\bbetter off (dead|without me)\b",
        r"\bplan to (die|end it)\b",
        r"\bgoing to end it\b",
    ]
]

_HARM_TO_OTHERS_PATTERNS: List[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bkill (him|her|them|someone|somebody)\b",
        r"\bhurt (him|her|them|someone|somebody) (badly|seriously)\b",
        r"\bgoing to (hurt|attack|kill) (him|her|them)\b",
        r"\bmake (him|her|them) pay\b.*\b(hurt|kill|attack)\b",
        r"\bwant(ing)? to hurt (someone|somebody|him|her|them)\b",
    ]
]


def assess_input(user_text: str) -> SafetyAssessment:
    """
    Screen one user message for crisis indicators.

    Checks self-harm patterns before harm-to-others patterns; if a message
    somehow matches both, self-harm takes priority since ElenchusAI's primary
    duty of care is to the user in front of it.
    """
    for pattern in _SELF_HARM_PATTERNS:
        if pattern.search(user_text):
            return SafetyAssessment(is_crisis=True, category=CrisisCategory.SELF_HARM)

    for pattern in _HARM_TO_OTHERS_PATTERNS:
        if pattern.search(user_text):
            return SafetyAssessment(
                is_crisis=True, category=CrisisCategory.HARM_TO_OTHERS
            )

    return SafetyAssessment(is_crisis=False)


# Fixed resource list. Deliberately US-focused, well-established, toll-free
# services; a deployment targeting other regions should localize this list.
# Kept as plain data (not prose baked into a template string) so a reviewer
# or future maintainer can update a single source of truth.
CRISIS_RESOURCES = [
    {
        "name": "988 Suicide & Crisis Lifeline",
        "contact": "Call or text 988",
        "notes": "Free, confidential, available 24/7 in the US.",
    },
    {
        "name": "Crisis Text Line",
        "contact": "Text HOME to 741741",
        "notes": "Free, 24/7 support over text message.",
    },
    {
        "name": "SAMHSA National Helpline",
        "contact": "1-800-662-4357",
        "notes": "Free, confidential treatment referral and information, 24/7.",
    },
    {
        "name": "Emergency services",
        "contact": "911 (or your local emergency number)",
        "notes": "Use this if someone is in immediate danger right now.",
    },
]


def build_crisis_response(category: CrisisCategory) -> str:
    """
    Compose the fixed, in-character-dropping message shown to the user.

    This text is never generated by the model. It always: (1) tells the user
    plainly that something in their message raised a safety concern, (2)
    names the concern in plain terms rather than clinical jargon, and (3)
    lists concrete human resources they can reach right now.
    """
    if category is CrisisCategory.SELF_HARM:
        concern_line = (
            "What you just wrote sounds like you might be thinking about "
            "hurting yourself, or that things feel unbearable right now."
        )
    else:
        concern_line = (
            "What you just wrote sounds like you might be thinking about "
            "hurting someone else."
        )

    resource_lines = "\n".join(
        f"- {r['name']}: {r['contact']} ({r['notes']})" for r in CRISIS_RESOURCES
    )

    return (
        "I need to pause the conversation we were having. "
        f"{concern_line} That matters more than anything else we were "
        "talking about, and it's not something I'm equipped to help with "
        "on my own. Please reach out to one of these right now:\n\n"
        f"{resource_lines}\n\n"
        "A licensed therapist or counselor can also help with this in a way "
        "I can't. If you'd like, we can keep talking here too -- but please "
        "reach out to one of the resources above first."
    )
