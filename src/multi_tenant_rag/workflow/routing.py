"""First-principles intent gates: skip LLM when the utterance is closed-form."""

from __future__ import annotations

import re
from typing import Literal

from multi_tenant_rag.workflow.state import ChatState

Intent = Literal["SMALL_TALK", "META", "DOC"]

# Closed greetings / pleasantries — never worth a model round-trip.
_GREETING_RE = re.compile(
    r"^\s*("
    r"hi+|hello+|hey+|howdy|yo|sup|hiya|"
    r"good\s+(morning|afternoon|evening)|"
    r"thanks?(?:\s+you)?|thank\s+you|ty|thx|"
    r"ok(?:ay)?|cool|great|awesome|cheers|bye|goodbye|see\s+ya|"
    r"what'?s\s+up|whats\s+up|how\s+are\s+you"
    r")[\s!.?]*$",
    re.IGNORECASE,
)

# Capability / access questions answerable from ACL alone.
_META_RE = re.compile(
    r"^\s*("
    r"what\s+can\s+(?:you|i)\s+(?:do|access|query|see|ask)|"
    r"what\s+companies?(?:\s+can\s+i\s+(?:access|query|see))?|"
    r"which\s+companies?(?:\s+can\s+i\s+(?:access|query|see))?|"
    r"what\s+(?:docs|documents)\s+can\s+i\s+(?:access|see|query)|"
    r"help(?:\s+me)?|"
    r"what\s+are\s+you|"
    r"who\s+are\s+you"
    r")\??\s*$",
    re.IGNORECASE,
)

# Follow-ups that need history rewrite before retrieval.
_REWRITE_HINT_RE = re.compile(
    r"\b("
    r"it|that|this|those|these|they|them|their|"
    r"same|previous|earlier|above|"
    r"you\s+said|as\s+above|"
    r"the\s+(?:company|quarter|period|figure|number|one)"
    r")\b|"
    r"\b(?:what|how)\s+about\b|"
    r"\band\s+(?:for|in|of)\b",
    re.IGNORECASE,
)

# Fresh interrogatives that should not expand just because they are short.
_FRESH_QUESTION_RE = re.compile(
    r"^\s*(?:what|when|where|who|which|"
    r"how\s+(?:much|many|did|does|do|is|are|was|were))\b",
    re.IGNORECASE,
)


def heuristic_intent(message: str) -> Intent | None:
    """Return a closed intent when safe; otherwise ``None`` (use LLM / DOC)."""

    text = message.strip()
    if not text:
        return "SMALL_TALK"
    if _GREETING_RE.match(text):
        return "SMALL_TALK"
    if _META_RE.match(text):
        return "META"
    return None


def needs_query_rewrite(message: str, history_turns: int) -> bool:
    """True when history exists and the utterance looks anaphoric."""

    return is_follow_up(message, history_turns > 0)


def is_follow_up(message: str, has_history: bool) -> bool:
    """Detect follow-ups that need prior-question context for embedding."""

    if not has_history:
        return False
    text = message.strip()
    if not text:
        return False
    if _REWRITE_HINT_RE.search(text):
        return True
    # Short replies like "higher or lower?" usually depend on the prior turn,
    # but fresh short factoids ("What was Apple revenue?") must stay standalone.
    if len(text.split()) <= 6:
        return not bool(_FRESH_QUESTION_RE.match(text))
    return False


def expand_search_query(current: str, last_user_question: str | None) -> str:
    """Concatenate prior question for embedding only; never recurse."""

    current_text = current.strip()
    previous = (last_user_question or "").strip()
    if not previous or not is_follow_up(current_text, True):
        return current_text
    if current_text.casefold().startswith(previous.casefold()):
        return current_text
    return f"{previous} {current_text}"


def route_intent(state: ChatState) -> str:
    """Select the next branch from classified intent."""

    intent = state.get("intent")
    if intent == "SMALL_TALK":
        return "chitchat_reply"
    if intent == "META":
        return "meta_reply"
    return "expand_query"
