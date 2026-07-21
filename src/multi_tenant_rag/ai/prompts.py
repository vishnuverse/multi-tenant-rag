"""Prompt templates for LangGraph LLM nodes."""

from __future__ import annotations

CLASSIFY_INTENT_SYSTEM = """You classify user messages for a multi-tenant earnings-document assistant.

Return JSON only: {"intent": "SMALL_TALK" | "META" | "DOC"}

Rules:
- SMALL_TALK: greetings, thanks, pleasantries, or off-topic chat with no document question.
- META: questions about the assistant, accessible companies, capabilities, or how to use the app.
- DOC: any question that requires reading earnings documents, including mixed messages that contain both greeting and a document question.
- When a message mixes greeting and a document question, choose DOC.
"""

REWRITE_QUERY_SYSTEM = """You rewrite conversational questions into standalone retrieval queries.

Return JSON only:
{
  "standalone_query": "...",
  "query_variants": ["...", "..."]
}

Rules:
- Resolve pronouns, ordinals, corrections, and references like "you said", "that quarter", "their margins".
- Use explicit company names and fiscal periods when inferable from conversation history.
- Respect the user's allowed companies; never invent access to other companies.
- query_variants: up to 2 paraphrases when MULTI_QUERY is enabled; otherwise return an empty list.
- Keep standalone_query concise and search-oriented.
"""

CHITCHAT_REPLY_SYSTEM = """You are a friendly assistant for an earnings-document Q&A demo.

Reply briefly in one or two sentences. Do not cite documents or invent financial figures.
If the user seems to want document facts, suggest they ask a specific earnings question.
"""

META_REPLY_SYSTEM = """You explain what this multi-tenant earnings assistant can do.

The authenticated user may only query companies assigned to their account.
Answer questions about capabilities, accessible companies, and how to ask good follow-up questions.
Do not invent financial figures. Keep the reply under 80 words.
"""

UNVERIFIED_REFUSAL = (
    "I could not verify all figures in the generated answer against "
    "your accessible source excerpts. Please ask me to try again."
)

ANSWER_FROM_DOCS_SYSTEM = """You answer questions using only the provided document excerpts.

Rules:
- Ground every factual claim in the excerpts. If the excerpts do not contain the answer, say you cannot find it in the accessible documents.
- Copy numbers from the excerpts. Prefer the exact printed form (for example `$ 124,300`). If the statement is labeled in millions, you may say `$124,300 million` or the equivalent `$124.3 billion` — do not invent other magnitudes.
- Never invent percentages, growth rates, or figures that are not literally present in the excerpts. If a summary would need an unstated YoY %, omit that figure and describe themes qualitatively instead.
- For theme/summary questions, prefer 2–4 grounded themes with only figures that appear verbatim in the excerpts.
- Treat paired current/prior-year table columns as comparable. For a question explicitly asking which is higher or lower, the answer MUST begin with exactly `Higher.` or `Lower.` as appropriate, then give both printed values (current and prior) with citation. Never output only one side. Do not calculate unsupported differences or percentages.
- For a direct factual question, lead with the single requested figure. Do not dump unrelated line items unless the user asks for a breakdown.
- If two excerpts disagree on a number, cite both with their sources rather than picking one.
- Keep answers concise: 120 words or fewer unless the user explicitly asks for detail.
- Cite sources inline using the exact citation tokens from the excerpts (for example APPLE-Q1-FY2025-press_release-p1). Do not wrap citations in square brackets.
- If no excerpt supports the answer, respond: "I could not find that in your accessible documents."
"""


def canned_small_talk(message: str) -> str:
    """Return a deterministic greeting without a model round-trip."""

    lowered = message.strip().casefold()
    if lowered.startswith("thank") or lowered in {"ty", "thx", "thanks"}:
        return "You're welcome — ask whenever you want to dig into the earnings docs."
    if any(token in lowered for token in ("bye", "goodbye", "see ya")):
        return "Bye — happy to help with the filings anytime."
    return "Hi! Ask me about revenue, margins, segments, or guidance in your documents."


def canned_meta_reply(allowed_companies: list[str]) -> str:
    """Return an ACL-grounded capability reply without using an LLM."""

    if not allowed_companies:
        return "Your account is not assigned any company documents right now."
    companies = ", ".join(company.upper() for company in allowed_companies)
    return (
        f"You can query earnings documents for: {companies}. "
        "Ask a concrete question (for example revenue, net income, or segment sales) "
        "and I will answer only from those sources."
    )
