"""Prompt construction for grounded, citing answers."""

from __future__ import annotations

from .retrieve import Hit

SYSTEM_PROMPT = """You are a legal research assistant for EU financial and data-protection law.
Answer ONLY from the numbered sources below. Cite the source after every claim, using its exact
label, e.g. [GDPR, Article 17(1)]. If the sources do not contain the answer, say so plainly and do
not guess or rely on memory. Never invent article numbers. This is research support, not legal advice."""


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk['citation']}\n{h.chunk['text']}" for i, h in enumerate(hits, 1))


def build_messages(history: list[dict], hits: list[Hit]) -> list[dict]:
    """System prompt with sources, then the conversation (caller's own system messages dropped)."""
    system = f"{SYSTEM_PROMPT}\n\nSOURCES:\n{format_sources(hits)}"
    turns = [m for m in history if m.get("role") != "system"]
    return [{"role": "system", "content": system}, *turns]


def sources_footer(hits: list[Hit]) -> str:
    seen: list[str] = []
    for h in hits:
        if h.chunk["citation"] not in seen:
            seen.append(h.chunk["citation"])
    return "\n\n---\n**Sources retrieved**\n" + "\n".join(f"- {c}" for c in seen)
