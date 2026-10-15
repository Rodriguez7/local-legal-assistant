"""Evaluation harness: does the system name the right article, and does it invent citations?

Every question asks which article of a named text covers a topic, so the gold answer is a single
article number. We measure it for three setups with the same questions and settings:
base model alone, base + RAG, and (later) fine-tuned + RAG.

Metrics (lower-is-better marked with a down arrow):
  correct    - the answer cites the gold article number
  wrong      - the answer cites article numbers but not the gold one (a confident hallucination)
  uncited    - the answer cites no article at all
  ungrounded - RAG only: a cited article number is not present in the retrieved sources
  hit@k      - RAG only: the gold article is among the retrieved chunks
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from .config import ROOT, Settings
from .rag import SYSTEM_PROMPT, build_messages
from .retrieve import Retriever

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

BASE_SYSTEM = (
    "You are a legal research assistant for EU financial and data-protection law. "
    "Answer concisely and cite the specific article number."
)

_CITE = re.compile(r"\bArt(?:icle|icles|\.)?\s*(\d+)", re.IGNORECASE)
_SOURCE_BLOCK = re.compile(r"\n+---\n\*\*Sources retrieved\*\*.*", re.DOTALL)

Chat = Callable[[list[dict]], str]


def extract_articles(answer: str) -> set[str]:
    """Article numbers cited in an answer (a service-appended sources footer is ignored)."""
    return set(_CITE.findall(_SOURCE_BLOCK.sub("", answer)))


@dataclass
class Result:
    id: str
    doc: str
    gold: str
    answer: str
    cited: list[str]
    correct: bool
    wrong: bool
    uncited: bool
    retrieved: list[str] = field(default_factory=list)
    hit: bool | None = None
    ungrounded: bool | None = None


def grade(q: dict, answer: str, retrieved: list[str] | None = None) -> Result:
    cited = extract_articles(answer)
    gold = q["gold_article"]
    r = Result(
        id=q["id"], doc=q["doc"], gold=gold, answer=answer, cited=sorted(cited, key=int),
        correct=gold in cited, wrong=bool(cited) and gold not in cited, uncited=not cited,
    )
    if retrieved is not None:
        r.retrieved = retrieved
        r.hit = gold in retrieved
        r.ungrounded = bool(cited - set(retrieved))
    return r


def ollama_chat(model: str, settings: Settings | None = None) -> Chat:
    settings = settings or Settings()

    def chat(messages: list[dict], temperature: float = 0.0) -> str:
        r = httpx.post(
            f"{settings.ollama_url}/v1/chat/completions",
            json={
                "model": model, "messages": messages, "temperature": temperature,
                "seed": 7, "stream": False,
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    return chat


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(
    chat: Chat | None,
    retriever: Retriever | None,
    questions: list[dict],
    k: int = 5,
) -> list[Result]:
    """chat=None gives a retrieval-only run; retriever=None gives the base model alone."""
    out = []
    for q in questions:
        hits = retriever.search(q["question"], k) if retriever else []
        retrieved = [h.chunk["article"] for h in hits] if retriever else None
        answer = ""
        if chat:
            msgs = (
                build_messages([{"role": "user", "content": q["question"]}], hits)
                if retriever
                else [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": q["question"]}]
            )
            answer = chat(msgs)
        out.append(grade(q, answer, retrieved))
    return out


def summarise(results: list[Result]) -> dict:
    n = len(results)

    def rate(pred) -> float | None:
        vals = [pred(r) for r in results if pred(r) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n": n,
        "correct": rate(lambda r: r.correct),
        "wrong": rate(lambda r: r.wrong),
        "uncited": rate(lambda r: r.uncited),
        "ungrounded": rate(lambda r: r.ungrounded),
        "hit@k": rate(lambda r: r.hit),
    }


def save(label: str, results: list[Result]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    path.write_text(json.dumps({"label": label, "summary": summarise(results),
                                "results": [asdict(r) for r in results]}, indent=1, ensure_ascii=False))
    return path


def report(results_dir: Path = RESULTS_DIR) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    lines = [
        "| Setup | Correct article | Wrong article | No citation | Ungrounded cite | Gold in top-k |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["summary"]
        lines.append(
            f"| {r['label']} | {pct(s['correct'])} | {pct(s['wrong'])} | {pct(s['uncited'])} "
            f"| {pct(s['ungrounded'])} | {pct(s['hit@k'])} |"
        )
    n = rows[0]["summary"]["n"] if rows else 0
    lines.append(f"\n{n} questions, temperature 0. Wrong = cited articles but not the gold one.")
    return "\n".join(lines)


__all__ = ["SYSTEM_PROMPT", "run", "grade", "summarise", "save", "report", "extract_articles"]
