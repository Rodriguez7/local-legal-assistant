"""Build the LoRA training set by self-distillation with rejection sampling.

What the fine-tune should teach is a *behaviour*: answer only from the supplied sources, cite them
with their exact label ("[GDPR, Article 17(1)]"), and refuse when the sources do not contain the
answer. The base model can do this when given extra formatting instructions and few-shot help, but
not reliably from the plain production prompt. So:

1. Pick a provision, have the local model write a practitioner question it answers.
2. Retrieve sources with the real retriever (realistic distractors included).
3. Generate the answer WITH the extra formatting instructions, then keep it only if every citation
   is an exact label of a retrieved source (rejection sampling).
4. Train on (plain production prompt -> accepted answer). About 15% of examples pair the question
   with unrelated sources and a refusal, so the model also learns to say "not in the sources".

Articles used by either evaluation set are excluded, so the evaluations stay held-out.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT, Settings
from .evaluate import QUESTIONS, load_questions
from .rag import build_messages
from .retrieve import Hit, Retriever, load_chunks

TRAIN_DIR = ROOT / "training"
HARD = QUESTIONS.with_name("questions_hard.jsonl")

Chat = Callable[[list[dict], float], str]

QUESTION_PROMPT = (
    "Below is a provision of EU law. Write ONE realistic question a compliance officer or lawyer "
    "at a fintech firm might ask whose answer is in this provision. Describe a situation or ask "
    "about the rule; do NOT mention the article number or the name of the regulation. "
    "Output only the question.\n\n{label}\n{text}"
)

FORMAT_RULES = (
    "\n\nFORMAT RULES: Write 2-5 sentences. After each claim, cite the source using its exact label "
    "in square brackets, copied from the source header, for example [GDPR, Article 17(1)]. Never cite "
    "with numbers such as [1] or [2]. Use only labels that appear in the sources."
)

REFUSALS = [
    "The provided sources do not contain the answer to this question, so I can't answer it reliably "
    "from them. Try rephrasing, or check the relevant regulation directly.",
    "I could not find this in the retrieved sources, and I won't guess. The question may concern a "
    "provision that was not retrieved, or a text outside this corpus.",
    "The sources I was given do not address this. I can only answer from the retrieved provisions, so "
    "I'd rather say so than risk citing the wrong article.",
]

_LABEL = re.compile(r"\[([^\[\]]+)\]")


def excluded_articles() -> set[tuple[str, str]]:
    qs = load_questions() + load_questions(HARD)
    return {(q["doc"], q["gold_article"]) for q in qs}


def accepted(answer: str, hits: list[Hit]) -> bool:
    """Every bracketed citation must be an exact retrieved label, and there must be at least one."""
    labels = {h.chunk["citation"].split(":")[0] for h in hits}
    cited = [m.strip() for m in _LABEL.findall(answer)]
    return bool(cited) and all(c in labels for c in cited)


def pick_chunks(chunks: list[dict], per_doc: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    banned = excluded_articles()
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        if (c["doc"], c["article"]) in banned or len(c["text"].split()) < 25:
            continue
        by_doc.setdefault(c["doc"], []).append(c)
    out = []
    for doc in sorted(by_doc):
        pool = by_doc[doc]
        out.extend(rng.sample(pool, min(per_doc, len(pool))))
    return out


def make_example(chunk: dict, chat: Chat, retriever: Retriever, rng: random.Random) -> dict | None:
    prompt = QUESTION_PROMPT.format(label=chunk["citation"], text=chunk["text"])
    q = chat([{"role": "user", "content": prompt}], 0.7)
    q = q.strip().strip('"')
    if not q.endswith("?") or len(q) < 25:
        return None
    hits = retriever.search(q, 5)
    if chunk["chunk_id"] not in {h.chunk["chunk_id"] for h in hits}:
        hits = [Hit(chunk, 1.0), *hits[:4]]  # keep it answerable
    msgs = build_messages([{"role": "user", "content": q}], hits)
    gen_msgs = [{**msgs[0], "content": msgs[0]["content"] + FORMAT_RULES}, *msgs[1:]]
    answer = chat(gen_msgs, 0.0).strip()
    if rng.random() < 0.15:  # unanswerable variant: unrelated sources, refusal target
        other = rng.sample([c for c in retriever.chunks if c["doc"] != chunk["doc"]], 5)
        msgs = build_messages([{"role": "user", "content": q}], [Hit(c, 0.0) for c in other])
        reply = {"role": "assistant", "content": rng.choice(REFUSALS)}
        return {"messages": [*msgs, reply], "kind": "refusal", "source": chunk["chunk_id"]}
    if not accepted(answer, hits):
        return None
    return {"messages": [*msgs, {"role": "assistant", "content": answer}], "kind": "answer",
            "source": chunk["chunk_id"]}


def build(chat: Chat, per_doc: int = 40, workers: int = 3, settings: Settings | None = None) -> Path:
    settings = settings or Settings()
    retriever = Retriever.load(settings)
    chosen = pick_chunks(load_chunks(settings), per_doc)
    TRAIN_DIR.mkdir(exist_ok=True)
    raw = TRAIN_DIR / "generated.jsonl"
    done = {json.loads(line)["source"] for line in raw.read_text().splitlines()} if raw.exists() else set()
    todo = [c for c in chosen if c["chunk_id"] not in done]
    print(f"{len(chosen)} chunks chosen, {len(todo)} to generate")

    def work(c: dict):
        return make_example(c, chat, retriever, random.Random(c["chunk_id"]))

    with ThreadPoolExecutor(workers) as pool, raw.open("a") as f:
        for i, ex in enumerate(pool.map(work, todo), 1):
            if ex:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)}")
    return split(raw)


def split(
    raw: Path,
    valid_frac: float = 0.1,
    seed: int = 7,
    count_tokens: Callable[[list[dict]], int] | None = None,
    max_tokens: int = 3000,
) -> Path:
    """Write train/valid files. With count_tokens, drop examples longer than max_tokens.

    An over-long example is truncated by the trainer from the right, which cuts off the answer;
    with prompt masking that leaves no target tokens and the loss becomes NaN.
    """
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    if count_tokens:
        kept = [r for r in rows if count_tokens(r["messages"]) <= max_tokens]
        print(f"dropped {len(rows) - len(kept)} examples over {max_tokens} tokens")
        rows = kept
    random.Random(seed).shuffle(rows)
    n_valid = max(1, int(len(rows) * valid_frac))
    for name, part in (("valid", rows[:n_valid]), ("train", rows[n_valid:])):
        with (TRAIN_DIR / f"{name}.jsonl").open("w") as f:
            for r in part:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    refusals = sum(r["kind"] == "refusal" for r in rows)
    print(f"train {len(rows) - n_valid}, valid {n_valid}, refusals {refusals}")
    return TRAIN_DIR / "train.jsonl"
