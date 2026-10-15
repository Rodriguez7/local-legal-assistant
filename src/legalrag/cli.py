"""Command line entry point: `legalrag fetch | parse | stats | index | search | serve`."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .config import ROOT, Settings, load_documents
from .fetch import fetch_all
from .parse import parse_all


def _stats(settings: Settings) -> int:
    chunks = parse_all(settings)
    arts: dict[str, set[str]] = collections.defaultdict(set)
    for c in chunks:
        arts[c.doc].add(c.article)
    bad = 0
    for doc in load_documents():
        found, want = len(arts[doc["id"]]), doc.get("articles")
        flag = "ok" if want is None or found == want else f"MISMATCH (expected {want})"
        bad += flag != "ok"
        print(f"{doc['short']:8} {found:4} articles  {flag}")
    print(f"{len(chunks)} chunks")
    return 1 if bad else 0


def _eval(args, settings: Settings) -> int:
    from . import evaluate as ev

    if args.command == "report":
        print(ev.report())
        return 0
    from .retrieve import Retriever

    qpath = {"hard": ev.QUESTIONS.with_name("questions_hard.jsonl"), None: ev.QUESTIONS}.get(args.questions)
    questions = ev.load_questions(qpath or Path(args.questions))[: args.limit]
    model = args.model or settings.chat_model
    chat = None if args.mode == "retrieval" else ev.ollama_chat(model, settings)
    retriever = None if args.mode == "base" else Retriever.load(settings)
    results = ev.run(chat, retriever, questions, settings.top_k)
    label = args.label or f"{model}+{args.mode}"
    print(json.dumps(ev.summarise(results), indent=1))
    if not args.limit and args.mode != "retrieval":
        print("saved", ev.save(label, results))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="legalrag")
    commands = [
        "fetch", "parse", "stats", "index", "search", "serve", "eval", "report", "build-data", "split-data",
    ]
    ap.add_argument("command", choices=commands)
    ap.add_argument("query", nargs="?", help="question for `search`")
    ap.add_argument("--mode", choices=["base", "rag", "retrieval"], default="rag", help="for `eval`")
    ap.add_argument("--model", help="Ollama model for `eval` (default: the configured chat model)")
    ap.add_argument("--label", help="name of the results file for `eval`")
    ap.add_argument("--questions", help="`eval` question set: a path or `hard`")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--force", action="store_true", help="re-download cached texts")
    args = ap.parse_args(argv)
    settings = Settings()
    if args.command == "fetch":
        for p in fetch_all(settings, force=args.force):
            print(p)
        return 0
    if args.command == "parse":
        print(f"{len(parse_all(settings))} chunks -> {settings.chunks_file}")
        return 0
    if args.command == "split-data":
        from transformers import AutoTokenizer

        from .trainset import TRAIN_DIR, split

        tok = AutoTokenizer.from_pretrained(ROOT / "models" / "qwen3-4b-instruct-2507")

        def count(messages: list[dict]) -> int:
            return len(tok.encode(tok.apply_chat_template(messages, tokenize=False)))

        split(TRAIN_DIR / "generated.jsonl", count_tokens=count)
        return 0
    if args.command == "build-data":
        from . import evaluate as ev
        from .trainset import build

        build(ev.ollama_chat(args.model or settings.chat_model, settings), per_doc=args.limit or 40)
        return 0
    if args.command in ("eval", "report"):
        return _eval(args, settings)
    if args.command == "serve":
        from .api import main as serve

        serve()
        return 0
    if args.command == "index":
        from .retrieve import Retriever

        r = Retriever.build(settings)
        print(f"indexed {len(r.chunks)} chunks -> {settings.embeddings_file}")
        return 0
    if args.command == "search":
        from .retrieve import Retriever

        if not args.query:
            ap.error("search needs a query")
        for h in Retriever.load(settings).search(args.query, settings.top_k):
            print(f"{h.score:.4f}  {h.chunk['citation']}")
        return 0
    return _stats(settings)


if __name__ == "__main__":
    raise SystemExit(main())
