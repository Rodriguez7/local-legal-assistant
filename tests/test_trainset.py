import json

from legalrag import trainset as ts
from legalrag.retrieve import Hit

CH = {
    "chunk_id": "x", "doc": "gdpr", "article": "17", "citation": "GDPR, Article 17(1): Erasure", "text": "t",
}
HITS = [Hit(CH, 1.0), Hit({**CH, "citation": "GDPR, Article 19: Notification"}, 0.5)]


def test_accepts_exact_labels():
    assert ts.accepted("Yes [GDPR, Article 17(1)] and [GDPR, Article 19].", HITS)


def test_rejects_numeric_unknown_or_missing_citations():
    assert not ts.accepted("Yes [1].", HITS)
    assert not ts.accepted("Yes [GDPR, Article 99].", HITS)
    assert not ts.accepted("Yes, no citation.", HITS)
    assert not ts.accepted("Yes [GDPR, Article 17(1)] [2].", HITS)


def test_eval_articles_excluded_from_training_pool():
    chunks = [
        {**CH, "chunk_id": "a", "article": "17", "text": "w " * 40},  # gold of an eval question
        {**CH, "chunk_id": "b", "article": "12", "text": "w " * 40},
    ]
    assert [c["chunk_id"] for c in ts.pick_chunks(chunks, 10)] == ["b"]


def test_split_writes_chat_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "TRAIN_DIR", tmp_path)
    raw = tmp_path / "generated.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": str(i)}], "kind": "answer", "source": str(i)}
        for i in range(20)
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows))
    ts.split(raw)
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(line) for line in (tmp_path / "valid.jsonl").read_text().splitlines()]
    assert len(train) == 18 and len(valid) == 2 and set(train[0]) == {"messages"}


def test_split_drops_over_long_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "TRAIN_DIR", tmp_path)
    raw = tmp_path / "generated.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "x" * n}], "kind": "answer", "source": str(n)}
        for n in (10, 20, 5000)
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows))
    ts.split(raw, count_tokens=lambda m: len(m[0]["content"]), max_tokens=100)
    kept = [
        json.loads(line)
        for f in ("train", "valid")
        for line in (tmp_path / f"{f}.jsonl").read_text().splitlines()
    ]
    assert len(kept) == 2
