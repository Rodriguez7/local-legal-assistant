"""Hybrid retrieval: dense embeddings + BM25, merged with reciprocal rank fusion (RRF).

Dense search finds paraphrases ("right to be forgotten"); BM25 finds exact terms and article
numbers ("Article 17"). Fusing the two ranked lists beats either alone on legal text.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx
import numpy as np

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60

Embedder = Callable[[Sequence[str]], np.ndarray]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for w in set(tokenize(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(w, 0)
                if f:
                    norm = 1 - self.b + self.b * self.len[i] / self.avg
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked lists of item ids; returns (id, score) best first."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def ollama_embedder(settings: Settings | None = None, batch: int = 32) -> Embedder:
    settings = settings or Settings()

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        with httpx.Client(timeout=300) as client:
            for i in range(0, len(texts), batch):
                r = client.post(
                    f"{settings.ollama_url}/api/embed",
                    json={"model": settings.embed_model, "input": list(texts[i : i + batch])},
                )
                r.raise_for_status()
                vecs.extend(r.json()["embeddings"])
        return np.asarray(vecs, dtype=np.float32)

    return embed


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


@dataclass
class Hit:
    chunk: dict
    score: float


class Retriever:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, embed: Embedder):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length; rebuild the index")
        self.chunks = chunks
        self.emb = _normalise(embeddings)
        self.embed = embed
        self.bm25 = BM25([self._doc_text(c) for c in chunks])

    @staticmethod
    def _doc_text(c: dict) -> str:
        return f"{c['citation']}\n{c['text']}"

    @classmethod
    def build(cls, settings: Settings | None = None, embed: Embedder | None = None) -> Retriever:
        """Embed every chunk and cache the matrix next to chunks.jsonl."""
        settings = settings or Settings()
        embed = embed or ollama_embedder(settings)
        chunks = load_chunks(settings)
        # nomic-embed-text expects task prefixes.
        emb = embed(["search_document: " + cls._doc_text(c) for c in chunks])
        np.save(settings.embeddings_file, emb)
        return cls(chunks, emb, embed)

    @classmethod
    def load(cls, settings: Settings | None = None, embed: Embedder | None = None) -> Retriever:
        settings = settings or Settings()
        embed = embed or ollama_embedder(settings)
        return cls(load_chunks(settings), np.load(settings.embeddings_file), embed)

    def search(self, query: str, k: int = 5, pool: int = 30) -> list[Hit]:
        qv = _normalise(self.embed(["search_query: " + query]))[0]
        dense = self.emb @ qv
        sparse = self.bm25.scores(query)
        d_rank = [int(i) for i in np.argsort(-dense)[:pool]]
        s_rank = [int(i) for i in np.argsort(-sparse)[:pool] if sparse[i] > 0]
        fused = rrf([d_rank, s_rank])[:k]
        return [Hit(self.chunks[i], s) for i, s in fused]


def load_chunks(settings: Settings) -> list[dict]:
    with settings.chunks_file.open() as f:
        return [json.loads(line) for line in f if line.strip()]
