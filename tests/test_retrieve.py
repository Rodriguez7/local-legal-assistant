import numpy as np
import pytest

from legalrag.retrieve import BM25, Retriever, rrf, tokenize

CHUNKS = [
    {"chunk_id": "a", "citation": "GDPR, Article 15: Right of access", "text": "access to personal data"},
    {"chunk_id": "b", "citation": "GDPR, Article 17: Right to erasure", "text": "erasure of personal data"},
    {"chunk_id": "c", "citation": "DORA, Article 5: Governance", "text": "ict risk management body"},
]
# Hand-made 3-d "embeddings": axis 0 = access, 1 = erasure, 2 = ict.
EMB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def fake_embed(texts):
    out = []
    for t in texts:
        t = t.lower()
        out.append([("access" in t) * 1.0, ("forgotten" in t or "erasure" in t) * 1.0, ("ict" in t) * 1.0])
    return np.array(out, dtype=np.float32)


def test_tokenize():
    assert tokenize("Article 17(1): Erasure!") == ["article", "17", "1", "erasure"]


def test_bm25_prefers_matching_doc():
    s = BM25([c["text"] for c in CHUNKS]).scores("erasure")
    assert int(np.argmax(s)) == 1 and s[0] == 0 == s[2]


def test_bm25_unknown_terms_score_zero():
    assert BM25(["a b", "c d"]).scores("zzz").sum() == 0


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 2]])
    assert [i for i, _ in fused][:2] == [1, 3]


def test_rrf_single_list_keeps_order():
    assert [i for i, _ in rrf([[4, 2, 9]])] == [4, 2, 9]


def test_dense_finds_paraphrase_without_shared_words():
    r = Retriever(CHUNKS, EMB, fake_embed)
    # "being forgotten" shares no token with chunk b's text, only the embedding links them.
    assert r.search("being forgotten", k=1)[0].chunk["chunk_id"] == "b"


def test_bm25_finds_exact_article_number():
    r = Retriever(CHUNKS, EMB, fake_embed)
    # The query vector is all zeros, so only BM25 can pull "c" up; fusion must still surface it.
    assert "c" in [h.chunk["chunk_id"] for h in r.search("Article 5", k=2)]


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        Retriever(CHUNKS, EMB[:2], fake_embed)
