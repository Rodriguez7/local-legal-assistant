import json

import numpy as np

from legalrag import evaluate as ev
from legalrag.retrieve import Retriever

Q = {"id": "q1", "doc": "gdpr", "question": "forgotten?", "gold_article": "17"}


def test_extract_articles_variants():
    text = "See Article 17(1), Art. 6 and Articles 15 and [GDPR, Article 20]."
    assert ev.extract_articles(text) == {"17", "6", "15", "20"}


def test_extract_ignores_sources_footer():
    text = "It is Article 17.\n\n---\n**Sources retrieved**\n- GDPR, Article 99: Penalties"
    assert ev.extract_articles(text) == {"17"}


def test_grade_correct_wrong_uncited():
    assert ev.grade(Q, "Article 17 applies").correct
    w = ev.grade(Q, "That is Article 15")
    assert w.wrong and not w.correct and not w.uncited
    assert ev.grade(Q, "I do not know").uncited


def test_grade_flags_the_art17_vs_art15_hallucination():
    r = ev.grade(Q, "The right to erasure is in Article 15.")
    assert r.wrong and r.cited == ["15"]


def test_grade_rag_hit_and_ungrounded():
    r = ev.grade(Q, "Article 17 and Article 99", retrieved=["17", "18"])
    assert r.hit and r.ungrounded
    assert not ev.grade(Q, "Article 17", retrieved=["17"]).ungrounded


def test_summarise_ignores_na_fields():
    s = ev.summarise([ev.grade(Q, "Article 17"), ev.grade(Q, "Article 15")])
    assert s["correct"] == 0.5 and s["wrong"] == 0.5 and s["hit@k"] is None


def test_run_base_vs_rag_uses_different_prompts():
    chunks = [{"chunk_id": "x", "doc": "gdpr", "article": "17", "citation": "GDPR, Article 17: Erasure",
               "text": "erasure"}]
    ret = Retriever(chunks, np.eye(1, dtype=np.float32), lambda t: np.ones((len(t), 1), np.float32))
    seen = []

    def chat(msgs):
        seen.append(msgs)
        return "Article 17"

    base = ev.run(chat, None, [Q])[0]
    rag = ev.run(chat, ret, [Q])[0]
    assert base.correct and base.hit is None
    assert rag.correct and rag.hit and "SOURCES" in seen[1][0]["content"]
    assert "SOURCES" not in seen[0][0]["content"]


def test_report_table(tmp_path):
    for label, ans in [("base", "Article 15"), ("rag", "Article 17")]:
        res = [ev.grade(Q, ans)]
        (tmp_path / f"{label}.json").write_text(
            json.dumps({"label": label, "summary": ev.summarise(res), "results": []})
        )
    table = ev.report(tmp_path)
    assert "| base | 0% | 100%" in table and "| rag | 100% | 0%" in table


def test_shipped_questions_are_well_formed():
    qs = ev.load_questions()
    assert len(qs) >= 40 and len({q["id"] for q in qs}) == len(qs)
    assert all(q["gold_article"].isdigit() for q in qs)
