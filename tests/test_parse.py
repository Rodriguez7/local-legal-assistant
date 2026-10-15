from pathlib import Path

import pytest

from legalrag.config import Settings, load_documents
from legalrag.parse import parse_document

DOC = {"id": "toy", "short": "Toy Act"}


def _art(num, title, paras):
    body = "".join(
        f'<div id="{num}.{i}"><p class="oj-normal">{i}. {t}</p></div>' for i, t in enumerate(paras, 1)
    )
    return (
        f'<div class="eli-subdivision" id="art_{num}"><p class="oj-ti-art">Article {num}</p>'
        f'<div class="eli-title"><p>{title}</p></div>{body}</div>'
    )


def _write(tmp_path, *arts):
    f = tmp_path / "toy.xhtml"
    f.write_text("<html><body>" + "".join(arts) + "</body></html>")
    return f


def test_short_article_is_single_chunk(tmp_path):
    chunks = parse_document(_write(tmp_path, _art(17, "Right to erasure", ["Erase it.", "Also this."])), DOC)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.citation == "Toy Act, Article 17: Right to erasure"
    assert c.chunk_id == "toy-art17" and c.paragraph is None
    assert "Erase it." in c.text


def test_long_article_splits_per_paragraph(tmp_path):
    long = "word " * 200
    chunks = parse_document(_write(tmp_path, _art(5, "Principles", [long, long])), DOC)
    assert [c.citation for c in chunks] == [
        "Toy Act, Article 5(1): Principles",
        "Toy Act, Article 5(2): Principles",
    ]
    assert chunks[1].chunk_id == "toy-art5-p2"


def test_non_article_divisions_ignored(tmp_path):
    f = _write(tmp_path, '<div class="eli-subdivision" id="art_x"><p class="oj-ti-art">Annex I</p></div>')
    assert parse_document(f, DOC) == []


def test_lettered_article_number(tmp_path):
    chunks = parse_document(_write(tmp_path, _art("12a", "Extra", ["Text."])), DOC)
    assert chunks[0].article == "12a"


@pytest.mark.skipif(not Settings().raw_dir.joinpath("gdpr.xhtml").exists(), reason="corpus not downloaded")
@pytest.mark.parametrize("doc", load_documents(), ids=lambda d: d["id"])
def test_article_counts_match_official(doc):
    path = Settings().raw_dir / f"{doc['id']}.xhtml"
    arts = {c.article for c in parse_document(Path(path), doc)}
    assert len(arts) == doc["articles"]


def test_gdpr_article_17_citation():
    path = Settings().raw_dir / "gdpr.xhtml"
    if not path.exists():
        pytest.skip("corpus not downloaded")
    doc = next(d for d in load_documents() if d["id"] == "gdpr")
    cites = [c.citation for c in parse_document(path, doc) if c.article == "17"]
    assert any("Right to erasure" in c for c in cites)
