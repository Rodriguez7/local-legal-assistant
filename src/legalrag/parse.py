"""Parse Official Journal XHTML (ELI markup) into citable chunks.

An article of 350 words or fewer is one chunk; a longer one is split per numbered paragraph.
Every chunk carries a citation header such as "GDPR, Article 17(1): Right to erasure".
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from lxml import html

from .config import Settings, load_documents

MAX_WORDS = 350


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    article: str
    paragraph: str | None
    title: str
    citation: str
    text: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _node_text(div) -> str:
    parts = []
    for p in div.xpath(".//p[contains(@class,'oj-normal')] | .//table"):
        parts.append(_clean(p.text_content()))
    return "\n".join(x for x in parts if x)


def parse_document(path: Path, doc: dict) -> list[Chunk]:
    tree = html.fromstring(path.read_bytes())
    short = doc["short"]
    chunks: list[Chunk] = []
    for art in tree.xpath('//div[contains(@class,"eli-subdivision") and starts-with(@id,"art_")]'):
        label = _clean("".join(art.xpath("./p[contains(@class,'oj-ti-art')]//text()")))
        m = re.search(r"Article\s+(\d+\w*)", label)
        if not m:
            continue
        num = m.group(1)
        title = _clean("".join(art.xpath(".//div[contains(@class,'eli-title')]//p//text()")))
        paras = art.xpath("./div[not(contains(@class,'eli-title'))]")
        body = "\n".join(_node_text(p) for p in paras) if paras else _node_text(art)
        base_id = f"{doc['id']}-art{num}"
        if len(body.split()) <= MAX_WORDS or len(paras) < 2:
            cite = f"{short}, Article {num}: {title}"
            chunks.append(Chunk(base_id, doc["id"], num, None, title, cite, body))
            continue
        for i, p in enumerate(paras, 1):
            text = _node_text(p)
            if not text:
                continue
            pm = re.match(r"(\d+)\.", text)
            par = pm.group(1) if pm else str(i)
            cite = f"{short}, Article {num}({par}): {title}"
            chunks.append(Chunk(f"{base_id}-p{par}", doc["id"], num, par, title, cite, text))
    return chunks


def parse_all(settings: Settings | None = None) -> list[Chunk]:
    settings = settings or Settings()
    out: list[Chunk] = []
    for doc in load_documents():
        out.extend(parse_document(settings.raw_dir / f"{doc['id']}.xhtml", doc))
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.chunks_file.write_text("\n".join(json.dumps(asdict(c)) for c in out))
    return out
