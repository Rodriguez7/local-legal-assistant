"""Download the legal texts from the EU Publications Office.

EUR-Lex itself puts a bot check in front of scripts. The Publications Office resolver below is
the official machine interface: it serves the same Official Journal text as XHTML.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from .config import Settings, load_documents

RESOLVER = "https://publications.europa.eu/resource/celex/{celex}"
HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "eng",
           "User-Agent": "legalrag/0.1 (research project; contact via GitHub Rodriguez7)"}


def fetch_all(settings: Settings | None = None, delay: float = 1.0, force: bool = False) -> list[Path]:
    settings = settings or Settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with httpx.Client(follow_redirects=True, timeout=120, headers=HEADERS) as client:
        for doc in load_documents():
            path = settings.raw_dir / f"{doc['id']}.xhtml"
            if path.exists() and not force:
                out.append(path)
                continue
            resp = client.get(RESOLVER.format(celex=doc["celex"]))
            resp.raise_for_status()
            path.write_bytes(resp.content)
            out.append(path)
            time.sleep(delay)
    return out
