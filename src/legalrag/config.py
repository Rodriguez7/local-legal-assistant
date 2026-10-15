"""Paths, model names and service URLs. Everything runs locally."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    chat_model: str = os.environ.get("LEGALRAG_CHAT_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    embed_model: str = os.environ.get("LEGALRAG_EMBED_MODEL", "nomic-embed-text")
    data_dir: Path = Path(os.environ.get("LEGALRAG_DATA", ROOT / "data"))
    top_k: int = 5

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chunks_file(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def embeddings_file(self) -> Path:
        return self.processed_dir / "embeddings.npy"


def load_documents() -> list[dict]:
    text = resources.files("legalrag").joinpath("corpus.yaml").read_text()
    return yaml.safe_load(text)["documents"]
