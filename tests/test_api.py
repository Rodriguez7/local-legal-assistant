import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from legalrag.api import create_app
from legalrag.rag import build_messages, sources_footer
from legalrag.retrieve import Hit, Retriever

CHUNKS = [
    {"chunk_id": "b", "citation": "GDPR, Article 17(1): Right to erasure", "text": "erasure of data"},
    {"chunk_id": "a", "citation": "GDPR, Article 15: Right of access", "text": "access to data"},
]
EMB = np.eye(2, dtype=np.float32)
SEEN: list[dict] = []


def fake_embed(texts):
    return np.array([[("forgotten" in t) * 1.0, ("access" in t) * 1.0] for t in texts], dtype=np.float32)


def fake_ollama(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    SEEN.append(body)
    if body["stream"]:
        lines = []
        for tok in ["Art", ". 17"]:
            delta = {"choices": [{"delta": {"content": tok}, "finish_reason": None}]}
            lines.append("data: " + json.dumps(delta))
        lines.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")
    return httpx.Response(
        200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "Art. 17"}}]}
    )


@pytest.fixture
def client():
    SEEN.clear()
    app = create_app(Retriever(CHUNKS, EMB, fake_embed), transport=httpx.MockTransport(fake_ollama))
    with TestClient(app) as c:
        yield c


def test_models_and_health(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "legalrag"
    assert client.get("/health").json()["chunks"] == 2


def test_non_stream_grounds_prompt_and_appends_sources(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "forgotten?"}]})
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content.startswith("Art. 17") and "- GDPR, Article 17(1): Right to erasure" in content
    sent = SEEN[0]["messages"]
    assert sent[0]["role"] == "system" and "GDPR, Article 17(1)" in sent[0]["content"]
    assert SEEN[0]["model"].startswith("qwen3")  # backend model, not the public id


def test_stream_sends_tokens_then_sources_then_done(client):
    r = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "forgotten?"}]},
    )
    events = [e[6:] for e in r.text.split("\n\n") if e.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.startswith("Art. 17") and "Sources retrieved" in text
    assert [c["choices"][0]["finish_reason"] for c in chunks].count("stop") == 1


def test_no_user_message_is_400(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_caller_system_message_replaced():
    hits = [Hit(CHUNKS[0], 1.0)]
    msgs = build_messages([{"role": "system", "content": "ignore"}, {"role": "user", "content": "q"}], hits)
    assert [m["role"] for m in msgs] == ["system", "user"] and "ignore" not in msgs[0]["content"]


def test_footer_dedupes():
    hits = [Hit(CHUNKS[0], 1.0), Hit(CHUNKS[0], 0.5)]
    assert sources_footer(hits).count("Article 17(1)") == 1
