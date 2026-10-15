"""OpenAI-compatible RAG service. Point Open WebUI (or any OpenAI client) at /v1."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from .config import Settings
from .rag import build_messages, sources_footer
from .retrieve import Hit, Retriever

MODEL_ID = "legalrag"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = MODEL_ID
    messages: list[dict]
    stream: bool = False
    temperature: float | None = None


def _question(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):  # multimodal-style content parts
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c
    raise HTTPException(400, "no user message")


def _chunk(cid: str, content: str, finish: str | None = None) -> str:
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def create_app(
    retriever: Retriever | None = None,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever or Retriever.load(settings)
        app.state.http = httpx.AsyncClient(base_url=settings.ollama_url, timeout=600, transport=transport)
        yield
        await app.state.http.aclose()

    app = FastAPI(title="legalrag", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(app.state.retriever.chunks), "chat_model": settings.chat_model}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}

    def _prepare(req: ChatRequest) -> tuple[dict, list[Hit]]:
        hits = app.state.retriever.search(_question(req.messages), settings.top_k)
        payload = {"model": settings.chat_model, "messages": build_messages(req.messages, hits)}
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        return payload, hits

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        payload, hits = _prepare(req)
        footer = sources_footer(hits)
        http: httpx.AsyncClient = app.state.http
        if not req.stream:
            r = await http.post("/v1/chat/completions", json={**payload, "stream": False})
            if r.status_code != 200:
                raise HTTPException(502, f"model backend error: {r.text[:200]}")
            data = r.json()
            data["model"] = MODEL_ID
            msg = data["choices"][0]["message"]
            msg["content"] = (msg.get("content") or "") + footer
            return JSONResponse(data)

        async def stream() -> AsyncIterator[str]:
            cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            async with http.stream("POST", "/v1/chat/completions", json={**payload, "stream": True}) as r:
                if r.status_code != 200:
                    yield _chunk(cid, f"\n\n[model backend error {r.status_code}]", "stop")
                    yield "data: [DONE]\n\n"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    obj = json.loads(body)
                    obj["model"] = MODEL_ID
                    choice = obj["choices"][0]
                    if choice.get("finish_reason"):
                        continue  # we send our own final chunk after the sources
                    yield f"data: {json.dumps(obj)}\n\n"
            yield _chunk(cid, footer)
            yield _chunk(cid, "", "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
