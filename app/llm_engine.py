"""
LLM engine: async, streaming inference against a local Ollama server.

Why httpx streaming instead of the `ollama` library's blocking calls:
  * FastAPI runs on an asyncio event loop. A blocking network/inference call
    would freeze the whole loop and make one user block every other user.
  * httpx.AsyncClient.stream() yields chunks without blocking the loop, so many
    WebSocket connections can be served concurrently. (On CPU the *model* still
    processes one generation at a time, but the server stays responsive and no
    connection is starved or dropped — see README "Concurrency".)

Ollama's /api/chat endpoint streams newline-delimited JSON. Each line looks like
  {"message": {"role": "assistant", "content": "Hel"}, "done": false}
and the final line carries timing stats and {"done": true}.
"""
import json
import os
from typing import AsyncGenerator, List

import httpx

from .config import settings


class LLMError(Exception):
    """Raised when the model backend fails, so callers can report it cleanly."""


class OllamaEngine:
    """Streams assistant tokens from a local Ollama model."""

    def __init__(self) -> None:
        self.host = settings.OLLAMA_HOST
        self.model = settings.MODEL_NAME

    async def stream_chat(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        """Yield the assistant's reply one token/chunk at a time."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": settings.TEMPERATURE,
                "num_predict": settings.MAX_TOKENS,
            },
        }
        try:
            timeout = httpx.Timeout(300.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise LLMError(
                            f"Ollama returned {resp.status_code}: {body.decode(errors='ignore')[:200]}"
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            # Skip a malformed line rather than killing the stream.
                            continue
                        if "error" in data:
                            raise LLMError(str(data["error"]))
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break
        except httpx.ConnectError as exc:
            raise LLMError(
                "Could not reach Ollama. Is it running? Try `ollama serve` and "
                f"`ollama pull {self.model}`."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"HTTP error talking to Ollama: {exc}") from exc


class MockEngine:
    """A fake engine that streams a canned reply word-by-word.

    Used by the test suite and lets you run the whole stack (server + UI) to see
    streaming without a model installed: set ENGINE=mock.
    """

    async def stream_chat(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        import asyncio

        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        reply = (
            f"[mock reply] I heard: '{last_user[:60]}'. I'm the mock engine, so "
            "I stream this fixed sentence to prove tokens arrive one by one."
        )
        for word in reply.split(" "):
            await asyncio.sleep(0.03)
            yield word + " "


def get_engine():
    """Pick the engine based on the ENGINE env var (defaults to real Ollama)."""
    if os.getenv("ENGINE", "ollama").lower() == "mock":
        return MockEngine()
    return OllamaEngine()
