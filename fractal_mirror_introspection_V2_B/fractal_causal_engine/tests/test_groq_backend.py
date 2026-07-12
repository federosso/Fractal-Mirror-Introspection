"""Test del backend Groq (V10.18.3).

Groq usa il protocollo OpenAI; non chiamiamo davvero l'API -- verifichiamo
solo il routing, la costruzione della richiesta e la gestione della key.
"""
from __future__ import annotations

import pytest

from fractal_causal_engine.llm import LLMClient, LLMConfig


def test_groq_without_key_raises_clear_error():
    client = LLMClient(LLMConfig(backend="groq", model="llama-3.3-70b-versatile"))
    with pytest.raises(RuntimeError, match="API key"):
        client.chat_ex([{"role": "user", "content": "ciao"}])


def test_groq_config_defaults():
    cfg = LLMConfig(backend="groq")
    # default sensato per l'endpoint
    assert cfg.groq_url == "https://api.groq.com/openai/v1"
    assert cfg.groq_api_key == ""


def test_groq_builds_request_with_auth_header(monkeypatch):
    """Verifica che _chat_groq costruisca la richiesta con Authorization e
    l'URL giusto, senza chiamare davvero la rete."""
    captured = {}

    def fake_post(self, url, payload, *, label, allow_retry_without_response_format,
                  extra_headers=None):
        captured["url"] = url
        captured["label"] = label
        captured["headers"] = extra_headers or {}
        from fractal_causal_engine.llm import ChatResult
        return ChatResult(content='{"ok": true}', finish_reason="stop")

    monkeypatch.setattr(LLMClient, "_post_openai_compatible", fake_post)
    client = LLMClient(LLMConfig(backend="groq", model="llama-3.3-70b-versatile",
                                 groq_api_key="gsk_test123"))
    result = client.chat_ex([{"role": "user", "content": "ciao"}])
    assert result.content == '{"ok": true}'
    assert captured["url"].endswith("/chat/completions")
    assert captured["label"] == "Groq"
    assert captured["headers"]["Authorization"] == "Bearer gsk_test123"


def test_groq_routing_in_chat_ex(monkeypatch):
    """chat_ex con backend='groq' instrada su _chat_groq."""
    called = {"groq": False}

    def fake_groq(self, messages, *, format_json=True, num_predict=None):
        called["groq"] = True
        from fractal_causal_engine.llm import ChatResult
        return ChatResult(content="{}", finish_reason="stop")

    monkeypatch.setattr(LLMClient, "_chat_groq", fake_groq)
    client = LLMClient(LLMConfig(backend="groq", groq_api_key="gsk_x"))
    client.chat_ex([{"role": "user", "content": "x"}])
    assert called["groq"] is True


# --- V10.19.2: User-Agent esplicito (fix Cloudflare 403 / error 1010) --------


def test_base_headers_include_user_agent():
    """Ogni richiesta HTTP deve portare uno User-Agent esplicito: il default
    'Python-urllib' viene bloccato da Cloudflare (Groq) con 403/1010."""
    from fractal_causal_engine.llm import _BASE_HTTP_HEADERS
    assert "User-Agent" in _BASE_HTTP_HEADERS
    assert _BASE_HTTP_HEADERS["User-Agent"]
    assert "urllib" not in _BASE_HTTP_HEADERS["User-Agent"].lower()


def test_groq_post_carries_user_agent(monkeypatch):
    """La richiesta a Groq include User-Agent insieme all'Authorization."""
    import urllib.request
    captured = {}

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'

    def fake_urlopen(req, timeout=None):
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LLMClient(LLMConfig(backend="groq", model="llama-3.3-70b-versatile",
                                 groq_api_key="gsk_test"))
    client.chat_ex([{"role": "user", "content": "ciao"}])
    # urllib normalizza le chiavi header in Title-Case -> confronto lower
    assert "user-agent" in captured["headers"]
    assert "authorization" in captured["headers"]
