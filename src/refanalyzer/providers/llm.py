"""LLM adapters for interpretation and comparison.

The LLM never replaces measurement: it receives measured numbers and
evidence images and is asked to interpret them. Providers:

* ``anthropic`` (default for ``auto``) - official Anthropic SDK
* ``openai``    - Chat Completions over HTTPS
* ``google``    - Gemini; only used when selected explicitly
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2

from ..config import api_key
from ..pipeline import Context
from .http import check, session

FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}


def _image_b64(path: str | Path, max_side: int = 1280) -> tuple[str, str]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.standard_b64encode(buf.tobytes()).decode(), "image/jpeg"


class LLM:
    provider = "none"

    def __init__(self, ctx: Context, role: str):
        self.ctx = ctx
        self.role = role  # "vision" or "reasoning"

    @property
    def model(self) -> str:
        raise NotImplementedError

    def complete(self, system: str, parts: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        raise NotImplementedError


def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


class AnthropicLLM(LLM):
    provider = "anthropic"

    @property
    def model(self) -> str:
        return self.ctx.config.get(f"llm.anthropic_{self.role}_model")

    def complete(self, system, parts, max_tokens=None):
        client = _anthropic_client()
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                data, mt = _image_b64(p["path"])
                content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.ctx.config.get("llm.max_output_tokens"),
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.ctx.config.get("llm.anthropic_effort")},
        }
        if self.model in FALLBACK_MODELS:
            # Server-side refusal fallback: re-routes a declined request to a
            # suitable model inside the same call.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        # Streaming avoids HTTP timeouts on long reports.
        with client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        self.ctx.costs.add_tokens("anthropic", msg.model or self.model, msg.usage.input_tokens, msg.usage.output_tokens,
                                  self.ctx.config.get("pricing.anthropic") or {}, note=self.role)
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise RuntimeError(f"model declined the request (category: {cat})")
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAILLM(LLM):
    provider = "openai"

    @property
    def model(self) -> str:
        return self.ctx.config.get(f"llm.openai_{self.role}_model")

    def complete(self, system, parts, max_tokens=None):
        s = session()
        s.headers["Authorization"] = f"Bearer {api_key('openai')}"
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            else:
                data, mt = _image_b64(p["path"])
                content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}})
        body = {"model": self.model, "max_completion_tokens": max_tokens or self.ctx.config.get("llm.max_output_tokens"),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        res = check(s.post("https://api.openai.com/v1/chat/completions", json=body, timeout=900), "OpenAI")
        u = res.get("usage") or {}
        self.ctx.costs.add_tokens("openai", self.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                                  self.ctx.config.get("pricing.openai") or {}, note=self.role)
        return res["choices"][0]["message"]["content"] or ""


class GoogleLLM(LLM):
    provider = "google"

    @property
    def model(self) -> str:
        return self.ctx.config.get("llm.google_model")

    def complete(self, system, parts, max_tokens=None):
        s = session()
        cparts = []
        for p in parts:
            if p["type"] == "text":
                cparts.append({"text": p["text"]})
            else:
                data, mt = _image_b64(p["path"])
                cparts.append({"inline_data": {"mime_type": mt, "data": data}})
        body = {"system_instruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": cparts}],
                "generationConfig": {"maxOutputTokens": max_tokens or self.ctx.config.get("llm.max_output_tokens")}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        res = check(s.post(url, params={"key": api_key("google")}, json=body, timeout=900), "Gemini")
        u = res.get("usageMetadata") or {}
        self.ctx.costs.add_tokens("google", self.model, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), {},
                                  note=self.role)
        return "".join(p.get("text", "") for p in res["candidates"][0]["content"]["parts"])


def make_llm(ctx: Context, role: str) -> LLM | None:
    provider = ctx.config.provider("llm")
    return {"anthropic": AnthropicLLM, "openai": OpenAILLM, "google": GoogleLLM}.get(provider, lambda *a: None)(ctx, role)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a model response."""
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = text.find(opener), text.rfind(closer)
        if 0 <= a < b:
            candidates.append(text[a:b + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None
