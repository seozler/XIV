"""The Claude adapter builds a valid SDK request (mocked transport, no network)."""

import json

import cv2
import numpy as np
import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")  # transport used by anthropic>=1.0

from refanalyzer.config import Config  # noqa: E402
from refanalyzer.pipeline import Context  # noqa: E402
from refanalyzer.providers import llm  # noqa: E402
from refanalyzer.workspace import Workspace  # noqa: E402


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_anthropic_request_shape(tmp_path, monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [],
               "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1200, "output_tokens": 0}}
        events = [
            {"type": "message_start", "message": msg},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"ok": true}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 300}},
            {"type": "message_stop"},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(llm, "_anthropic_client", lambda: client)
    img = tmp_path / "k.jpg"
    cv2.imwrite(str(img), np.zeros((20, 30, 3), np.uint8))
    ws = Workspace(tmp_path, "v")
    ctx = Context(ws=ws, config=Config.load(), video_hash="h", stage="t")
    out = llm.AnthropicLLM(ctx, "vision").complete("sys", [{"type": "text", "text": "hi"}, {"type": "image", "path": str(img)}])
    assert llm.extract_json(out) == {"ok": True}
    body = seen["body"]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen["beta"]
    assert body["messages"][0]["content"][1]["type"] == "image"
    item = ctx.costs.items[0]
    assert item["input_tokens"] == 1200 and item["output_tokens"] == 300
    assert item["total_usd"] == pytest.approx(1200 / 1e6 * 2 + 300 / 1e6 * 10)


def test_extract_json_variants():
    assert llm.extract_json('text ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert llm.extract_json('prefix {"tasks": [1, 2]} suffix') == {"tasks": [1, 2]}
    assert llm.extract_json("no json here") is None
