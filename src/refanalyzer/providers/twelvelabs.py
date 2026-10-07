"""Twelve Labs Pegasus adapter (video-native semantic analysis).

Flow (API v1.3): find or create an index with the Pegasus model, upload the
video as an indexing task, wait until it is ready, then call ``/analyze``
with a prompt. Pegasus output is semantic and timestamped but is not used
for millisecond-level animation parameters.
"""

from __future__ import annotations

from pathlib import Path

from ..config import api_key
from ..pipeline import Context
from .http import check, poll, session


def analyze(ctx: Context, video: Path, prompt: str, duration_h: float) -> dict:
    cfg = ctx.config.get("twelvelabs")
    base = cfg["base_url"]
    s = session()
    s.headers["x-api-key"] = api_key("twelvelabs")
    idx = check(s.get(f"{base}/indexes", params={"index_name": cfg["index_name"]}, timeout=60), "Twelve Labs indexes")
    data = idx.get("data") or []
    if data:
        index_id = data[0]["_id"]
    else:
        created = check(s.post(f"{base}/indexes", json={
            "index_name": cfg["index_name"],
            "models": [{"model_name": cfg["model"], "model_options": ["visual", "audio"]}]}, timeout=60),
            "Twelve Labs create index")
        index_id = created["_id"]
    with video.open("rb") as fh:
        task = check(s.post(f"{base}/tasks", data={"index_id": index_id}, files={"video_file": fh}, timeout=1800),
                     "Twelve Labs upload")
    done = poll(lambda: check(s.get(f"{base}/tasks/{task['_id']}", timeout=60), "Twelve Labs task"),
                lambda d: d.get("status") == "ready",
                lambda d: d.get("status") if d.get("status") == "failed" else None, 10, 3600, "Twelve Labs indexing")
    res = check(s.post(f"{base}/analyze", json={"video_id": done["video_id"], "prompt": prompt, "temperature": 0.2,
                                                "stream": False}, timeout=900), "Twelve Labs analyze")
    out_tokens = (res.get("usage") or {}).get("output_tokens", 0)
    ctx.costs.add("twelvelabs", "pegasus video analysis", duration_h, "video hour",
                  ctx.config.get("pricing.twelvelabs_per_video_hour"))
    ctx.costs.add("twelvelabs", "pegasus output tokens", out_tokens / 1e6, "M tokens",
                  ctx.config.get("pricing.twelvelabs_output_per_mtok"))
    return {"provider": "twelvelabs", "model": cfg["model"], "index_id": index_id, "video_id": done["video_id"],
            "text": res.get("data") or res.get("text") or "", "usage": res.get("usage")}
