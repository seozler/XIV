"""Analysis workspace: output layout, content hashing and the reuse index.

Layout (one directory per reference video, keyed by content hash so the
same file is never analysed twice)::

    <root>/<video_id>/
      input/      reference video + current-video-system.txt
      metadata/   video.json, frames.json, scenes.json, manifest.json
      visual/     frames/, ocr.json, objects.json, motion.json, colors.json, design.json
      audio/      stems, transcript.json, speech_style.json, loudness.json, beats.json, sfx.json
      combined/   timeline.json, synchronization.json, evidence.json, slices.json
      reports/    reference-analysis.md, system-comparison.md, improvement-tasks.json, costs.json
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import read_json, write_json

SUBDIRS = ["input", "metadata", "visual", "visual/frames", "visual/evidence", "audio", "combined", "reports", "cache"]


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class Workspace:
    def __init__(self, root: str | os.PathLike, video_id: str):
        self.root = Path(root)
        self.video_id = video_id
        self.dir = self.root / video_id
        for sub in SUBDIRS:
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @classmethod
    def for_video(cls, root: str | os.PathLike, video: str | os.PathLike) -> tuple["Workspace", str]:
        digest = sha256_file(video)
        ws = cls(root, f"ref_{digest[:12]}")
        return ws, digest

    @classmethod
    def open(cls, root: str | os.PathLike, video_id: str) -> "Workspace":
        if not (Path(root) / video_id).is_dir():
            raise FileNotFoundError(f"no analysis workspace {video_id!r} under {root}")
        return cls(root, video_id)

    def path(self, rel: str) -> Path:
        return self.dir / rel

    def read(self, rel: str, default: Any = None) -> Any:
        return read_json(self.path(rel), default)

    def write(self, rel: str, data: Any, **kw: Any) -> Path:
        return write_json(self.path(rel), data, **kw)

    def rel(self, path: str | os.PathLike) -> str:
        return os.path.relpath(path, self.dir)

    # -- inputs -----------------------------------------------------------
    def import_input(self, src: str | os.PathLike, name: str | None = None) -> Path:
        src = Path(src)
        dst = self.path(f"input/{name or src.name}")
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return dst
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return dst

    @property
    def video_path(self) -> Path:
        meta = self.read("metadata/video.json") or {}
        name = meta.get("input_file")
        if name and self.path(f"input/{name}").exists():
            return self.path(f"input/{name}")
        for p in sorted(self.path("input").iterdir()):
            if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
                return p
        raise FileNotFoundError("reference video not found in workspace input/")

    @property
    def system_spec_path(self) -> Path | None:
        p = self.path("input/current-video-system.txt")
        return p if p.exists() else None

    # -- manifest ---------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        return self.read("metadata/manifest.json", {"stages": {}})

    def costs(self) -> dict[str, Any]:
        """Aggregate the per-stage API cost records kept in the manifest."""
        items = []
        for name, rec in self.manifest().get("stages", {}).items():
            for it in rec.get("costs") or []:
                items.append({**it, "stage": name})
        known = [i["total_usd"] for i in items if i.get("total_usd") is not None]
        return {"items": items, "known_total_usd": round(sum(known), 6),
                "unpriced_items": sum(1 for i in items if i.get("total_usd") is None),
                "excluded": ["CPU server time", "storage", "network egress", "retried failed calls"]}

    def update_manifest(self, stage: str, record: dict[str, Any]) -> None:
        with self._lock:
            man = self.manifest()
            man.setdefault("stages", {})[stage] = record
            man["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.write("metadata/manifest.json", man)


class Library:
    """Index over every analysed reference, so later runs can reuse results."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.index_path = self.root / "index.json"

    def load(self) -> dict[str, Any]:
        return read_json(self.index_path, {"videos": {}})

    def register(self, video_id: str, entry: dict[str, Any]) -> None:
        idx = self.load()
        prev = idx["videos"].get(video_id, {})
        prev.update(entry)
        prev["updated_at"] = datetime.now(timezone.utc).isoformat()
        idx["videos"][video_id] = prev
        write_json(self.index_path, idx)

    def list(self) -> list[dict[str, Any]]:
        idx = self.load()
        return [dict(video_id=k, **v) for k, v in sorted(idx["videos"].items(), key=lambda kv: kv[1].get("updated_at", ""))]


class CostTracker:
    """Accumulates paid API usage so every run ends with a cost breakdown."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, provider: str, item: str, quantity: float, unit: str, unit_price: float | None,
            note: str = "") -> None:
        total = None if unit_price is None else quantity * unit_price
        with self._lock:
            self.items.append({
                "provider": provider, "item": item, "quantity": quantity, "unit": unit,
                "unit_price_usd": unit_price, "total_usd": total, "note": note,
            })

    def add_tokens(self, provider: str, model: str, input_tokens: int, output_tokens: int,
                   price_table: dict[str, list[float]], note: str = "") -> None:
        price = price_table.get(model)
        total = None
        if price:
            total = input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]
        with self._lock:
            self.items.append({
                "provider": provider, "item": model, "quantity": input_tokens + output_tokens,
                "unit": "tokens", "input_tokens": input_tokens, "output_tokens": output_tokens,
                "unit_price_usd": None, "total_usd": total, "note": note,
            })

    def summary(self) -> dict[str, Any]:
        known = [i["total_usd"] for i in self.items if i["total_usd"] is not None]
        return {
            "items": self.items,
            "known_total_usd": round(sum(known), 6),
            "unpriced_items": sum(1 for i in self.items if i["total_usd"] is None),
            "excluded": ["CPU server time", "storage", "network egress", "retried failed calls"],
        }
