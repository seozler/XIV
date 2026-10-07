"""Dialogue / music / effects separation adapters.

* ``musicai`` - Music AI (pay-per-use). Uses a workflow created in the
  Music AI dashboard (e.g. one built on the Cinematic Stems module); set
  ``musicai.workflow_dme`` and map its output names in ``musicai.stem_keys``.
* ``demucs``  - local Demucs (MIT). Splits vocals vs accompaniment only, so
  music and effects stay mixed together; flagged as such.
* ``none``    - no separation; downstream stages analyse the mix and lower
  their confidence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..config import api_key
from ..pipeline import Context
from .http import check, download, poll, session


class MusicAI:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.cfg = ctx.config.get("musicai")
        self.key = api_key("musicai")
        self.s = session()
        self.s.headers["Authorization"] = self.key

    def upload(self, path: Path) -> str:
        data = check(self.s.get(f"{self.cfg['base_url']}/upload", timeout=60), "Music AI upload URL")
        with path.open("rb") as fh:
            r = self.s.put(data["uploadUrl"], data=fh, timeout=600)
        check(r, "Music AI upload")
        return data["downloadUrl"]

    def run_workflow(self, workflow: str, input_url: str, name: str) -> dict:
        job = check(self.s.post(f"{self.cfg['base_url']}/job", json={
            "name": name, "workflow": workflow, "params": {"inputUrl": input_url}}, timeout=60), "Music AI job")
        jid = job["id"]
        res = poll(lambda: check(self.s.get(f"{self.cfg['base_url']}/job/{jid}", timeout=60), "Music AI status"),
                   lambda d: d.get("status") == "SUCCEEDED",
                   lambda d: d.get("error") or ("job failed" if d.get("status") == "FAILED" else None),
                   self.cfg["poll_seconds"], self.cfg["timeout_seconds"], "Music AI job")
        return res.get("result") or {}


def separate(ctx: Context, provider: str, mix: Path, duration_min: float) -> dict:
    out_dir = ctx.ws.path("audio")
    if provider == "musicai":
        mai = MusicAI(ctx)
        wf = ctx.config.get("musicai.workflow_dme")
        if not wf:
            raise RuntimeError("musicai.workflow_dme is not set (create a dialogue/music/effects workflow "
                               "in the Music AI dashboard and put its slug in refanalyzer.toml)")
        url = mai.upload(mix)
        result = mai.run_workflow(wf, url, f"refanalyzer-{ctx.ws.video_id}")
        stems = {}
        for stem, key in ctx.config.get("musicai.stem_keys").items():
            if key in result and isinstance(result[key], str) and result[key].startswith("http"):
                raw = download(mai.s, result[key], out_dir / f"{stem}.raw")
                stems[stem] = _to_wav(raw, out_dir / f"{stem}.wav")
        ctx.costs.add("musicai", "dialogue/music/effects separation", duration_min, "minute",
                      ctx.config.get("pricing.musicai_dme_per_min"))
        missing = [k for k in ctx.config.get("musicai.stem_keys") if k not in stems]
        return {"stems": stems, "kind": "dme", "missing": missing, "raw_result_keys": sorted(result)}
    if provider == "demucs":
        tmp = ctx.ws.path("cache/demucs")
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(tmp), str(mix)],
                       check=True, capture_output=True)
        base = next(tmp.glob(f"htdemucs/{mix.stem}"))
        stems = {"dialogue": _to_wav(base / "vocals.wav", out_dir / "dialogue.wav"),
                 "music_and_effects": _to_wav(base / "no_vocals.wav", out_dir / "music_and_effects.wav")}
        return {"stems": stems, "kind": "vocals/accompaniment", "missing": ["music", "effects"]}
    return {"stems": {}, "kind": "none", "missing": ["dialogue", "music", "effects"]}


def _to_wav(src: Path, dst: Path) -> Path:
    from .. import media

    if src.suffix.lower() == ".wav" and src == dst:
        return dst
    media.run([media.require("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(src), "-ac", "2", "-ar", "44100",
               "-c:a", "pcm_s16le", str(dst)])
    if src.suffix == ".raw":
        src.unlink(missing_ok=True)
    return dst
