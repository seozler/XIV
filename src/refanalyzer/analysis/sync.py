"""Stage: sync - audio/visual temporal correlation.

For each pairing rule (beat<->cut, whoosh<->entrance, spoken word<->text
appearance, ...) every visual event is matched to the nearest audio event
within a tolerance window. Reported per rule: hit rate, signed deltas
(visual minus audio: negative = picture leads sound), and a chance baseline
so a "pattern" is only claimed when hits clearly exceed what random timing
would produce (one-sided binomial test).
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from scipy.stats import binomtest

from ..models import INFERRED, OBSERVED, prov
from ..pipeline import Context


def _norm(w: str) -> str:
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", w).lower())


def visual_events(ws) -> list[dict]:
    ev: list[dict] = []
    sc = ws.read("metadata/scenes.json") or {}
    for c in sc.get("cuts", []):
        ev.append({"kind": "cut", "t": c["t"], "ref": f"cut@{c['t']:.3f}"})
    for f in sc.get("fades", []) + sc.get("dissolves", []):
        if not f["type"].endswith("_hold"):
            ev.append({"kind": "transition", "t": f["start"], "ref": f["type"]})
    for el in (ws.read("visual/objects.json") or {}).get("elements", []):
        a = el["animation"]
        if a.get("start_s") is None:
            continue
        role = el["role"]
        ev.append({"kind": {"entrance": "entrance", "exit": "exit", "move": "move"}.get(role, "transient"),
                   "t": a["start_s"], "ref": el["id"], "labels": a["labels"]})
        if a.get("end_s") is not None and role in ("entrance", "move"):
            ev.append({"kind": "settle", "t": a["end_s"], "ref": el["id"]})
        sc_curve = a["curves"].get("scale")
        if sc_curve and sc_curve["measured"].get("overshoot_ratio", 0) > 0.02:
            ev.append({"kind": "emphasis", "t": sc_curve["measured"]["peak_velocity_at_s"], "ref": el["id"]})
    for tr in (ws.read("visual/ocr.json") or {}).get("tracks", []):
        ev.append({"kind": "text_appear", "t": tr["start"], "ref": tr["id"], "text": tr["content"],
                   "precise": tr.get("linked_element") is not None})
    for m in (ws.read("visual/motion.json") or {}).get("camera_moves", []):
        ev.append({"kind": "camera_move", "t": m["start"], "ref": f"{m['type']}_{m['direction']}"})
    return sorted(ev, key=lambda e: e["t"])


def audio_events(ws) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    b = ws.read("audio/beats.json") or {}
    out["beat"] = [{"t": t} for t in b.get("beats") or []]
    out["downbeat"] = [{"t": t} for t in b.get("downbeats_4_4") or []]
    out["music_section"] = [{"t": t} for t in b.get("sections") or []]
    for e in (ws.read("audio/sfx.json") or {}).get("events", []):
        out.setdefault(f"sfx_{e['type']}", []).append({"t": e["start"], "conf": e["confidence"]})
    out["sfx_any"] = [x for k, v in out.items() if k.startswith("sfx_") for x in v]
    tr = ws.read("audio/transcript.json") or {}
    words = tr.get("words") or []
    out["word"] = [{"t": w["start"], "word": w["word"]} for w in words]
    segs = []
    for w in words:
        if segs and w["start"] - segs[-1]["end"] < 0.35:
            segs[-1]["end"] = w["end"]
        else:
            segs.append({"t": w["start"], "end": w["end"]})
    out["speech_onset"] = segs
    out["pause_end"] = [{"t": b2["t"], "pause": b2["t"] - a["end"]} for a, b2 in zip(segs, segs[1:]) if b2["t"] - a["end"] > 0.5]
    return out


RULES = [
    ("cut", "beat", "Cuts land on music beats (beat-synchronised editing)"),
    ("cut", "downbeat", "Cuts land on downbeats / bar lines"),
    ("cut", "music_section", "Scene changes follow musical section changes"),
    ("entrance", "beat", "Element entrances start on beats"),
    ("settle", "beat", "Elements settle on beats"),
    ("emphasis", "beat", "Scale emphasis (overshoot) on beats"),
    ("entrance", "sfx_whoosh", "Whoosh accompanies element entrances"),
    ("move", "sfx_whoosh", "Whoosh accompanies element moves"),
    ("cut", "sfx_whoosh", "Whoosh used as transition sound"),
    ("transition", "sfx_whoosh", "Whoosh accompanies fades/dissolves"),
    ("entrance", "sfx_pop", "Pop accompanies element entrances"),
    ("settle", "sfx_impact", "Impact accents an element landing"),
    ("emphasis", "sfx_impact", "Impact on scale emphasis"),
    ("entrance", "sfx_any", "Any SFX on element entrances"),
    ("entrance", "speech_onset", "Visuals enter as a new phrase starts"),
    ("cut", "speech_onset", "Cuts align with phrase starts"),
    ("cut", "pause_end", "Cuts placed in narration pauses"),
]


def match(visual: list[dict], audio: list[dict], tol: float) -> tuple[list[dict], int]:
    at = np.array([a["t"] for a in audio])
    hits = []
    for v in visual:
        if not len(at):
            break
        i = int(np.argmin(np.abs(at - v["t"])))
        d = v["t"] - at[i]
        if abs(d) <= tol:
            hits.append({"visual": v["ref"], "visual_t": v["t"], "audio_t": float(at[i]), "delta_ms": float(d * 1000)})
    return hits, len(visual)


def chance_rate(audio: list[dict], tol: float, duration: float) -> float:
    """Probability a random instant falls within tol of some audio event."""
    if not audio or duration <= 0:
        return 0.0
    ts = np.sort([a["t"] for a in audio])
    covered = 0.0
    lo = hi = None
    for t in ts:
        a, b = max(0.0, t - tol), min(duration, t + tol)
        if hi is None or a > hi:
            if hi is not None:
                covered += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    if hi is not None:
        covered += hi - lo
    return min(1.0, covered / duration)


def text_speech_sync(ws, window: float = 4.0) -> list[dict]:
    """On-screen words vs the moment they are spoken."""
    words = (ws.read("audio/transcript.json") or {}).get("words") or []
    out = []
    if not words:
        return out
    for tr in (ws.read("visual/ocr.json") or {}).get("tracks", []):
        toks = [_norm(t) for t in tr["content"].split() if len(_norm(t)) >= 3]
        for tok in toks:
            cands = [w for w in words if _norm(w["word"]) == tok and abs(w["start"] - tr["start"]) <= window]
            if not cands:
                continue
            w = min(cands, key=lambda w: abs(w["start"] - tr["start"]))
            out.append({"text_track": tr["id"], "token": tok, "text_appear_s": tr["start"], "spoken_s": w["start"],
                        "delta_ms": (tr["start"] - w["start"]) * 1000, "timing_precise": bool(tr.get("linked_element"))})
            break
    return out


def run(ctx: Context) -> dict:
    ws = ctx.ws
    dur = float(ws.read("metadata/video.json")["duration_s"])
    tol = float(ctx.config.get("analysis.sync_tolerance_ms")) / 1000
    vis = visual_events(ws)
    aud = audio_events(ws)
    patterns = []
    pairs = []
    for vk, ak, desc in RULES:
        v = [e for e in vis if e["kind"] == vk]
        a = aud.get(ak, [])
        if len(v) < 2 or not a:
            continue
        hits, n = match(v, a, tol)
        p0 = chance_rate(a, tol, dur)
        k = len(hits)
        pval = float(binomtest(k, n, p0, alternative="greater").pvalue) if 0 < p0 < 1 else (0.0 if k else 1.0)
        deltas = [h["delta_ms"] for h in hits]
        rec = {
            "rule": f"{vk}~{ak}", "description": desc, "visual_events": n, "matched": k,
            "hit_rate": k / n, "chance_rate": p0, "lift": (k / n) / p0 if p0 > 0 else None, "p_value": pval,
            "delta_ms": {"median": float(np.median(deltas)), "mean": float(np.mean(deltas)),
                         "sd": float(np.std(deltas))} if deltas else None,
            "pattern_detected": bool(k >= 2 and pval < 0.05 and k / n >= 0.4),
            "examples": hits[:8],
        }
        pairs += [{**h, "rule": rec["rule"]} for h in hits]
        patterns.append(rec)
    ts_sync = text_speech_sync(ws)
    if ts_sync:
        d = [x["delta_ms"] for x in ts_sync]
        patterns.append({"rule": "text_appear~spoken_word", "description": "On-screen text vs the spoken word",
                         "visual_events": len(ts_sync), "matched": len(ts_sync), "hit_rate": 1.0,
                         "delta_ms": {"median": float(np.median(d)), "mean": float(np.mean(d)), "sd": float(np.std(d))},
                         "pattern_detected": len(ts_sync) >= 2 and float(np.std(d)) < 700,
                         "examples": ts_sync[:10]})
    # Pacing: does the picture hold still during narration pauses?
    fm = ws.read("visual/frame_metrics.json")["columns"]
    t = np.asarray(fm["t"])
    res = np.asarray(fm["residual_frac"])
    pause_motion = []
    for p in aud.get("pause_end", []):
        sel = (t >= p["t"] - p["pause"]) & (t < p["t"])
        if sel.any():
            pause_motion.append(float(res[sel].mean()))
    pacing = {
        "visual_events_per_minute": len(vis) / (dur / 60) if dur else None,
        "mean_motion_during_pauses": float(np.mean(pause_motion)) if pause_motion else None,
        "mean_motion_overall": float(res.mean()) if len(res) else None,
    }
    ws.write("combined/synchronization.json", {
        "tolerance_ms": tol * 1000, "patterns": patterns, "pairs": pairs, "text_speech": ts_sync, "pacing": pacing,
        "provenance": {"matching": prov("numpy", "nearest-event matching within tolerance", 0.85, OBSERVED),
                       "patterns": prov("scipy.stats.binomtest", "hit rate vs chance coverage of audio events",
                                        0.7, INFERRED)},
    })
    return {"outputs": ["combined/synchronization.json"],
            "notes": [f"{sum(p['pattern_detected'] for p in patterns)} sync patterns detected"]}
